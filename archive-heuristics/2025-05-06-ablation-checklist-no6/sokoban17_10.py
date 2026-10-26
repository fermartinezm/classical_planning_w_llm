from fnmatch import fnmatch
# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C).
       Returns None if parsing fails or format is unexpected."""
    try:
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Location string does not follow the expected 'loc_R_C' format
            return None
    except (ValueError, IndexError):
        # Parts could not be converted to integers or split failed
        return None

def make_location_str(r, c):
     """Reconstructs a location string from row and column."""
     return f'loc_{r}_{c}'

def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two location strings."""
    coords1 = parse_location(loc1_str)
    coords2 = parse_location(loc2_str)
    if coords1 is None or coords2 is None:
        # This indicates a problem with location naming convention
        # Return infinity as distance cannot be computed
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The sum of Manhattan distances for each box from its current location
       to its goal location. This estimates the minimum number of pushes required
       for all boxes combined, assuming no obstacles and perfect robot positioning.
    2. The minimum Manhattan distance from the robot's current location to
       any of the required push locations for the first step towards the goal
       for any box that needs moving. This estimates the cost for the robot
       to get into position to start moving the closest box.

    # Assumptions
    - Location names follow the 'loc_R_C' format allowing Manhattan distance calculation.
    - The primary cost is moving boxes to goals, and the robot must assist.
    - The heuristic is non-admissible; it ignores obstacles (other boxes, walls)
      and complex robot repositioning needed for subsequent pushes or different boxes.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task definition.

    # Step-by-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, heuristic is 0.
    2. Identify the current location of the robot and all relevant boxes (those with goals).
    3. Initialize total box distance and minimum robot-to-push-position distance to 0 and infinity, respectively.
    4. For each box that has a goal location:
       a. If the box is not at its goal location:
          i. Mark that at least one box needs moving.
          ii. Calculate the Manhattan distance from the box's current location to its goal location. Add this to the total box distance.
          iii. Determine the coordinates of the location(s) where the robot must be positioned to make the *first* push towards the goal (based on Manhattan distance path). There might be one (if moving purely horizontal or vertical) or two (if moving diagonally).
          iv. Calculate the Manhattan distance from the robot's current location to each of these required push locations.
          v. Update the minimum robot-to-push-position distance found so far across all boxes that need moving.
    5. The robot's contribution to the heuristic is the minimum distance calculated in step 4.v (if any box needed moving, otherwise 0).
    6. The total heuristic value is the sum of the total box distance (step 4.ii sum) and the robot's contribution (step 5).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not explicitly used for graph traversal in this Manhattan distance heuristic,
        # but could be used to validate location names if needed.
        # self.static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically '(at box loc)'
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
            # Handle potential other goal types if necessary, but 'at' is standard for box goals.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Check if goal is reached
        # The goal is a frozenset of facts. Check if all goal facts are in the state.
        if self.goals <= state:
             return 0

        # Find current locations of robot and boxes
        robot_location = None
        box_locations = {} # Map box name to its location string

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                 box, location = parts[1], parts[2]
                 # Only track boxes that are relevant to the goals
                 if box in self.goal_locations:
                    box_locations[box] = location
            # We don't need 'clear' facts for this heuristic calculation

        # If robot location is not found, something is wrong with the state representation.
        # Return infinity as heuristic cannot be computed reliably.
        if robot_location is None:
             # print("Warning: Robot location not found in state.") # Optional warning
             return float('inf')

        # Parse robot location coordinates once
        robot_coords = parse_location(robot_location)
        if robot_coords is None:
             # print(f"Warning: Could not parse robot location '{robot_location}'.") # Optional warning
             return float('inf')


        total_box_distance = 0
        min_robot_to_any_push_loc = float('inf')
        any_box_needs_moving = False

        # Iterate through the boxes that have a goal location
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If a box with a goal is not found in the state, it's an invalid state representation.
            # Return infinity as heuristic cannot be computed.
            if current_location is None:
                 # print(f"Warning: Goal box {box} not found in state.") # Optional warning
                 return float('inf')

            if current_location != goal_location:
                any_box_needs_moving = True

                # Calculate Manhattan distance for this box
                box_dist = manhattan_distance(current_location, goal_location)
                if box_dist == float('inf'):
                     # Location parsing failed for box or goal
                     return float('inf')
                total_box_distance += box_dist

                # Calculate required push locations for the first step towards goal
                current_coords = parse_location(current_location)
                goal_coords = parse_location(goal_location)

                # This check is technically redundant if manhattan_distance already checked,
                # but good for clarity.
                if current_coords is None or goal_coords is None:
                     return float('inf')

                r_b, c_b = current_coords
                r_g, c_g = goal_coords
                push_locs_coords = [] # Store coordinates of potential push locations

                # If vertical movement is needed towards the goal
                if r_b != r_g:
                    # Robot must be on the opposite side of the box from the goal row
                    r_push = r_b + (1 if r_g > r_b else -1) # +1 if goal is below, -1 if goal is above
                    push_locs_coords.append((r_push, c_b))

                # If horizontal movement is needed towards the goal
                if c_b != c_g:
                    # Robot must be on the opposite side of the box from the goal column
                    c_push = c_b + (1 if c_g > c_b else -1) # +1 if goal is right, -1 if goal is left
                    push_locs_coords.append((r_b, c_push))

                # Calculate minimum distance from robot to any of these potential first-push locations for this box
                min_md_robot_to_this_box_push_loc = float('inf')

                for push_coords in push_locs_coords:
                    push_loc_str = make_location_str(*push_coords)
                    # Calculate distance assuming the location string is valid for md function
                    dist = manhattan_distance(robot_location, push_loc_str)
                    # manhattan_distance returns inf if push_loc_str is not parseable,
                    # which is the desired behavior here.
                    min_md_robot_to_this_box_push_loc = min(min_md_robot_to_this_box_push_loc, dist)

                # Update the overall minimum robot distance to any first-push location across all boxes
                if min_md_robot_to_this_box_push_loc != float('inf'):
                    min_robot_to_any_push_loc = min(min_robot_to_any_push_loc, min_md_robot_to_this_box_push_loc)


        # If no boxes need moving, the robot contribution is 0.
        # Otherwise, it's the minimum distance to get into position for the first push of any box.
        robot_contribution = min_robot_to_any_push_loc if any_box_needs_moving else 0

        # The heuristic is the sum of box distances (minimum pushes) plus the
        # cost for the robot to get into position for the first push of the
        # most easily reachable box that needs moving.
        # This is a non-admissible estimate designed for greedy search.
        return total_box_distance + robot_contribution
