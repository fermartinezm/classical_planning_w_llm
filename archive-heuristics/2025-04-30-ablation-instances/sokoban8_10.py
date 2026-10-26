from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, though PDDL facts should follow this
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for safety
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """Parses a location string like 'loc_row_col' into (row, col)."""
    try:
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        else:
            # Handle potential variations or errors
            raise ValueError(f"Unexpected location format: {loc_str}")
    except (ValueError, IndexError) as e:
        # print(f"Error parsing location string '{loc_str}': {e}")
        raise # Re-raise the exception after printing/logging

def manhattan_distance(loc1_str, loc2_str):
    """Calculates Manhattan distance between two location strings."""
    r1, c1 = parse_location(loc1_str)
    r2, c2 = parse_location(loc2_str)
    return abs(r1 - r2) + abs(c1 - c2)


# Define the heuristic class
# Inherit from Heuristic if the base class is provided and required by the framework
class sokobanHeuristic: # Or class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban puzzle by summing:
    1. The Manhattan distance for each box from its current location to its goal location.
    2. The minimum Manhattan distance from the robot's current location to any box that is not yet at its goal.

    # Assumptions
    - Locations are named in the format 'loc_row_col'.
    - Boxes are distinguishable and have specific goal locations defined in the task goals.
    - The cost of moving the robot to a box and pushing a box one step are roughly equivalent for heuristic purposes (unit cost).
    - The grid defined by 'adjacent' facts corresponds to Manhattan distances based on the 'loc_row_col' naming.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location by searching for the '(at-robot ?l)' fact in the state.
    2. Identify the current location of each box that has a goal location by searching for '(at ?b ?l)' facts in the state and checking against the known boxes from the goals.
    3. Initialize the total heuristic value `h` to 0.
    4. Create a list to keep track of boxes that are not yet at their goal locations.
    5. Iterate through each box for which a goal location is defined:
       - Get the box's current location and its goal location.
       - If the box's current location is different from its goal location:
         - Calculate the Manhattan distance between the box's current location and its goal location.
         - Add this distance to `h`. This component estimates the minimum number of push actions required for this box.
         - Add the box to the list of boxes that still need to be moved.
    6. After processing all boxes, check if there are any boxes remaining in the list of boxes that need to be moved.
    7. If there are boxes that still need to be moved:
       - Calculate the Manhattan distance from the robot's current location to each of these boxes.
       - Find the minimum of these distances.
       - Add this minimum distance to `h`. This component estimates the cost for the robot to reach the nearest box it needs to interact with.
    8. If all boxes were already at their goals (the list of boxes to move is empty), the heuristic remains 0.
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not directly used in this heuristic's calculation,
        # but the location naming convention derived from them is assumed.
        # self.static = task.static # Not strictly needed for this heuristic logic

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            # Assuming goal is (at boxX locY)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location
            # Note: This assumes goals are simple (at box loc) facts.
            # If goals can be complex (e.g., (and ...)), task.goals should provide the flattened set.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, the state is likely invalid or unreachable.
        # Return infinity to prune this branch.
        if robot_location is None:
             return float('inf')

        # Find current locations of all objects that are goal boxes
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 obj, loc = parts[1], parts[2]
                 # Only consider objects that are boxes with defined goals
                 if obj in self.box_goals:
                     box_locations[obj] = loc

        total_heuristic = 0
        boxes_not_at_goal = []

        # Calculate sum of Manhattan distances for boxes to their goals
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)

            # If a box with a goal is not found in the state, it's an invalid state.
            if current_loc is None:
                 return float('inf')

            if current_loc != goal_loc:
                try:
                    dist = manhattan_distance(current_loc, goal_loc)
                    total_heuristic += dist
                    boxes_not_at_goal.append(box)
                except ValueError:
                    # Handle parsing error if location format is unexpected
                    return float('inf') # Cannot calculate heuristic

        # If all boxes are at goal, heuristic is 0
        if not boxes_not_at_goal:
            return 0

        # Calculate minimum Manhattan distance from robot to a box that needs moving
        min_robot_to_box_dist = float('inf')
        try:
            for box in boxes_not_at_goal:
                box_loc = box_locations[box]
                dist = manhattan_distance(robot_location, box_loc)
                min_robot_to_box_dist = min(min_robot_to_box_dist, dist)
        except ValueError:
             # Handle parsing error
             return float('inf') # Cannot calculate heuristic

        # Add the minimum robot distance to the total heuristic
        total_heuristic += min_robot_to_box_dist

        return total_heuristic

