# Import necessary modules
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format, maybe log a warning or raise an error
        # For robustness, return empty list or handle in caller
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
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """
    Parses a location string like 'loc_row_col' into a (row, col) tuple of integers.
    Assumes the format is always 'loc_int_int'.
    Returns None if parsing fails.
    """
    if not isinstance(loc_str, str):
        return None
    try:
        parts = loc_str.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            return (int(parts[1]), int(parts[2]))
        else:
            return None
    except ValueError:
        return None

def manhattan_distance(loc1_tuple, loc2_tuple):
    """
    Calculates the Manhattan distance between two (row, col) tuples.
    Returns float('inf') if either input is None.
    """
    if loc1_tuple is None or loc2_tuple is None:
        return float('inf')
    r1, c1 = loc1_tuple
    r2, c2 = loc2_tuple
    return abs(r1 - r2) + abs(c1 - c2)

# Assuming Heuristic base class is imported from heuristics.heuristic_base
# class sokobanHeuristic(Heuristic): # Use this line in the actual planner environment

# Placeholder for the actual Heuristic base class import
# from heuristics.heuristic_base import Heuristic

# Define the heuristic class inheriting from Heuristic
class sokobanHeuristic: # Replace with 'class sokobanHeuristic(Heuristic):' in the actual planner environment
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each
    box not at its goal location, the Manhattan distance from the box to its
    goal plus the minimum Manhattan distance from the robot to a position
    adjacent to the box from which it can be pushed towards the goal.

    # Assumptions
    - Location names follow the format 'loc_row_col', allowing coordinate extraction.
    - Manhattan distance on these coordinates is a reasonable approximation
      of the actual path distance on the grid defined by 'adjacent' facts.
    - The heuristic ignores potential deadlocks (situations where a box
      cannot be moved further) and complex interactions between multiple boxes.
    - The cost of moving the robot to the *first* push position for each
      off-goal box is included, but subsequent robot repositioning costs
      for the same box or costs related to moving around other boxes are
      only partially captured by the Manhattan distance.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
      Assumes goal conditions are primarily of the form '(at ?b ?l)' for boxes.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location. Parse its string representation
       into (row, col) coordinates. If parsing fails, return infinity.
    2. Identify the current location for each box that has a specified goal
       location. Store as a dictionary {box_name: loc_str}.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each box for which a goal location is defined:
       a. Get the box's current location string from the state information.
          If the box is not found in the state, return infinity (invalid state).
       b. Parse the box's current location string and its goal location string
          into (row, col) coordinates. Return infinity if parsing fails for either.
       c. If the box's current coordinates are the same as its goal coordinates,
          its contribution to the heuristic is 0. Continue to the next box.
       d. If the box is not at its goal:
          i. Calculate the Manhattan distance between the box's current location
             and its goal location. This represents the minimum number of push
             actions required for this box. Add this distance to the box's
             contribution.
          ii. Determine the set of locations adjacent to the box from which
              it could be pushed one step towards its goal. If the box needs
              to move vertically (row changes), the robot must be vertically
              aligned but one step away on the opposite side. If the box needs
              to move horizontally (column changes), the robot must be horizontally
              aligned but one step away on the opposite side. If the box needs
              both vertical and horizontal movement, there are two such potential
              first-push locations (one for a vertical push, one for a horizontal push).
          iii. Calculate the minimum Manhattan distance from the robot's current
               location to any of the potential first-push locations identified
               in the previous step. If no push locations are determined (should
               not happen for an off-goal box with valid coordinates), return infinity.
          iv. Add this minimum robot distance to the box's contribution.
       e. Add the box's total contribution (box distance + minimum robot distance
          to first push position) to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # Call the base class constructor. In the actual planner environment,
        # this would be: super().__init__(task)
        # For standalone code generation, we simulate the necessary initialization:
        self.goals = task.goals
        self.static = task.static


        # Store goal locations for each box.
        # Assumes goal facts are like '(at box1 loc_2_4)'
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Assuming the first argument of 'at' in a goal is a box
                box_name = parts[1]
                goal_location_str = parts[2]
                self.goal_locations[box_name] = goal_location_str

        # Identify the set of box names we need to track based on goals
        self.box_names = set(self.goal_locations.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Identify robot's current location
        robot_location_str = None
        # 2. Identify current box locations
        box_current_locations = {} # {box_name: loc_str}

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_location_str = parts[1]
            elif parts and parts[0] == "at" and len(parts) == 3:
                 obj_name = parts[1]
                 loc_str = parts[2]
                 # Check if this object is one of the boxes we care about (i.e., has a goal)
                 if obj_name in self.box_names:
                     box_current_locations[obj_name] = loc_str

        # If robot location is not found, state is invalid
        if robot_location_str is None:
             return float('inf')

        robot_coords = parse_location(robot_location_str)
        if robot_coords is None:
             return float('inf') # Parsing failed

        total_heuristic = 0

        # 4. Iterate through each box with a goal
        for box_name, goal_location_str in self.goal_locations.items():
            current_location_str = box_current_locations.get(box_name)

            # If a box is not found in the state, state is invalid
            if current_location_str is None:
                 return float('inf')

            # 4b. Parse locations
            current_coords = parse_location(current_location_str)
            goal_coords = parse_location(goal_location_str)

            if current_coords is None or goal_coords is None:
                 return float('inf') # Parsing failed

            br, bc = current_coords
            gr, gc = goal_coords

            # 4c. If box is at goal, continue
            if (br, bc) == (gr, gc):
                continue

            # 4d. If box is not at goal:
            # i. Box distance (minimum pushes)
            box_dist = manhattan_distance(current_coords, goal_coords)
            box_contribution = box_dist

            # ii. Determine potential first push locations
            push_locations_coords = []
            # Needs vertical movement?
            if br > gr: # Needs UP push (robot must be below)
                push_locations_coords.append((br + 1, bc))
            elif br < gr: # Needs DOWN push (robot must be above)
                push_locations_coords.append((br - 1, bc))

            # Needs horizontal movement?
            if bc > gc: # Needs LEFT push (robot must be right)
                push_locations_coords.append((br, bc + 1))
            elif bc < gc: # Needs RIGHT push (robot must be left)
                push_locations_coords.append((br, bc - 1))

            # iii. Calculate minimum robot distance to a push location
            min_robot_dist_to_push = float('inf')

            # This list should not be empty if the box is not at the goal
            # and coordinates were parsed successfully.
            if not push_locations_coords:
                 # This indicates an unexpected condition, possibly an invalid goal/location setup
                 # or a box that cannot be pushed towards its goal based on coordinates.
                 # Treat as unsolvable from here.
                 return float('inf')

            for pr, pc in push_locations_coords:
                dist = manhattan_distance(robot_coords, (pr, pc))
                min_robot_dist_to_push = min(min_robot_dist_to_push, dist)

            # Add robot distance to box contribution
            box_contribution += min_robot_dist_to_push

            # 4e. Add box's total contribution to total heuristic
            total_heuristic += box_contribution

        # 5. Return total heuristic
        return total_heuristic
