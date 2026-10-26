# Required import for inheriting from Heuristic
from heuristics.heuristic_base import Heuristic
# Used for absolute value calculation
import math

# Helper functions used by the heuristic class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    # Returns a list of strings [predicate, arg1, arg2]
    return fact[1:-1].split()

def get_coords(location_name):
    """Parses 'loc_R_C' string into (R, C) tuple."""
    # Assumes location names are always in 'loc_R_C' format based on examples.
    # R is row, C is column.
    try:
        parts = location_name.split('_')
        # Check for expected format: starts with 'loc', has 3 parts, last two are digits
        if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
             # Convert row and column parts to integers
             return (int(parts[1]), int(parts[2]))
        else:
             # If the format is unexpected, raise a ValueError.
             # This helps diagnose issues with the input PDDL instance.
             raise ValueError(f"Unexpected location format: {location_name}")
    except (ValueError, IndexError) as e:
         # Catch potential errors from split or int conversion
         raise ValueError(f"Could not parse location coordinates from '{location_name}': {e}")


def sign(x):
    """Returns the sign of x: -1 for negative, 1 for positive, 0 for zero."""
    return (x > 0) - (x < 0)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each misplaced box,
    the Manhattan distance from the box's current location to its goal location
    and the Manhattan distance from the robot's current location to a position
    adjacent to the box from which it could push the box towards its goal.

    # Assumptions
    - Locations are on a grid and can be parsed from 'loc_R_C' format, where R is the row and C is the column.
    - The cost of moving the robot or pushing a box one step is 1.
    - The heuristic ignores dynamic obstacles (other boxes, robot position affecting clear)
      and static obstacles (walls implicitly defined by lack of adjacent facts)
      for distance calculations, making it non-admissible but fast.
    - For a box needing to move diagonally (both row and column differ from goal),
      the robot's approach cost is the minimum distance to a position allowing
      either a vertical or horizontal push towards the goal.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Static facts (like 'adjacent' or 'clear' in the initial state) are not explicitly used
      beyond identifying the grid structure implicitly via location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location string from the state. Parse its coordinates (r_r, c_r).
    2. Initialize total heuristic value to 0.
    3. For each box that has a goal location defined in the task:
       a. Find the box's current location string from the state.
       b. If the box's current location is the same as its goal location, continue to the next box (contribution is 0).
       c. If the box is misplaced:
          i. Parse the box's current location coordinates (r_b, c_b) and its goal location coordinates (r_g, c_g).
          ii. Calculate the box's Manhattan distance to the goal: `box_dist = abs(r_b - r_g) + abs(c_b - c_g)`. This represents the minimum number of pushes needed for the box itself, ignoring obstacles.
          iii. Calculate the robot's estimated distance to a suitable push position:
              - Determine the set of potential locations adjacent to the box from which the robot could initiate a push that moves the box towards its goal.
              - If the box needs to move vertically (r_b != r_g), a potential push start location for a vertical push is (r_b - sign(r_g - r_b), c_b).
              - If the box needs to move horizontally (c_b != c_g), a potential push start location for a horizontal push is (r_b, c_b - sign(c_g - c_b)).
              - If both vertical and horizontal movement are needed (r_b != r_g and c_b != c_g), both calculated locations are potential first push starts.
              - Calculate the Manhattan distance from the robot's current location (r_r, c_r) to each identified potential push start location.
              - The `robot_approach_dist` is the minimum of these calculated distances.
          iv. Add `box_dist + robot_approach_dist` to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_box_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal facts are typically (at boxX loc_Y_Z)
            # Ensure the fact is 'at', has 2 arguments, and the first argument starts with 'box'
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_box_locations[box] = location
            # Ignore other types of goal predicates if any exist.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot and box locations in the current state.
        robot_location = None
        current_box_locations = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot" and len(args) == 1:
                robot_location = args[0]
            elif predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                current_box_locations[box] = location

        # If robot location is not found, the state is likely invalid or not reachable.
        # Returning infinity is a common practice for unreachable/invalid states in heuristics.
        if robot_location is None:
             # This should ideally not happen in valid states generated by the planner.
             return float('inf')

        # Parse robot coordinates
        try:
            r_r, c_r = get_coords(robot_location)
        except ValueError as e:
            # Handle error if robot location format is unexpected
            print(f"Error parsing robot location coordinates: {e}")
            return float('inf')


        total_heuristic = 0

        # Calculate heuristic for each box that has a goal defined in the task
        for box, goal_location in self.goal_box_locations.items():
            current_location = current_box_locations.get(box)

            # If box is not found in the current state (shouldn't happen for objects defined)
            # or is already at its goal location, its contribution is 0.
            if current_location is None or current_location == goal_location:
                continue

            # Parse box and goal coordinates
            try:
                r_b, c_b = get_coords(current_location)
                r_g, c_g = get_coords(goal_location)
            except ValueError as e:
                 # Handle error if location format is unexpected for current or goal location
                 print(f"Error parsing box or goal location coordinates for {box}: {e}")
                 return float('inf')


            # 1. Box distance (minimum pushes)
            # This is the Manhattan distance on the grid.
            box_dist = abs(r_b - r_g) + abs(c_b - c_g)

            # 2. Robot distance to a suitable push position
            # The robot needs to reach a location adjacent to the box (r_b, c_b)
            # from which it can push towards the goal (r_g, c_g).

            potential_push_starts_coords = []

            # Determine potential push directions based on goal relative to box
            dr = sign(r_g - r_b) # -1 if goal is up, 1 if goal is down, 0 if same row
            dc = sign(c_g - c_b) # -1 if goal is left, 1 if goal is right, 0 if same column

            # If box needs to move vertically (dr is not 0)
            if dr != 0:
                # Robot must be on the opposite side vertically: row r_b - dr
                target_r = r_b - dr
                target_c = c_b
                potential_push_starts_coords.append((target_r, target_c))

            # If box needs to move horizontally (dc is not 0)
            if dc != 0:
                # Robot must be on the opposite side horizontally: column c_b - dc
                target_r = r_b
                target_c = c_b - dc
                potential_push_starts_coords.append((target_r, target_c))

            # If the box is not at the goal (checked by `continue` above),
            # then either dr != 0 or dc != 0 (or both), so potential_push_starts_coords
            # will contain at least one coordinate pair.

            robot_approach_dist = float('inf') # Initialize with infinity

            # Calculate minimum robot Manhattan distance to any potential push start location
            for target_r, target_c in potential_push_starts_coords:
                 dist = abs(r_r - target_r) + abs(c_r - target_c)
                 robot_approach_dist = min(robot_approach_dist, dist)

            # If robot_approach_dist is still infinity, it means potential_push_starts_coords was empty,
            # which implies box_dist was 0, but we already handled that case.
            # This check is mostly for safety against unexpected logic paths.
            if robot_approach_dist == float('inf'):
                 # This should logically not be reached if box_dist > 0.
                 # Return infinity to indicate a problem.
                 # print(f"Internal Warning: Could not determine valid push position candidates for box {box}")
                 return float('inf')


            # Add the combined distance for this box to the total heuristic
            total_heuristic += box_dist + robot_approach_dist

        # The total heuristic is the sum of costs for all misplaced boxes.
        # If all boxes are at their goals, the loop finishes with total_heuristic = 0.
        return total_heuristic
