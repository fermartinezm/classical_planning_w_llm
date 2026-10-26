# Add necessary imports
from fnmatch import fnmatch
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic

# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
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
    Parses a location string like 'loc_X_Y' into a tuple (X, Y).
    Assumes the input string is just the location object name, e.g., 'loc_4_4'.
    """
    parts = loc_str.split('_')
    if len(parts) != 3 or parts[0] != 'loc':
        # This should not happen with valid PDDL location object names
        raise ValueError(f"Unexpected location format: {loc_str}")

    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # Handle cases where row/col parts are not integers
        raise ValueError(f"Invalid row or column in location string: {loc_str}")


def manhattan_distance(loc1_str, loc2_str):
    """
    Calculates Manhattan distance between two location strings.
    Assumes location strings are in 'loc_row_col' format.
    """
    try:
        r1, c1 = parse_location(loc1_str)
        r2, c2 = parse_location(loc2_str)
        return abs(r1 - r2) + abs(c1 - c2)
    except ValueError as e:
        # Re-raise with context
        raise ValueError(f"Could not calculate Manhattan distance between {loc1_str} and {loc2_str}: {e}")


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box from its current location to its goal location,
    and adding the Manhattan distance from the robot to the nearest box that
    is not yet at its goal.

    # Assumptions
    - The grid structure is implicitly defined by location names following
      the format 'loc_row_col', allowing Manhattan distance calculation.
    - The goal is defined primarily by the final locations of specific boxes.
    - This heuristic is non-admissible; it ignores obstacles (other boxes, walls),
      the specific position the robot needs to be in to push a box, and the
      requirement for the target location to be clear. It provides a simple
      estimate based on distances.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
      Stores this mapping in `self.box_goals`.
    - Static facts are available but not used in this simple version of the heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the current state (`node.state`) and the task definition (`node.task`).
    2. Check if the current state is the actual goal state using `node.task.goal_reached(state)`. If yes, return 0.
    3. Identify the current location of each box that is specified in the goal conditions from the state facts. Store these in `current_box_locations`.
    4. Calculate the sum of Manhattan distances for all boxes that are not yet at their goal locations:
       - Initialize `sum_box_distances` to 0.
       - Initialize an empty list `boxes_not_at_goal_locations` to store locations of boxes needing movement.
       - Iterate through the `self.box_goals` mapping (pre-computed in `__init__`).
       - For each box, get its current location from `current_box_locations` and its goal location from `self.box_goals`.
       - If the current location is different from the goal location:
         - Calculate the Manhattan distance between the current and goal locations.
         - Add this distance to `sum_box_distances`.
         - Add the box's current location to `boxes_not_at_goal_locations`.
       - Handle potential errors during distance calculation by returning infinity.
    5. Find the robot's current location from the state facts.
    6. Calculate the Manhattan distance from the robot's current location to the nearest box that is not at its goal:
       - Initialize `min_robot_dist_to_box` to infinity.
       - If the robot location is found and there are boxes not at their goals:
         - Iterate through the locations in `boxes_not_at_goal_locations`.
         - For each box location, calculate the Manhattan distance from the robot's location to this box location.
         - Update `min_robot_dist_to_box` with the minimum distance found.
         - Handle potential errors during distance calculation by returning infinity.
       - If `min_robot_dist_to_box` is still infinity (meaning no boxes need moving or robot location not found), the robot component is 0. Otherwise, the robot component is `min_robot_dist_to_box`.
    7. The preliminary heuristic value is `sum_box_distances + robot_component`.
    8. Since we already checked that the state is not the goal state (step 2), the heuristic must be strictly positive. Return `max(1, h_value)` to ensure this.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for boxes.
        """
        # Store goal locations for each box specified in the goal.
        self.box_goals = {}
        for goal in task.goals:
            # Goal facts are typically (at <box> <location>)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.box_goals[box] = location
            # Note: Assuming goals primarily contain (at box location) facts.
            # Other goal types (like robot position) are ignored by this heuristic
            # in terms of calculating the distance sum, but the final goal check
            # accounts for them.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        This is the sum of Manhattan distances for boxes not at their goals
        plus the distance from the robot to the nearest box not at its goal.
        """
        state = node.state  # Current world state.
        task = node.task # Access the task object

        # Check if the goal is reached first. If so, heuristic is 0.
        if task.goal_reached(state):
             return 0

        # Find current locations of all boxes that are in the goals.
        current_box_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                 obj, location = args
                 # Only track locations for objects that are boxes in our goals
                 if obj in self.box_goals:
                     current_box_locations[obj] = location

        # Calculate sum of Manhattan distances for boxes not at goal.
        sum_box_distances = 0
        boxes_not_at_goal_locations = [] # Collect locations for robot distance calculation
        for box, goal_location in self.box_goals.items():
             current_location = current_box_locations.get(box)

             # If a box from the goal is not found in the state, something is wrong.
             # This shouldn't happen in valid states during search.
             if current_location is None:
                 # This state is likely invalid or corrupted. Return infinity.
                 # print(f"Warning: Box '{box}' from goal not found in state.") # Optional warning
                 return float('inf')

             if current_location != goal_location:
                try:
                    distance = manhattan_distance(current_location, goal_location)
                    sum_box_distances += distance
                    boxes_not_at_goal_locations.append(current_location)
                except ValueError as e:
                    # Handle cases where location format is unexpected during distance calc
                    # print(f"Error calculating box distance for {box}: {e}") # Optional error print
                    return float('inf') # Propagate error

        # Find robot location
        robot_location = None
        for fact in state:
             if match(fact, "at-robot", "*"):
                  robot_location = get_parts(fact)[1]
                  break

        # Calculate robot distance to nearest box not at goal
        robot_component = 0
        if robot_location and boxes_not_at_goal_locations:
             min_robot_dist_to_box = float('inf')
             for box_loc in boxes_not_at_goal_locations:
                  try:
                       dist = manhattan_distance(robot_location, box_loc)
                       min_robot_dist_to_box = min(min_robot_dist_to_box, dist)
                   # Handle cases where location format is unexpected during distance calc
                  except ValueError as e:
                       # print(f"Error calculating robot-box distance: {e}") # Optional error print
                       return float('inf') # Propagate error

             if min_robot_dist_to_box != float('inf'):
                  robot_component = min_robot_dist_to_box
             # If no boxes need moving, min_robot_dist_to_box remains inf, robot_component remains 0.

        # Total heuristic value
        h_value = sum_box_distances + robot_component

        # Since we already checked that the state is not the goal state (step 2),
        # the heuristic must be strictly positive.
        # If h_value is 0 here, it means sum_box_distances is 0 AND robot_component is 0.
        # This implies all boxes are at goal locations AND the robot is at one of those locations.
        # If this state is NOT the final goal state (e.g., goal requires robot elsewhere),
        # the remaining cost is at least 1 (a robot move).
        # Returning max(1, h_value) ensures the heuristic is >= 1 for non-goal states.
        return max(1, h_value)
