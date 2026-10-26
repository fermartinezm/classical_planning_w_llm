from fnmatch import fnmatch
# Assuming Heuristic base class is available in a path like 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running within the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for unexpected input format
        return []
    return fact[1:-1].split()

def get_coords(location_str):
    """Parses a location string like 'loc_R_C' into a (row, col) tuple of integers."""
    try:
        parts = location_str.split('_')
        # Expecting format like 'loc_1_1' -> ['loc', '1', '1']
        if len(parts) == 3 and parts[0] == 'loc':
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected location format, e.g., not starting with 'loc_' or wrong number of parts
            return None
    except (ValueError, IndexError):
        # Handle cases where parts[1] or parts[2] are not valid integers
        return None

def manhattan_distance(loc1_str, loc2_str):
    """Calculates the Manhattan distance between two location strings 'loc_R_C'."""
    coords1 = get_coords(loc1_str)
    coords2 = get_coords(loc2_str)

    # If parsing failed for either location, consider them infinitely far apart
    if coords1 is None or coords2 is None:
        return float('inf')

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box to its goal location and adding the Manhattan distance
    from the robot to the closest misplaced box.

    # Assumptions
    - The locations are structured like a grid and named in the format 'loc_R_C',
      where R is the row and C is the column.
    - Manhattan distance on this grid is a reasonable approximation of path distance.
    - The primary cost drivers are moving boxes to their goals and the robot
      reaching the boxes to push them.
    - The heuristic is not strictly admissible but aims to be informative for
      greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goal conditions.
    - Stores these goal locations in a dictionary mapping box objects to their
      target location strings.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Find the current location of the robot by looking for the `(at-robot ?l)` fact.
       - Find the current location of each box by looking for `(at ?b ?l)` facts. Store these in a dictionary.

    2. Initialize Heuristic Components:
       - Initialize `total_box_distance` to 0. This will sum the distances of all misplaced boxes to their goals.
       - Initialize `min_robot_box_distance` to infinity (`float('inf')`). This will track the distance from the robot to the nearest box that is not yet at its goal.
       - Initialize a flag `misplaced_boxes_exist` to `False`.

    3. Calculate Box-Goal Distances and Find Closest Misplaced Box:
       - Iterate through the `self.box_goals` dictionary (which maps each box to its target location).
       - For each box and its goal location:
         - Get the box's current location from the `box_locations` dictionary extracted from the state.
         - If the box's current location is known and is *not* the same as its goal location:
           - Set `misplaced_boxes_exist` to `True`.
           - Calculate the Manhattan distance between the box's current location and its goal location using the `manhattan_distance` helper function.
           - Add this distance to `total_box_distance`.
           - Calculate the Manhattan distance between the robot's current location and the box's current location using `manhattan_distance`.
           - Update `min_robot_box_distance` to be the minimum of its current value and the newly calculated robot-box distance. Handle cases where robot location is not found or location strings are invalid (distance will be infinity).

    4. Handle Goal State:
       - If `misplaced_boxes_exist` is still `False` after checking all boxes, it means all boxes are at their goal locations. In this case, the state is a goal state, and the heuristic value is 0. Return 0 immediately.

    5. Combine Distances for Non-Goal States:
       - If there are misplaced boxes (`misplaced_boxes_exist` is `True`):
         - The heuristic value is the sum of `total_box_distance` and `min_robot_box_distance`.
         - If `min_robot_box_distance` is still infinity (e.g., robot location wasn't found, or all misplaced box locations were unparseable), add a large penalty to `total_box_distance` to indicate a potentially problematic state.

    6. Return the Result:
       - Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            # Goal facts are typically like '(at box1 loc_2_4)'
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        robot_location = None
        box_locations = {}

        # Extract robot and box locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif predicate == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                box_locations[box] = location

        total_box_distance = 0
        min_robot_box_distance = float('inf')
        misplaced_boxes_exist = False

        # Calculate sum of box-goal distances and find closest misplaced box
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box) # Get current location of the box

            # Check if the box exists in the current state and is not at its goal
            if current_loc is not None and current_loc != goal_loc:
                misplaced_boxes_exist = True
                box_dist = manhattan_distance(current_loc, goal_loc)

                # If box-goal distance is infinite (e.g., invalid location string),
                # the state is likely unsolvable or malformed for this heuristic.
                # Return infinity or a very large number.
                if box_dist == float('inf'):
                     return float('inf')

                total_box_distance += box_dist

                # Calculate robot-box distance for this misplaced box
                if robot_location: # Ensure robot location was found
                    robot_dist = manhattan_distance(robot_location, current_loc)
                    # Only update min_robot_box_distance if robot_dist is finite
                    if robot_dist != float('inf'):
                         min_robot_box_distance = min(min_robot_box_distance, robot_dist)


        # If all boxes are at their goals, the heuristic is 0
        if not misplaced_boxes_exist:
            return 0

        # If there are misplaced boxes, add the minimum robot-box distance.
        # If min_robot_box_distance is still infinity, it means the robot location
        # was not found or could not be parsed, or no misplaced box location
        # could be parsed. This indicates a problem, so add a large penalty.
        if min_robot_box_distance == float('inf'):
             # This case should ideally not happen in valid states with misplaced boxes
             # and a robot, but serves as a safeguard against parsing issues or
             # potentially unreachable states (though Manhattan doesn't check reachability graph).
             # Adding a large value penalizes such states heavily in greedy search.
             return total_box_distance + 1000000 # Add a large penalty

        return total_box_distance + min_robot_box_distance

