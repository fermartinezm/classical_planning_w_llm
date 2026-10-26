from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
# No need for math module, abs is a built-in function

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(loc_str):
    """
    Parses a location string like 'loc_R_C' into a (row, col) tuple.
    Returns None if the format is not recognized.
    """
    # Assumes location names are always in the format loc_R_C
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # R or C are not integers
            return None
    # Format is unexpected
    return None

def manhattan_distance(loc1_str, loc2_str, location_coords):
    """
    Calculates Manhattan distance between two location strings using pre-parsed coordinates.
    Returns float('inf') if either location string is not found in location_coords.
    """
    coords1 = location_coords.get(loc1_str)
    coords2 = location_coords.get(loc2_str)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance if coordinates are unknown
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the Manhattan
    distances of each box to its goal location. It also adds the Manhattan
    distance from the robot to the nearest misplaced box. This combination
    attempts to account for both the box movement cost (approximated by box-goal distance)
    and the robot's need to reach a box to push it (approximated by robot-box distance).
    It is designed for greedy best-first search and is not admissible.

    # Assumptions
    - Location names are consistently in the format 'loc_R_C' where R and C are integers.
    - The grid implied by location names is consistent with the adjacency facts (though adjacency is not directly used in distance calculation, only location names).
    - The primary cost driver is moving boxes, which requires the robot to reach them.
    - All box objects that need to reach a goal are listed in the PDDL instance goal facts using the '(at ?box ?location)' predicate.
    - All location objects are listed under the 'location' type in the PDDL instance :objects section.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Parses all location names defined in the task's :objects section into (row, col) coordinates and stores them in a dictionary for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the robot from the state facts. If the robot's location cannot be found or parsed, the state is likely invalid or unsolvable, return infinity.
    2. Identify the current location of each box that has a goal location defined. Store these in a dictionary. If a box with a goal is not found in the current state facts or its location cannot be parsed, return infinity.
    3. Initialize the total heuristic value to 0.
    4. Initialize a variable `min_robot_to_misplaced_box_dist` to infinity. This will track the minimum distance from the robot to any box that is not yet at its goal.
    5. Initialize a counter `misplaced_boxes_count` to 0.
    6. Iterate through each box and its goal location stored during initialization (`self.goal_locations`):
       - Get the box's current location from the `current_box_locations` dictionary.
       - If the box's current location is the same as its goal location, this box is satisfied; continue to the next box.
       - If the box is not at its goal:
         - Increment `misplaced_boxes_count`.
         - Calculate the Manhattan distance between the box's current location and its goal location using the pre-parsed coordinates (`self.location_coords`). If distance calculation fails (e.g., location not parsed), return infinity. Add this distance to the `total_heuristic`. This term estimates the minimum number of pushes required for this box.
         - Calculate the Manhattan distance between the robot's current location and the box's current location. If distance calculation fails, return infinity. Update `min_robot_to_misplaced_box_dist` with the minimum of its current value and this calculated distance. This term estimates the robot's effort to reach this specific box.
    7. After iterating through all goal boxes, if `misplaced_boxes_count` is greater than 0:
         # If min_robot_to_misplaced_box_dist is still inf here, it means
         # misplaced boxes exist but robot_location or their locations were unparseable,
         # which is handled by returning inf earlier.
         total_heuristic += min_robot_to_misplaced_box_dist
    8. If `misplaced_boxes_count` is 0, the state is a goal state, and the `total_heuristic` will be 0 (as nothing was added in step 6 or 7).
    9. Return the final `total_heuristic` value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and parsing locations.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal facts are typically (at ?box ?location)
            if predicate == "at" and len(args) == 2:
                box, location = args
                # Assuming any object in an 'at' goal is a box we need to move
                self.goal_locations[box] = location

        # Pre-parse all location names into (row, col) coordinates.
        self.location_coords = {}
        # Get all objects defined in the problem
        if hasattr(task, 'objects') and task.objects:
             if 'location' in task.objects:
                 for loc_str in task.objects['location']:
                      coords = parse_location(loc_str)
                      if coords:
                          self.location_coords[loc_str] = coords
                      # else: print(f"Warning: Could not parse location string '{loc_str}'")
             # else: print("Warning: No objects of type 'location' found in task definition.")
        # else: print("Warning: Task object has no 'objects' attribute or it is empty.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        if robot_location is None or robot_location not in self.location_coords:
             # Robot location is unknown or unparseable
             return float('inf') # Cannot solve

        # Find current box locations for boxes that have a goal
        current_box_locations = {}
        for fact in state:
            # Look for (at ?o ?l) facts
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Only consider objects that are boxes with defined goals
                 if obj in self.goal_locations:
                     current_box_locations[obj] = loc

        total_heuristic = 0
        min_robot_to_misplaced_box_dist = float('inf')
        misplaced_boxes_count = 0

        # Iterate through all boxes that need to reach a goal
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box with a goal is not found in the state or its location is unparseable
            if current_location is None or current_location not in self.location_coords:
                 return float('inf') # Cannot solve

            # Check if the box is already at its goal
            if current_location != goal_location:
                misplaced_boxes_count += 1

                # Cost 1: Distance for the box to reach its goal (minimum pushes)
                box_to_goal_dist = manhattan_distance(current_location, goal_location, self.location_coords)
                if box_to_goal_dist == float('inf'):
                    # This should ideally not happen if current_location and goal_location
                    # were found in self.location_coords, but included for robustness.
                    return float('inf')
                total_heuristic += box_to_goal_dist

                # Cost 2: Distance for the robot to reach this box
                robot_to_box_dist = manhattan_distance(robot_location, current_location, self.location_coords)
                if robot_to_box_dist == float('inf'):
                     # This should ideally not happen if robot_location and current_location
                     # were found in self.location_coords, but included for robustness.
                     return float('inf')

                min_robot_to_misplaced_box_dist = min(min_robot_to_misplaced_box_dist, robot_to_box_dist)

        # Add the cost for the robot to reach the nearest misplaced box
        # This term is only added if there are misplaced boxes.
        if misplaced_boxes_count > 0:
             # If min_robot_to_misplaced_box_dist is still inf here, it means
             # misplaced boxes exist but robot_location or their locations were unparseable,
             # which is handled by returning inf earlier.
             total_heuristic += min_robot_to_misplaced_box_dist

        # If misplaced_boxes_count is 0, total_heuristic is 0, and min_robot_to_misplaced_box_dist is inf.
        # The condition `misplaced_boxes_count > 0` prevents adding inf.
        # So, the heuristic is 0 if and only if all boxes are at their goals.

        return total_heuristic
