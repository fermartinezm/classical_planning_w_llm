# The code should be placed within a file named sokobanHeuristic.py
# and be importable by the planner.
# It should inherit from heuristics.heuristic_base.Heuristic

from heuristics.heuristic_base import Heuristic

import re
import math
# fnmatch is not strictly needed for this specific heuristic implementation
# but was present in example heuristics, so keeping it might be safer
# if the base class or environment implicitly relies on it, though unlikely.
# Let's remove it to keep dependencies minimal if not used.
# from fnmatch import fnmatch

def get_parts(fact):
    """Helper function to parse PDDL fact string into parts."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

# The match helper function is also not used in the final heuristic logic.
# def match(fact, *args):
#     """Helper function to check if a fact matches a pattern."""
#     parts = get_parts(fact)
#     return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_location(location_str):
    """Parses a location string like 'loc_row_col' into (row, col)."""
    match = re.match(r'loc_(\d+)_(\d+)', location_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # If it doesn't match the expected format, it's not a location we can parse coordinates for
    raise ValueError(f"Invalid location string format: {location_str}")

def manhattan_distance(loc1_str, loc2_str, location_coords):
    """Calculates Manhattan distance between two location strings."""
    # Assumes loc1_str and loc2_str are valid keys in location_coords
    r1, c1 = location_coords[loc1_str]
    r2, c2 = location_coords[loc2_str]
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the Manhattan distances of all misplaced boxes to their respective goal
    locations, and adding the Manhattan distance from the robot to the
    nearest misplaced box. This aims to capture both the box movement cost
    and the robot's effort to get to a box that needs moving. It ignores
    complexities like required robot positioning for pushes, blockages by
    other boxes or walls, and deadlocks. This heuristic is non-admissible
    but is efficiently computable and provides a more informed estimate
    than just box distances alone.

    Assumptions:
    - Location names follow the format 'loc_row_col' where row and col are integers.
    - The underlying grid structure implied by 'loc_row_col' corresponds to
      the adjacency relations defined in the PDDL problem, allowing Manhattan
      distance to serve as a reasonable proxy for shortest path distance
      on the location graph.
    - The goal is defined by the final positions of specific boxes.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic pre-processes the task
    information:
    1. It extracts the goal location for each box from the task's goal facts.
    2. It identifies all unique location objects mentioned in the initial state
       and static facts and parses their row and column coordinates, storing
       them in a dictionary mapping location names to (row, col) tuples. This
       allows for quick coordinate lookup during heuristic calculation. It also
       ensures that goal locations themselves are parsed, even if they don't
       appear in initial/static 'at' facts (e.g., an empty goal cell).

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state from the search node.
    2. Identify the current location of the robot and each box by iterating
       through the facts in the state and finding facts of the form
       `(at-robot ?location)` and `(at ?box ?location)`.
    3. Initialize the total heuristic value (`box_distance_sum`) to 0 and
       the minimum robot distance to a misplaced box (`min_robot_distance`)
       to infinity.
    4. Keep track of whether any boxes are misplaced (`any_misplaced_boxes`).
    5. Iterate through each box for which a goal location was identified during
       initialization.
    6. For a given box, retrieve its current location from the state and its
       goal location from the pre-processed goal information.
    7. If the box's current location is different from its goal location:
       a. Set `any_misplaced_boxes` to True.
       b. Check if the box's current location is valid (not None and coordinates parsed).
          If not, return infinity as the state is likely malformed or unparseable.
       c. Check if the goal location for this box is valid (coordinates parsed).
          If not, return infinity.
       d. Check if the robot's current location is valid (not None and coordinates parsed).
          If not, return infinity.
       e. Calculate the Manhattan distance between the box's current location
          and its goal location using the pre-processed location coordinates.
          Add this distance to `box_distance_sum`.
       f. Calculate the Manhattan distance between the robot's current location
          and the box's current location.
       g. Update `min_robot_distance` if this distance is smaller.
    8. After processing all boxes:
       a. If `any_misplaced_boxes` is True, the total heuristic is
          `box_distance_sum + min_robot_distance`. If `min_robot_distance`
          is still infinity (should not happen if checks passed), return infinity.
       b. If `any_misplaced_boxes` is False (meaning all boxes are at their goals),
          the total heuristic is 0.
    9. Return the total heuristic value.
    """
    def __init__(self, task):
        # Extract goal locations for boxes
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                location_name = parts[2]
                self.goal_locations[box_name] = location_name

        # Extract location coordinates
        self.location_coords = {}
        # Locations appear in initial state (robot, boxes, clear) and static (adjacent)
        all_facts = set(task.initial_state) | set(task.static)
        location_pattern = r'loc_\d+_\d+' # Regex to find location strings

        # Parse locations found in initial/static facts
        for fact in all_facts:
            location_strings = re.findall(location_pattern, fact)
            for loc_str in location_strings:
                if loc_str not in self.location_coords:
                    try:
                        self.location_coords[loc_str] = parse_location(loc_str)
                    except ValueError:
                        pass # Ignore unparseable strings

        # Ensure goal locations are also parsed, even if they don't appear in initial/static 'at' facts
        for goal_loc in self.goal_locations.values():
             if goal_loc not in self.location_coords:
                 try:
                     self.location_coords[goal_loc] = parse_location(goal_loc)
                 except ValueError:
                     # print(f"Warning: Could not parse goal location {goal_loc}. Heuristic might be inaccurate.")
                     pass # We don't add it if unparseable. Distance calculation will return inf.


    def __call__(self, node):
        state = node.state

        current_box_locations = {}
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj_name = parts[1]
                loc_name = parts[2]
                if obj_name in self.goal_locations: # Only track boxes relevant to goals
                     current_box_locations[obj_name] = loc_name
            elif parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]

        box_distance_sum = 0
        min_robot_distance = math.inf
        any_misplaced_boxes = False

        # Sum Manhattan distances for misplaced boxes and find min robot distance
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If box is not found in state or not at goal
            if current_location is None or current_location != goal_location:
                 any_misplaced_boxes = True

                 # --- Validate locations before calculating distance ---
                 # Check current box location
                 if current_location is None or current_location not in self.location_coords:
                     # Box is missing from state or its location is unparsed - invalid state?
                     # print(f"Error: Box {box} location {current_location} not found/parsed in state.")
                     return math.inf # Penalize invalid states

                 # Check goal location for this box
                 if goal_location not in self.location_coords:
                     # Goal location for this box was not parsed - indicates problem setup issue
                     # print(f"Error: Goal location {goal_location} for box {box} not parsed.")
                     return math.inf # Cannot compute distance to unparsed goal

                 # Check robot location
                 if robot_location is None or robot_location not in self.location_coords:
                     # Robot location not found in state or unparsed - invalid state?
                     # print("Error: Robot location not found/parsed in state.")
                     return math.inf # Penalize invalid states
                 # --- End Validation ---


                 # Calculate box-goal distance
                 # manhattan_distance assumes valid keys, which are guaranteed by checks above
                 md_box_goal = manhattan_distance(current_location, goal_location, self.location_coords)
                 box_distance_sum += md_box_goal

                 # Calculate robot-box distance
                 md_robot_box = manhattan_distance(robot_location, current_location, self.location_coords)
                 min_robot_distance = min(min_robot_distance, md_robot_box)


        # Calculate final heuristic value
        if any_misplaced_boxes:
            # If min_robot_distance is still inf, it means any_misplaced_boxes was True,
            # but we failed to calculate robot distance for any misplaced box.
            # This should be caught by the validation checks inside the loop,
            # causing an early return of inf. This check here is a final safeguard.
            if min_robot_distance == math.inf:
                 # This implies a logic error or unhandled state issue if we reach here
                 # with any_misplaced_boxes True but min_robot_distance is inf.
                 # print("Internal Error: Reached end with any_misplaced_boxes True but min_robot_distance inf.")
                 return math.inf

            return box_distance_sum + min_robot_distance
        else:
            # All boxes are at their goal locations
            return 0
