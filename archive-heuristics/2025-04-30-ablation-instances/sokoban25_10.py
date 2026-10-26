from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import re

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` allowed in args.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(location_name):
    """
    Extracts row and column from location names like 'loc_r_c'.
    Returns a tuple (row, col) or None if the format doesn't match.
    """
    match = re.match(r'loc_(\d+)_(\d+)', location_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # If location names don't follow this pattern, Manhattan distance is not applicable.
    # Returning None allows the heuristic to handle this case (e.g., return infinity).
    return None

def manhattan_distance(loc1_name, loc2_name):
    """
    Calculates Manhattan distance between two locations based on their names.
    Assumes location names are in 'loc_r_c' format.
    Returns float('inf') if coordinates cannot be extracted.
    """
    coords1 = get_coords(loc1_name)
    coords2 = get_coords(loc2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance for unknown formats
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It considers the Manhattan distance of each
    misplaced box to its goal and the Manhattan distance of the robot to the
    closest misplaced box. It is designed for greedy best-first search and
    is not admissible.

    # Assumptions
    - The goal specifies a unique target location for each box that appears
      in an `(at ?b ?l)` goal predicate.
    - Location names follow the format 'loc_row_col' allowing Manhattan distance calculation.
    - The cost of moving the robot and pushing a box are related to grid distance.
    - The heuristic simplifies the pushing mechanics and ignores obstacles (other boxes, walls not explicitly marked as non-adjacent).

    # Heuristic Initialization
    - Extracts the goal location for each specific box from the task goals.
    - Static facts (like 'adjacent') are not explicitly used for distance calculation in this Manhattan-distance based heuristic, but the location naming convention implies the grid structure assumed by the example files.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state from the search node.
    2. Identify the robot's current location by finding the fact `(at-robot ?l)`.
    3. Identify the current location of each box that has a goal location, by finding facts `(at ?b ?l)`.
    4. Determine which of these boxes are not currently at their assigned goal locations. Collect these as 'misplaced boxes'.
    5. If the list of misplaced boxes is empty, the current state is a goal state, and the heuristic value is 0.
    6. If there are misplaced boxes:
       a. Calculate the sum of Manhattan distances for each misplaced box from its current location to its goal location. This provides a lower bound on the number of pushes required for the boxes themselves in a relaxed grid without obstacles.
       b. Calculate the minimum Manhattan distance from the robot's current location to the current location of any of the misplaced boxes. This estimates the minimum robot movement needed to reach a box it can start pushing.
       c. The heuristic value is the sum of the minimum robot-to-misplaced-box distance and the sum of misplaced box-to-goal distances. This non-admissible combination attempts to capture both the robot's initial travel cost to engage with a box and the total box movement effort required.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Store goal locations for each box that appears in a goal predicate.
        # Assuming goals are of the form (at box_name loc_name)
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Check for facts like (at box1 loc_goal_1) in the goals
            if parts and parts[0] == "at" and len(parts) == 3:
                 # We assume the first argument after 'at' is the object (box)
                 # and the second is the location.
                 obj_name, loc_name = parts[1], parts[2]
                 # In PDDL, ?o is typed as 'box'. We rely on the goal structure
                 # implying which objects are the boxes we need to move.
                 self.goal_locations[obj_name] = loc_name

        # Static facts are available in task.static but are not directly used
        # in this Manhattan distance based heuristic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                # The robot location is the second part of the fact string
                parts = get_parts(fact)
                if len(parts) > 1:
                    robot_loc = parts[1]
                break # Assuming only one robot

        # If robot location cannot be found, the state is likely invalid or terminal (stuck)
        if robot_loc is None or get_coords(robot_loc) is None:
             # Cannot calculate distances if robot location is unknown or malformed
             return float('inf') # Indicate a high cost/problematic state

        # Find current box locations for boxes that have a goal
        box_locations = {}
        for fact in state:
            # Match facts like (at box1 loc_3_5)
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 # Only consider objects that are identified as boxes with goals
                 if obj_name in self.goal_locations:
                     box_locations[obj_name] = loc_name

        misplaced_boxes = []
        sum_box_distances = 0

        # Calculate sum of distances for misplaced boxes
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # If a box with a goal is not found in the current state's 'at' facts,
            # it's an unexpected situation. Assuming valid states, all boxes
            # with goals should be present in 'at' facts unless they are at their goal.
            # However, the goal check below handles the 'at goal' case.
            # If current_loc is None, it might indicate a problem with the state representation
            # or the box definition/parsing. For robustness, we could treat this
            # as infinite cost or ignore the box if it wasn't in the initial state's 'at' facts.
            # A simpler approach for this heuristic is to only consider boxes found in the state.
            # If a box with a goal isn't in box_locations, it won't be added to misplaced_boxes.
            # This is okay if the goal state check `not misplaced_boxes` is the primary
            # way to identify goal states.

            # Check if the box is at its goal location
            if current_loc is not None and current_loc != goal_loc:
                # Box is misplaced
                misplaced_boxes.append(box)
                dist = manhattan_distance(current_loc, goal_loc)
                if dist == float('inf'):
                    # Cannot calculate distance for this box's path
                    return float('inf')
                sum_box_distances += dist

        # If no boxes are misplaced, the goal is reached.
        if not misplaced_boxes:
            return 0

        # Calculate minimum distance from robot to any misplaced box
        min_robot_distance = float('inf')
        for box in misplaced_boxes:
            box_loc = box_locations.get(box) # Get location again, should exist for misplaced boxes
            if box_loc is not None: # Defensive check
                dist = manhattan_distance(robot_loc, box_loc)
                if dist == float('inf'):
                    # Cannot calculate distance from robot to this box
                    return float('inf')
                min_robot_distance = min(min_robot_distance, dist)

        # If min_robot_distance is still inf, it means no misplaced boxes had valid locations,
        # which contradicts misplaced_boxes being non-empty. This case should ideally not happen
        # in a valid problem state, but we return inf defensively.
        if min_robot_distance == float('inf'):
             return float('inf')


        # Combine distances:
        # The heuristic is the sum of the minimum distance for the robot to reach
        # any misplaced box, plus the sum of the Manhattan distances for all
        # misplaced boxes to reach their respective goals.
        total_heuristic = min_robot_distance + sum_box_distances

        return total_heuristic

