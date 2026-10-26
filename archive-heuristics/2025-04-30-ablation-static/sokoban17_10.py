# Assuming Heuristic base class is provided or defined elsewhere
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing locations and calculating distance
def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    # Example: 'loc_1_1' -> ('1', '1') -> (1, 1)
    parts = loc_str.split('_')
    # Assumes format 'loc_row_col'
    # Robustness check: ensure format is as expected
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # This should not happen with valid PDDL, but handle defensively
            return None # Return None for invalid format
    else:
        # Handle unexpected location string format
        return None # Return None for invalid format


def manhattan_distance(loc1_tuple, loc2_tuple):
    """Calculates Manhattan distance between two location tuples (R, C)."""
    # Ensure inputs are valid tuples
    if loc1_tuple is None or loc2_tuple is None:
        # Cannot calculate distance if locations are invalid
        return float('inf') # Indicate impossibility or large cost

    r1, c1 = loc1_tuple
    r2, c2 = loc2_tuple
    return abs(r1 - r2) + abs(c1 - c2)


# Assuming Heuristic base class is available and imported elsewhere
# from heuristics.heuristic_base import Heuristic
# If inheriting, use: class sokobanHeuristic(Heuristic):
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to reach a goal state
    by summing the Manhattan distances of all misplaced boxes to their goals
    and adding the minimum Manhattan distance from the robot to any misplaced box.

    # Assumptions
    - Locations are arranged on a grid-like structure where Manhattan distance
      provides a reasonable, albeit non-admissible, estimate of path distance.
    - The primary costs are moving boxes (pushes) and moving the robot to
      get into position to push.
    - The heuristic does not account for complex interactions like deadlocks
      or needing to move boxes away from goals temporarily.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  In the constructor (`__init__`), iterate through the task's goal
        conditions (`task.goals`). For each goal fact of the form `(at box_name loc_R_C)`,
        parse the location string `loc_R_C` into a `(R, C)` tuple and store
        it in a dictionary mapping box names to their goal location tuples.
    2.  In the call method (`__call__`), access the current state (`node.state`).
    3.  Iterate through the facts in the current state to find the robot's
        current location and the current location of each box. Parse these
        locations into `(R, C)` tuples using the `parse_location` helper function.
    4.  Initialize `h_boxes` to 0. This will sum the distances for misplaced boxes.
    5.  Initialize an empty list `misplaced_boxes` to store the names of boxes
        that are not at their goal locations.
    6.  Iterate through the `goal_locations` dictionary (created in `__init__`).
        For each box name and its goal location tuple:
        -   Find the box's current location tuple from the parsed state information (`box_locations`).
        -   If the box's current location was successfully found and parsed, and it is different from its goal location:
            -   Calculate the Manhattan distance between the current location
                and the goal location using the `manhattan_distance` helper function.
            -   Add this distance to `h_boxes`.
            -   Add the box name to the `misplaced_boxes` list.
    7.  Initialize `h_robot` to 0. This will store the robot's contribution.
    8.  If the `misplaced_boxes` list is not empty (meaning there are boxes
        to move) and the robot's location was successfully found and parsed:
        -   Find the minimum Manhattan distance from the robot's current location
            tuple to the current location tuple of any box in the `misplaced_boxes` list.
        -   Set `h_robot` to this minimum distance.
    9.  The total heuristic value is `h_boxes + h_robot`. Return this value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # If inheriting from Heuristic, call super().__init__(task)
        # super().__init__(task) # Uncomment if inheriting from Heuristic
        self.task = task # Store task reference

        # Store goal locations for each box. Map box name (str) to goal location (tuple).
        self.goal_locations = {}
        for goal in self.task.goals:
            # goal is a string like '(at box1 loc_2_4)'
            # Check if the goal fact is an 'at' predicate with 3 parts
            parts = goal[1:-1].split() # Remove parentheses and split
            if len(parts) == 3 and parts[0] == 'at':
                box_name = parts[1]
                goal_loc_str = parts[2]
                # Ensure the second argument is a location string and parse it
                if goal_loc_str.startswith('loc_'):
                     parsed_loc = parse_location(goal_loc_str)
                     if parsed_loc is not None: # Check if parsing was successful
                        self.goal_locations[box_name] = parsed_loc
                # else: Handle unexpected goal format? Assume valid PDDL goals.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Extract robot and box locations from the current state
        robot_location = None # Will be a tuple (R, C) or None if not found/parsed
        box_locations = {}    # Map box name (str) to current location (tuple or None)

        for fact in state:
            parts = fact[1:-1].split()
            if len(parts) == 2 and parts[0] == 'at-robot':
                robot_location = parse_location(parts[1])
            elif len(parts) == 3 and parts[0] == 'at':
                box_name = parts[1]
                box_loc_str = parts[2]
                # Ensure the location string format and parse it
                if box_loc_str.startswith('loc_'):
                    parsed_loc = parse_location(box_loc_str)
                    if parsed_loc is not None: # Check if parsing was successful
                        box_locations[box_name] = parsed_loc
            # We don't need 'clear' facts for this heuristic calculation

        # Calculate H_boxes: Sum of Manhattan distances for misplaced boxes
        h_boxes = 0
        misplaced_boxes = [] # List of box names not at their goals

        # Iterate through the boxes we know about from the goals
        for box_name, goal_loc_tuple in self.goal_locations.items():
            current_loc_tuple = box_locations.get(box_name) # Get current location from state

            # Check if the box's current location was found/parsed and is not at its goal
            if current_loc_tuple is not None and current_loc_tuple != goal_loc_tuple:
                h_boxes += manhattan_distance(current_loc_tuple, goal_loc_tuple)
                misplaced_boxes.append(box_name)

        # Calculate H_robot: Minimum Manhattan distance from robot to any misplaced box
        h_robot = 0
        # Only calculate if there are boxes to move AND we found the robot's location
        if misplaced_boxes and robot_location is not None:
            min_robot_dist = float('inf')
            for box_name in misplaced_boxes:
                box_loc_tuple = box_locations.get(box_name) # Get box location (should exist if in misplaced_boxes)
                if box_loc_tuple is not None: # Defensive check
                    # Calculate distance from robot to the box
                    dist = manhattan_distance(robot_location, box_loc_tuple)
                    min_robot_dist = min(min_robot_dist, dist)
            # If min_robot_dist is still inf, it means no misplaced box location was found/parsed,
            # which shouldn't happen if misplaced_boxes is not empty and box_locations was populated.
            # But setting h_robot to 0 if min_robot_dist is inf is safer.
            if min_robot_dist != float('inf'):
                 h_robot = min_robot_dist
            # else: h_robot remains 0

        # Total heuristic is the sum of box distances and robot distance to the closest box
        return h_boxes + h_robot
