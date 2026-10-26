from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_location(location_string):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    parts = location_string.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not integers
            # print(f"Warning: Could not parse location string {location_string} as integers.")
            return None
    else:
        # Handle unexpected location string format
        # print(f"Warning: Unexpected location string format: {location_string}")
        return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    Manhattan distances between each box's current location and its goal location.
    It provides a lower bound on the number of grid cells each box needs to traverse
    to reach its target, assuming unobstructed movement for the box itself.

    # Assumptions
    - The grid structure is implied by location names like 'loc_R_C', where R and C
      represent row and column numbers.
    - The cost of moving a box is related to the grid distance it needs to travel.
    - The heuristic ignores the robot's position, the need for the robot to maneuver
      to push the box, and potential obstacles (other boxes, walls/non-clear locations).
      It only considers the box's distance to its goal.
    - Each box mentioned in the goal has a specific target location.
    - All objects (including boxes) are located somewhere in the state.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
      It specifically looks for facts of the form `(at ?box ?location)` within the goals.
    - Stores these box-goal mappings in a dictionary `self.box_goals`.
    - Static facts (like 'adjacent') are not used in this specific Manhattan distance
      calculation, as the grid coordinates are parsed directly from location names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value `total_cost` to 0.
    2. Create a mapping `current_locations` from object names to their current location strings
       by iterating through the facts in the current state (`node.state`) and extracting
       `at` predicates of the form `(at ?obj ?location)`. This step identifies where
       each object (including boxes) is currently located.
    3. Iterate through the `self.box_goals` dictionary. This dictionary holds the target
       location string for each box that needs to be moved to a specific goal. For each
       `box` and its `goal_location_str`:
       a. Check if the `box` exists as a key in the `current_locations` mapping. This confirms
          that the box's current position is known in the state. (This check is mostly
          for robustness; in valid states, all objects should have a location).
       b. If the box's current location `current_location_str` is found in `current_locations`:
          i. Parse the `current_location_str` (e.g., 'loc_4_4') into a tuple of integers
             representing its grid coordinates `(current_row, current_col)` using the
             `parse_location` helper function.
          ii. Parse the `goal_location_str` (e.g., 'loc_2_4') into a tuple of integers
              representing its grid coordinates `(goal_row, goal_col)` using the
              `parse_location` helper function.
          iii. If both parsing steps were successful (i.e., both `current_coords` and
               `goal_coords` are not `None`):
              - Calculate the Manhattan distance between the current coordinates and the
                goal coordinates: `distance = abs(current_row - goal_row) + abs(current_col - goal_col)`.
              - Add this calculated `distance` to the `total_cost`.
          iv. If parsing failed for either the current or goal location string, it indicates
              an unexpected format. The heuristic simply skips adding cost for this box
              in such a case, assuming malformed data is not standard.
       c. If the box is not found in `current_locations`, this would indicate a problem
          with the state representation where a box is not located anywhere. The heuristic
          assumes valid states where all objects have locations.
    4. After iterating through all boxes with goals, the final `total_cost` represents
       the sum of Manhattan distances for all boxes to their respective goals. Return
       this `total_cost` as the heuristic value for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each box.
        """
        # The base class constructor stores task.goals and task.static
        super().__init__(task)

        # Store goal locations for each box.
        self.box_goals = {}
        # Goals are a frozenset of facts, e.g., frozenset({'(at box1 loc_2_4)'})
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Check if the fact is of the form (at ?obj ?loc) and has 3 parts
            if parts and parts[0] == "at" and len(parts) == 3:
                obj_name = parts[1]
                loc_name = parts[2]
                # Assuming any object in an 'at' goal predicate is a box we need to move
                self.box_goals[obj_name] = loc_name

        # Static facts (like 'adjacent') are not used in this specific heuristic
        # self.static = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Find current locations of all objects (specifically boxes we care about).
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Look for facts of the form (at ?obj ?loc)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name = parts[1]
                 loc_name = parts[2]
                 # Only store location if the object is one of the boxes we need to move
                 if obj_name in self.box_goals:
                     current_locations[obj_name] = loc_name
            # We don't need the robot's location for this simple heuristic

        total_cost = 0  # Initialize heuristic cost.

        # Calculate the sum of Manhattan distances for each box to its goal.
        for box, goal_location_str in self.box_goals.items():
            # Ensure the box is actually present in the current state (it should be)
            if box in current_locations:
                current_location_str = current_locations[box]

                current_coords = parse_location(current_location_str)
                goal_coords = parse_location(goal_location_str)

                # Ensure both current and goal locations were parsed successfully
                if current_coords is not None and goal_coords is not None:
                    # Calculate Manhattan distance: |row1 - row2| + |col1 - col2|
                    distance = abs(current_coords[0] - goal_coords[0]) + abs(current_coords[1] - goal_coords[1])
                    total_cost += distance
                # else:
                    # If parsing fails, the location string format is unexpected.
                    # We could log a warning or add a penalty, but skipping is simple.

            # else:
                # If a box from the goals is not found in the state, this is unexpected
                # in a standard planning problem state representation. Assume valid states.

        return total_cost
