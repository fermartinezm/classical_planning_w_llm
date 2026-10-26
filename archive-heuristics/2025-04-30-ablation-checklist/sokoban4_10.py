from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_location(loc_str):
    """
    Parses a location string like 'loc_R_C' into a tuple of integers (R, C).
    Assumes location names follow this format.
    """
    parts = loc_str.split('_')
    if len(parts) == 3 and parts[0] == 'loc':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers if necessary,
            # but based on examples, they seem to be.
            pass
    # Return None or raise error for unexpected formats
    return None # Or raise ValueError(f"Unexpected location format: {loc_str}")


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the difficulty of the current state by summing
    the Manhattan distances of all boxes from their current locations to
    their respective goal locations.

    # Assumptions
    - Each box has a unique goal location specified in the task's goals.
    - Location names follow the format 'loc_R_C' where R and C are integers
      representing row and column coordinates.
    - The heuristic ignores the robot's position and potential obstacles
      (other boxes, walls, clear predicates) for simplicity and efficiency.
      This makes it non-admissible but fast to compute.

    # Heuristic Initialization
    - Extract the goal location for each box from the task's goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of every box.
    2. For each box:
       a. Determine its specific goal location (stored during initialization).
       b. If the box is not currently at its goal location:
          i. Parse the current location string into (row, col) coordinates.
          ii. Parse the goal location string into (row, col) coordinates.
          iii. Calculate the Manhattan distance between the current and goal coordinates:
              `abs(current_row - goal_row) + abs(current_col - goal_col)`.
          iv. Add this distance to a running total.
    3. The heuristic value is the total sum of Manhattan distances for all boxes
       not yet at their goals.
    4. If all boxes are at their goals, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal location for each box.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each box.
        # We assume goals are of the form (at ?box ?location)
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Check if the object is a box (simple check based on name format)
                # A more robust way would be to parse the domain's object types,
                # but assuming 'box' objects are the ones in 'at' goals is reasonable
                # for this domain based on the examples.
                if obj_name.startswith("box"): # Simple check, could be improved
                     self.box_goals[obj_name] = loc_name

        # Note: This heuristic does not need static facts like 'adjacent' or 'clear'
        # because it uses Manhattan distance, which is a grid-based calculation
        # ignoring the actual graph connectivity and obstacles.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of Manhattan distances of boxes to their goals.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Find the current location of each box in the state.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1], parts[2]
                 if obj_name.startswith("box"): # Only track boxes
                    current_box_locations[obj_name] = loc_name

        total_distance = 0  # Initialize the heuristic cost.

        # Calculate the sum of Manhattan distances for boxes not at their goals.
        for box, goal_location in self.box_goals.items():
            current_location = current_box_locations.get(box) # Get current location, None if box not found (shouldn't happen in valid states)

            if current_location is not None and current_location != goal_location:
                # Parse locations into coordinates
                current_coords = parse_location(current_location)
                goal_coords = parse_location(goal_location)

                if current_coords is not None and goal_coords is not None:
                    # Calculate Manhattan distance
                    distance = abs(current_coords[0] - goal_coords[0]) + \
                               abs(current_coords[1] - goal_coords[1])
                    total_distance += distance
                # else: Handle parsing error if necessary

        return total_distance

# Example usage (assuming 'task' and 'node' objects are available as described)
# task = ... # An instance of the Task class
# heuristic = sokobanHeuristic(task)
# state = ... # A frozenset representing a state
# node = type('Node', (object,), {'state': state})() # Create a dummy node object
# h_value = heuristic(node)
# print(f"Heuristic value: {h_value}")
