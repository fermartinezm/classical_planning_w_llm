from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve a Sokoban puzzle by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that each move or push action has a cost of 1.
    The heuristic is admissible in a simplified version of Sokoban where boxes can move freely without obstacles, but it is not guaranteed
    to be admissible in the full Sokoban domain due to the complexities of box interactions and deadlocks. However, it provides a reasonable
    and efficiently computable estimate for guiding greedy best-first search.

    # Assumptions:
    - The cost of each move and push action is assumed to be 1.
    - The heuristic does not explicitly consider obstacles like walls or other boxes when calculating distances.
    - It assumes that boxes can be moved independently to their goal locations, which is not always possible in Sokoban due to potential deadlocks.
    - Location names are assumed to be in the format 'loc_row_column' (e.g., 'loc_2_4'), allowing extraction of row and column indices for Manhattan distance calculation.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal conditions from the task definition to identify the goal location for each box.
    - It stores the goal locations in a dictionary where keys are box names and values are their goal locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. Initialize the total heuristic value to 0.
    2. Iterate through each goal condition in the task's goal definition.
    3. For each goal condition, identify the box and its target goal location. We assume goal conditions are of the form '(at boxX loc_goal_X)'.
    4. For each box, find its current location in the current state by checking facts of the form '(at boxX loc_current_X)'.
    5. Extract the row and column numbers from both the current location and the goal location strings.
       Assuming location strings are in the format 'loc_row_column', split the string by '_' and parse the second and third parts as integers.
    6. Calculate the Manhattan distance between the current location and the goal location for the box:
       Manhattan distance = |row_goal - row_current| + |column_goal - column_current|.
    7. Add the calculated Manhattan distance to the total heuristic value.
    8. After iterating through all goal boxes, the total heuristic value represents the estimated number of actions needed to reach the goal state from the current state.
    9. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic by extracting goal box locations.
        """
        super().__init__(task)
        self.box_goals = {}
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box_name = parts[1]
                goal_location = parts[2]
                self.box_goals[box_name] = goal_location

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state based on Manhattan distances of boxes to their goal locations.
        """
        state = node.state
        heuristic_value = 0
        box_current_locations = {}

        for fact in state:
            if match(fact, "at", "*", "*") and not match(fact, "at-robot", "*"): # To avoid considering robot's 'at' fact
                parts = get_parts(fact)
                box_name = parts[1]
                current_location = parts[2]
                box_current_locations[box_name] = current_location

        for box_name, goal_location in self.box_goals.items():
            if box_name in box_current_locations: # Ensure box is still in the state (should always be true in Sokoban)
                current_location = box_current_locations[box_name]
                heuristic_value += self._manhattan_distance(current_location, goal_location)
            else:
                # This case should ideally not happen in Sokoban, but for robustness, we can assign a high penalty or handle as needed.
                # For now, we will assume the box is missing, which is a very bad state, so return a large heuristic value.
                return float('inf') # Or some large number if infinity is not suitable

        return heuristic_value

    def _get_location_coords(self, location_str):
        """
        Helper function to extract row and column coordinates from a location string (e.g., 'loc_2_4').
        """
        parts = location_str.split('_')
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except (IndexError, ValueError):
            raise ValueError(f"Invalid location string format: {location_str}. Expected 'loc_row_column'.")

    def _manhattan_distance(self, loc1_str, loc2_str):
        """
        Calculate the Manhattan distance between two locations given as strings.
        """
        row1, col1 = self._get_location_coords(loc1_str)
        row2, col2 = self._get_location_coords(loc2_str)
        return abs(row1 - row2) + abs(col1 - col2)

