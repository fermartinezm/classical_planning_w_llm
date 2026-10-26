from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_location_coordinates(location_name):
    """
    Extracts row and column coordinates from a location name like 'loc_R_C'.
    Returns a tuple (row, column) or None if the format is incorrect.
    """
    match = re.match(r'loc_(\d+)_(\d+)', location_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of moves required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.

    # Assumptions:
    - The heuristic assumes that each box can be moved independently towards its goal location, ignoring potential blockages by other boxes or walls.
    - It only considers the movement of boxes and does not account for robot movements explicitly, assuming robot movements are always necessary to push boxes.
    - It assumes location names are in the format 'loc_R_C' where R and C are row and column numbers, allowing for Manhattan distance calculation.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal conditions from the task definition to determine the goal location for each box.
    - It stores these goal locations in a dictionary for efficient lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract goal box locations:
       - Iterate through the goal predicates in the task definition.
       - For each goal predicate of the form '(at box_name goal_location)', store the goal location for the corresponding box_name.

    2. For a given state, iterate through all boxes:
       - For each box, find its current location in the state using predicates of the form '(at box_name current_location)'.
       - Retrieve the pre-calculated goal location for this box.

    3. Calculate Manhattan distance:
       - Parse the row and column coordinates from both the current location and the goal location names (assuming 'loc_R_C' format).
       - Calculate the Manhattan distance as the sum of the absolute differences of row and column coordinates: |row_goal - row_current| + |col_goal - col_current|.

    4. Sum Manhattan distances:
       - Sum the Manhattan distances calculated for all boxes.
       - This sum represents the estimated number of moves required to move all boxes to their goal locations.

    5. Return the total Manhattan distance as the heuristic value.
       - If the current state is a goal state (all boxes are at their goal locations), the heuristic value will be 0 because all Manhattan distances will be 0.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic.
        Extracts goal box locations from the task definition.
        """
        self.goal_locations = {}
        for goal in task.goals:
            parts = goal[1:-1].split() # Remove brackets and split
            if parts[0] == 'at':
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_locations[box_name] = goal_location

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the sum of Manhattan distances of each box to its goal location.
        """
        state = node.state
        heuristic_value = 0

        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split() # Remove brackets and split
            if parts[0] == 'at' and parts[1] not in ['robot']: # Identify box locations
                box_name = parts[1]
                current_location = parts[2]
                box_locations[box_name] = current_location

        for box_name, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box_name)
            if current_location:
                current_coords = get_location_coordinates(current_location)
                goal_coords = get_location_coordinates(goal_location)

                if current_coords and goal_coords:
                    manhattan_distance = abs(goal_coords[0] - current_coords[0]) + abs(goal_coords[1] - current_coords[1])
                    heuristic_value += manhattan_distance
        return heuristic_value
