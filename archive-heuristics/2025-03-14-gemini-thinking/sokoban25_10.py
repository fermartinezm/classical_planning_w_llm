from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_location_coordinates(location_str):
    """
    Extracts the row and column coordinates from a location string like 'loc_x_y'.
    Returns a tuple (row, column) of integers, or None if parsing fails.
    """
    match = re.match(r'loc_(\d+)_(\d+)', location_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Sokoban domain.
    It calculates the sum of Manhattan distances for each box from its current location to its goal location.

    # Assumptions:
    - Each box has a unique goal location specified in the problem definition.
    - Location names are in the format 'loc_row_column'.
    - The heuristic focuses on moving boxes to their goal locations and does not explicitly consider robot movements or potential deadlocks.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task goals.
    - Stores these goal locations in a dictionary for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal condition in the task goals.
    3. For each goal condition, identify if it is an 'at' predicate for a box.
    4. Extract the box name and its goal location from the goal predicate.
    5. Store the goal location for each box in a dictionary `goal_locations = {box_name: goal_location}`.
    6. For the given state, iterate through the facts.
    7. Identify facts that are 'at' predicates for boxes in the current state.
    8. For each box in the current state, get its current location and its pre-calculated goal location from `goal_locations`.
    9. If a goal location is found for the box, calculate the Manhattan distance between the current location and the goal location.
       - Parse the row and column coordinates from both location strings (e.g., 'loc_x_y' to (x, y)).
       - Manhattan distance = |goal_row - current_row| + |goal_column - current_column|.
    10. Add the Manhattan distance for each box to the total heuristic value.
    11. If a box does not have a goal location defined (which ideally should not happen based on assumptions), skip it or handle it appropriately (e.g., assign a large distance).
    12. Return the total heuristic value. This value represents the estimated number of moves needed to push all boxes to their goal locations.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations."""
        self.goals = task.goals
        self.goal_locations = {}

        for goal in self.goals:
            parts = goal[1:-1].split() # Remove parentheses and split
            if parts[0] == 'at':
                box_name = parts[1]
                goal_location = parts[2]
                self.goal_locations[box_name] = goal_location

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split() # Remove parentheses and split
            if parts[0] == 'at' and parts[1] not in ['robot', '-robot']: # Identify box locations
                box_name = parts[1]
                current_location = parts[2]
                box_locations[box_name] = current_location

        for box_name, goal_location in self.goal_locations.items():
            if box_name in box_locations:
                current_location = box_locations[box_name]

                goal_coords = get_location_coordinates(goal_location)
                current_coords = get_location_coordinates(current_location)

                if goal_coords and current_coords:
                    distance = abs(goal_coords[0] - current_coords[0]) + abs(goal_coords[1] - current_coords[1])
                    heuristic_value += distance
                else:
                    # Handle cases where location parsing fails (optional - for robustness)
                    heuristic_value += 1000 # Assign a large penalty if location parsing fails

        return heuristic_value
