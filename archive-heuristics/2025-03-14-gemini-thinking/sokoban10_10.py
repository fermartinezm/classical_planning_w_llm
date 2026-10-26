from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve a Sokoban puzzle by calculating the sum of Manhattan distances
    for each box from its current location to its goal location. It assumes that each move or push action has a unit cost.

    # Assumptions:
    - The heuristic assumes a grid-based world, where Manhattan distance is a reasonable estimate of the shortest path in the absence of obstacles.
    - It does not consider obstacles (walls or other boxes) when calculating distances, which makes it non-admissible but efficient.
    - It assumes that each box needs to be moved to its specified goal location.

    # Heuristic Initialization
    - The heuristic initializes by parsing the goal predicates from the task definition to identify the goal location for each box.
    - It stores these goal locations in a dictionary for quick access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the given state.
    2. For each box, retrieve its pre-defined goal location from the initialized goal locations.
    3. Parse the location names to extract row and column coordinates. Location names are assumed to be in the format 'loc_row_col'.
    4. Calculate the Manhattan distance between the current location and the goal location for each box.
       Manhattan distance is calculated as the sum of the absolute differences of their row and column coordinates: |row1 - row2| + |col1 - col2|.
    5. Sum up the Manhattan distances for all boxes. This sum represents the estimated number of moves/pushes needed to reach the goal state.
    6. Return the total sum as the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic by extracting goal box locations.
        """
        self.goal_box_locations = {}
        for goal in task.goals:
            predicate, *args = self._parse_fact(goal)
            if predicate == 'at':
                box_name, location_name = args
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state based on Manhattan distances of boxes to their goal locations.
        """
        state = node.state
        current_box_locations = {}
        for fact in state:
            predicate, *args = self._parse_fact(fact)
            if predicate == 'at':
                if args[0] in self.goal_box_locations: # Only consider boxes that are part of the goal
                    box_name, location_name = args
                    current_box_locations[box_name] = location_name

        total_manhattan_distance = 0
        for box_name, goal_location in self.goal_box_locations.items():
            if box_name not in current_box_locations: # Should not happen in valid sokoban problems, but for robustness
                continue

            current_location = current_box_locations[box_name]

            current_loc_coords = self._parse_location_name(current_location)
            goal_loc_coords = self._parse_location_name(goal_location)

            if current_loc_coords and goal_loc_coords: # Ensure parsing was successful
                manhattan_distance = abs(current_loc_coords[0] - goal_loc_coords[0]) + abs(current_loc_coords[1] - goal_loc_coords[1])
                total_manhattan_distance += manhattan_distance

        return total_manhattan_distance

    def _parse_fact(self, fact_str):
        """
        Parses a PDDL fact string and returns a list of its components.
        Removes parentheses and splits the string by spaces.
        """
        return fact_str[1:-1].split()

    def _parse_location_name(self, location_name):
        """
        Parses a location name in the format 'loc_row_col' and returns a tuple (row, col) of integers.
        Returns None if parsing fails.
        """
        match = re.match(r'loc_(\d+)_(\d+)', location_name)
        if match:
            row, col = map(int, match.groups())
            return (row, col)
        return None
