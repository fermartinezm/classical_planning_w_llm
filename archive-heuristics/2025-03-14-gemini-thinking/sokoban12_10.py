from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of moves required to move all boxes to their goal locations based on the Manhattan distance.
    It calculates the sum of Manhattan distances for each box from its current location to its closest goal location.

    # Assumptions:
    - Each move (move or push) has a cost of 1.
    - Moving boxes independently towards their goals is a reasonable approximation of the actual cost.
    - The location names follow a pattern like 'loc_r_c' where r and c are row and column indices.

    # Heuristic Initialization
    - Extracts the goal box locations from the task goals.
    - Parses location names to extract row and column information for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal locations for each box from the task goals.
    2. For each box that is part of the goal:
       - Determine the current location of the box from the current state.
       - Determine the goal location of the box from the initialized goal locations.
       - Parse the current and goal location names to extract row and column indices.
       - Calculate the Manhattan distance between the current and goal locations:
         `|row_current - row_goal| + |col_current - col_goal|`.
    3. Sum up the Manhattan distances for all boxes that have goal locations.
    4. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the sokoban heuristic.
        Extracts goal box locations and prepares for Manhattan distance calculation.
        """
        self.goal_box_locations = {}
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at':
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the sum of Manhattan distances of each box to its goal location.
        """
        state = node.state
        heuristic_value = 0

        for box_name, goal_location_name in self.goal_box_locations.items():
            current_box_location_name = None
            for fact in state:
                parts = self._get_parts(fact)
                if parts[0] == 'at' and parts[1] == box_name:
                    current_box_location_name = parts[2]
                    break

            if current_box_location_name:
                current_location_coords = self._parse_location_name(current_box_location_name)
                goal_location_coords = self._parse_location_name(goal_location_name)

                if current_location_coords and goal_location_coords:
                    manhattan_distance = abs(current_location_coords[0] - goal_location_coords[0]) + abs(current_location_coords[1] - goal_location_coords[1])
                    heuristic_value += manhattan_distance

        return heuristic_value

    def _get_parts(self, fact):
        """
        Extract the components of a PDDL fact by removing parentheses and splitting the string.
        """
        return fact[1:-1].split()

    def _parse_location_name(self, location_name):
        """
        Parses a location name (e.g., 'loc_3_4') and extracts row and column indices.
        Returns a tuple (row, column) or None if parsing fails.
        """
        parts = location_name.split('_')
        if len(parts) == 3 and parts[0] == 'loc':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None
        return None
