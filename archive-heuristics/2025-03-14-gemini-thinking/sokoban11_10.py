from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It is based on the sum of Manhattan distances of each box from its current location to its goal location.

    # Assumptions:
    - The heuristic assumes that each box needs to be moved independently to its goal location,
      ignoring potential interactions or blockages between boxes.
    - It considers only the distance and does not account for deadlocks or the robot's path.

    # Heuristic Initialization
    - The heuristic initialization extracts the goal locations for each box from the task goals.
    - It also parses the static facts to understand the grid structure, although this is not directly used in the current heuristic calculation but could be useful for more advanced heuristics.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract goal locations for each box from the task's goal description.
    2. For a given state, identify the current location of each box.
    3. For each box, calculate the Manhattan distance between its current location and its goal location.
       To do this:
       - Parse the location names (e.g., 'loc_2_3') to extract row and column numbers.
       - Calculate the Manhattan distance as the sum of the absolute differences of row and column numbers.
    4. Sum up the Manhattan distances for all boxes. This sum is the heuristic estimate.
    5. If the current state is a goal state (all boxes are at their goal locations), the heuristic value is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal box locations and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.goal_box_locations = {}

        # Extract goal locations for each box
        for goal_fact in self.goals:
            parts = self._parse_fact(goal_fact)
            if parts[0] == 'at':
                box_name = parts[1]
                location_name = parts[2]
                self.goal_box_locations[box_name] = location_name

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        current_box_locations = {}

        # Extract current box locations from the state
        for fact in state:
            parts = self._parse_fact(fact)
            if parts[0] == 'at' and parts[1] in self.goal_box_locations: # Check if it's a box and if we have a goal for it
                box_name = parts[1]
                location_name = parts[2]
                current_box_locations[box_name] = location_name

        if not self.goal_box_locations:
            return 0 # No boxes to move, already at goal?

        if self._is_goal_state(state):
            return 0

        total_distance = 0
        for box_name, goal_location in self.goal_box_locations.items():
            current_location = current_box_locations.get(box_name)
            if current_location is None:
                # Box location not found in state, something is wrong or box is not in the problem.
                # To avoid errors, return a high heuristic value.
                return float('inf')

            goal_loc_coords = self._parse_location_name(goal_location)
            current_loc_coords = self._parse_location_name(current_location)

            if goal_loc_coords and current_loc_coords:
                distance = abs(goal_loc_coords[0] - current_loc_coords[0]) + abs(goal_loc_coords[1] - current_loc_coords[1])
                total_distance += distance
            else:
                # Cannot parse location names, return a high heuristic value.
                return float('inf')

        return total_distance

    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string and returns a list of its parts."""
        fact_str = fact_str.strip()
        if fact_str.startswith('(') and fact_str.endswith(')'):
            fact_str = fact_str[1:-1]
        return fact_str.split()

    def _parse_location_name(self, location_name):
        """Parses a location name like 'loc_r_c' and returns (r, c) as integers, or None if parsing fails."""
        match = re.match(r'loc_(\d+)_(\d+)', location_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def _is_goal_state(self, state):
        """Check if the current state satisfies all goal conditions."""
        for goal_fact in self.goals:
            if goal_fact not in state:
                return False
        return True


