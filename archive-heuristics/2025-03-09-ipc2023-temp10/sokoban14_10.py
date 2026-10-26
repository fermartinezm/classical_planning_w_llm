from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions:
    - Each push action moves a box one location closer to its goal.
    - The robot can always move to the required position to push a box.
    - Only one box can occupy a location.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Extract the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each box from the state.
    2. For each box, find its goal location.
    3. Calculate the Manhattan distance between the current location and the goal location of each box.
       The Manhattan distance is calculated as the sum of the absolute differences of the x and y coordinates.
       To find the x and y coordinates, we extract them from the location names. For example, loc_2_3 has x=2 and y=3.
    4. Sum the Manhattan distances for all boxes. This sum is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        # Extract adjacency information (not used in this heuristic, but could be useful for future improvements)
        self.adjacencies = {}
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                if loc1 not in self.adjacencies:
                    self.adjacencies[loc1] = {}
                self.adjacencies[loc1][direction] = loc2

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        heuristic_value = 0

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Calculate Manhattan distance for each box
        for box, current_location in box_locations.items():
            if box in self.box_goals:
                goal_location = self.box_goals[box]

                # Extract x and y coordinates from location names
                current_x, current_y = map(int, current_location.split('_')[1:])
                goal_x, goal_y = map(int, goal_location.split('_')[1:])

                # Calculate Manhattan distance
                manhattan_distance = abs(current_x - goal_x) + abs(current_y - goal_y)
                heuristic_value += manhattan_distance
            else:
                # Box has no goal, this should not happen, but handle it gracefully
                return float('inf')

        # If the goal is reached, the heuristic value is 0
        if node.state == self.goals:
            return 0

        return heuristic_value
