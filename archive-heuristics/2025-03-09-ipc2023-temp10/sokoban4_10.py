from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions:
    - The heuristic assumes that the robot can always reach a position to push a box towards its goal.
    - It does not consider obstacles or walls, which may make some movements impossible.
    - It assumes that each box has a unique goal location.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, find its goal location.
    3. Calculate the Manhattan distance between the box's current location and its goal location.
       The Manhattan distance is calculated using the row and column indices of the locations.
    4. Sum the Manhattan distances for all boxes.
    5. If the state is a goal state, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        # Extract adjacency information for calculating Manhattan distance.
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
        """
        Estimate the number of actions needed to move all boxes to their goal locations.
        """
        state = node.state

        # Check if the state is a goal state.
        if self.is_goal_state(state):
            return 0

        # Extract current box locations.
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] not in ['robot']:
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Calculate the sum of Manhattan distances.
        total_distance = 0
        for box, current_location in box_locations.items():
            goal_location = self.box_goals[box]
            total_distance += self.manhattan_distance(current_location, goal_location)

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """
        Calculate the Manhattan distance between two locations.
        """
        loc1_parts = loc1.split('_')
        loc2_parts = loc2.split('_')

        row1 = int(loc1_parts[1])
        col1 = int(loc1_parts[2])
        row2 = int(loc2_parts[1])
        col2 = int(loc2_parts[2])

        return abs(row1 - row2) + abs(col1 - col2)

    def is_goal_state(self, state):
        """
        Check if the given state is a goal state.
        """
        for goal in self.goals:
            if goal not in state:
                return False
        return True
