from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions:
    - Each push action moves a box one location closer to its goal.
    - The robot can always move to push a box.
    - Moving the robot to push a box does not contribute to the heuristic value.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each box from the state.
    2. For each box, calculate the Manhattan distance between its current location and its goal location.
       - The Manhattan distance is calculated as the sum of the absolute differences of the row and column indices of the two locations.
    3. Sum the Manhattan distances for all boxes.
    4. Return the total sum as the heuristic value.
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

        # Extract adjacency information
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

        # If the state is a goal state, return 0
        if self.is_goal_state(state):
            return 0

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] in self.box_goals:
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Calculate the sum of Manhattan distances
        total_distance = 0
        for box, current_location in box_locations.items():
            goal_location = self.box_goals[box]
            total_distance += self.manhattan_distance(current_location, goal_location)

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        row1, col1 = map(int, loc1.split('_')[1:])
        row2, col2 = map(int, loc2.split('_')[1:])
        return abs(row1 - row2) + abs(col1 - col2)

    def is_goal_state(self, state):
        """Check if the given state is a goal state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
