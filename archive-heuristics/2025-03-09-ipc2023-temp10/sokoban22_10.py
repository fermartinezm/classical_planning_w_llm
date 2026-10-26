from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of moves required to push all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its closest goal location.

    # Assumptions
    - The robot can always move to a position to push a box (ignores potential deadlocks).
    - Moving a box to its goal location is always beneficial.
    - The cost of moving the robot is ignored.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Store the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, find the Manhattan distance to each of its goal locations.
    3. Assign each box to its closest goal location.
    4. Sum the Manhattan distances for all boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                if box not in self.box_goals:
                    self.box_goals[box] = []
                self.box_goals[box].append(location)

        # Extract adjacency information
        self.adj = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.adj:
                    self.adj[l1] = []
                self.adj[l1].append(l2)
                if l2 not in self.adj:
                    self.adj[l2] = []
                self.adj[l2].append(l1)

    def __call__(self, node):
        """
        Estimate the number of moves required to push all boxes to their goal locations.
        """
        state = node.state

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # If the state is a goal state, return 0
        if node.state >= self.goals:
            return 0

        total_distance = 0
        for box, current_location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals[box]:
                # Calculate Manhattan distance
                current_x, current_y = map(int, current_location.split('_')[1:])
                goal_x, goal_y = map(int, goal_location.split('_')[1:])
                distance = abs(current_x - goal_x) + abs(current_y - goal_y)
                min_distance = min(min_distance, distance)
            total_distance += min_distance

        return total_distance
