from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box and its closest goal location.

    # Assumptions
    - The robot can move freely to push boxes.
    - Only one box can be moved at a time.
    - The heuristic is admissible if we only consider the number of pushes.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Extract the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, calculate the Manhattan distance to each of its goal locations.
    3. Assign each box to its closest goal location.
    4. Sum the Manhattan distances for all boxes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Adjacency information between locations.
        """
        self.goals = task.goals
        self.static = task.static

        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                if box not in self.box_goals:
                    self.box_goals[box] = []
                self.box_goals[box].append(location)

        self.adj = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.adj:
                    self.adj[l1] = {}
                self.adj[l1][parts[3]] = l2

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        if all(goal in state for goal in self.goals):
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
