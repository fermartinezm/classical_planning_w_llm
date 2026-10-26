from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its closest goal location.

    # Assumptions:
    - Moving the robot is less costly than moving a box.
    - The robot can always reach a position to push a box.
    - The heuristic does not consider deadlocks.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, calculate the Manhattan distance to each of its goal locations.
    3. Assign each box to its closest goal location.
    4. Sum the Manhattan distances for all boxes to their assigned goal locations.
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
        Estimate the number of actions needed to move all boxes to their goal locations.
        """
        state = node.state
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                try:
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
                except:
                    pass

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
