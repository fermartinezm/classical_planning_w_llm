from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of moves required to solve a Sokoban problem.
    It calculates the sum of Manhattan distances between each box and its closest goal location.
    This sum is then scaled by a factor to account for the pushes needed.

    # Assumptions
    - The robot can move freely to position itself for pushes.
    - The heuristic is not admissible.
    - The heuristic assumes that each box can be moved independently.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Store the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes and the robot from the state.
    2. For each box, find the Manhattan distance to each goal location.
    3. Choose the minimum Manhattan distance for each box.
    4. Sum the minimum distances for all boxes.
    5. Multiply the sum by a scaling factor (e.g., 2) to account for the fact that each box move requires the robot to move as well.
    """

    def __init__(self, task):
        """
        Initialize the heuristic.

        Extract goal locations and adjacency information from the task.
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

        self.adjacencies = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.adjacencies:
                    self.adjacencies[loc1] = []
                self.adjacencies[loc1].append(loc2)
                if loc2 not in self.adjacencies:
                    self.adjacencies[loc2] = []
                self.adjacencies[loc2].append(loc1)

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        """
        state = node.state

        box_locations = {}
        robot_location = None

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                if parts[1] != 'robot':
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
                else:
                    robot_location = parts[2]
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        if self.goal_reached(state):
            return 0

        total_distance = 0
        for box, current_location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals[box]:
                distance = self.manhattan_distance(current_location, goal_location)
                min_distance = min(min_distance, distance)
            total_distance += min_distance

        # Scale the total distance to account for pushes and robot movements
        scaling_factor = 2
        return total_distance * scaling_factor

    def manhattan_distance(self, loc1, loc2):
        """
        Calculate the Manhattan distance between two locations.
        """
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def goal_reached(self, state):
        """
        Check if the goal has been reached.
        """
        return self.goals <= state
