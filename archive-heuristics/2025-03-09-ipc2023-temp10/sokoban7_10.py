from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class sokoban7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban problem by considering the Manhattan distances
    between each box and its closest goal location, and the robot's distance to the nearest box that needs to be moved.
    It also considers how many boxes are in goal locations.

    # Assumptions
    - The cost of moving a box is proportional to the Manhattan distance to its nearest goal.
    - The robot must be adjacent to a box to push it.
    - The heuristic is admissible if we only consider the cost of moving boxes to their goals.
    - The heuristic is not admissible if we consider the robot's movements.

    # Heuristic Initialization
    - Extract the initial box locations, goal locations, and adjacency information from the task.
    - Build a data structure to efficiently find adjacent locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of the robot and all boxes from the state.
    2. For each box, find the Manhattan distance to its closest goal location.
    3. Sum these distances to get an estimate of the box movement cost.
    4. Find the closest box to the robot that is not in a goal location.
    5. Calculate the Manhattan distance between the robot and this box.
    6. Add this distance to the box movement cost to get the final heuristic value.
    7. If all boxes are at goal locations, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.static = task.static

        self.box_goals = {}
        self.box_locations = {}
        self.robot_location = None

        # Extract goal locations for each box
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals.setdefault(box, []).append(location)

        # Build adjacency dictionary
        self.adj = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1)

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state

        # Extract box and robot locations from the state
        box_locations = {}
        robot_location = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                if parts[1] != 'robot':
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        # If the state satisfies the goal, return 0
        if node.task.goal_reached(state):
            return 0

        # Calculate the sum of Manhattan distances between each box and its closest goal
        box_distances_sum = 0
        for box, location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals.get(box, []):
                min_distance = min(min_distance, self.manhattan_distance(location, goal_location))
            box_distances_sum += min_distance

        # Find the closest box to the robot that is not in a goal location
        min_robot_distance = float('inf')
        for box, location in box_locations.items():
            if location not in self.box_goals.get(box, []):
                distance = self.manhattan_distance(robot_location, location)
                min_robot_distance = min(min_robot_distance, distance)

        # Add the robot distance to the box distances sum
        heuristic_value = box_distances_sum + min_robot_distance

        return heuristic_value

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)
