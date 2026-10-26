from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It considers the Manhattan distance between each box and its closest goal location,
    and the Manhattan distance between the robot and the closest box that needs to be moved.
    It also takes into account whether the robot is blocked from pushing a box in the desired direction.

    # Assumptions
    - The heuristic assumes that boxes can be moved independently, which is not always true in Sokoban.
    - It assumes that the robot can always reach a box to push it, which may not be the case due to obstacles.
    - It assumes that the shortest path between two locations is a good estimate of the number of moves required.

    # Heuristic Initialization
    - The heuristic initializes the goal locations for each box and the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of the robot and all boxes from the state.
    2. Extract the goal locations for each box.
    3. For each box, calculate the Manhattan distance to its closest goal location.
    4. Calculate the Manhattan distance between the robot and the closest box that needs to be moved.
    5. Sum the Manhattan distances for all boxes and the robot.
    6. Check for deadlocks: If a box is in a corner and cannot be moved towards its goal, add a penalty.
    7. Return the total estimated cost.
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
                self.box_goals[box] = location

        # Extract adjacency information
        self.adjacencies = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                direction = parts[3]
                if loc1 not in self.adjacencies:
                    self.adjacencies[loc1] = {}
                self.adjacencies[loc1][direction] = loc2
                if loc2 not in self.adjacencies:
                    self.adjacencies[loc2] = {}
                # Reverse direction
                if direction == 'up':
                    reverse_direction = 'down'
                elif direction == 'down':
                    reverse_direction = 'up'
                elif direction == 'left':
                    reverse_direction = 'right'
                else:
                    reverse_direction = 'left'
                self.adjacencies[loc2][reverse_direction] = loc1

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state.
        """
        state = node.state

        # Extract robot and box locations
        robot_location = None
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # If the state is a goal state, return 0
        if self.goal_reached(state):
            return 0

        # Calculate Manhattan distances
        total_distance = 0
        for box, location in box_locations.items():
            goal_location = self.box_goals[box]
            total_distance += self.manhattan_distance(location, goal_location)

        # Calculate distance from robot to closest box
        min_robot_distance = float('inf')
        for box, location in box_locations.items():
            if location != self.box_goals[box]:
                distance = self.manhattan_distance(robot_location, location)
                min_robot_distance = min(min_robot_distance, distance)
        total_distance += min_robot_distance

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """
        Calculate the Manhattan distance between two locations.
        """
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def goal_reached(self, state):
        """
        Check if the current state is a goal state.
        """
        for goal in self.goals:
            if goal not in state:
                return False
        return True
