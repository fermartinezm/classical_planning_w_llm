from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its closest goal location.
    Additionally, it considers the minimum number of moves required for the robot to reach a position from which it can push each box towards its goal.

    # Assumptions
    - The heuristic assumes that the shortest path between two locations can be approximated by the Manhattan distance.
    - It assumes that each box can be moved independently to its goal location, ignoring potential blockages by other boxes.
    - It assumes that the robot can always reach a position to push a box, although this might not always be true in complex scenarios.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Extract the adjacency information between locations to calculate Manhattan distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, calculate the Manhattan distance to each of its goal locations.
    3. Assign each box to its closest goal location based on Manhattan distance.
    4. Calculate the minimum number of moves required for the robot to reach a position from which it can push each box towards its goal. This is done by calculating the Manhattan distance between the robot's current location and the location adjacent to the box in the direction of the goal.
    5. Sum the Manhattan distances for all boxes and the robot moves to get the final heuristic value.
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
                inverse_direction = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}.get(direction)
                if inverse_direction:
                    self.adjacencies[loc2][inverse_direction] = loc1

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        box_locations = {}
        robot_location = None

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        if not box_locations or not robot_location:
            return float('inf')

        total_distance = 0
        for box, current_location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals[box]:
                distance = self.manhattan_distance(current_location, goal_location)
                min_distance = min(min_distance, distance)
            total_distance += min_distance

            # Calculate robot move cost
            best_push_distance = float('inf')
            for goal_location in self.box_goals[box]:
                dx = self.get_dx(current_location, goal_location)
                dy = self.get_dy(current_location, goal_location)

                push_location = None
                if dx > 0:
                    push_location = self.get_adjacent(current_location, 'left')
                elif dx < 0:
                    push_location = self.get_adjacent(current_location, 'right')
                elif dy > 0:
                    push_location = self.get_adjacent(current_location, 'up')
                elif dy < 0:
                    push_location = self.get_adjacent(current_location, 'down')

                if push_location:
                    robot_distance = self.manhattan_distance(robot_location, push_location)
                    best_push_distance = min(best_push_distance, robot_distance)
            total_distance += best_push_distance

        if self.goal_reached(state):
            return 0
        else:
            return total_distance

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def get_dx(self, loc1, loc2):
        """Calculate the difference in x-coordinates between two locations."""
        x1 = int(loc1.split('_')[2])
        x2 = int(loc2.split('_')[2])
        return x2 - x1

    def get_dy(self, loc1, loc2):
        """Calculate the difference in y-coordinates between two locations."""
        y1 = int(loc1.split('_')[1])
        y2 = int(loc2.split('_')[1])
        return y2 - y1

    def get_adjacent(self, location, direction):
        """Get the adjacent location in the given direction."""
        if location in self.adjacencies and direction in self.adjacencies[location]:
            return self.adjacencies[location][direction]
        return None
