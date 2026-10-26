from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.
    Additionally, it adds a penalty if the robot is not adjacent to any box, encouraging the robot to move closer to a box.

    # Assumptions
    - The heuristic assumes that each box must be moved to a specific goal location.
    - It considers only the Manhattan distance, ignoring obstacles and the robot's path.
    - It assumes that the robot needs to be adjacent to a box to push it.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes and the robot from the state.
    2. For each box, calculate the Manhattan distance between its current location and its goal location.
    3. Sum up the Manhattan distances for all boxes.
    4. Check if the robot is adjacent to any box. If not, add a penalty equal to the minimum Manhattan distance between the robot and any box.
    5. If the state is a goal state, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Adjacency information between locations.
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

        # Extract adjacency information between locations.
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
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the state is a goal state.
        if self.is_goal_state(state):
            return 0

        # Extract current locations of boxes and the robot.
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

        # Calculate the sum of Manhattan distances for all boxes.
        total_distance = 0
        for box, goal_location in self.box_goals.items():
            current_location = box_locations[box]
            total_distance += self.manhattan_distance(current_location, goal_location)

        # Check if the robot is adjacent to any box.
        robot_adjacent_to_box = False
        for box_location in box_locations.values():
            if self.is_adjacent(robot_location, box_location):
                robot_adjacent_to_box = True
                break

        # Add a penalty if the robot is not adjacent to any box.
        if not robot_adjacent_to_box:
            min_robot_distance = float('inf')
            for box_location in box_locations.values():
                distance = self.manhattan_distance(robot_location, box_location)
                min_robot_distance = min(min_robot_distance, distance)
            total_distance += min_robot_distance

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def is_adjacent(self, loc1, loc2):
        """Check if two locations are adjacent."""
        if loc1 not in self.adjacencies:
            return False
        for direction in self.adjacencies[loc1]:
            if self.adjacencies[loc1][direction] == loc2:
                return True
        return False

    def is_goal_state(self, state):
        """Check if the given state is a goal state."""
        return self.goals <= state
