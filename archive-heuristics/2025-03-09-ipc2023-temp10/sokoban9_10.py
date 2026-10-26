from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box and its closest goal location.
    It also considers the distance between the robot and the closest box that needs to be moved.

    # Assumptions:
    - The heuristic assumes that the robot can always reach a box to push it.
    - It does not account for deadlocks or situations where a box is trapped.
    - It assumes that each box has a unique goal location.

    # Heuristic Initialization
    - Extract the initial box locations.
    - Extract the goal box locations.
    - Extract the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. Extract the goal locations of all boxes from the task goals.
    3. Calculate the Manhattan distance between each box's current location and its goal location.
    4. Sum the Manhattan distances for all boxes.
    5. Find the robot's current location.
    6. Find the closest box to the robot that is not at its goal location.
    7. Calculate the Manhattan distance between the robot and the closest box.
    8. Add the robot-to-box distance to the sum of box-to-goal distances.
    9. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        self.adjacencies = {}
        for fact in self.static_facts:
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

        def extract_location(fact):
            """Extract the location from a fact string."""
            return fact[1:-1].split()[2]

        def extract_object(fact):
            """Extract the object from a fact string."""
            return fact[1:-1].split()[1]

        def manhattan_distance(loc1, loc2):
            """Calculate the Manhattan distance between two locations."""
            x1, y1 = map(int, loc1.split('_')[1:])
            x2, y2 = map(int, loc2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)

        box_locations = {}
        robot_location = None

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at':
                if parts[1] == 'box1' or parts[1] == 'box2' or parts[1] == 'box3' or parts[1] == 'box4':  # added support for multiple boxes
                    box = parts[1]
                    location = parts[2]
                    box_locations[box] = location
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        if all(box_locations.get(box) == self.box_goals.get(box) for box in self.box_goals):
            return 0  # Goal state

        total_distance = 0
        for box, location in box_locations.items():
            goal_location = self.box_goals.get(box)
            if goal_location:
                total_distance += manhattan_distance(location, goal_location)

        # Find the closest box to the robot
        min_robot_distance = float('inf')
        for box, location in box_locations.items():
            goal_location = self.box_goals.get(box)
            if goal_location and location != goal_location:
                distance = manhattan_distance(robot_location, location)
                min_robot_distance = min(min_robot_distance, distance)

        if min_robot_distance != float('inf'):
            total_distance += min_robot_distance

        return total_distance
