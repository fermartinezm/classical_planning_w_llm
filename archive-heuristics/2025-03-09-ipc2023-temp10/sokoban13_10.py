from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of moves required to solve a Sokoban problem.
    It considers the Manhattan distance between each box and its goal location,
    and the Manhattan distance between the robot and the closest box.
    It also considers whether the robot is blocking a box from reaching its goal.

    # Assumptions
    - Moving a box requires pushing it, which also requires the robot to be adjacent to the box.
    - The heuristic is not admissible, as it may underestimate the true cost.

    # Heuristic Initialization
    - Extract the goal locations for each box.
    - Extract the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of the robot and all boxes.
    2. For each box, find its goal location and calculate the Manhattan distance to it.
    3. Sum the Manhattan distances for all boxes to their respective goals.
    4. Find the closest box to the robot and calculate the Manhattan distance between them.
    5. Add the distance between the robot and the closest box to the total distance.
    6. Check if the robot is blocking any box from reaching its goal. If so, add a penalty.
    7. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                self.box_goals[parts[1]] = parts[2]

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
        """Estimate the number of moves required to solve the Sokoban problem."""
        state = node.state

        robot_location = None
        box_locations = {}

        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_location = parts[1]
            elif parts[0] == 'at':
                box_locations[parts[1]] = parts[2]

        if not robot_location:
            return float('inf')

        total_distance = 0
        for box, location in box_locations.items():
            if box in self.box_goals:
                goal_location = self.box_goals[box]
                total_distance += self.manhattan_distance(location, goal_location)
            else:
                # Box has no goal. The state is unsolvable.
                return float('inf')

        min_robot_distance = float('inf')
        for box, location in box_locations.items():
            distance = self.manhattan_distance(robot_location, location)
            min_robot_distance = min(min_robot_distance, distance)

        total_distance += min_robot_distance

        # Check for blocking boxes
        for box, location in box_locations.items():
            goal_location = self.box_goals[box]
            if self.is_blocking(location, goal_location, robot_location):
                total_distance += 5  # Add a penalty for blocking boxes

        # Check if the goal is reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        return total_distance

    def manhattan_distance(self, loc1, loc2):
        """Calculate the Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def is_blocking(self, box_location, goal_location, robot_location):
        """Check if the robot is blocking the box from reaching its goal."""
        # This is a simplified check and can be improved
        x_box, y_box = map(int, box_location.split('_')[1:])
        x_goal, y_goal = map(int, goal_location.split('_')[1:])
        x_robot, y_robot = map(int, robot_location.split('_')[1:])

        # Check if the robot is on the line between the box and the goal
        if x_box == x_goal and x_box == x_robot:
            if (y_box < y_robot < y_goal) or (y_goal < y_robot < y_box):
                return True
        if y_box == y_goal and y_box == y_robot:
            if (x_box < x_robot < x_goal) or (x_goal < x_robot < x_box):
                return True

        return False
