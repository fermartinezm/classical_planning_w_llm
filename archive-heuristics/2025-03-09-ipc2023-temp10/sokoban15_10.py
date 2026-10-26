from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to solve a Sokoban problem by considering the Manhattan distances
    between each box and its goal location, and the robot's distance to the nearest box. It also considers
    the number of boxes and the number of goals.

    # Assumptions
    - The cost of moving a box is higher than the cost of moving the robot.
    - The heuristic is not admissible.
    - The heuristic assumes that all boxes must be at their goal locations simultaneously.

    # Heuristic Initialization
    - Extract the initial locations of the robot and boxes.
    - Extract the goal locations of the boxes.
    - Store the adjacency information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of the robot and boxes from the state.
    2. Calculate the Manhattan distance between each box and its closest goal location.
    3. Calculate the minimum Manhattan distance from the robot to any box.
    4. Sum the Manhattan distances for all boxes.
    5. Add the minimum robot-to-box distance to the sum.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, initial state, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

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
        """
        Estimate the cost to reach the goal state from the given state.
        """
        state = node.state

        # Extract robot location
        robot_location = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_location = parts[1]
                break

        # Extract box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] in self.box_goals:
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # If any box location is missing, return a large value
        if len(box_locations) != len(self.box_goals):
            return 1000000

        # If the state is a goal state, return 0
        goal_reached = True
        for box, goal_location in self.box_goals.items():
            if box not in box_locations or box_locations[box] != goal_location:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Calculate Manhattan distances between boxes and their goal locations
        total_box_distance = 0
        for box, box_location in box_locations.items():
            goal_location = self.box_goals[box]
            box_distance = self.manhattan_distance(box_location, goal_location)
            total_box_distance += box_distance

        # Calculate the minimum Manhattan distance from the robot to any box
        min_robot_distance = float('inf')
        for box, box_location in box_locations.items():
            robot_distance = self.manhattan_distance(robot_location, box_location)
            min_robot_distance = min(min_robot_distance, robot_distance)

        # Return the estimated cost
        return total_box_distance + min_robot_distance

    def manhattan_distance(self, loc1, loc2):
        """
        Calculate the Manhattan distance between two locations.
        """
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)
