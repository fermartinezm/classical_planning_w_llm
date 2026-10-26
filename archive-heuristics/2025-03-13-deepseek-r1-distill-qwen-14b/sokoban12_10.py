from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal locations. It calculates the Manhattan distance from each box's current position to its goal and from the robot's current position to the box's current position. The sum of these distances for all boxes gives the heuristic value.

    # Assumptions:
    - The grid layout is consistent and can be parsed from static facts.
    - Boxes can be pushed one at a time.
    - The Manhattan distance is used as an approximation of the required movements.

    # Heuristic Initialization
    - Extract goal locations for each box from the task's goals.
    - Parse static facts to create a grid map of location coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot.
    2. For each box, determine its current location and whether it's at the goal.
    3. For each box not at the goal, compute:
       a. The Manhattan distance from the box's current location to its goal.
       b. The Manhattan distance from the robot's current location to the box's current location.
    4. Sum these distances for all boxes not at their goals.
    5. The total sum is the heuristic value.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) >= 3 and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Parse static facts to get location coordinates
        self.location_coords = {}
        for fact in self.static:
            if fact.startswith('(adjacent'):
                parts = fact[1:-1].split()
                if parts[0] == 'adjacent' and len(parts) >= 3:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    self.location_coords[loc1] = self.parse_loc(loc1)
                    self.location_coords[loc2] = self.parse_loc(loc2)

    def parse_loc(self, loc_str):
        """Parse a location string like 'loc_2_3' into (x, y) coordinates."""
        parts = loc_str.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return (x, y)

    def manhattan_distance(self, loc1, loc2):
        """Compute the Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_positions = {}

        # Find robot's current location
        for fact in state:
            if fact.startswith('(at-robot'):
                robot_loc_str = fact[1:-1].split()[1]
                robot_loc = self.location_coords[robot_loc_str]
                break

        # Find current positions of boxes
        for fact in state:
            if fact.startswith('(at'):
                parts = fact[1:-1].split()
                if len(parts) >= 3 and parts[1].startswith('box'):
                    box = parts[1]
                    loc = parts[2]
                    box_positions[box] = self.location_coords[loc]

        total_cost = 0

        # For each box, calculate the required distance
        for box, current_loc in box_positions.items():
            if box in self.goal_locations:
                goal_loc_str = self.goal_locations[box]
                goal_loc = self.location_coords[goal_loc_str]
                if current_loc == goal_loc:
                    continue  # Already at goal

                # Distance from box to goal
                box_to_goal = self.manhattan_distance(current_loc, goal_loc)

                # Distance from robot to box
                robot_to_box = self.manhattan_distance(robot_loc, current_loc)

                total_cost += box_to_goal + robot_to_box

        return total_cost
