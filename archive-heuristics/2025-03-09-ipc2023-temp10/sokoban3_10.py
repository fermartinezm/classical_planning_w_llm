from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class sokoban3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.
    Additionally, it adds a penalty if the robot is not adjacent to the box it needs to push.

    # Assumptions:
    - The heuristic assumes that each box can be moved independently without considering other boxes' positions.
    - It also assumes that the robot can always reach a position adjacent to the box it needs to push.
    - The heuristic does not account for deadlocks or situations where a box is trapped.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Store the adjacency information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes and the robot from the state.
    2. For each box, calculate the Manhattan distance between its current location and its goal location.
       The Manhattan distance is calculated as the sum of the absolute differences of the x and y coordinates of the two locations.
       To determine the x and y coordinates, we assume that the location names follow the format "loc_x_y".
    3. If the robot is not adjacent to a box that needs to be moved, add a penalty to the heuristic value.
       The penalty is the Manhattan distance between the robot and the closest location adjacent to the box.
    4. Sum the Manhattan distances for all boxes to get the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.box_goals = {}
        for goal in self.goals:
            if goal.startswith('(at box'):
                parts = goal[1:-1].split()
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        self.adjacency = {}
        for fact in static_facts:
            if fact.startswith('(adjacent'):
                parts = fact[1:-1].split()
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = []
                self.adjacency[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        box_locations = {}
        robot_location = None

        for fact in state:
            if fact.startswith('(at box'):
                parts = fact[1:-1].split()
                box = parts[1]
                location = parts[2]
                box_locations[box] = location
            elif fact.startswith('(at-robot'):
                parts = fact[1:-1].split()
                robot_location = parts[1]

        if all(box_locations.get(box) == self.box_goals.get(box) for box in self.box_goals):
            return 0

        total_distance = 0
        for box, current_location in box_locations.items():
            goal_location = self.box_goals[box]

            x1, y1 = map(int, current_location.split('_')[1:])
            x2, y2 = map(int, goal_location.split('_')[1:])
            total_distance += abs(x1 - x2) + abs(y1 - y2)

            # Check if robot is adjacent to the box
            adjacent_locations = []
            for loc in self.adjacency.keys():
                if current_location in loc and loc in self.adjacency:
                    adjacent_locations.extend(self.adjacency[loc])
            
            is_adjacent = False
            for adj_loc in adjacent_locations:
                if adj_loc in self.adjacency and robot_location in adj_loc:
                    is_adjacent = True
                    break
            
            if not is_adjacent:
                # Calculate Manhattan distance to closest adjacent location
                min_robot_distance = float('inf')
                for adj_loc in self.adjacency[current_location]:
                    x3, y3 = map(int, adj_loc.split('_')[1:])
                    xr, yr = map(int, robot_location.split('_')[1:])
                    robot_distance = abs(xr - x3) + abs(yr - y3)
                    min_robot_distance = min(min_robot_distance, robot_distance)
                total_distance += min_robot_distance

        return total_distance
