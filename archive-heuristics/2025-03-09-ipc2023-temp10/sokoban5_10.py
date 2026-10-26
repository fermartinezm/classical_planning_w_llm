from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box and its closest goal location,
    plus a penalty if the robot is far from the boxes.

    # Assumptions
    - The heuristic assumes that each box can be moved independently to its goal location.
    - It does not take into account blocking situations or deadlocks.
    - It assumes that the robot can always reach a box to push it.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task goals.
    - Extract the adjacency information between locations from the static facts.
    - Build a dictionary mapping each location to its coordinates (row, col).
    - Build a dictionary mapping each location to its adjacent locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes and the robot from the state.
    2. For each box, find the Manhattan distance to its closest goal location.
    3. Sum up the Manhattan distances for all boxes.
    4. Calculate the minimum distance from the robot to any of the boxes.
    5. If the robot is far from the boxes (distance > 2), add a penalty to the heuristic value.
       This encourages the robot to stay close to the boxes.
    6. If the state is a goal state, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.box_goals = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                box = parts[1]
                location = parts[2]
                self.box_goals.setdefault(box, []).append(location)

        self.adjacencies = {}
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                self.adjacencies.setdefault(loc1, []).append(loc2)
                self.adjacencies.setdefault(loc2, []).append(loc1)

        # Extract location coordinates from object names (e.g., loc_2_3 -> (2, 3))
        self.location_coords = {}
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.location_coords:
                    try:
                        row, col = map(int, loc1.split('_')[1:])
                        self.location_coords[loc1] = (row, col)
                    except:
                        pass
                if loc2 not in self.location_coords:
                    try:
                        row, col = map(int, loc2.split('_')[1:])
                        self.location_coords[loc2] = (row, col)
                    except:
                        pass

    def __call__(self, node):
        """Estimate the number of actions needed to move all boxes to their goal locations."""
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

        if all(fact in state for fact in self.goals):
            return 0

        total_distance = 0
        for box, location in box_locations.items():
            min_distance = float('inf')
            for goal_location in self.box_goals.get(box, []):
                if location in self.location_coords and goal_location in self.location_coords:
                    row1, col1 = self.location_coords[location]
                    row2, col2 = self.location_coords[goal_location]
                    distance = abs(row1 - row2) + abs(col1 - col2)
                    min_distance = min(min_distance, distance)
                else:
                    min_distance = 10 # Assign a default value if location coordinates are not available
            total_distance += min_distance

        # Robot proximity penalty
        min_robot_distance = float('inf')
        if robot_location:
            for location in box_locations.values():
                if robot_location in self.location_coords and location in self.location_coords:
                    row1, col1 = self.location_coords[robot_location]
                    row2, col2 = self.location_coords[location]
                    distance = abs(row1 - row2) + abs(col1 - col2)
                    min_robot_distance = min(min_robot_distance, distance)
                else:
                    min_robot_distance = 10 # Assign a default value if location coordinates are not available

        if min_robot_distance > 2:
            total_distance += 5  # Add a penalty if the robot is far from the boxes

        return total_distance
