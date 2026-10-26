from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.
    This provides a lower bound on the number of moves required, as it doesn't account for obstacles or the robot's movements.

    # Assumptions
    - The heuristic assumes that boxes can be moved independently without considering other boxes' positions.
    - It also assumes that the robot can always reach a position to push a box in the desired direction.
    - It does not consider deadlocks or unsolvable configurations.

    # Heuristic Initialization
    - The heuristic extracts the goal locations for each box from the task's goal conditions.
    - It also extracts the adjacency information between locations from the static facts to compute Manhattan distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all boxes from the state.
    2. For each box, find its goal location.
    3. Calculate the Manhattan distance between the box's current location and its goal location.
       - The Manhattan distance is calculated by summing the number of horizontal and vertical moves required to reach the goal,
         based on the adjacency information extracted from the static facts.
    4. Sum the Manhattan distances for all boxes.
    5. If the state is a goal state, return 0.
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

        # Extract adjacency information for Manhattan distance calculation
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

    def __call__(self, node):
        """
        Estimate the number of actions needed to move all boxes to their goal locations.
        """
        state = node.state

        # Check if the state is a goal state
        if self.is_goal_state(state):
            return 0

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] in self.box_goals:
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Calculate the sum of Manhattan distances for all boxes
        total_distance = 0
        for box, goal_location in self.box_goals.items():
            if box in box_locations:
                current_location = box_locations[box]
                distance = self.manhattan_distance(current_location, goal_location)
                total_distance += distance
            else:
                # Box is not in the state, which means the state is not reachable from initial state.
                return float('inf')

        return total_distance

    def manhattan_distance(self, start, end):
        """
        Calculate the Manhattan distance between two locations based on adjacency information.
        """
        visited = {start}
        queue = [(start, 0)]  # (location, distance)

        while queue:
            current_location, distance = queue.pop(0)

            if current_location == end:
                return distance

            if current_location in self.adjacencies:
                for direction, next_location in self.adjacencies[current_location].items():
                    if next_location not in visited:
                        visited.add(next_location)
                        queue.append((next_location, distance + 1))

        # If the goal is unreachable, return infinity
        return float('inf')

    def is_goal_state(self, state):
        """
        Check if the given state is a goal state.
        """
        for goal in self.goals:
            if goal not in state:
                return False
        return True
