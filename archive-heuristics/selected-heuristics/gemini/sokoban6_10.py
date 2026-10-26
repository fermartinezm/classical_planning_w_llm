from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their goal locations.
    It calculates the sum of Manhattan distances between each box's current location and its goal location.

    # Assumptions
    - Each push action moves a box one location closer to its goal.
    - The robot can always move to push a box.
    - The heuristic does not take into account obstacles or other boxes.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goal conditions.
    - Extract the adjacency information from the static facts to calculate Manhattan distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each box from the state.
    2. For each box, find its goal location.
    3. Calculate the Manhattan distance between the box's current location and its goal location.
       - Use the adjacency information to determine the steps needed to move between locations.
    4. Sum the Manhattan distances for all boxes.
    5. Return the total sum as the heuristic value.
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

    def __call__(self, node):
        """
        Estimate the number of actions needed to move all boxes to their goal locations.
        """
        state = node.state

        # Extract current box locations
        box_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] != 'robot':
                box = parts[1]
                location = parts[2]
                box_locations[box] = location

        # Calculate total Manhattan distance
        total_distance = 0
        for box, goal_location in self.box_goals.items():
            current_location = box_locations.get(box)
            if current_location is None:
                # Box is not in the state, which means the state is not reachable
                return float('inf')
            distance = self.manhattan_distance(current_location, goal_location)
            total_distance += distance

        return total_distance

    def manhattan_distance(self, start, end):
        """
        Calculate the Manhattan distance between two locations using the adjacency information.
        """
        # This implementation assumes a grid-like structure and uses a simple BFS to find the shortest path.
        queue = [(start, 0)]  # (location, distance)
        visited = {start}

        while queue:
            location, distance = queue.pop(0)

            if location == end:
                return distance

            if location in self.adjacencies:
                for direction, neighbor in self.adjacencies[location].items():
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        visited.add(neighbor)

        # If no path is found, return infinity
        return float('inf')
