from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.

    # Assumptions:
    - There is only one vehicle in the domain.
    - Packages can be either on the ground or inside the vehicle.
    - The vehicle can move between any two connected locations in one drive action.

    # Heuristic Initialization
    - Extracts the goal locations for each package.
    - Builds a graph of the road network from static facts.
    - Precomputes the shortest path distances between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine if it is already at its goal location. If yes, no actions are needed.
    2. If the package is not at the goal, calculate the drive actions needed for the vehicle to reach the package's current location.
    3. Calculate the drive actions needed from the package's location to the goal location.
    4. Sum the drive actions, plus 2 actions for pickup and drop.
    5. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal locations and building the road network."""
        self.goals = task.goals
        self.static = task.static

        # Extract all locations from static facts
        self.locations = set()
        for fact in self.static:
            if fact.startswith('(road'):
                l1, l2 = fact[1:-1].split()
                self.locations.add(l1)
                self.locations.add(l2)
        
        # Build adjacency list for the road network
        self.graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            if fact.startswith('(road'):
                l1, l2 = fact[1:-1].split()
                self.graph[l1].add(l2)
                self.graph[l2].add(l1)
        
        # Precompute shortest paths between all pairs of locations using BFS
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = {loc: True}
            while queue:
                current, dist = queue.popleft()
                self.distances[loc][current] = dist
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Find the vehicle's current location
        vehicle_location = None
        for fact in state:
            if fact.startswith('(at'):
                parts = fact[1:-1].split()
                if parts[1] == 'vehicle':
                    vehicle_location = parts[2]
                    break
        
        if vehicle_location is None:
            return 0  # No vehicle found, should not happen in transport domain

        total_actions = 0

        # Process each package
        for package in self.goal_locations:
            # Determine the package's current location
            package_location = None
            for fact in state:
                if fact.startswith('(at'):
                    parts = fact[1:-1].split()
                    if parts[1] == package:
                        package_location = parts[2]
                        break
                elif fact.startswith('(in'):
                    parts = fact[1:-1].split()
                    if parts[2] == package:
                        package_location = vehicle_location
                        break
            
            if package_location is None:
                continue  # Package is not present, should not happen

            goal_location = self.goal_locations[package]

            if package_location == goal_location:
                continue  # No actions needed

            # Calculate drive actions from vehicle to package location
            if package_location in self.distances[vehicle_location]:
                d1 = self.distances[vehicle_location][package_location]
            else:
                d1 = 0  # No path found, assume 0 (should not happen)

            # Calculate drive actions from package location to goal
            if goal_location in self.distances[package_location]:
                d2 = self.distances[package_location][goal_location]
            else:
                d2 = 0  # No path found, assume 0 (should not happen)

            total_actions += d1 + d2 + 2  # +2 for pickup and drop

        return total_actions
