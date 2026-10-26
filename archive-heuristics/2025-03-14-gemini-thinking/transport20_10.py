from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their respective goal locations. It considers the necessary drive, pick-up, and drop actions.

    # Assumptions:
    - Each package needs to be picked up, transported, and dropped at its goal location.
    - The heuristic estimates the driving cost based on the shortest path in terms of roads between locations.
    - It assumes that a vehicle is always available to transport a package once it's picked up.
    - It does not explicitly consider vehicle capacity or the capacity-predecessor constraints in detail for heuristic calculation, focusing on the sequence of actions for each package.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Pre-computes the road network as an adjacency list to efficiently calculate shortest paths between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location in the current state:
    1. Determine the current location of the package. If the package is 'in' a vehicle, the vehicle's location is considered the package's current location for pathfinding purposes.
    2. Identify the goal location for the package.
    3. Calculate the shortest path (number of roads) between the current location and the goal location using Breadth-First Search (BFS) on the road network. If the current location and goal location are the same, the path length is 0.
    4. Estimate the cost for this package as the shortest path length + 2 (for one pick-up and one drop action).
    5. Sum up the estimated costs for all packages that are not at their goal locations. This sum is the total heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Builds a graph representing the road network from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                self.goal_locations[parts[1]] = parts[2]

        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            parts = fact[1:-1].split()
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_network[l1].append(l2)
                self.road_network[l2].append(l1) # Roads are bidirectional

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not at its goal location, estimate the number of actions
        (drive, pick-up, drop) needed to reach the goal.
        """
        state = node.state
        heuristic_value = 0

        current_package_locations = {}
        vehicle_locations = {}

        for fact in state:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == 'at':
                obj_type = None
                for obj_name in parts[1:]:
                    if obj_name in task.objects and task.objects[obj_name] == 'package':
                        current_package_locations[parts[1]] = parts[2]
                        obj_type = 'package'
                        break
                    elif obj_name in task.objects and task.objects[obj_name] == 'vehicle':
                        vehicle_locations[parts[1]] = parts[2]
                        obj_type = 'vehicle'
                        break
            elif predicate == 'in':
                current_package_locations[parts[1]] = parts[2] # package is in vehicle, location is vehicle

        for package, goal_location in self.goal_locations.items():
            current_location = None
            if package in current_package_locations:
                current_location = current_package_locations[package]
            else:
                # Package location not found in state, assume initial location (worst case, heuristic will be high)
                initial_location_fact = next((fact for fact in task.initial_state if fact.startswith(f'(at {package} ')), None)
                if initial_location_fact:
                    current_location = initial_location_fact[1:-1].split()[2]
                else:
                    continue # Cannot determine current location, skip this package for heuristic

            if current_location != goal_location:
                shortest_path = self.shortest_path_length(current_location, goal_location)
                heuristic_value += shortest_path + 2 # 2 for pick-up and drop

        return heuristic_value

    def shortest_path_length(self, start_location, goal_location):
        """
        Calculate the shortest path length between two locations using BFS on the road network.
        Returns the path length, or infinity if no path exists.
        """
        if start_location == goal_location:
            return 0

        queue = collections.deque([(start_location, 0)]) # (location, distance)
        visited = {start_location}

        while queue:
            current_location, distance = queue.popleft()

            if current_location == goal_location:
                return distance

            for neighbor in self.road_network[current_location]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf') # No path found
