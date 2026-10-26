from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations. It considers the shortest path between locations and whether packages are already in transit or on the ground.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between locations.
    - The goal is to have all packages at their specified target locations.
    - If a package is already at its goal, no actions are needed for it.

    # Heuristic Initialization
    - Extracts static facts including road connections and package goal locations.
    - Precomputes the shortest paths between all relevant locations using Dijkstra's algorithm.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it is inside a vehicle.
    2. Find the shortest path from the current location to the goal location using precomputed paths.
    3. Calculate the number of drive actions needed as the length of the shortest path minus one.
    4. If the package is on the ground, add one action for picking it up.
    5. If the package is inside a vehicle, add one action for dropping it at the goal.
    6. Sum the actions required for all packages to estimate the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and precomputing shortest paths."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract road connections
        self.roads = {}
        for fact in static_facts:
            if fnmatch(fact, '(road * *)'):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

        # Precompute shortest paths between all locations using Dijkstra's algorithm
        self.locations = set()
        for fact in static_facts:
            if fnmatch(fact, '(road * *)'):
                parts = fact[1:-1].split()
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Precompute all pairs shortest paths
        self.shortest_paths = {}
        for loc in self.locations:
            self.shortest_paths[loc] = {}
            visited = set()
            queue = [(0, loc)]
            while queue:
                dist, current = heapq.heappop(queue)
                if current in visited:
                    continue
                visited.add(current)
                for neighbor in self.roads.get(current, []):
                    if neighbor not in visited:
                        new_dist = dist + 1
                        if neighbor not in self.shortest_paths[loc] or new_dist < self.shortest_paths[loc][neighbor]:
                            self.shortest_paths[loc][neighbor] = new_dist
                            heapq.heappush(queue, (new_dist, neighbor))

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_actions = 0

        # Track where packages and vehicles are currently located
        current_locations = {}
        in_vehicle = {}
        for fact in state:
            if fact.startswith('(at ') and ' ' in fact:
                parts = fact[1:-1].split()
                obj = parts[1]
                loc = parts[2]
                current_locations[obj] = loc
            elif fact.startswith('(in ') and ' ' in fact:
                parts = fact[1:-1].split()
                package = parts[1]
                vehicle = parts[2]
                in_vehicle[package] = vehicle

        # For each package, determine its current location and whether it's in a vehicle
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations:
                # Package is already at goal
                continue

            current_loc = current_locations[package]
            if current_loc == goal_location:
                continue  # Already at goal

            # Determine if the package is in a vehicle
            if package in in_vehicle:
                # Package is inside a vehicle; need to drop it
                vehicle = in_vehicle[package]
                # Vehicle must be at current_loc, which is the package's current location
                # So, we need to move the vehicle to the goal location
                # First, check if the vehicle is already at the goal location
                if current_loc == goal_location:
                    continue  # Already at goal
                # Calculate the drive actions needed
                if goal_location in self.shortest_paths[current_loc]:
                    path_length = self.shortest_paths[current_loc][goal_location]
                    total_actions += 1  # Drop action
                    total_actions += path_length  # Drive actions
                else:
                    # No path exists; heuristic is infinity (problem unsolvable)
                    return float('inf')
            else:
                # Package is on the ground; need to pick it up and move it
                # First, check if the vehicle is already at the goal location
                # Vehicle must be at current_loc to pick up the package
                # Then, drive to the goal location
                # So, total actions: pick up (1) + drive actions (path_length) + drop (1)
                # But if the vehicle is already at current_loc, no need to move it
                # So, we assume a vehicle is available at current_loc
                # So, actions: pick up (1) + drive actions (path_length) + drop (1)
                if current_loc == goal_location:
                    continue  # Already at goal
                if goal_location in self.shortest_paths[current_loc]:
                    path_length = self.shortest_paths[current_loc][goal_location]
                    total_actions += 1  # Pick up
                    total_actions += path_length  # Drive actions
                    total_actions += 1  # Drop
                else:
                    # No path exists; heuristic is infinity (problem unsolvable)
                    return float('inf')

        return total_actions
