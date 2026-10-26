from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

class transport7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers the distance vehicles need to drive to pick up and deliver packages, as well as the required pick-up and drop actions. The heuristic assumes that each package can be transported by the best possible vehicle with available capacity.

    # Assumptions
    - Roads are bidirectional, allowing for undirected path calculations.
    - Vehicles can carry multiple packages if their capacity allows, but each pick-up and drop action is counted individually.
    - The shortest path between locations is used to estimate drive actions.
    - If a vehicle cannot currently pick up a package due to capacity, it is ignored, potentially overestimating the cost.

    # Heuristic Initialization
    - Extracts road network from static facts to precompute shortest paths between all locations.
    - Extracts capacity-predecessor relationships to determine valid vehicle capacities for picking up packages.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - For each package, determine if it's in a vehicle or at a location.
       - For each vehicle, note its current location and capacity.
    2. **Check Package Status**:
       - If a package is already at its goal, no cost is added.
       - If a package is in a vehicle, calculate the drive distance from the vehicle's current location to the goal and add a drop action.
       - If a package is not in a vehicle, find the best vehicle (minimal total drive distance) that can pick it up and compute the cost of pick-up, drive to goal, and drop.
    3. **Sum Costs**:
       - Accumulate the costs for all packages to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.goal_locations = {}
        self.predecessors = set()
        self.road_map = defaultdict(list)
        self.shortest_paths = {}

        # Extract goal locations for each package
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('p'):
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Extract static facts: roads and capacity-predecessors
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_map[l1].append(l2)
                self.road_map[l2].append(l1)
            elif parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.predecessors.add(s2)

        # Precompute shortest paths between all locations using BFS
        locations = set(self.road_map.keys())
        for loc in locations:
            self.shortest_paths[loc] = {}
            queue = deque([loc])
            self.shortest_paths[loc][loc] = 0
            visited = set([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.road_map[current]:
                    if neighbor not in visited:
                        self.shortest_paths[loc][neighbor] = self.shortest_paths[loc][current] + 1
                        visited.add(neighbor)
                        queue.append(neighbor)

    def __call__(self, node):
        state = node.state
        current_package_locations = {}
        current_package_vehicles = {}
        current_vehicle_locations = {}
        current_vehicle_capacities = {}

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if len(parts) == 0:
                continue
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj.startswith('p'):
                    current_package_locations[obj] = loc
                elif obj.startswith('v'):
                    current_vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                package = parts[1]
                vehicle = parts[2]
                current_package_vehicles[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle = parts[1]
                capacity = parts[2]
                current_vehicle_capacities[vehicle] = capacity

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            if package in current_package_vehicles:
                # Package is in a vehicle
                vehicle = current_package_vehicles[package]
                vehicle_loc = current_vehicle_locations.get(vehicle, None)
                if vehicle_loc is None:
                    continue  # Vehicle location unknown, skip
                # Drive from vehicle's current location to goal
                distance = self.shortest_paths.get(vehicle_loc, {}).get(goal_loc, float('inf'))
                if distance == float('inf'):
                    distance = 1000  # Penalize unreachable goals
                total_cost += distance + 1  # Drop action
            elif package in current_package_locations:
                current_loc = current_package_locations[package]
                if current_loc == goal_loc:
                    continue  # Already at goal
                # Find best vehicle to pick up the package
                min_cost = float('inf')
                for vehicle, capacity in current_vehicle_capacities.items():
                    if capacity not in self.predecessors:
                        continue  # Vehicle cannot pick up
                    vehicle_loc = current_vehicle_locations.get(vehicle, None)
                    if vehicle_loc is None:
                        continue
                    # Distance from vehicle to package's current location
                    dist1 = self.shortest_paths.get(vehicle_loc, {}).get(current_loc, float('inf'))
                    # Distance from package's location to goal
                    dist2 = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                    if dist1 == float('inf') or dist2 == float('inf'):
                        cost = 1000  # Penalize unreachable paths
                    else:
                        cost = dist1 + dist2 + 2  # Pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost == float('inf'):
                    min_cost = 1000  # Fallback cost
                total_cost += min_cost
            else:
                # Package not found in state (invalid state)
                pass

        return total_cost
