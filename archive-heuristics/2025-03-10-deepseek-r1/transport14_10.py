import math
from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transport14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations. It considers the minimal driving steps between locations, pick-up and drop actions, and vehicle capacity constraints.

    # Assumptions:
    - Roads are directed, and shortest paths are precomputed.
    - Vehicles can carry packages if their capacity allows.
    - Each package is handled independently, potentially overestimating when multiple packages can be transported together.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build a road graph from static facts and precompute shortest paths between all locations.
    - Extract capacity-predecessor relationships to determine valid pick-up and drop actions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If already at the goal, contribute 0.
        b. If in a vehicle:
            i. Check if the vehicle can drop the package (based on capacity).
            ii. Compute driving steps from vehicle's location to goal.
            iii. Add drive steps + 1 (drop action).
        c. If not in a vehicle:
            i. Find the closest vehicle that can pick it up (considering capacity and distance).
            ii. Compute drive steps for the vehicle to reach the package.
            iii. Compute drive steps from package's location to goal.
            iv. Add drive steps + 2 (pick-up and drop).
    2. Sum the costs for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                self.goal_locations[parts[1]] = parts[2]

        # Build road graph and capacity predecessors
        self.road_map = defaultdict(list)
        self.capacity_predecessors = {}
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'road':
                self.road_map[parts[1]].append(parts[2])
            elif parts[0] == 'capacity-predecessor':
                self.capacity_predecessors[parts[2]] = parts[1]

        # Precompute shortest paths
        self.shortest_paths = {}
        all_locations = set(self.road_map.keys())
        for connections in self.road_map.values():
            all_locations.update(connections)
        for loc in all_locations:
            self.shortest_paths[loc] = self._bfs(loc)

    def _bfs(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.road_map.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state
        vehicles = {}
        vehicle_capacities = {}
        package_info = {}

        # Extract vehicle locations and capacities
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('v'):
                vehicles[parts[1]] = parts[2]
            elif parts[0] == 'capacity' and parts[1].startswith('v'):
                vehicle_capacities[parts[1]] = parts[2]

        # Determine package locations and vehicle status
        for package in self.goal_locations:
            current_loc = None
            in_vehicle = False
            # Check if package is in a vehicle
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == 'in' and parts[1] == package:
                    vehicle = parts[2]
                    current_loc = vehicles.get(vehicle)
                    in_vehicle = True
                    break
            if not in_vehicle:
                # Find package's current location
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == 'at' and parts[1] == package:
                        current_loc = parts[2]
                        break
            package_info[package] = (current_loc, in_vehicle)

        total_cost = 0
        s_prev_values = set(self.capacity_predecessors.values())

        for package, (current_loc, in_vehicle) in package_info.items():
            goal_loc = self.goal_locations.get(package)
            if current_loc == goal_loc:
                continue
            if current_loc is None or goal_loc is None:
                return float('inf')

            if in_vehicle:
                # Find vehicle and check capacity
                vehicle = None
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == 'in' and parts[1] == package:
                        vehicle = parts[2]
                        break
                if not vehicle:
                    return float('inf')
                capacity = vehicle_capacities.get(vehicle)
                if not capacity or capacity not in s_prev_values:
                    return float('inf')

                drive_steps = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                if drive_steps == float('inf'):
                    return float('inf')
                total_cost += drive_steps + 1
            else:
                # Find closest capable vehicle
                min_drive = float('inf')
                for v, v_loc in vehicles.items():
                    cap = vehicle_capacities.get(v)
                    if not cap or cap not in self.capacity_predecessors:
                        continue
                    steps = self.shortest_paths.get(v_loc, {}).get(current_loc, float('inf'))
                    if steps < min_drive:
                        min_drive = steps
                if min_drive == float('inf'):
                    return float('inf')

                drive_to_goal = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                if drive_to_goal == float('inf'):
                    return float('inf')

                total_cost += min_drive + drive_to_goal + 2

        return total_cost
