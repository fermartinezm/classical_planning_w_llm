from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


class transport23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to transport all packages to their goal locations. 
    For each package, it calculates the minimal steps needed considering whether it is in a vehicle or on the ground, 
    the vehicles' capacities, and the shortest path between locations using the road network.

    # Assumptions
    - Roads are directed, and the shortest paths between locations are precomputed.
    - Each package can be picked up by at least one vehicle with sufficient capacity.
    - Vehicles can carry multiple packages, but the heuristic counts each package's steps separately.

    # Heuristic Initialization
    - Extract road network from static facts and precompute all-pairs shortest paths using BFS.
    - Extract capacity predecessor relationships to determine valid vehicle capacities for picking up packages.
    - Extract goal locations for each package from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Precompute Shortest Paths**: For each location, use BFS to find the shortest path to all other locations.
    2. **Parse Current State**: Determine the current location of each package and vehicle, and whether packages are in a vehicle.
    3. **Evaluate Each Package**:
        - If the package is already at its goal, cost is 0.
        - If in a vehicle, calculate drive steps from the vehicle's current location to the goal plus a drop action.
        - If on the ground, find the vehicle with the minimal cost to pick it up (drive to package, pick-up, drive to goal, drop).
    4. **Sum Costs**: Aggregate the costs for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.goal_locations = {}
        self.capacity_predecessors = {}
        self.roads = defaultdict(list)
        self.shortest_paths = defaultdict(dict)  # start: {end: steps}

        # Extract goal locations for each package
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at' and len(parts) >= 3:
                package = parts[1]
                loc = parts[2]
                self.goal_locations[package] = loc

        # Extract static facts: roads and capacity-predecessors
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'road' and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                self.roads[l1].append(l2)
            elif parts[0] == 'capacity-predecessor' and len(parts) >= 3:
                s1, s2 = parts[1], parts[2]
                self.capacity_predecessors[s2] = s1

        # Precompute shortest paths between all locations using BFS
        all_locations = set()
        for l1 in self.roads:
            all_locations.add(l1)
            all_locations.update(self.roads[l1])
        all_locations = list(all_locations)

        for start in all_locations:
            visited = {}
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited[current] = dist
                for neighbor in self.roads.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            self.shortest_paths[start] = visited

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_package_locs = {}
        current_vehicle_locs = {}
        vehicle_caps = {}
        in_vehicle = {}

        for fact in state:
            parts = fact.strip('()').split()
            if not parts:
                continue
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj.startswith('p'):
                    current_package_locs[obj] = loc
                elif obj.startswith('v'):
                    current_vehicle_locs[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                vehicle_caps[vehicle] = cap

        total_cost = 0
        for package, goal_loc in self.goal_locations.items():
            if package in in_vehicle:
                # Package is in a vehicle
                vehicle = in_vehicle[package]
                vehicle_loc = current_vehicle_locs.get(vehicle, None)
                if vehicle_loc is None:
                    total_cost += 1000
                    continue
                steps = self.shortest_paths.get(vehicle_loc, {}).get(goal_loc, float('inf'))
                if steps == float('inf'):
                    total_cost += 1000
                else:
                    total_cost += steps + 1  # drive steps + drop
            else:
                # Package is on the ground
                current_loc = current_package_locs.get(package, None)
                if current_loc == goal_loc:
                    continue
                if current_loc is None:
                    total_cost += 1000
                    continue
                min_cost = float('inf')
                for vehicle in current_vehicle_locs:
                    cap = vehicle_caps.get(vehicle, None)
                    if cap not in self.capacity_predecessors:
                        continue  # Vehicle cannot pick up
                    vehicle_loc = current_vehicle_locs[vehicle]
                    d1 = self.shortest_paths.get(vehicle_loc, {}).get(current_loc, float('inf'))
                    d2 = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
                    if d1 != float('inf') and d2 != float('inf'):
                        cost = d1 + d2 + 2  # pick-up and drop
                        if cost < min_cost:
                            min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost
                else:
                    total_cost += 1000  # No valid path found
        return total_cost
