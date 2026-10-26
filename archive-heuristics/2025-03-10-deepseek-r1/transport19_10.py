from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transport19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the shortest path for vehicles to drive to required locations, and actions for picking up and dropping packages.
    Vehicle capacities are taken into account to determine if a package can be picked up.

    Assumptions:
    - Roads are directed, and the shortest path is computed using BFS.
    - A vehicle can pick up a package only if its current capacity is not the minimal (i.e., it has a predecessor in the capacity hierarchy).
    - Each package is handled independently, potentially overestimating when vehicles need to transport multiple packages.

    Heuristic Initialization:
    - Extracts road network from static facts to build a directed graph.
    - Extracts capacity-predecessor relationships to determine valid capacities for picking up packages.
    - Extracts goal locations for each package from the task's goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each package, check if it is already at its goal location.
    2. If the package is in a vehicle:
        a. Compute the shortest path from the vehicle's current location to the goal.
        b. Add drive actions plus one drop action.
    3. If the package is not in a vehicle:
        a. For each vehicle with valid capacity, compute the shortest path to the package's location and then to the goal.
        b. Add drive actions, pick-up, and drop actions.
        c. Use the minimum cost across all valid vehicles.
    4. Sum the costs for all packages to get the heuristic value.
    """

    def __init__(self, task):
        self.goal_locations = {}
        self.roads = defaultdict(list)
        self.capacity_predecessors = {}
        self.min_capacities = set()

        # Extract goal locations for each package
        for goal in task.goals:
            if not goal.startswith('(at '):
                continue
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                package = parts[1]
                loc = parts[2]
                self.goal_locations[package] = loc

        # Extract road network from static facts
        for fact in task.static:
            if fact.startswith('(road '):
                parts = fact[1:-1].split()
                if parts[0] == 'road' and len(parts) == 3:
                    from_loc, to_loc = parts[1], parts[2]
                    self.roads[from_loc].append(to_loc)

        # Extract capacity-predecessor relationships and determine minimal capacities
        all_capacities = set()
        for fact in task.static:
            if fact.startswith('(capacity-predecessor '):
                parts = fact[1:-1].split()
                if parts[0] == 'capacity-predecessor' and len(parts) == 3:
                    s1, s2 = parts[1], parts[2]
                    self.capacity_predecessors[s2] = s1
                    all_capacities.add(s1)
                    all_capacities.add(s2)
        # Minimal capacities are those not present as keys in capacity_predecessors
        self.min_capacities = {s for s in all_capacities if s not in self.capacity_predecessors}

    def shortest_path(self, start, end):
        """Compute the shortest path length from start to end using BFS."""
        if start == end:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            loc, dist = queue.popleft()
            if loc == end:
                return dist
            if loc in visited:
                continue
            visited.add(loc)
            for neighbor in self.roads.get(loc, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found (assumed solvable)

    def __call__(self, node):
        state = node.state
        current_packages = {}
        current_vehicles = {}
        current_capacities = {}

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if not parts:
                continue
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj.startswith('p'):
                    current_packages[obj] = ('at', loc)
                elif obj.startswith('v'):
                    current_vehicles[obj] = loc
            elif parts[0] == 'in':
                package = parts[1]
                vehicle = parts[2]
                current_packages[package] = ('in', vehicle)
            elif parts[0] == 'capacity':
                vehicle = parts[1]
                capacity = parts[2]
                current_capacities[vehicle] = capacity

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            if package not in current_packages:
                continue  # Package not in state (invalid case)
            current_info = current_packages[package]

            # Check if package is already at goal
            if current_info[0] == 'at' and current_info[1] == goal_loc:
                continue

            # Compute cost based on current state
            if current_info[0] == 'in':
                # Package is in a vehicle: drive to goal and drop
                vehicle = current_info[1]
                vehicle_loc = current_vehicles.get(vehicle, None)
                if not vehicle_loc:
                    continue
                distance = self.shortest_path(vehicle_loc, goal_loc)
                total_cost += distance + 1  # drive + drop
            else:
                # Package is at a location: find best vehicle to pick up
                package_loc = current_info[1]
                min_cost = float('inf')
                for vehicle, vehicle_loc in current_vehicles.items():
                    # Check if vehicle can pick up (capacity is not minimal)
                    vehicle_cap = current_capacities.get(vehicle, None)
                    if vehicle_cap in self.min_capacities:
                        continue  # Vehicle cannot pick up
                    # Compute distances
                    d1 = self.shortest_path(vehicle_loc, package_loc)
                    d2 = self.shortest_path(package_loc, goal_loc)
                    cost = d1 + d2 + 2  # pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost
