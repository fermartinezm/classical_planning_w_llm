from fnmatch import fnmatch
import itertools

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Transport22Heuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations.
    Considers drive actions, pick-up/drop actions, vehicle capacities, and road network distances.

    # Assumptions
    - Each package requires a separate pick-up and drop action.
    - Vehicles can only pick up packages if their current capacity allows.
    - Roads are directional; shortest paths are precomputed using Floyd-Warshall.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds a road graph from static facts and computes all-pairs shortest paths.
    - Identifies valid vehicle capacities for picking up packages using capacity-predecessor relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package:
        a. If already at goal: contribute 0.
        b. If in a vehicle: compute drive steps from vehicle's location to goal + 1 (drop).
        c. If not in a vehicle: find the best vehicle (min drive steps to package, then to goal) + 2 (pick-up and drop).
    2. Sum the minimal actions for all packages.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build road graph and compute shortest paths
        road_facts = [fact for fact in task.static if match(fact, 'road', '*', '*')]
        locations = set()
        for fact in road_facts:
            parts = get_parts(fact)
            locations.add(parts[1])
            locations.add(parts[2])
        self.locations = list(locations)
        n = len(self.locations)
        self.location_index = {loc: i for i, loc in enumerate(self.locations)}

        INF = float('inf')
        self.distance = [[INF] * n for _ in range(n)]
        for i in range(n):
            self.distance[i][i] = 0
        for fact in road_facts:
            parts = get_parts(fact)
            l1, l2 = parts[1], parts[2]
            i, j = self.location_index[l1], self.location_index[l2]
            self.distance[i][j] = 1  # Each road is one action

        # Floyd-Warshall algorithm for shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distance[i][k] + self.distance[k][j] < self.distance[i][j]:
                        self.distance[i][j] = self.distance[i][k] + self.distance[k][j]

        # Extract allowed capacities (s2 where capacity-predecessor exists)
        self.allowed_capacities = set()
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                parts = get_parts(fact)
                self.allowed_capacities.add(parts[2])

    def __call__(self, node):
        state = node.state
        package_at = {}
        package_in_vehicle = {}
        vehicle_location = {}
        vehicle_capacity = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_at[obj] = loc
                else:
                    vehicle_location[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
            elif parts[0] == 'capacity':
                vehicle, cap = parts[1], parts[2]
                vehicle_capacity[vehicle] = cap

        total = 0
        for package, goal_loc in self.goal_locations.items():
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                if vehicle not in vehicle_location:
                    continue
                current_loc = vehicle_location[vehicle]
                if current_loc == goal_loc:
                    total += 1  # Drop action only
                    continue
                if current_loc not in self.location_index or goal_loc not in self.location_index:
                    continue
                i = self.location_index[current_loc]
                j = self.location_index[goal_loc]
                steps = self.distance[i][j]
                if steps != float('inf'):
                    total += steps + 1  # Drive steps + drop
            else:
                if package not in package_at or package_at[package] == goal_loc:
                    continue
                current_loc = package_at[package]
                min_cost = float('inf')
                for vehicle in vehicle_location:
                    if vehicle not in vehicle_capacity or vehicle_capacity[vehicle] not in self.allowed_capacities:
                        continue
                    v_loc = vehicle_location[vehicle]
                    if v_loc not in self.location_index or current_loc not in self.location_index:
                        continue
                    i = self.location_index[v_loc]
                    j = self.location_index[current_loc]
                    drive1 = self.distance[i][j]
                    if drive1 == float('inf'):
                        continue
                    k = self.location_index[goal_loc]
                    drive2 = self.distance[j][k]
                    if drive2 == float('inf'):
                        continue
                    cost = drive1 + drive2 + 2  # pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost != float('inf'):
                    total += min_cost
        return total
