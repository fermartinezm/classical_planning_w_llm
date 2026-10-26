from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions required to move all packages to their goal locations.
    For each package, the heuristic considers the minimal driving distance for a vehicle to pick it up,
    transport it to the goal, and includes pick-up/drop actions. Vehicle capacities are considered, and
    penalties are applied if no available vehicle can carry the package.

    # Assumptions
    - Roads are bidirectional and form a connected graph.
    - Vehicles can carry multiple packages up to their capacity.
    - A vehicle must return to a location to drop a package there.
    - If a vehicle is full, a large penalty is added to encourage freeing capacity.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static road information.
    - Build a size hierarchy from capacity-predecessor facts to determine vehicle capacities.
    - Extract goal locations for each package.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package at goal: contribute 0.
    2. For each package in a vehicle:
        a. Calculate driving distance from vehicle's current location to goal.
        b. Add 1 action for dropping the package.
    3. For each package on the ground:
        a. Find the closest vehicle with available capacity.
        b. Calculate driving distance from vehicle's location to package and then to goal.
        c. Add 2 actions for pick-up and drop.
        d. If no vehicle is available, add a large penalty.
    4. Sum all costs for a total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize with precomputed shortest paths and size capacities."""
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('p'):
                self.goal_locations[parts[1]] = parts[2]

        # Build road graph from static facts
        self.road_graph = defaultdict(list)
        self.all_locations = set()
        for fact in task.static:
            if match(fact, 'road', '*', '*'):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1)
                self.all_locations.update([l1, l2])

        # Precompute all-pairs shortest paths
        self.shortest_paths = {}
        for location in self.all_locations:
            self.shortest_paths[location] = {}
            queue = deque([(location, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.shortest_paths[location][current] = dist
                for neighbor in self.road_graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

        # Build size hierarchy
        size_predecessors = {}
        sizes = set()
        for fact in task.static:
            if match(fact, 'capacity-predecessor', '*', '*'):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                size_predecessors[s2] = s1
                sizes.add(s1)
                sizes.add(s2)
        
        self.size_value = {}
        for size in sizes:
            current = size
            count = 0
            while current in size_predecessors:
                current = size_predecessors[current]
                count += 1
            self.size_value[size] = count

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        package_locations = {}
        vehicle_locations = {}
        vehicle_capacities = {}
        vehicle_loads = defaultdict(int)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'):
                    package_locations[obj] = loc
                else:
                    vehicle_locations[obj] = loc
            elif parts[0] == 'capacity':
                veh, size = parts[1], parts[2]
                vehicle_capacities[veh] = size
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                package_locations[pkg] = veh
                vehicle_loads[veh] += 1

        total = 0
        for pkg, goal_loc in self.goal_locations.items():
            current = package_locations.get(pkg)
            if current == goal_loc:
                continue

            if current in vehicle_locations:  # Package is in a vehicle
                veh = current
                veh_loc = vehicle_locations.get(veh, None)
                if veh_loc is None:
                    continue  # Vehicle location unknown; skip
                distance = self.shortest_paths.get(veh_loc, {}).get(goal_loc, float('inf'))
                total += distance + 1  # Drop action
            else:  # Package is on the ground
                pkg_loc = current
                if pkg_loc == goal_loc:
                    continue
                min_cost = float('inf')
                for veh in vehicle_locations:
                    size = vehicle_capacities.get(veh)
                    if size is None:
                        continue
                    numeric_cap = self.size_value.get(size, 0)
                    load = vehicle_loads.get(veh, 0)
                    if numeric_cap - load <= 0:
                        continue
                    veh_loc = vehicle_locations[veh]
                    distance_a = self.shortest_paths.get(veh_loc, {}).get(pkg_loc, float('inf'))
                    distance_b = self.shortest_paths.get(pkg_loc, {}).get(goal_loc, float('inf'))
                    cost = distance_a + distance_b + 2  # Pick-up and drop
                    if cost < min_cost:
                        min_cost = cost
                if min_cost == float('inf'):
                    total += 1000  # Penalty for no available vehicle
                else:
                    total += min_cost

        return total
