from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    Summary:
    This heuristic estimates the minimum number of actions to transport all packages
    from their current locations to their respective goal locations. It considers:
        - Current locations of packages (either in a vehicle or at a location).
        - Current locations of vehicles.
        - The road connectivity between locations.
        - Vehicle capacities and their hierarchical ordering (capacity-predecessor relation).
    It computes an approximate cost of driving, picking up, and dropping required to place
    each package at its goal location.

    Assumptions:
    - Vehicles have limited capacity, represented via the `capacity` and `capacity-predecessor` predicates.
    - Packages can be either at some location or inside exactly one vehicle.
    - Vehicles can move only between locations connected by roads.
    - Each pickup or drop costs 1 action; each drive step (edge on road) costs 1 action.
    - The heuristic does not presume an admissible cost but tries to minimize expanded nodes,
      by estimating realistic costs including travel distances on shortest paths.

    Heuristic Initialization:
    - Extract the goals mapping packages to their goal locations.
    - Extract static facts:
        * Roads as a bidirectional graph of locations.
        * Capacity-predecessor relationships forming a chain of capacity levels for vehicles.
    - Precompute all-pairs shortest paths between locations using Floyd-Warshall algorithm.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each package:
       - Identify current status: in vehicle or at some location.
       - If in a vehicle:
         * Identify the vehicle's current location.
         * Determine the capacity level of that vehicle.
       - If the package is already at the goal location, cost is 0 for that package.

    2. For packages not at goal:
       - Estimate cost to get the package to the goal location, considering:
         a) If package is in a vehicle:
            - Drive vehicle from current vehicle location to goal location (shortest path distance).
            - Drop package there (cost 1).
         b) If package is at a location:
            - Find vehicles that can hold the package (capacity constraints).
              Use the smallest vehicle capable (lowest capacity >= package size).
            - Estimate:
              -- Drive from vehicle location to package location (to pick up).
              -- Pick up package (cost 1).
              -- Drive from package location (or vehicle starting location) to goal location.
              -- Drop package (cost 1).
            - Take the minimal such cost over all suitable vehicles.

    3. Sum these estimates over all packages to produce the heuristic estimate.

    4. If no suitable vehicle can carry a package (should not happen in well-formed instances),
       treat that package with a high default cost.

    The heuristic efficiently approximates travel actions and pickup/drop actions required,
    combining shortest path distances and capacity constraints.

    """

    def __init__(self, task):
        """Initialize heuristic by extracting goal info, road graph, and capacity ordering."""
        self.goals = {}
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "at":
                package, goal_loc = parts[1], parts[2]
                self.goals[package] = goal_loc

        # Extract road connectivity (undirected)
        self.locations = set()
        roads = set()
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                roads.add((l1, l2))
                roads.add((l2, l1))
        self.locations = sorted(self.locations)

        # Build adjacency list for roads
        self.adj = {loc: set() for loc in self.locations}
        for l1, l2 in roads:
            self.adj[l1].add(l2)

        # Precompute all-pairs shortest paths using Floyd-Warshall
        self.dist = {l: {m: float('inf') for m in self.locations} for l in self.locations}
        for loc in self.locations:
            self.dist[loc][loc] = 0
        for l1 in self.locations:
            for l2 in self.adj[l1]:
                self.dist[l1][l2] = 1  # each road edge costs 1

        # Floyd-Warshall
        for k in self.locations:
            for i in self.locations:
                dist_i_k = self.dist[i][k]
                if dist_i_k == float('inf'):
                    continue
                for j in self.locations:
                    if dist_i_k + self.dist[k][j] < self.dist[i][j]:
                        self.dist[i][j] = dist_i_k + self.dist[k][j]

        # Extract capacity-predecessor relation into a list ordered from smallest to largest capacity
        # We expect a chain c0 < c1 < c2 < ...
        cap_preds = []
        for fact in task.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, c1, c2 = get_parts(fact)
                cap_preds.append((c1, c2))
        # Build order from transitive closure
        self.cap_order = {}
        # Find all capacity objects
        capacity_objs = set()
        for c1, c2 in cap_preds:
            capacity_objs.add(c1)
            capacity_objs.add(c2)
        capacity_objs = list(capacity_objs)
        # Build graph and find minimal capacities, order them
        pred_graph = {c: set() for c in capacity_objs}
        for c1, c2 in cap_preds:
            pred_graph[c2].add(c1)  # c2 has predecessor c1
        # Topological sort or ordering by repeatedly removing minimal nodes
        ordered = []
        capacity_objs_set = set(capacity_objs)
        while capacity_objs_set:
            # find minimal nodes with no predecessor in capacity_objs_set
            mins = [c for c in capacity_objs_set if pred_graph[c].isdisjoint(capacity_objs_set)]
            if not mins:
                # cycle or error (shouldn't happen)
                break
            mins.sort()
            for m in mins:
                ordered.append(m)
                capacity_objs_set.remove(m)
        self.capacity_order = ordered  # from smallest to largest capacity

        # Build capacity level map: capacity -> level (index)
        self.cap_level = {c: i for i, c in enumerate(self.capacity_order)}

    def __call__(self, node):
        """Compute heuristic estimate of actions needed from given node state to goal."""
        state = node.state

        # Quick check: if goal is reached, heuristic is 0
        if self.goals.items() <= { (get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "at" }:
            return 0

        # Extract package locations:
        # For each package:
        # - If (in p v) then package is in vehicle v
        # - else if (at p l), package at location l

        package_locations = {}  # package -> location or vehicle
        package_in_vehicle = set()
        vehicles_at = {}  # vehicle -> location
        vehicle_capacity = {}  # vehicle -> capacity object

        # Extract capacities for vehicles (latest known capacity fact)
        # capacity facts are updated due to pick/drop (capacity changes)
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, v, c = get_parts(fact)
                vehicle_capacity[v] = c

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj.startswith("v"):  # vehicle
                    vehicles_at[obj] = loc
                else:
                    package_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_locations[package] = vehicle
                package_in_vehicle.add(package)

        # To estimate package size: Use capacity levels of vehicle carrying it, or minimal capacity found
        # Heuristic simplification: assume smallest vehicle capacity can carry all packages,
        # but realistically should consider vehicle capacity for pick-up feasibility.
        # Due to PDDL, no direct package size info is given.
        # We use a heuristic: package must fit into the vehicle that holds it or vehicle chosen for pickup.

        # We can approximate package size by assuming it requires capacity level 0 (smallest)
        # or maybe any vehicle with capacity >= smallest capacity can carry any package.

        # To ensure heuristic is finite, we assume all packages can be carried by at least one vehicle.

        # Build list of vehicles by capacity level
        vehicles_by_capacity_level = {}
        for v, c in vehicle_capacity.items():
            level = self.cap_level.get(c, -1)
            if level >= 0:
                vehicles_by_capacity_level.setdefault(level, []).append(v)

        min_cap_level = min(vehicles_by_capacity_level.keys(), default=0)

        total_cost = 0

        # Helper to get shortest driving cost between two locations, returns float('inf') if no path
        def drive_cost(l1, l2):
            if l1 not in self.dist or l2 not in self.dist[l1]:
                return float('inf')
            return self.dist[l1][l2]

        for p, goal_loc in self.goals.items():
            # If package already at goal location, cost 0 for this package
            # Check if package is in a vehicle, then its effective current location is vehicle's location
            if p not in package_locations:
                # No info about package - assume large cost
                total_cost += 100
                continue

            current = package_locations[p]

            if current == goal_loc:
                # Package at goal location
                continue

            if current in vehicles_at:
                # package is in vehicle
                vehicle = current
                vehicle_loc = vehicles_at.get(vehicle, None)
                if vehicle_loc is None:
                    # vehicle location unknown: pessimistic cost
                    travel = 5
                else:
                    travel = drive_cost(vehicle_loc, goal_loc)
                    if travel == float('inf'):
                        travel = 10  # if no path, penalize
                # heuristic cost: drive vehicle + drop package
                cost = travel + 1  # drop action
                total_cost += cost
            else:
                # package is at a location
                pkg_loc = current

                # Find all vehicles that can carry the package by capacity:
                # We approximate package requires min capacity level 0 (smallest)
                # Find vehicles with capacity level >= min_cap_level

                # To simplify: For all vehicles, estimate cost of:
                # move vehicle to package location (drive)
                # pick up (1)
                # drive from package location to goal location
                # drop (1)
                # Take minimal such cost

                best_package_cost = float('inf')
                for cap_level, vehicles in vehicles_by_capacity_level.items():
                    # vehicle capacity >= min capacity needed (assumed 0 here)
                    for v in vehicles:
                        if v not in vehicles_at:
                            continue
                        vehicle_loc = vehicles_at[v]

                        to_package = drive_cost(vehicle_loc, pkg_loc)
                        if to_package == float('inf'):
                            continue
                        package_to_goal = drive_cost(pkg_loc, goal_loc)
                        if package_to_goal == float('inf'):
                            continue

                        # total cost = drive to package + pick + drive to goal + drop
                        cost = to_package + 1 + package_to_goal + 1

                        if cost < best_package_cost:
                            best_package_cost = cost

                if best_package_cost == float('inf'):
                    # No vehicle reachable/package unreachable, penalize with large number
                    best_package_cost = 20

                total_cost += best_package_cost

        return int(total_cost)
