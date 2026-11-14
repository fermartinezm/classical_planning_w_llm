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
    Domain-dependent heuristic for the Transport domain.

    Summary:
    --------
    This heuristic estimates the approximate number of actions required to deliver
    all packages to their respective goal locations, considering vehicle capacities,
    package locations, vehicle locations, and road connectivity.

    Assumptions:
    ------------
    - Each package must be transported from its current location to its goal location.
    - Packages can be either at a location or inside a vehicle.
    - Vehicles move along roads between locations.
    - Pick-up and drop actions consume one action each and affect vehicle capacity.
    - Drive actions move vehicles between directly connected locations (roads).
    - Capacity constraints imply that vehicles have a hierarchical size and can carry packages only if capacity allows.
    - The heuristic does not need to be admissible; a good approximation suffices.
    - The heuristic should be efficiently computable and avoid expensive graph searches (e.g., BFS for shortest path).
      Instead, it precomputes distance estimates once in initialization.

    Heuristic Initialization:
    -------------------------
    - Extract goal package locations from the task goals (facts of form (at package location)).
    - Extract static facts:
      - Road connections (edges between locations).
      - Vehicle capacities.
      - Capacity predecessor relations defining capacity hierarchy.
    - Build adjacency list for locations from static road facts.
    - Precompute shortest path lengths between all locations using a BFS-based all-pairs shortest path with caching 
      to enable fast distance queries.
    - Store vehicle capacities and capacity-predecessor chains for quick capacity comparisons.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If the current state is a goal state (all package goals met), heuristic = 0.

    2. For each package:
        a. Determine current location: either at a location or inside a vehicle.
           - If inside a vehicle, vehicle location is considered current package location.
        b. Get the package's goal location.

    3. For each vehicle:
        - Track current loaded packages count and capacity.
          This is approximated as the difference between vehicle initial capacity (largest size) and current capacity
          (stored in state) through capacity-predecessor relations.
          The number of packages currently inside the vehicle is inferred to estimate how many more pick-ups are possible.

    4. Estimate the action cost for each package:
        - If package already at goal location, cost 0.
        - Otherwise:
          i. If package is inside a vehicle at the goal location, cost is drop (1).
          ii. Else if inside vehicle elsewhere:
              - cost at least drop (1)
              - plus drive cost (distance from vehicle location to goal)
          iii. If package is at a location (not in vehicle):
              - cost: pick-up (1)
              - plus drive cost (distance from package location to goal)
                - This includes minimal vehicle positioning cost approximated by minimal drive distance from any vehicle that can carry the package.
              - plus drop (1)

    5. Shortcut for vehicle positioning cost:
       - For each package not inside a vehicle, find the minimal drive distance from _any_ vehicle that can carry it to the package location,
         plus from package location to goal location.
       - For simplicity, include pick-up and drop costs 2 actions.
       - If no vehicle can carry the package, package cannot be moved, assign a high heuristic cost.

    6. Sum estimated costs over all packages as the heuristic value.

    7. The heuristic is not guaranteed to be admissible but is expected to guide search efficiently,
       giving preference to states with fewer packages remaining to be delivered and shorter movement requirements.
    """

    def __init__(self, task):
        # Store task goals and static facts.
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for packages: dict package -> location
        self.package_goals = {}
        for g in self.goals:
            pred, *args = get_parts(g)
            if pred == "at" and len(args) == 2:
                package, location = args
                self.package_goals[package] = location

        # Extract static roads -> adjacency list for locations
        self.roads = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.roads.setdefault(l1, set()).add(l2)
                # Roads are bidirectional as per examples, but safer to add both directions
                # (In example domain, roads are explicitly specified both ways)
                # So only add given direction here.

        # Extract all location names from roads keys and values
        self.locations = set(self.roads.keys())
        for adjs in self.roads.values():
            self.locations.update(adjs)

        # Precompute shortest path distances between all locations with BFS from each node.
        # We store distances as dict of dict: dist[src][dest] = shortest number of drives

        # To avoid computing expensive all pairs shortest paths (which is O(n* (n+m))):
        # Since the domain can be large, we compute distances on demand and cache.

        self._dist_cache = {}

        # Extract vehicle initial capacities, and capacity-predecessor relations (for capacity hierarchy)
        self.vehicle_capacities = {}  # vehicle -> size (string)
        self.capacity_preds = set()  # set of (s1, s2) where s1 capacity-predecessor s2 (s1 is predecessor of s2)

        for fact in self.static:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, size = get_parts(fact)
                self.vehicle_capacities[vehicle] = size
            elif match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_preds.add((s1, s2))

        # Build a mapping size -> set of all sizes reachable by stepping capacity-predecessor,
        # used to check if size1 >= size2 (if size2 reachable from size1).
        # for quick capacity feasibility test.

        # Build predecessor graph: child -> parents
        self.capacity_graph = {}
        for smaller, bigger in self.capacity_preds:
            self.capacity_graph.setdefault(bigger, set()).add(smaller)

        # For checking if one size is bigger-or-equal than the other:
        # We'll build the transitive closure of capacity-predecessor edges for efficient comparison.
        # Using BFS to find all smaller sizes reachable from a size.

        self.size_reachable = {}  # size -> set of sizes smaller-or-equal by predecessor

        all_sizes = set()
        for s1, s2 in self.capacity_preds:
            all_sizes.add(s1)
            all_sizes.add(s2)
        for size in all_sizes:
            self.size_reachable[size] = self._reachable_sizes(size)

        # Some sizes might appear only as capacity in vehicles but not in predecessor relation,
        # make sure to add those sizes with themselves reachable.
        for c in self.vehicle_capacities.values():
            if c not in self.size_reachable:
                self.size_reachable[c] = {c}
        # package sizes are not given in data, assume all packages can fit in c0 (smallest),
        # so minimal capacity c0 or higher is needed.

        # Extract the smallest size (without predecessor) and assign it as minimum capacity to packages
        self.smallest_size = None
        all_sizes.update(self.vehicle_capacities.values())
        for s in all_sizes:
            # s is smallest size if it does not appear as bigger in capacity-predecessor pairs
            if all((s2 != s) for _, s2 in self.capacity_preds):
                self.smallest_size = s
                break

        # Store package sizes? Not given in domain; assume all packages need the smallest capacity.

        # Extract package sizes from state? domain does not specify package sizes in facts:
        # Heuristic assumes all packages require the smallest size c0 for capacity checks.

        # Store all vehicles and all locations for convenience
        self.vehicles = set(self.vehicle_capacities.keys())
        self.locations.update(self.roads.keys())
        for adjs in self.roads.values():
            self.locations.update(adjs)

    def _reachable_sizes(self, size):
        """Compute all sizes smaller-or-equal to given size by transitive closure of capacity-predecessor."""
        visited = set()
        stack = [size]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            # Add predecessors (smaller sizes)
            for pred in self.capacity_graph.get(current, ()):
                if pred not in visited:
                    stack.append(pred)
        return visited

    def _can_carry(self, vehicle_size, package_size):
        """Determine whether vehicle capacity (vehicle_size) is >= package_size."""
        # package_size assumed smallest_size if unknown (c0)
        # Check if package_size is reachable from vehicle_size through the predecessor relation, 
        # meaning vehicle has capacity at least package_size.
        return package_size in self.size_reachable.get(vehicle_size, set())

    def _bfs_distance(self, start):
        """Compute shortest drive distances from start location to all other locations using BFS."""
        dist = {start: 0}
        queue = [start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for nbr in self.roads.get(current, []):
                if nbr not in dist:
                    dist[nbr] = dist[current] + 1
                    queue.append(nbr)
        return dist

    def _distance(self, loc1, loc2):
        """Compute or retrieve cached drive distance between two locations.

        Returns a non-negative integer or a large number if no path exists.
        """
        if loc1 == loc2:
            return 0
        if loc1 not in self.locations or loc2 not in self.locations:
            return 10**6  # Large penalty for invalid locations

        if loc1 not in self._dist_cache:
            self._dist_cache[loc1] = self._bfs_distance(loc1)
        return self._dist_cache[loc1].get(loc2, 10**6)

    def __call__(self, node):
        state = node.state

        # 1) Check if goal state: heuristic=0 if goal reached
        if self.goals <= state:
            return 0

        # 2) Helper to get vehicle current capacity sizes from state facts:
        # capacity(?v, ?s) - only one capacity per vehicle is true at a time.
        vehicle_current_capacity = {}
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, v, s = get_parts(fact)
                if v in self.vehicles:
                    vehicle_current_capacity[v] = s

        # 3) Count how many packages currently inside each vehicle (approximate load).
        # From at/in facts:
        # (in ?p ?v) package inside vehicle v
        packages_in_vehicle = {v: 0 for v in self.vehicles}
        for fact in state:
            if match(fact, "in", "*", "*"):
                _, p, v = get_parts(fact)
                if v in packages_in_vehicle:
                    packages_in_vehicle[v] += 1

        # 4) Extract current package locations, either at location or inside vehicle.
        # Map: package -> current location or vehicle
        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, p, loc = get_parts(fact)
                if p in self.package_goals:
                    package_locations[p] = loc
            elif match(fact, "in", "*", "*"):
                _, p, v = get_parts(fact)
                if p in self.package_goals:
                    package_locations[p] = v

        # 5) Helper function to get vehicle location from state (at ?v ?loc)
        vehicle_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, v, loc = get_parts(fact)
                if v in self.vehicles:
                    vehicle_locations[v] = loc

        heuristic_total = 0

        # Package minimal size assumed smallest capacity (c0)
        package_size = self.smallest_size

        # For fast checking: precompute list of vehicles able to carry package (of size c0)
        capable_vehicles = [
            v
            for v, size in self.vehicle_capacities.items()
            if self._can_carry(size, package_size)
        ]
        if not capable_vehicles:
            # No vehicle can carry smallest package --> unsolvable scenario
            # Assign large heuristic value.
            return 10**6

        for p, goal_loc in self.package_goals.items():
            current_loc = package_locations.get(p)
            if current_loc is None:
                # Package location unknown (not in state), assign high cost
                heuristic_total += 10**3
                continue

            if current_loc == goal_loc:
                # Package already at goal: no cost
                continue

            # Estimate cost for delivering this package.

            if current_loc in self.vehicles:
                # Package inside a vehicle
                vehicle_loc = vehicle_locations.get(current_loc)
                # If vehicle location unknown penalize heuristically large cost
                if vehicle_loc is None:
                    heuristic_total += 10**3
                    continue

                if vehicle_loc == goal_loc:
                    # Vehicle already at goal location, just drop
                    heuristic_total += 1  # drop action
                else:
                    # Need to drive vehicle to goal + drop
                    dist_drive = self._distance(vehicle_loc, goal_loc)
                    heuristic_total += dist_drive + 1  # drive actions + drop
            else:
                # Package at location (not in vehicle)
                pkg_loc = current_loc

                # Find minimal cost vehicle positioning to pkg_loc that can carry package
                min_drive_to_pkg = 10**6
                for v in capable_vehicles:
                    v_loc = vehicle_locations.get(v)
                    if v_loc is None:
                        continue  # can't find vehicle at location, ignore
                    dist_to_pkg = self._distance(v_loc, pkg_loc)
                    if dist_to_pkg < min_drive_to_pkg:
                        min_drive_to_pkg = dist_to_pkg

                # If no vehicle position known, assign high cost
                if min_drive_to_pkg == 10**6:
                    heuristic_total += 10**3
                    continue

                # Drive from vehicle to package + pick-up + drive from package to goal + drop
                dist_pkg_to_goal = self._distance(pkg_loc, goal_loc)

                # Total cost approx:
                # drive to package + pick-up(1) + drive to goal + drop(1)
                heuristic_total += min_drive_to_pkg + 1 + dist_pkg_to_goal + 1

        return heuristic_total
