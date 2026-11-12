from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, removing surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Transport domain.
    It estimates the number of actions (drive, pick-up, drop) required to achieve
    all goal (at package location) facts from a given state. The heuristic sums
    per-package optimistic delivery costs, but accounts for vehicle locations,
    available capacity (inferred from capacity facts and the capacity-predecessor
    chain), and graph distances along directed roads. The heuristic is not
    admissible but is designed to be informative and cheap to compute.

    Assumptions
    -----------
    - Every package goal is of the form (at <package> <location>).
    - Roads are directed edges; driving between adjacent locations costs 1.
    - Picking up or dropping a single package each costs 1.
    - Vehicles carry packages; vehicle current free capacity is encoded with
      (capacity <vehicle> <size>) and sizes are ordered by
      (capacity-predecessor <smaller> <larger>).
    - If no vehicle has free capacity, a vehicle that is carrying at least one
      package can first drop one package at the pickup location (cost 1) to
      free space; this is modeled explicitly with a small extra cost.
    - If there is truly no way to move a package (disconnected graph or no
      vehicle capable of ever picking it up), the heuristic uses a large finite
      penalty to keep values finite.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - self.goals: a mapping of package -> goal_location from task.goals.
    - self.road_adj: adjacency list of directed roads from static facts.
    - self.sizes_order: numeric mapping for size tokens derived from
      capacity-predecessor facts (smallest size -> 0, next -> 1, ...).
      This numeric mapping lets us interpret (capacity v cX) as an integer
      "free slots" level for that vehicle.
    - Location set used for distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each node/state:
    1. If the state's facts already satisfy all goal facts, return 0.
    2. Parse the state to obtain:
       - package_locations: for each package, either a location token or a vehicle token.
       - vehicle_locations: current location for every vehicle.
       - vehicle_loads: set of packages carried by each vehicle.
       - vehicle_capacity_level: numeric free-capacity level for each vehicle (0 means no free space).
    3. For graph distances between locations, use BFS on the directed road graph. Distances are cached per source.
    4. For each package that is not yet at its goal:
       - If the package is currently in a vehicle V:
         * Compute driving distance from V's current location to the goal location (number of drive actions).
         * Cost = drive_distance + 1 (one drop).
       - If the package is on the ground at location P:
         * Consider all vehicles V. For vehicles with free capacity (> 0):
             cost_v = dist(V_loc, P) + 1 (pick) + dist(P, goal) + 1 (drop).
           (distances of 0 are allowed when vehicle is already at the place).
         * If at least one vehicle with capacity > 0 exists, take the minimum cost_v.
         * Otherwise, if there are vehicles that carry at least one package, assume a vehicle can
           drive to P, drop one carried package (cost 1) to free space, then pick the desired package.
           For those vehicles:
             cost_v = dist(V_loc, P) + 1 (drop an existing carried package) + 1 (pick) +
                      dist(P, goal) + 1 (drop at goal).
           Take the minimum over such vehicles and add it to the heuristic.
         * If neither type of vehicle exists or a required distance is infinite, assign a large finite
           penalty for that package to keep the heuristic finite.
    5. Sum the per-package estimates and return the sum. Action costs used are:
       drive_cost = 1, pick_cost = 1, drop_cost = 1.

    Notes on accuracy and performance
    --------------------------------
    - The estimate treats each package independently and therefore may overcount shared
      drive actions when one vehicle can carry multiple packages together. This is a
      deliberate trade-off for runtime and simplicity; the heuristic remains informative by
      using true road distances and current vehicle positions/capacities.
    - BFS distance caching makes repeated distance queries efficient.
    """

    # large finite penalty for unreachable or highly constrained cases
    LARGE_PENALTY = 1000

    def __init__(self, task):
        """Extract goals, roads and capacity-predecessor information from the task."""
        self.goals = {}  # package -> desired location
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "at":
                _, pkg, loc = parts
                self.goals[pkg] = loc

        # Extract road adjacency from static facts
        self.road_adj = defaultdict(list)  # loc -> list of neighbor locs
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                _, a, b = parts
                self.road_adj[a].append(b)
                self.locations.add(a)
                self.locations.add(b)

        # Extract the capacity-predecessor relation and compute numeric levels.
        # capacity-predecessor s1 s2 -> s2 has predecessor s1 (s1 < s2)
        predecessor = {}  # larger_size -> smaller_size
        sizes = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor":
                _, s1, s2 = parts
                predecessor[s2] = s1
                sizes.add(s1)
                sizes.add(s2)

        # Compute numeric level for each size token by following predecessor chain:
        # minimal sizes (no predecessor entry) map to 0, next to 1, etc.
        self.size_level = {}

        def compute_level(s):
            if s in self.size_level:
                return self.size_level[s]
            if s not in predecessor:
                self.size_level[s] = 0
            else:
                self.size_level[s] = compute_level(predecessor[s]) + 1
            return self.size_level[s]

        for s in sizes:
            compute_level(s)

        # Cache for BFS distances per source location
        self._dist_cache = {}

    def _bfs_distances_from(self, src):
        """
        Return a dict mapping locations to shortest directed distance (number of drives)
        from src to each reachable location. Caches result in self._dist_cache.
        """
        if src in self._dist_cache:
            return self._dist_cache[src]

        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.road_adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """
        Directed shortest-path distance (in number of drive actions) from location a to b.
        If no path exists, return None.
        """
        if a == b:
            return 0
        if a not in self.locations or b not in self.locations:
            # If either is unknown as a location in the static road graph, we
            # cannot compute a path; return None (treat later as LARGE_PENALTY).
            return None
        dist_from_a = self._bfs_distances_from(a)
        return dist_from_a.get(b, None)

    def __call__(self, node):
        """
        Compute and return the heuristic estimate for the node.

        Steps:
        - If all goals are satisfied in this state, return 0.
        - Otherwise compute per-package delivery estimates (using current vehicle
          locations, loads and capacities) and return their sum.
        """
        state = node.state

        # 1) If goals already satisfied, heuristic is 0.
        # task.goals are strings like "(at p1 l2)" and state contains such strings.
        if all(goal in state for goal in self.goals_to_fact_strings()):
            return 0

        # Parse current state:
        package_loc = {}  # package -> either location or vehicle token
        vehicle_loc = {}  # vehicle -> location token
        vehicle_loads = defaultdict(set)  # vehicle -> set of packages it carries
        vehicle_capacity_level = {}  # vehicle -> numeric capacity level (free slots)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                # 'at' is used for both packages and vehicles
                if obj.startswith("p"):  # package naming convention assumed
                    package_loc[obj] = loc
                else:
                    # treat anything else as possibly a vehicle; safe to store vehicle locations
                    vehicle_loc[obj] = loc
            elif pred == "in":
                pkg, veh = parts[1], parts[2]
                package_loc[pkg] = veh
                vehicle_loads[veh].add(pkg)
            elif pred == "capacity":
                veh, size = parts[1], parts[2]
                # map size token to numeric level; if unknown size, assume 0
                vehicle_capacity_level[veh] = self.size_level.get(size, 0)

        # For vehicles that don't have explicit capacity fact in the current state, assume level 0
        for v in vehicle_loc:
            if v not in vehicle_capacity_level:
                vehicle_capacity_level[v] = 0

        total_cost = 0

        # For every package that has a goal, compute estimated cost if it's not already at goal.
        for pkg, goal_loc in self.goals.items():
            # If goal already satisfied (fact present), skip (should have been caught earlier but double-check)
            goal_fact = f"(at {pkg} {goal_loc})"
            if goal_fact in state:
                continue

            # Determine current location of package
            if pkg not in package_loc:
                # Package not present in state? Unlikely, but assign large penalty.
                total_cost += self.LARGE_PENALTY
                continue
            cur = package_loc[pkg]

            # Case A: package is inside a vehicle
            if cur in vehicle_loc:
                veh = cur
                vloc = vehicle_loc.get(veh)
                if vloc is None:
                    # Unknown vehicle location -> penalize
                    total_cost += self.LARGE_PENALTY
                    continue
                d = self._distance(vloc, goal_loc)
                if d is None:
                    # unreachable -> large but finite penalty
                    total_cost += self.LARGE_PENALTY
                else:
                    # drive actions + one drop
                    total_cost += d + 1

            # Case B: package is on the ground at a location
            else:
                pkg_loc = cur
                # Quick check: if no path from package location to goal, even with vehicle, penalize
                d_pkg_goal = self._distance(pkg_loc, goal_loc)
                if d_pkg_goal is None:
                    # No path from pickup to goal (even if vehicle could move), assign large penalty
                    total_cost += self.LARGE_PENALTY
                    continue

                best_cost = None

                # 1) Try vehicles that have free capacity (> 0)
                for v, vloc in vehicle_loc.items():
                    cap = vehicle_capacity_level.get(v, 0)
                    if cap <= 0:
                        continue
                    # distance from vehicle to package
                    d1 = self._distance(vloc, pkg_loc)
                    if d1 is None:
                        continue
                    # distance from package to goal (we already computed)
                    d2 = d_pkg_goal
                    cost = d1 + 1 + d2 + 1  # drive to package, pick, drive to goal, drop
                    if best_cost is None or cost < best_cost:
                        best_cost = cost

                if best_cost is not None:
                    total_cost += best_cost
                    continue

                # 2) No vehicle with free capacity: try vehicles that carry at least one package
                best_cost_drop_then_pick = None
                for v, vloc in vehicle_loc.items():
                    if len(vehicle_loads.get(v, ())) == 0:
                        continue
                    d1 = self._distance(vloc, pkg_loc)
                    if d1 is None:
                        continue
                    d2 = d_pkg_goal
                    # drive to package, drop an existing carried package (1) to free space,
                    # pick the desired package (1), drive to goal (d2), drop (1)
                    cost = d1 + 1 + 1 + d2 + 1
                    if best_cost_drop_then_pick is None or cost < best_cost_drop_then_pick:
                        best_cost_drop_then_pick = cost

                if best_cost_drop_then_pick is not None:
                    total_cost += best_cost_drop_then_pick
                    continue

                # 3) No vehicle with capacity or carrying packages: try any vehicle regardless (assume it can pick)
                # This is a fallback to keep heuristic finite (but penalize).
                fallback_best = None
                for v, vloc in vehicle_loc.items():
                    d1 = self._distance(vloc, pkg_loc)
                    if d1 is None:
                        continue
                    d2 = d_pkg_goal
                    cost = d1 + 1 + d2 + 1 + 2  # extra penalty to reflect uncertain capacity handling
                    if fallback_best is None or cost < fallback_best:
                        fallback_best = cost
                if fallback_best is not None:
                    total_cost += fallback_best
                else:
                    # No vehicle reachable at all -> large penalty
                    total_cost += self.LARGE_PENALTY

        return int(total_cost)

    def goals_to_fact_strings(self):
        """Return the goal facts as strings, for quick membership checking in a state."""
        return {f"(at {pkg} {loc})" for pkg, loc in self.goals.items()}