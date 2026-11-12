from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into parts."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact matches a simple pattern (supports '*' wildcard per part).
    Example: match("(at p1 l1)", "at", "*", "l1") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the PDDL "transport" domain.
    The heuristic estimates the number of actions required to move all packages
    to their goal locations. It models the main action types explicitly:
      - drive (each road traversal counts as 1),
      - pick-up (1),
      - drop (1).
    For packages already inside vehicles, it counts the drives required for the
    vehicle to reach the package's goal and a single drop. For packages on the
    ground it estimates the cheapest vehicle to pick them up and deliver them
    (drive-to-package + pick + drive-to-goal + drop). Vehicles with zero free
    capacity are still considered by assuming they can free one slot with a
    single (optimistic) drop action; this keeps the heuristic finite and
    makes it less pessimistic.

    Assumptions
    -----------
    - Roads are directed and represented by (road l_from l_to) static facts.
    - capacity-predecessor facts form chains like c0 <- c1 <- c2 ... where
      predecessor means "one less free slot". We map sizes to integers by
      traversing predecessor pointers (c0 -> 0, c1 -> 1, ...).
    - Each drive action corresponds to traversing a single road fact.
    - Pick-up and drop each have unit cost.
    - Vehicles may be temporarily assumed to free capacity with one drop if they
      currently report zero free capacity (optimistic approximation).
    - The heuristic is not necessarily admissible; it is tuned to be informative
      and cheaply computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    In __init__ the heuristic:
      - Extracts goal (at ...) facts and stores goal location per package.
      - Parses static road facts into a directed adjacency list.
      - Parses capacity-predecessor facts and builds a predecessor mapping so
        that each size symbol can be mapped to an integer count of free slots.
      - Initializes a cache for BFS distance results between locations.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state we compute the heuristic as follows:

    1. If the state's facts already satisfy all goal facts, return 0.

    2. Parse the current state to find:
       - where each package currently is: either "(at p L)" or "(in p V)".
       - where each vehicle is: "(at v L)".
       - each vehicle's reported capacity: "(capacity v size)" -> integer free slots.

    3. For every package with a goal location different from its current grounded
       location:
       - If the package is inside a vehicle V:
         * Compute d = shortest number of drives from V's location to the
           package's goal location using the directed road graph (BFS).
         * Cost contribution for this package := d + 1 (drop).
           If no path exists, a large penalty is used (see implementation).
       - If the package is on the ground at location L:
         * For each vehicle V, compute an optimistic cost to use V:
             cost_V = (extra_free_cost) + dist(V_loc, L) + 1 (pick)
                     + dist(L, goal) + 1 (drop)
             where extra_free_cost = 0 if V currently reports >=1 free slot,
             otherwise = 1 representing an optimistic drop to free a slot.
         * Take the minimum cost_V across vehicles as this package's cost.
         * If no path exists for any required drive, large penalty is used.

    4. Sum the per-package costs to form the heuristic value.

    Notes on efficiency and design choices:
      - All-pairs distances are computed lazily via BFS from sources and cached.
      - Mapping sizes to integer capacity is done by walking predecessor links.
      - The heuristic treats packages independently and does not attempt a full
        multi-package vehicle routing optimization; this keeps computation fast
        while remaining informative.
    """

    def __init__(self, task):
        # Store goal atom strings and extract package -> goal_location mapping
        self.goals = task.goals
        self.goal_locations = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "at" and len(parts) >= 3:
                pkg, loc = parts[1], parts[2]
                self.goal_locations[pkg] = loc

        # Static facts
        static_facts = task.static

        # Build road adjacency (directed)
        self.adj = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                self.adj.setdefault(a, []).append(b)

        # Ensure adjacency keys exist for all locations that may appear later
        for loc in list(self.locations):
            self.adj.setdefault(loc, [])

        # Build capacity predecessor mapping: pred[successor] = predecessor
        # capacity-predecessor s1 s2 means s1 is predecessor of s2 (s2 -> s1)
        self.cap_pred = {}
        self.sizes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor" and len(parts) >= 3:
                s1, s2 = parts[1], parts[2]
                self.cap_pred[s2] = s1
                self.sizes.add(s1)
                self.sizes.add(s2)

        # Precompute integer mapping for sizes by walking predecessor chain.
        # size_to_int(s) returns number of steps to minimal size (minimal has 0).
        self.size_to_int_cache = {}

        def size_to_int(s):
            if s in self.size_to_int_cache:
                return self.size_to_int_cache[s]
            cur = s
            count = 0
            # Walk predecessor chain: s -> pred(s) -> pred(pred(s)) ...
            while cur in self.cap_pred:
                cur = self.cap_pred[cur]
                count += 1
                # guard for accidental cycles: break with a large number
                if count > 1000:
                    break
            self.size_to_int_cache[s] = count
            return count

        # Materialize mapping for known sizes (in static facts)
        for s in self.sizes:
            size_to_int(s)
        self.size_to_int = size_to_int

        # Distance cache for BFS from a given source location: source -> {dest:dist}
        self._dist_cache = {}

        # Large penalty used when no path exists; chosen to be large but finite
        self.INF = 10 ** 6

    def _ensure_location_in_graph(self, loc):
        """Ensure a location key exists in adjacency (some locations appear only in state)."""
        if loc not in self.adj:
            self.adj.setdefault(loc, [])
            self.locations.add(loc)

    def _bfs_distances_from(self, src):
        """Return a dict mapping each reachable location to shortest distance (#drives) from src."""
        if src in self._dist_cache:
            return self._dist_cache[src]
        # Ensure src node exists in adjacency
        self._ensure_location_in_graph(src)
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """
        Shortest number of drive actions to go from location a to b.
        Returns 0 if a == b, INF if b unreachable from a.
        """
        if a == b:
            return 0
        dmap = self._bfs_distances_from(a)
        return dmap.get(b, self.INF)

    def __call__(self, node):
        """
        Compute heuristic value for node.

        Implementation overview:
          - If all goals satisfied, return 0.
          - Parse the state to find package locations, vehicle locations, vehicle
            capacities, and which packages are in which vehicles.
          - For each package not at its goal, compute an optimistic delivery cost
            (see class docstring) and sum.
        """
        state = node.state

        # 1) If goal already reached, return 0
        if self.goals <= state:
            return 0

        # 2) Parse state: package_at (ground), package_in (vehicle),
        # vehicle_at, vehicle_capacity_size
        package_at = {}  # package -> location (ground)
        package_in = {}  # package -> vehicle
        vehicle_at = {}  # vehicle -> location
        vehicle_capacity = {}  # vehicle -> integer free slots

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            # (at ?x ?v - location)
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # determine whether obj is vehicle or package by naming is not necessary;
                # we store both, and later consult goal list to see which objects are packages.
                if obj.startswith("v") or obj.startswith("vehicle"):
                    vehicle_at[obj] = loc
                else:
                    # could be package or vehicle; if this object also appears in package goals,
                    # consider it package; otherwise, keep in vehicle_at if name suggests vehicle.
                    # Simpler: if object appears on left in any goal 'at' facts it's a package.
                    if obj in self.goal_locations:
                        package_at[obj] = loc
                    else:
                        # when unknown, still record as vehicle if later used
                        vehicle_at[obj] = loc
            # (in ?x - package ?v - vehicle)
            elif pred == "in" and len(parts) >= 3:
                pkg, veh = parts[1], parts[2]
                package_in[pkg] = veh
            # (capacity ?v ?s)
            elif pred == "capacity" and len(parts) >= 3:
                veh, size = parts[1], parts[2]
                try:
                    cap_int = self.size_to_int(size)
                except Exception:
                    cap_int = 0
                vehicle_capacity[veh] = cap_int

        # Ensure that all vehicles that appear in state have a capacity entry (default 0)
        for v in vehicle_at:
            vehicle_capacity.setdefault(v, 0)

        # Also ensure adjacency contains all locations found in state (so BFS won't miss them)
        for loc in set(package_at.values()) | set(vehicle_at.values()) | set(self.goal_locations.values()):
            self._ensure_location_in_graph(loc)

        total_cost = 0

        # For each package that has a goal, compute cost if not already at goal
        for pkg, goal_loc in self.goal_locations.items():
            # Is the package already at the goal location on the ground?
            if ("(at " + pkg + " " + goal_loc + ")") in state:
                continue  # no cost

            # Determine where the package currently is
            if pkg in package_in:
                # package is in vehicle
                veh = package_in[pkg]
                veh_loc = vehicle_at.get(veh)
                if veh_loc is None:
                    # If vehicle location unknown in state, we conservatively assume it is somewhere;
                    # give a moderate penalty to account for uncertainty.
                    total_cost += 2  # optimistic small cost
                    continue
                # cost = drives needed to get vehicle to goal + drop
                d = self._distance(veh_loc, goal_loc)
                if d >= self.INF:
                    total_cost += self.INF
                else:
                    total_cost += d + 1  # drives + drop
            elif pkg in package_at:
                pkg_loc = package_at[pkg]
                # If package is on the ground, choose the best vehicle to pick and deliver it.
                best_cost = self.INF
                # If there are no vehicles known, fall back to driving package (impossible), give penalty
                if not vehicle_at:
                    # Try to estimate by assuming a virtual vehicle must travel from some nearby place:
                    d = self._distance(pkg_loc, goal_loc)
                    if d >= self.INF:
                        best_cost = self.INF
                    else:
                        # pick+drop + drives: drive from package to goal (via vehicle) counted twice roughly
                        best_cost = d + 2  # optimistic
                else:
                    for veh, veh_loc in vehicle_at.items():
                        cap = vehicle_capacity.get(veh, 0)
                        extra_free = 0 if cap >= 1 else 1  # optimistic single drop to free a slot
                        d1 = self._distance(veh_loc, pkg_loc)
                        d2 = self._distance(pkg_loc, goal_loc)
                        if d1 >= self.INF or d2 >= self.INF:
                            # skip unreachable option (or treat as very expensive)
                            option_cost = self.INF
                        else:
                            option_cost = extra_free + d1 + 1 + d2 + 1  # extra + drive to pkg + pick + drive to goal + drop
                        if option_cost < best_cost:
                            best_cost = option_cost
                total_cost += best_cost
            else:
                # Package not present in parsed facts (weird state). Fall back to penalty.
                total_cost += 2

            # early cutoff if cost already enormous
            if total_cost >= self.INF:
                return self.INF

        # Return integer heuristic (finite for typical solvable states)
        return int(total_cost)