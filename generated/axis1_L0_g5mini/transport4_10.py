from math import ceil
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'transport' domain. The heuristic estimates
    the number of actions (drives, pick-ups, drops) needed to place every package
    at its goal location. It uses the road graph (static) to compute shortest
    driving distances between locations and uses the capacity-predecessor chain
    (static) to interpret vehicle capacities as integer free-slot counts.

    Assumptions
    -----------
    - capacity-predecessor facts form a linear-ish ordering of "size" symbols
      so they can be assigned integer ranks (free-slot counts). The lowest-rank
      size corresponds to zero free slots.
    - Driving between two directly connected locations (road facts) costs 1
      action per road edge (one 'drive' operator per road fact traversal).
    - Picking up a package costs 1 action, dropping a package costs 1 action.
    - Vehicles with zero free slots can free one slot by dropping one carried
      package at a cost approximated by a small penalty (we add 2 actions of
      penalty to allow for this in the heuristic). This is a pragmatic
      approximation to keep the heuristic finite and informative.
    - Multiple packages can be carried by the same vehicle; to allow for
      bundling (reducing trips), the heuristic divides the raw summed estimate
      by the maximum vehicle capacity seen in the problem (at least 1).
      This is an optimistic (non-admissible) correction intended to reduce
      over-counting when vehicles can transport multiple packages in one trip.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and pre-process static information from the
    task:
    - Build the directed road adjacency list from static "(road L1 L2)" facts.
    - Collect the complete set of location symbols found in the static facts.
    - Parse all capacity-predecessor facts to build a successor graph for size
      symbols and compute integer ranks for each size (used to interpret
      capacity facts, mapping a size symbol to an integer free-slot count).
    - Precompute BFS-based shortest-path distances between all pairs of
      locations that appear in the static graph (for fast repeated queries).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. Quick goal check: if all goal facts are satisfied in the state, return 0.
    2. Parse the state to find:
       - For each package: whether it is at a location or inside a vehicle.
       - For each vehicle: its current location and its capacity size symbol.
    3. Convert vehicle capacity size symbols to integer free-slot counts using
       the precomputed size ranks. If a vehicle has no capacity fact found,
       treat its free slots as 0 (conservative).
    4. For each package with a goal location different from its current
       (or not already at the goal):
       - If the package is already in a vehicle v:
         * Cost estimate = shortest-drive-distance(vehicle_location, goal_location)
                          + 1 (drop)
       - If the package is at some location L:
         * Consider every vehicle v and compute:
           drive_to_pick = shortest-drive-distance(vehicle_location, L)
           drive_to_goal = shortest-drive-distance(L, goal_location)
           base_cost_v = drive_to_pick + 1 (pick) + drive_to_goal + 1 (drop)
           If vehicle v currently has 0 free slots, add a small penalty (2).
         * Use the minimum base_cost_v across vehicles as the estimate for this package.
       - If no path exists for required driving, treat the distance as a large
         finite penalty so the heuristic remains finite.
    5. Sum the per-package estimates to obtain a raw action estimate.
    6. Apply an optimistic bundling correction: divide the raw estimate by the
       maximum vehicle capacity (in integer free slots) seen in the problem,
       rounding up to keep the heuristic integral. If max capacity < 1,
       behave as if max capacity = 1.
    7. Return the computed integer heuristic value.

    Notes on Tradeoffs
    ------------------
    - This heuristic is not admissible (it can over-optimistically divide by
      max capacity and apply approximate penalties for freeing slots), but it
      is designed to be informative and fast to compute to guide greedy
      best-first search effectively.
    - The choices around capacity penalties and bundling are pragmatic and aim
      to reduce node expansions by reflecting important domain structure:
      driving distances and capacities.
    """

    def __init__(self, task):
        # Store goal facts (as full fact strings)
        self.goals = task.goals
        # Static facts from the task (frozenset of strings)
        static_facts = task.static

        # Build road adjacency from static facts: directed edges
        self.adj = defaultdict(list)
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                _, l1, l2 = parts
                self.adj[l1].append(l2)
                # remember these locations
                self.locations.add(l1)
                self.locations.add(l2)

        # Also include any locations that appear in goal facts (they might not be in any road facts)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                _, obj, loc = parts
                self.locations.add(loc)

        # Parse capacity-predecessor static facts to build size-successor graph
        succ = defaultdict(list)
        pred_count = defaultdict(int)
        sizes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor" and len(parts) == 3:
                _, s1, s2 = parts
                succ[s1].append(s2)
                pred_count[s2] += 1
                sizes.add(s1)
                sizes.add(s2)

        # Compute integer ranks for sizes (rank = number of predecessors distance from minimal sizes)
        # Start with sizes that have pred_count == 0 (minimal sizes)
        self.size_rank = {}
        queue = deque()
        for s in sizes:
            if pred_count.get(s, 0) == 0:
                self.size_rank[s] = 0
                queue.append(s)
        # BFS/topological propagation (the graph should be acyclic in sensible problems)
        while queue:
            cur = queue.popleft()
            for nd in succ.get(cur, []):
                # if we can assign a better rank, do so
                proposed = self.size_rank[cur] + 1
                if nd not in self.size_rank or proposed > self.size_rank[nd]:
                    self.size_rank[nd] = proposed
                pred_count[nd] -= 1
                if pred_count[nd] <= 0:
                    queue.append(nd)

        # Precompute BFS shortest path distances between all locations that we know
        # Ensure adjacency has entries for isolated locations we collected
        for loc in list(self.locations):
            if loc not in self.adj:
                self.adj[loc] = []

        # Precompute distances as a dict of dicts: dist[a][b] = integer distance or INF if unreachable
        self.INF = 10 ** 6
        self.dist = {}
        for src in self.adj.keys():
            self.dist[src] = self._bfs_distances(src)

        # Precompute maximum possible capacity (largest rank) among known sizes (used for bundling)
        self.max_size_rank = max(self.size_rank.values()) if self.size_rank else 0

    def _bfs_distances(self, src):
        """Return a dict of shortest distances from src to all known locations (directed)."""
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        # For completeness, return mapping for every encountered location; caller treats missing as INF
        return d

    def _loc_distance(self, a, b):
        """Return shortest distance (integer) between a and b or INF if unreachable or unknown."""
        if a == b:
            return 0
        if a in self.dist and b in self.dist[a]:
            return self.dist[a][b]
        return self.INF

    def __call__(self, node):
        state = node.state

        # 1) Quick check: if state already satisfies goals, return 0
        if self.goals <= state:
            return 0

        # Parse state facts: find package locations (either at L or in V), vehicle positions, and capacities.
        package_location = {}  # maps package -> location (if at) or vehicle (if in)
        package_in_vehicle = {}  # maps package -> bool
        vehicle_location = {}  # maps vehicle -> location
        vehicle_capacity_rank = {}  # maps vehicle -> integer free slots according to size_rank

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # 'at' is used both for vehicles and packages/locatables
                package_location[obj] = loc
                # If it's a vehicle, register its location
                # We heuristically detect vehicles by name prefix 'v' commonly used, but better to treat all objects:
                # we will later read capacity facts to decide which objects are vehicles.
            elif pred == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                package_location[pkg] = veh
                package_in_vehicle[pkg] = True
            elif pred == "capacity" and len(parts) == 3:
                veh, size_sym = parts[1], parts[2]
                vehicle_capacity_rank[veh] = self.size_rank.get(size_sym, 0)
                # vehicle's location may also be present as an 'at' fact in the state; we will pick it up below

        # Collect vehicle locations from 'at' facts explicitly (vehicles appear as objects in capacity facts usually)
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicle_capacity_rank:
                    vehicle_location[obj] = loc
                # also record any other locations discovered
                self.locations.add(loc)
                if loc not in self.adj:
                    # ensure adjacency entry exists so distances can be queried later (will be INF)
                    self.adj.setdefault(loc, [])

        # Fallback: if some vehicle had no explicit capacity in state but appears as 'at', assume zero capacity
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                veh_candidate = parts[1]
                if veh_candidate not in vehicle_capacity_rank and veh_candidate in vehicle_location:
                    # If the object looks like a vehicle (we already have it as a vehicle_location), but no capacity fact,
                    # assume 0 free slots. This keeps heuristic finite.
                    vehicle_capacity_rank.setdefault(veh_candidate, 0)

        # Build list of packages from goals (we only need to ensure each goal package is counted)
        # Goals are facts like '(at p1 l2)'
        packages_to_consider = []
        goal_for_package = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                pkg, loc = parts[1], parts[2]
                packages_to_consider.append(pkg)
                goal_for_package[pkg] = loc

        # If no vehicles exist in the problem (vehicle_capacity_rank empty), heuristic should still be finite:
        # but without vehicles packages cannot be moved, we will return a large cost to reflect infeasibility.
        if not vehicle_capacity_rank:
            # Return a large finite heuristic (not infinite) but >0 so solver can identify problem difficulty.
            # This keeps the heuristic finite as required by the checklist.
            return len(packages_to_consider) * 100

        # For each package not already at its goal, compute a minimal local estimate of cost to deliver it.
        total_cost = 0
        for pkg in packages_to_consider:
            goal_loc = goal_for_package[pkg]

            # Determine current package state
            if pkg not in package_location:
                # Package whereabouts unknown in state (shouldn't happen), penalize heavily but finite.
                total_cost += 100
                continue

            cur = package_location[pkg]

            # If the package is already at the goal location in-state, skip
            if (cur == goal_loc) and (not package_in_vehicle.get(pkg, False)):
                continue

            # Case A: package is inside a vehicle
            if package_in_vehicle.get(pkg, False):
                veh = cur
                veh_loc = vehicle_location.get(veh)
                if not veh_loc:
                    # Vehicle location unknown => penalize
                    drive_cost = self.INF // 100
                else:
                    drive_cost = self._loc_distance(veh_loc, goal_loc)
                # Needs one drop action
                cost_pkg = drive_cost + 1
                # If unreachable, add large but finite penalty
                if drive_cost >= self.INF:
                    cost_pkg = 1000
                total_cost += cost_pkg
                continue

            # Case B: package is at some location L and needs to be picked up and moved
            pick_loc = cur
            best_cost_for_pkg = self.INF
            for veh, veh_loc in vehicle_location.items():
                # distance for vehicle to reach package
                d1 = self._loc_distance(veh_loc, pick_loc)
                # distance to drive from pick_loc to goal
                d2 = self._loc_distance(pick_loc, goal_loc)
                if d1 >= self.INF or d2 >= self.INF:
                    continue  # unreachable with this vehicle
                base = d1 + 1 + d2 + 1  # drive-to-pick + pick + drive-to-goal + drop
                # penalty if vehicle has no free slots currently
                free_slots = vehicle_capacity_rank.get(veh, 0)
                if free_slots <= 0:
                    # Add small penalty to reflect the need to drop something first to free space.
                    # This is an approximation; keep it small to remain informative.
                    base += 2
                if base < best_cost_for_pkg:
                    best_cost_for_pkg = base

            if best_cost_for_pkg >= self.INF:
                # no vehicle can reach this package/goal pair; penalize but keep finite
                best_cost_for_pkg = 1000

            total_cost += best_cost_for_pkg

        # Apply optimistic bundling: vehicles can carry multiple packages in one route.
        # We use the maximum vehicle capacity (as integer free slots / rank) seen in the problem
        # as a simple divisor to account for possible bundling of deliveries.
        max_cap = max(vehicle_capacity_rank.values()) if vehicle_capacity_rank else 0
        # size_rank is number of free slots; a rank of 0 means zero free slots.
        # For bundling, we consider capacity as (max_cap) but at least 1 to avoid division by zero.
        bundling_divisor = max(1, max_cap)
        heuristic_value = int(ceil(float(total_cost) / float(bundling_divisor)))

        # Ensure heuristic is at least 0 and finite
        if heuristic_value < 0:
            heuristic_value = 0
        return heuristic_value