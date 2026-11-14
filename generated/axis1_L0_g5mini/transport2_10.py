from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Return components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Return True if fact matches the pattern args (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the transport domain estimating the number of actions (drives,
    pick-ups and drops) required to move all packages to their goal locations.

    Assumptions
    -----------
    - The planner's state representation uses facts like "(at obj loc)" and "(in pkg veh)".
    - Roads are directed edges "(road l1 l2)". Driving between adjacent locations costs 1.
    - Picking up a package costs 1 action and dropping costs 1 action.
    - Vehicle capacities are represented by labels (e.g. c0..cN) together with
      facts "(capacity v cX)". The predicate "(capacity-predecessor s1 s2)"
      indicates s1 is the predecessor of s2 in a capacity chain.
      We interpret the chain as ordered [smallest ... largest] and map each
      capacity label to an integer level: level(c) = index in that order.
      A pick-up reduces the level by 1; a drop increases it by 1.
    - Vehicles with capacity level >= 1 can pick up another package directly.
      Vehicles with capacity level == 0 need an extra (unmodelled) drop action
      somewhere before picking; this is approximated by adding a small penalty.
    - We do a greedy assignment of packages to vehicles (nearest-vehicle heuristic),
      decrementing vehicle capacity levels as packages are assigned. This captures
      limited capacity interactions cheaply without solving a full assignment/route problem.
    - If a location pair is disconnected in the road graph, we treat the distance
      as large but finite to keep the heuristic well-defined.

    Heuristic Initialization
    ------------------------
    - Extract directed road adjacency from static facts.
    - Extract capacity-predecessor relations and build a topological ordering
      of capacity labels to map each size label to an integer level.
    - Extract package goal locations from the task goals.
    - Prepare caches for BFS distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state:
    1. If the state's facts already satisfy the goal set, return 0.
    2. Parse the state to find:
       - For each package: whether it is at a location or currently in a vehicle.
       - For each vehicle: its current location and its capacity label (mapped to a level).
    3. For each package whose current location != goal location:
       - If the package is already inside a vehicle v:
           * Cost estimate = drive_distance(vehicle_location, goal_location) + 1 (drop).
       - If the package is on the ground at location L:
           * For each vehicle v, estimate:
               cost_v = drive_distance(vehicle_location, L)  # move to package
                        + 1  # pick-up
                        + drive_distance(L, goal_location)  # move to goal
                        + 1  # drop
               If vehicle's current capacity level < 1, add a small penalty to cost_v
               to approximate the extra work needed to free capacity (e.g., dropping
               something elsewhere).
           * Choose the vehicle with minimum cost_v.
       - If no vehicle provides a finite route, use a large finite penalty.
    4. To account for capacity interactions, we perform a greedy assignment:
       - Sort packages by their best-single-vehicle cost (cheapest first) and
         assign each package to the vehicle that yields that cost, decrementing
         that vehicle's capacity level (if > 0) so subsequent assignments see reduced capacity.
       - Packages already inside vehicles do not consume additional capacity.
    5. Sum the per-package estimated costs to produce the heuristic value.
    6. Return the summed estimate (an integer). The estimate is not admissible but is
       computationally cheap and captures distances, pick/drop costs and limited capacity.

    Notes on tuning:
    - The penalty for assigning to a vehicle with 0 capacity is intentionally small
      (penalty_for_full = 2) because the planner will typically be able to free space;
      the heuristic prefers vehicles that already have capacity.
    - The BFS distance is cached per start location to avoid repeated expensive graph searches.
    """

    def __init__(self, task):
        """Extract static information: roads, capacity chain and goal locations."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency list for directed roads
        self.adj = defaultdict(list)
        self.locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj[l1].append(l2)
                self.locations.add(l1)
                self.locations.add(l2)

        # Build capacity-predecessor graph and compute an ordering (topological)
        # predecessor fact: (capacity-predecessor s1 s2) meaning s1 -> s2
        preds = defaultdict(set)
        succs = defaultdict(set)
        sizes = set()
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                succs[s1].add(s2)
                preds[s2].add(s1)
                sizes.add(s1)
                sizes.add(s2)

        # Topological order of sizes: smallest first (those without predecessors)
        order = []
        # Kahn's algorithm
        no_pred = [s for s in sizes if len(preds[s]) == 0]
        # If there were no capacity-predecessor facts (degenerate), sizes remains empty.
        while no_pred:
            s = no_pred.pop()
            order.append(s)
            for s2 in list(succs[s]):
                preds[s2].remove(s)
                succs[s].remove(s2)
                if len(preds[s2]) == 0:
                    no_pred.append(s2)
        # It's possible sizes is empty if domain has no capacity-predecessor facts
        # In that case we will still proceed with an empty map; vehicles will be treated as capacity 0.
        self.size_to_level = {size: idx for idx, size in enumerate(order)}

        # Extract goal locations for packages (assume goals are only 'at' facts for packages)
        self.goal_locations = {}
        for fact in self.goals:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.goal_locations[obj] = loc

        # BFS distance cache: start_location -> {location: distance}
        self._dist_cache = {}

        # penalty for assigning to a vehicle with no available capacity (heuristic fudge)
        self.penalty_for_full = 2

    def _bfs_distances(self, start):
        """Return dict of distances from start to every reachable location using directed roads."""
        if start in self._dist_cache:
            return self._dist_cache[start]
        dist = {}
        q = deque()
        dist[start] = 0
        q.append(start)
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[start] = dist
        return dist

    def _distance(self, a, b):
        """Return directed graph distance from location a to b, or a large number if unreachable."""
        if a == b:
            return 0
        if a not in self.locations:
            # a unknown as a location in the static graph -> unreachable in road graph
            return INF
        dist_map = self._bfs_distances(a)
        return dist_map.get(b, INF)

    def __call__(self, node):
        """
        Compute the heuristic estimate for node.state.
        Steps implemented as described in the docstring above.
        """
        state = node.state

        # 1) If goal already reached, h = 0
        if self.goals <= state:
            return 0

        # 2) Parse state: collect 'at' and 'in' facts
        at_map = {}     # object -> location
        in_map = {}     # package -> vehicle
        vehicle_capacity_label = {}  # vehicle -> size label
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif pred == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                in_map[pkg] = veh
            elif pred == "capacity" and len(parts) == 3:
                veh, size = parts[1], parts[2]
                vehicle_capacity_label[veh] = size

        # 3) Build vehicle locations and capacity levels
        vehicle_location = {}
        vehicle_level = {}
        for obj, loc in at_map.items():
            # treat any object that has a capacity label as a vehicle
            if obj in vehicle_capacity_label:
                vehicle_location[obj] = loc
        # map capacity labels to integer levels
        for veh, label in vehicle_capacity_label.items():
            vehicle_level[veh] = self.size_to_level.get(label, 0)

        # If a vehicle has no recorded location (shouldn't happen), treat as unreachable
        # and give it no capacity
        for veh in vehicle_capacity_label:
            vehicle_location.setdefault(veh, None)
            vehicle_level.setdefault(veh, 0)

        # 4) Determine packages that still need to be transported
        packages_to_move = []
        for pkg, goal_loc in self.goal_locations.items():
            # Determine pkg current physical location (in a vehicle or on ground)
            if pkg in in_map:
                current_holder = in_map[pkg]  # vehicle
                # If a package is in a vehicle, check where that vehicle is
                vloc = vehicle_location.get(current_holder)
                if vloc is not None and vloc == goal_loc:
                    # already effectively at goal (vehicle at goal and package in vehicle)
                    continue
                # Otherwise, package still needs to be dropped at goal
                packages_to_move.append(pkg)
            else:
                # On ground
                ploc = at_map.get(pkg)
                if ploc is not None and ploc == goal_loc:
                    continue
                packages_to_move.append(pkg)

        # 5) For each package compute best vehicle option (greedy assignment)
        # Prepare a mutable copy of vehicle capacity levels for assignment
        mutable_vehicle_level = dict(vehicle_level)

        # Precompute candidate costs for package->vehicle pairs
        pkg_best_options = {}  # pkg -> list of (cost, vehicle, details)
        for pkg in packages_to_move:
            # determine current package position
            if pkg in in_map:
                veh = in_map[pkg]
                vloc = vehicle_location.get(veh)
                goal_loc = self.goal_locations[pkg]
                if vloc is None:
                    # vehicle location unknown => unreachable
                    best_cost = INF
                    best_list = []
                else:
                    d = self._distance(vloc, goal_loc)
                    # drop cost 1
                    best_cost = d + 1
                    best_list = [(best_cost, veh, {"type": "in-vehicle", "vloc": vloc})]
                pkg_best_options[pkg] = best_list
                continue

            # package on ground
            ploc = at_map.get(pkg)
            goal_loc = self.goal_locations[pkg]
            candidates = []
            # For every vehicle, estimate the cost to serve this package
            for veh, vloc in vehicle_location.items():
                if vloc is None:
                    continue
                # distance vehicle->package + pick + distance package->goal + drop
                d1 = self._distance(vloc, ploc)
                d2 = self._distance(ploc, goal_loc)
                if d1 >= INF or d2 >= INF:
                    continue
                base_cost = d1 + 1 + d2 + 1  # drives + pick + drop
                # add penalty if vehicle appears full
                level = mutable_vehicle_level.get(veh, 0)
                if level < 1:
                    base_cost += self.penalty_for_full
                candidates.append((base_cost, veh, {"type": "ground", "vloc": vloc}))
            # If no vehicle had a route, give a large fallback (finite)
            if not candidates:
                candidates = [(INF, None, {"type": "unreachable"})]
            # sort candidates by cost ascending
            candidates.sort(key=lambda x: x[0])
            pkg_best_options[pkg] = candidates

        # 6) Greedy assignment: pick cheapest package first, assign it to the best vehicle,
        # decrement capacity when appropriate, sum estimated costs.
        # We compute a score for sorting: minimal candidate cost.
        pkg_min_costs = []
        for pkg, cand in pkg_best_options.items():
            if cand:
                pkg_min_costs.append((cand[0][0], pkg))
            else:
                pkg_min_costs.append((INF, pkg))
        pkg_min_costs.sort()  # cheapest packages first

        total_cost = 0
        for _, pkg in pkg_min_costs:
            candidates = pkg_best_options.get(pkg, [(INF, None, {})])
            # select the current best candidate considering current mutable vehicle levels
            best_choice = None
            best_cost = INF
            best_details = None
            for cost, veh, details in candidates:
                if veh is None:
                    # unreachable fallback candidate
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = None
                        best_details = details
                    continue
                # recompute penalty if vehicle level changed
                level = mutable_vehicle_level.get(veh, 0)
                adj_cost = cost
                # If original candidate cost included a penalty for full and now vehicle
                # has gained capacity (unlikely in greedy), we could reduce cost; recompute:
                # Detect original candidate type to adjust: if candidate is of type 'ground'
                if details.get("type") == "ground":
                    # recalc precise cost without referring to earlier penalty:
                    vloc = vehicle_location.get(veh)
                    ploc = at_map.get(pkg)
                    goal_loc = self.goal_locations[pkg]
                    if vloc is None or ploc is None:
                        continue
                    d1 = self._distance(vloc, ploc)
                    d2 = self._distance(ploc, goal_loc)
                    if d1 >= INF or d2 >= INF:
                        continue
                    adj_cost = d1 + 1 + d2 + 1
                    if level < 1:
                        adj_cost += self.penalty_for_full
                elif details.get("type") == "in-vehicle":
                    # package already inside veh; cost unaffected by other assignments
                    vloc = vehicle_location.get(in_map.get(pkg))
                    goal_loc = self.goal_locations[pkg]
                    if vloc is None:
                        adj_cost = INF
                    else:
                        d = self._distance(vloc, goal_loc)
                        adj_cost = d + 1
                # pick smallest adjusted cost
                if adj_cost < best_cost:
                    best_cost = adj_cost
                    best_choice = veh
                    best_details = details

            # Assign package to best_choice and update capacities if needed
            total_cost += best_cost if best_cost < INF else INF // 10

            # If assigned to a vehicle and package is on the ground and vehicle had capacity > 0, decrement it
            if best_choice is not None and best_details is not None:
                if best_details.get("type") == "ground":
                    if mutable_vehicle_level.get(best_choice, 0) > 0:
                        mutable_vehicle_level[best_choice] -= 1
                    # if level was 0 we leave it at 0 (we applied a penalty instead)
                # If package was already in a vehicle, it does not consume additional capacity

        # Return final heuristic estimate as integer
        # cap the result at a reasonable finite number
        if total_cost >= INF:
            return INF // 10
        return int(total_cost)