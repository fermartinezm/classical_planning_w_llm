from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 100000


def get_parts(fact):
    """Return components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact (string) matches a pattern given by args (wildcards '*' allowed).
    Example: match("(at p1 l1)", "at", "*", "l1") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `transport` domain. The heuristic estimates the
    number of actions (drive, pick-up, drop) still required to achieve the goal 'at'
    facts for packages. It combines a shortest-path estimate on the road network with
    a greedy assignment of packages to vehicles subject to vehicle remaining capacities.

    Assumptions
    -----------
    - `drive`, `pick-up`, and `drop` each count as 1 action.
    - The capacity predicate expresses the number of "free slots" via a size hierarchy
      (capacity-predecessor c0 c1 means c0 is one unit smaller than c1). We derive an
      integer level for each size; that integer is treated as the number of remaining
      slots for the vehicle when the vehicle has that size fact.
    - Vehicles can pick up packages from ground locations (at facts) and drop them at
      goal locations. Packages may also already be inside vehicles.
    - If a package is currently inside vehicle V, we only consider V as the carrier of
      that package (we do not model intermediate dropping and reloading by other vehicles).
    - The road graph is directed as given by static (road l1 l2) facts. Drive cost
      equals the number of road edges traversed.
    - If locations are not reachable according to the static road graph, a large finite
      cost is returned (INF) so the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - goal locations for each package (self.goal_locations).
    - the directed road adjacency mapping (self.roads).
    - the capacity-predecessor relations to compute an integer level for each size token
      (self.size_level). The level is computed by walking predecessor links until no
      predecessor is found; the minimal size has level 0, its successor level 1, etc.
    - a cache (self._dist_cache) for BFS distance maps from a source location to all
      reachable locations in the static road graph.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts hold in the state, return 0.
    2. Parse the state and extract:
       - where every package currently is (either at a location or in a vehicle),
       - where every vehicle currently is (from (at v L) facts),
       - current capacity size fact for vehicles (converted to integer remaining slots).
    3. Precompute (and cache) shortest-path distances on the static road graph between
       any required locations using BFS. Unreachable locations are assigned INF.
    4. For each package:
       - If package already at its goal location: cost 0.
       - If package is inside vehicle V: estimated cost is
            distance(vehicle_location, goal_location) + 1 (drop)
         (if unreachable: INF).
       - If package is on the ground at location P_loc: for a candidate vehicle V at V_loc,
         estimated cost to deliver the package by V is:
            distance(V_loc, P_loc) + 1 (pick) + distance(P_loc, goal_loc) + 1 (drop)
         If V is already at P_loc some terms become 0. If distances are unreachable, INF is used.
    5. To account for vehicle capacity limits, we perform a greedy assignment:
       - Build a cost table for every (package, vehicle) pair listed above (except that
         packages already in a vehicle are only eligible for their current vehicle).
       - Repeatedly pick the available (package, vehicle) pair with minimal cost, assign
         the package to that vehicle, and decrement that vehicle's remaining-slot count.
       - Stop when either all packages are assigned or no vehicle has remaining slots.
    6. If any packages remain unassigned after capacities exhausted, we add for each the
       minimal possible delivery cost ignoring capacity (i.e., the cheapest vehicle if any)
       — this keeps the heuristic finite even if capacities are insufficient.
    7. The heuristic is the sum of the assigned package delivery costs.
    """

    def __init__(self, task):
        # Goal facts (set of strings)
        self.goals = task.goals

        # Build map package -> goal_location from goals of the form "(at pX lY)".
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # Treat as package goal if object appears in an at-goal; store regardless
                self.goal_locations[obj] = loc

        # Extract directed road adjacency from static facts
        self.roads = {}
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.roads.setdefault(l1, set()).add(l2)

        # Extract capacity-predecessor relations and compute integer levels for sizes.
        # capacity-predecessor s1 s2 means s1 is the immediate predecessor of s2.
        pred_map = {}  # maps s2 -> s1
        sizes = set()
        for fact in task.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                pred_map[s2] = s1
                sizes.add(s1)
                sizes.add(s2)
        # Also include any size names that might appear in task.facts (just in case)
        for fact in task.facts:
            parts = get_parts(fact)
            for p in parts[1:]:
                if p.startswith("c"):
                    sizes.add(p)

        # Compute level for each size by walking predecessors backwards to minimal size.
        self.size_level = {}
        for s in sizes:
            level = 0
            cur = s
            # Walk backwards: while cur has a predecessor pred_map[cur], increment level
            while cur in pred_map:
                cur = pred_map[cur]
                level += 1
                # safety: avoid infinite loops in malformed input
                if level > 1000:
                    break
            self.size_level[s] = level

        # Distance cache per source location -> dict(dest -> dist)
        self._dist_cache = {}

    def _bfs_from(self, src):
        """
        BFS on directed road graph starting at src. Returns a dict dest->distance (#edges).
        Uses self.roads adjacency. Unseen nodes are unreachable.
        """
        if src in self._dist_cache:
            return self._dist_cache[src]

        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.roads.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """
        Return the shortest directed distance (number of drive actions) from location a to b.
        If a == b -> 0. If unreachable -> INF.
        """
        if a == b:
            return 0
        dist_map = self._bfs_from(a)
        return dist_map.get(b, INF)

    def __call__(self, node):
        state = node.state

        # 1) Heuristic must be 0 for goal states
        if self.goals <= state:
            return 0

        # Parse state:
        # - at_map: maps object -> location for facts (at X L)
        # - in_map: maps package -> vehicle for facts (in p v)
        # - vehicle_capacity_slots: maps vehicle -> integer capacity slots (derived from size levels)
        at_map = {}
        in_map = {}
        vehicle_capacity_slots = {}
        # collect all vehicles discovered
        vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif pred == "in" and len(parts) >= 3:
                package, vehicle = parts[1], parts[2]
                in_map[package] = vehicle
            elif pred == "capacity" and len(parts) >= 3:
                vehicle, size = parts[1], parts[2]
                vehicles.add(vehicle)
                slots = self.size_level.get(size, 0)
                # Ensure non-negative slots
                vehicle_capacity_slots[vehicle] = max(0, slots)

        # For vehicles without an explicit capacity fact in this state, assume 0 slots.
        for fact in state:
            # ensure we collect any vehicle names from at facts if they weren't in capacity
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj = parts[1]
                # we can't know whether obj is vehicle or package; but if object has capacity fact it's a vehicle
                if obj not in vehicle_capacity_slots and obj.startswith("v"):
                    vehicles.add(obj)
                    vehicle_capacity_slots.setdefault(obj, 0)

        # For all vehicles, get their current location (from at_map). If unknown, they are unreachable.
        vehicle_locations = {v: at_map.get(v, None) for v in vehicles}

        # Build list of packages that appear in goal list (we only estimate for goals)
        packages = list(self.goal_locations.keys())

        # For each package compute current status and a baseline cost estimate if no vehicles existed:
        # We'll compute assignment costs shortly.
        # We'll maintain a set of packages already at goal to skip.
        package_info = {}  # pkg -> dict with keys: 'at' (loc) or 'in' (veh), 'goal' (loc)
        for pkg in packages:
            goal_loc = self.goal_locations[pkg]
            if match(f"(at {pkg} {goal_loc})"):
                # Note: match against constant string won't find it in state; instead check state membership
                pass
            # Determine if the package is already at the goal in the current state
            if f"(at {pkg} {goal_loc})" in state:
                package_info[pkg] = {"status": "done", "goal": goal_loc}
                continue
            if pkg in in_map:
                package_info[pkg] = {"status": "in-vehicle", "vehicle": in_map[pkg], "goal": goal_loc}
            elif pkg in at_map:
                package_info[pkg] = {"status": "at-location", "location": at_map[pkg], "goal": goal_loc}
            else:
                # Unknown location: be conservative and treat as unreachable on ground (large penalty)
                package_info[pkg] = {"status": "unknown", "location": None, "goal": goal_loc}

        # If there are no vehicles known and some packages are not done, heuristic must still be finite.
        # We will use INF-based distances to give large but finite heuristic.

        # Precompute relevant distances (only when needed)
        # We'll lazily compute distances via self._distance()

        # Cost building:
        # For packages 'in-vehicle', we only consider their current vehicle as carrier.
        # For 'at-location', we consider every vehicle as possible carrier.
        # For 'unknown', treat cost INF (but attempt to find minimal via vehicles - will likely be INF).
        # Cost formula:
        # - package in vehicle V: cost = distance(V_loc, goal_loc) + 1 (drop)
        # - package at L picked by vehicle V: cost = distance(V_loc, L) + 1 (pick) + distance(L, goal) + 1 (drop)
        # All distances use BFS on static road directed graph and are INF if unreachable.

        # Build cost table dict: (pkg, veh) -> cost
        cost_table = {}
        # Also track the minimal possible cost for each package across all vehicles (for fallback when capacities exhausted)
        min_cost_for_pkg = {}

        # Precompute any missing vehicle_locations keys as None -> treat as unreachable
        for v in vehicle_capacity_slots:
            vehicle_locations.setdefault(v, None)

        for pkg, info in package_info.items():
            if info.get("status") == "done":
                min_cost_for_pkg[pkg] = 0
                continue

            goal_loc = info["goal"]
            best_cost = INF

            if info["status"] == "in-vehicle":
                veh = info["vehicle"]
                v_loc = vehicle_locations.get(veh, None)
                if v_loc is None:
                    c = INF
                else:
                    d = self._distance(v_loc, goal_loc)
                    c = d + 1  # drop
                cost_table[(pkg, veh)] = c
                best_cost = min(best_cost, c)
                # Do not create costs for other vehicles in this case.
            elif info["status"] == "at-location":
                loc = info["location"]
                if loc is None:
                    # unknown location on the ground
                    for veh in vehicle_capacity_slots:
                        cost_table[(pkg, veh)] = INF
                    best_cost = INF
                else:
                    # Consider all vehicles as carriers
                    for veh in vehicle_capacity_slots:
                        v_loc = vehicle_locations.get(veh, None)
                        if v_loc is None:
                            c = INF
                        else:
                            d1 = self._distance(v_loc, loc)
                            d2 = self._distance(loc, goal_loc)
                            c = d1 + 1 + d2 + 1  # travel to package + pick + travel to goal + drop
                        cost_table[(pkg, veh)] = c
                        if c < best_cost:
                            best_cost = c
            else:  # unknown status
                # Attempt to compute something using vehicles if any; else INF
                for veh in vehicle_capacity_slots:
                    v_loc = vehicle_locations.get(veh, None)
                    if v_loc is None:
                        c = INF
                    else:
                        # assume vehicle will have to go from v_loc to goal and perform one pick/drop somewhere
                        # This is a weak fallback
                        d = self._distance(v_loc, info.get("goal"))
                        c = d + 2
                    cost_table[(pkg, veh)] = c
                    if c < best_cost:
                        best_cost = c

            min_cost_for_pkg[pkg] = best_cost

        # Greedy assignment respecting vehicle slots.
        remaining_pkgs = {p for p, info in package_info.items() if info.get("status") != "done"}
        # Copy capacities to mutable dict
        veh_slots = dict(vehicle_capacity_slots)

        total_cost = 0

        # Quick exit: if there are no packages to solve, return 0
        if not remaining_pkgs:
            return 0

        # Build list of eligible (pkg,veh) pairs; for packages currently in vehicle only that vehicle is eligible
        # We'll repeatedly select the minimal-cost eligible pair.
        while remaining_pkgs and any(slots > 0 for slots in veh_slots.values()):
            best_pair = None
            best_pair_cost = INF
            for pkg in remaining_pkgs:
                info = package_info[pkg]
                if info.get("status") == "in-vehicle":
                    only_veh = info["vehicle"]
                    # Only consider its current vehicle if that vehicle exists
                    if veh_slots.get(only_veh, 0) <= 0:
                        # even if the vehicle has zero remaining slots, the package is already inside it
                        # and delivering it still consumes no extra slot; to model this, allow assignment
                        # even if slots == 0. This special-case ensures packages already inside vehicles
                        # are not blocked artificially.
                        pass
                    # cost might be missing if vehicle not known; handle safely
                    c = cost_table.get((pkg, only_veh), INF)
                    if c < best_pair_cost:
                        best_pair_cost = c
                        best_pair = (pkg, only_veh)
                else:
                    # package on ground or unknown: consider all vehicles with slots > 0
                    for veh, slots in veh_slots.items():
                        if slots <= 0:
                            continue
                        c = cost_table.get((pkg, veh), INF)
                        if c < best_pair_cost:
                            best_pair_cost = c
                            best_pair = (pkg, veh)

            if best_pair is None:
                # No eligible pair found (e.g., no vehicles with free slots). Break to fallback.
                break

            pkg, veh = best_pair
            c = best_pair_cost

            # Add cost for this package
            total_cost += c

            # Remove package from remaining set
            remaining_pkgs.remove(pkg)

            # If package was not already in-vehicle, consuming a slot is required.
            if package_info[pkg].get("status") != "in-vehicle":
                # Decrement slot for the vehicle used (if vehicle exists)
                if veh in veh_slots:
                    veh_slots[veh] = max(0, veh_slots[veh] - 1)
                # else: vehicle unknown, nothing to decrement

            # If package was in-vehicle but the vehicle had 0 slots, we still allowed assignment above
            # because the package is already inside (no extra capacity consumption).

        # For any packages left unassigned (e.g., not enough capacity across vehicles), add fallback minimal cost
        for pkg in list(remaining_pkgs):
            # use precomputed minimal cost across vehicles (may be INF)
            c = min_cost_for_pkg.get(pkg, INF)
            total_cost += c
            remaining_pkgs.remove(pkg)

        # Ensure heuristic is finite; if total_cost is INF or larger, clip to large finite number
        if total_cost >= INF:
            return INF
        return int(total_cost)