from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return tokens of a fact string like '(at p1 l1)' -> ['at', 'p1', 'l1']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the "transport" domain. It estimates the
    number of actions (drives, pick-ups, drops) required to achieve all goal
    "at" facts for packages by considering:
      - shortest driving distances (in terms of drive actions) between locations,
      - vehicles' current capacities (using the capacity-predecessor chain),
      - current positions of vehicles and packages (at / in).
    The heuristic is intentionally non-admissible (it may overestimate),
    but is designed to be informative and efficiently computable so that
    greedy best-first search expands fewer nodes.

    Assumptions
    -----------
    - Roads in the static facts represent directed drive edges (road l1 l2).
      Driving from l1 to l2 costs exactly 1 action (a single drive action).
    - Capacity levels are totally ordered via capacity-predecessor facts. A
      pick-up reduces the vehicle's capacity level to its predecessor; a drop
      increases it correspondingly. The number of consecutive picks available
      for a vehicle equals its index within this ordering relative to the
      minimal capacity.
    - We approximate multi-package interactions by summing individual package
      transport costs (assigning each package to the best vehicle given the
      current state). This may double-count drive actions but remains a good
      guide for greedy search.
    - If no vehicle currently has an immediate free slot (cannot pick because
      its capacity is at the minimal level), the heuristic allows using any
      vehicle but adds a small penalty to account for the need to free space
      (via drops) — ensuring the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and precompute:
      - goal mapping: package -> goal location (from task.goals).
      - adjacency list of roads (from task.static).
      - set of locations (from road facts plus goal locations).
      - shortest-path distances (in number of drive actions) between all
        locations using BFS on the directed road graph.
      - capacity ordering and an integer index for each size symbol based on
        capacity-predecessor facts. The minimal capacity receives index 0; each
        successor increases the index by 1. This index allows quick computation
        of how many picks a vehicle can still perform (free slots = index -
        min_index).
    These precomputations keep the per-state heuristic computation efficient.

    Step-By-Step Thinking for Computing Heuristic (for a given state)
    -----------------------------------------------------------------
    1. If the state already satisfies all goal facts, return 0.
    2. Parse the current state to find:
        - where each vehicle is located (from 'at' facts for vehicles),
        - where each package is (either 'at' a location or 'in' a vehicle),
        - each vehicle's current capacity size (from 'capacity' facts).
    3. For each package that is not yet at its goal location:
        a) If it is already in a vehicle v:
           - Estimate cost = (driving distance from v's location to the goal)
                           + 1 (a drop action).
        b) If it is at some location loc:
           - For each vehicle v compute the cost to use v to transport the
             package:
               drive_v_to_pkg = dist(v_loc, loc)
               pick_cost = 1
               drive_pkg_to_goal = dist(loc, goal_loc)
               drop_cost = 1
             candidate_cost = drive_v_to_pkg + pick_cost + drive_pkg_to_goal + drop_cost
           - Consider only vehicles with at least one free slot (according to
             the capacity index). If none have free slots, allow using the
             cheapest candidate and add a small penalty (2) to reflect the
             need to free capacity first (e.g., by dropping another item).
           - If some distances are unreachable in the road graph, a large but
             finite cost is used to keep the heuristic finite.
    4. Sum the per-package estimated costs to produce the final heuristic
       value. The heuristic may overcount shared vehicle movement (intentionally
       simplifying interactions), but it is efficient and informative.

    Implementation notes
    --------------------
    - The heuristic is finite for solvable states because unreachable edges
      are mapped to a large finite penalty instead of infinity.
    - All parsing of PDDL-style facts removes surrounding parentheses via
      get_parts so tokens are extracted correctly.
    """

    def __init__(self, task):
        # Goals: map package -> desired location (we only handle at facts)
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.goal_locations[obj] = loc

        # Extract static road graph and capacity-predecessor relations.
        static = task.static

        # Build adjacency for directed roads
        self.adj = defaultdict(list)
        self.locations = set()
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.adj[a].append(b)
                self.locations.add(a)
                self.locations.add(b)

        # Also include goal locations and any possible location tokens that might
        # not appear in road facts (this keeps distance computations safe).
        for loc in self.goal_locations.values():
            self.locations.add(loc)

        # Compute shortest-path distances between all locations (BFS per node).
        # Distances are integer number of drive actions.
        self.dist = {}  # (src, dst) -> distance
        # For unreachable pairs we will use a finite large penalty.
        self.max_unreach_penalty = max(10, len(self.locations) + 5)

        for src in self.locations:
            # BFS from src
            q = deque()
            q.append(src)
            seen = {src: 0}
            while q:
                u = q.popleft()
                for v in self.adj.get(u, []):
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        q.append(v)
            # fill dist entries for this source
            for dst in self.locations:
                if dst in seen:
                    self.dist[(src, dst)] = seen[dst]
                else:
                    self.dist[(src, dst)] = self.max_unreach_penalty

        # Parse capacity-predecessor to establish integer indices for sizes.
        # Build successor mapping pred -> [succ,...]
        succs = defaultdict(list)
        preds = defaultdict(list)
        size_symbols = set()
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor" and len(parts) == 3:
                s1, s2 = parts[1], parts[2]
                succs[s1].append(s2)
                preds[s2].append(s1)
                size_symbols.add(s1)
                size_symbols.add(s2)
        # Also scan task.facts for any capacity symbols not appearing above
        for fact in task.facts:
            parts = get_parts(fact)
            if parts[0] == "capacity" and len(parts) == 3:
                size_symbols.add(parts[2])

        # Topological/level assignment: minimal capacities (no predecessor) get index 0.
        self.size_index = {}
        # nodes with no predecessor are minima
        queue = deque()
        for s in size_symbols:
            if not preds.get(s):
                self.size_index[s] = 0
                queue.append(s)
        # Process successors, assign index = parent_index + 1
        while queue:
            s = queue.popleft()
            idx = self.size_index[s]
            for suc in succs.get(s, []):
                # assign only if not assigned or found larger index via another path
                new_idx = idx + 1
                if self.size_index.get(suc, -1) < new_idx:
                    self.size_index[suc] = new_idx
                # once we've set its index, we still push to propagate further
                queue.append(suc)

        # If there are any unassigned sizes (due to unexpected topology), give them index 0
        for s in size_symbols:
            if s not in self.size_index:
                self.size_index[s] = 0

        # Track minimal size index (should be 0 but compute defensively)
        self.min_size_index = min(self.size_index.values()) if self.size_index else 0

    def _drive_distance(self, a, b):
        """Return precomputed drive distance (finite)."""
        # If a or b are unknown locations, return penalty.
        if a not in self.locations or b not in self.locations:
            return self.max_unreach_penalty
        return self.dist.get((a, b), self.max_unreach_penalty)

    def __call__(self, node):
        state = node.state

        # Fast goal check: if all goals already satisfied, heuristic is 0.
        # task goals are not directly available here, but the heuristic was
        # initialized with goal_locations. We check presence of those facts in state.
        # If every (at pkg loc) for goals is in state -> return 0.
        goal_facts_needed = set(f"(at {p} {l})" for p, l in self.goal_locations.items())
        if goal_facts_needed <= state:
            return 0

        # Parse state: vehicle locations, package locations (at or in), vehicle capacities
        vehicle_loc = {}      # vehicle -> location
        package_at = {}       # package -> location (if at)
        package_in = {}       # package -> vehicle (if in)
        vehicle_capacity_size = {}  # vehicle -> size symbol

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # distinguish vehicles and packages by naming conventions is unsafe,
                # but domain types give that 'vehicle' objects have capacity facts;
                # we will treat any object that also appears in capacity facts as vehicle.
                # For now, record both; vehicle entries will be used when capacity exists.
                vehicle_loc[obj] = loc  # may overwrite package at but packages won't have capacity facts
                package_at[obj] = loc
            elif pred == "in" and len(parts) == 3:
                obj, v = parts[1], parts[2]
                package_in[obj] = v
            elif pred == "capacity" and len(parts) == 3:
                v, size = parts[1], parts[2]
                vehicle_capacity_size[v] = size
                # ensure the vehicle location dict has an entry (may be added from at facts)
                if v not in vehicle_loc:
                    # no location known from 'at' for vehicle; leave absent
                    pass

        # Filter vehicle_loc to only those that are actual vehicles (present in capacity facts)
        actual_vehicles = set(vehicle_capacity_size.keys())
        vehicle_loc = {v: vehicle_loc[v] for v in list(vehicle_loc.keys()) if v in actual_vehicles}

        # For objects that were only in package_at, they might have been overwritten in vehicle_loc;
        # rebuild package_at properly: a package's at-fact is any 'at' fact whose subject is not a vehicle.
        # We'll rebuild package_at by scanning state again.
        package_at = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj not in actual_vehicles:
                    package_at[obj] = loc

        # Compute vehicle free slots from capacity indices.
        vehicle_free_slots = {}
        for v, size in vehicle_capacity_size.items():
            idx = self.size_index.get(size, self.min_size_index)
            # number of picks available until minimal capacity:
            free = idx - self.min_size_index
            vehicle_free_slots[v] = max(0, free)

        # For each package not at goal, compute estimated cost.
        total_cost = 0
        for pkg, goal_loc in self.goal_locations.items():
            # check if package already at goal
            if f"(at {pkg} {goal_loc})" in state:
                continue

            # determine current package status
            if pkg in package_in:
                # package is inside a vehicle
                veh = package_in[pkg]
                veh_loc = vehicle_loc.get(veh, None)
                if veh_loc is None:
                    # unknown vehicle location -> apply penalty but finite
                    drive_cost = self.max_unreach_penalty
                else:
                    drive_cost = self._drive_distance(veh_loc, goal_loc)
                # one drop action
                cost = drive_cost + 1
                total_cost += cost
            elif pkg in package_at:
                pkg_loc = package_at[pkg]
                # If the package is already at some location but not the goal,
                # compute best vehicle to take it.
                best = None
                for v in actual_vehicles:
                    v_loc = vehicle_loc.get(v, None)
                    if v_loc is None:
                        # unknown location - treat as high cost candidate
                        d1 = self.max_unreach_penalty
                    else:
                        d1 = self._drive_distance(v_loc, pkg_loc)
                    d2 = self._drive_distance(pkg_loc, goal_loc)
                    cand = d1 + 1 + d2 + 1  # drive to pkg + pick + drive to goal + drop
                    # prefer vehicles that have free slot
                    if vehicle_free_slots.get(v, 0) >= 1:
                        # prefer these: lower value considered strictly better
                        score = cand
                    else:
                        # penalize vehicles without free slot by adding a small penalty
                        score = cand + 2
                    if best is None or score < best:
                        best = score
                if best is None:
                    # no vehicles at all? use a large finite fallback
                    best = self.max_unreach_penalty + 4
                total_cost += best
            else:
                # Package not present in state? treat as worst-case finite penalty.
                total_cost += self.max_unreach_penalty + 3

        # Ensure heuristic returns integer
        return int(total_cost)