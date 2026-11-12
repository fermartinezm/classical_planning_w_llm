from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `spanner` domain. The heuristic estimates the
    number of actions required to reach a goal state (all goal nuts tightened) from
    a given state. It considers walking along links, picking up usable spanners, and
    tightening nuts. Each PDDL action is counted as one unit cost (walk, pickup_spanner,
    tighten_nut).

    Assumptions
    -----------
    - There is (at least) one man who can walk, pick up any number of spanners, and
      tighten nuts. If multiple men exist, the heuristic uses one man (the one
      identified in the state) as the active agent; multiple men are not exploited.
    - Spanners are single-use with respect to the `usable` predicate: once used by
      tighten_nut they become unusable. A usable spanner can be carried or lying on the ground.
    - Links in the static facts are directed (walk is allowed if there is a `link start end`).
    - Distances are measured by the minimal number of walk actions along directed links.
    - If there are fewer usable spanners (carried + on-ground) than the number of remaining
      nuts that need to be tightened, the heuristic returns a large finite value (indicating
      difficulty/impossibility under current resources).

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - Goal facts (task.goals)
    - The directed adjacency graph of locations from the static `link` facts (task.static).
      This adjacency is used to compute shortest path distances (in number of walk actions)
      between locations by BFS.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state), the heuristic proceeds as follows:

    1. Quick goal check:
       - If all goal facts are already true in the state, return 0.

    2. Identify relevant objects and facts from the state:
       - Determine which nuts still need tightening: for each goal '(tightened N)', if it is
         not present in the state it is considered a remaining nut to handle. The location
         of each such nut is taken from an '(at N L)' fact in the state.
       - Identify usable spanners:
         * Carried usable spanners: those that appear in '(carrying M S)' and also '(usable S)'.
         * Ground usable spanners: those that appear as '(at S L)' and also '(usable S)'.
       - Determine the (single) man to reason about:
         * Prefer an object that appears as the first argument in a 'carrying' fact if present.
         * Otherwise, choose a non-location object from an 'at' fact that is not identified as a nut or spanner.
       - Determine the man's current location from '(at MAN LOC)'.

    3. Precompute distances:
       - Use BFS on the directed adjacency graph to compute shortest path distances between
         locations on demand. If a destination is unreachable, that option is considered invalid.

    4. Greedy sequential assignment (approximate but efficient):
       - The heuristic assumes the man performs a sequence of subplans. For each remaining nut,
         it computes the minimal incremental cost to tighten one nut given the current man's
         location and available usable spanners, then commits to that choice and updates the
         man's location and the set of available spanners. This is a fast, greedy approximation
         of an optimal multi-goal routing and pickup schedule.
       - Options considered for each nut:
         A) Use a carried usable spanner: cost = distance(man_loc -> nut_loc) + 1 (tighten).
         B) Fetch a ground usable spanner S: cost = distance(man_loc -> spanner_loc) + 1 (pickup) +
            distance(spanner_loc -> nut_loc) + 1 (tighten).
       - After choosing an option, the chosen spanner (if any) is removed from availability,
         and the man location is updated to the nut location (since tighten_nut leaves the man
         at the nut).
       - If at any point no usable spanners are available for remaining nuts, the heuristic
         returns a large finite value (10000) to indicate infeasibility with current usable spanners.

    5. Sum up incremental costs and return the total as the heuristic estimate.

    Notes on admissibility and efficiency
    -------------------------------------
    - This heuristic is not guaranteed to be admissible (it may overestimate), but it is
      designed to be informative and cheaply computable to guide greedy best-first search.
    - Time complexity is small: BFS distances are computed only between locations that
      appear in the greedy choices; BFS is linear in the size of the link graph and
      the greedy loop runs once per remaining nut.
    """

    def __init__(self, task):
        # store goal facts
        self.goals = set(task.goals)
        # build directed adjacency of locations from static link facts
        self.adj = {}  # location -> list of successor locations
        self.static = set(task.static)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, []).append(b)
        # a reasonable large value for infeasible/very expensive estimates
        self.INFTY = 10000

    def _bfs_distances(self, start):
        """
        Compute BFS distances (in number of walk actions) from start location
        to all reachable locations in the directed graph self.adj.
        Returns a dict loc -> distance. If start is not known in the adjacency,
        still returns {start: 0} (start may be isolated).
        """
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = set(node.state)  # frozenset -> mutable set for lookups

        # 1) If all goals satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # 2) Identify remaining nuts (from goals that are not yet true)
        remaining_nut_names = []
        nut_locations = {}  # nut -> location
        # find all '(tightened N)' goals missing => these nuts must be tightened
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened":
                nut = parts[1]
                if goal not in state:
                    remaining_nut_names.append(nut)
        # get location for each remaining nut from '(at nut loc)'
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in remaining_nut_names:
                    nut_locations[obj] = loc
        # If any nut's location unknown (state malformed), give a large estimate
        for nut in remaining_nut_names:
            if nut not in nut_locations:
                return self.INFTY

        # 3) Identify usable spanners: carried usable and ground usable
        carried_usable = []  # list of spanner names
        ground_usable = {}   # spanner name -> location
        # First collect usable spanners
        usable_set = {get_parts(f)[1] for f in state if get_parts(f)[0] == "usable"}
        # find carried spanners and their carriers
        carriers = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "carrying" and len(parts) >= 3:
                man, sp = parts[1], parts[2]
                carriers[sp] = man
        # collect carried usable spanners
        for sp in usable_set:
            if sp in carriers:
                carried_usable.append(sp)
        # collect ground usable spanners and their location
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in usable_set and obj not in carriers:
                    ground_usable[obj] = loc

        # 4) Identify man's current location.
        # Try to infer man(s): objects that appear as first arg in carrying facts are men.
        men = set(carriers.values())
        man_name = None
        man_loc = None
        if men:
            man_name = next(iter(men))
        else:
            # no carrying facts: infer man by looking for an object that is at a location and
            # that is not a location, not a usable spanner, and not a nut.
            # To build set of location names, use static adjacency keys and values plus any
            # second args from at facts in the state.
            loc_names = set(self.adj.keys())
            for succs in self.adj.values():
                loc_names.update(succs)
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) >= 3:
                    loc_names.add(parts[2])
            # now check 'at' facts for a candidate man
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) >= 3:
                    obj, loc = parts[1], parts[2]
                    if loc in loc_names and obj not in usable_set and obj not in remaining_nut_names:
                        # not a spanner (usable_set) and not a nut (remaining_nut_names) -> candidate man
                        man_name = obj
                        break
        # now get man's location
        if man_name:
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) >= 3 and parts[1] == man_name:
                    man_loc = parts[2]
                    break
        if man_loc is None:
            # cannot determine man's location -> return large estimate
            return self.INFTY

        # 5) Greedy sequential assignment of spanners to nuts, summing minimal incremental costs.
        total_cost = 0
        # convert to mutable structures for allocation
        remaining_nuts = list(remaining_nut_names)
        carried_count = len(carried_usable)
        ground_spanners = dict(ground_usable)  # copy
        current_loc = man_loc

        # if there are fewer usable spanners than remaining nuts -> infeasible
        total_usable_available = carried_count + len(ground_spanners)
        if total_usable_available < len(remaining_nuts):
            return self.INFTY

        # cache BFS distance maps from locations we visit
        dist_cache = {}

        while remaining_nuts:
            best_option = None  # tuple (cost, nut, option_type, spanner_name or None)
            # compute distances from current_loc once
            if current_loc not in dist_cache:
                dist_cache[current_loc] = self._bfs_distances(current_loc)
            dist_from_current = dist_cache[current_loc]

            for nut in remaining_nuts:
                nut_loc = nut_locations.get(nut)
                # Option A: use a carried usable spanner (if available)
                if carried_count > 0:
                    d = dist_from_current.get(nut_loc, None)
                    if d is not None:
                        cost_a = d + 1  # walk cost + tighten
                        if best_option is None or cost_a < best_option[0]:
                            best_option = (cost_a, nut, "carried", None)
                # Option B: fetch a ground usable spanner
                for sp, s_loc in ground_spanners.items():
                    # compute dist to spanner location (from current)
                    d1 = dist_from_current.get(s_loc, None)
                    if d1 is None:
                        continue
                    # compute dist from spanner location to nut location
                    if s_loc not in dist_cache:
                        dist_cache[s_loc] = self._bfs_distances(s_loc)
                    d2 = dist_cache[s_loc].get(nut_loc, None)
                    if d2 is None:
                        continue
                    cost_b = d1 + 1 + d2 + 1  # walk to sp + pickup + walk to nut + tighten
                    if best_option is None or cost_b < best_option[0]:
                        best_option = (cost_b, nut, "ground", sp)

            if best_option is None:
                # no viable option for some nut (unreachable) -> return large estimate
                return self.INFTY

            cost_chosen, nut_chosen, opt_type, sp_chosen = best_option
            total_cost += cost_chosen

            # update state of available spanners and man's location
            nut_loc = nut_locations[nut_chosen]
            if opt_type == "carried":
                carried_count -= 1
            else:  # used a ground spanner
                # remove that spanner from availability
                ground_spanners.pop(sp_chosen, None)

            # after tightening, the man ends up at the nut's location
            current_loc = nut_loc
            # remove the nut from remaining list
            remaining_nuts.remove(nut_chosen)

            # safety: if cost grows beyond INFTY, cut off
            if total_cost >= self.INFTY:
                return self.INFTY

        return int(total_cost)