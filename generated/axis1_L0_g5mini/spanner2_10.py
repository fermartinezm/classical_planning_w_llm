from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check whether a fact matches a pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `spanner` domain. It estimates the number
    of actions required to tighten all remaining loose nuts in the current state.

    Assumptions
    -----------
    - Each usable spanner can be used exactly once (tighten_nut removes `usable`).
    - Men can carry multiple spanners at the same time (domain allows multiple carrying facts).
    - There are only the actions provided in the domain (walk, pickup_spanner, tighten_nut).
    - Movement cost is measured in number of walk actions (each `link` traversed costs 1).
    - Pickup and tighten each cost 1 action.
    - If there are fewer usable spanners available than loose nuts, the state is unsolvable;
      the heuristic returns a very large value in that case.
    - The heuristic is not guaranteed to be admissible. It is a greedy, efficient estimate
      that accounts for distances on the location graph and which spanners are currently usable
      and/or carried.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Store the set of goal facts (so we can return 0 for goal states).
    - Parse static facts to extract the directed adjacency graph of locations from `link` facts.
    - Precompute a list of known locations (keys in the adjacency map can be used for BFS).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a node (state), the heuristic computes an estimated action cost as follows:

    1. Quick goal check:
       - If all goal facts are already true in the state, return 0.

    2. Parse the state:
       - Find all men and their current locations via `(at ?m ?loc)`.
       - Find all spanners and determine whether they are:
           a) at a location `(at spannerX locationY)`, or
           b) carried by some man `(carrying man spannerX)`.
       - Identify which spanners are currently `usable` via `(usable spannerX)`.
       - Find loose nuts `(loose nutX)` and their locations `(at nutX locationZ)`.

    3. Feasibility check:
       - If the total number of usable spanners (both carried and on the ground) is
         smaller than the number of loose nuts, return a large cost (interpreted as unsolvable).

    4. Distance computations:
       - Use BFS on the directed link graph to compute shortest path lengths between locations.
       - Distances are cached and an unreachable pair is treated as INF.

    5. Greedy assignment loop:
       - Repeatedly pick the next (man, nut, spanner) combination that minimizes the immediate
         incremental action cost from the current state:
           * If the chosen man already carries a usable spanner, the cost to tighten a nut is:
               dist(man_loc, nut_loc) + 1 (tighten)
           * If the spanner is on the ground at sp_loc, cost is:
               dist(man_loc, sp_loc) + 1 (pickup) + dist(sp_loc, nut_loc) + 1 (tighten)
       - After choosing the minimal-cost option:
           * Add the cost to the heuristic total.
           * Update that man's location to the nut's location (tighten occurs there).
           * Remove the chosen spanner from the available usable set.
           * Remove the tightened nut from the loose set.
       - Continue until all loose nuts are assigned or no feasible moves exist.

    6. Return the accumulated cost as the heuristic estimate.

    Notes on efficiency and behavior
    --------------------------------
    - The greedy assignment is polynomial-time and avoids solving the combinatorial routing
      problem exactly, which keeps the heuristic efficiently computable while producing
      informative guidance.
    - The BFS distances are computed on demand and cached to avoid repeated traversal costs.
    """

    def __init__(self, task):
        """Extract goal facts and static directed links between locations."""
        self.goals = set(task.goals)
        # Parse static facts to build directed adjacency for locations
        self.adj = defaultdict(list)  # directed adjacency: loc -> [neighbouring locs]
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj[l1].append(l2)

        # Cache for BFS distances: (start, goal) -> distance or INF
        self._dist_cache = {}

    def _distance(self, start, goal):
        """Directed shortest-path distance in number of walk actions using BFS."""
        INF = 10**6
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]
        # BFS
        if start not in self.adj:
            self._dist_cache[key] = INF
            return INF
        q = deque()
        q.append((start, 0))
        visited = {start}
        while q:
            loc, d = q.popleft()
            for nb in self.adj.get(loc, []):
                if nb == goal:
                    self._dist_cache[key] = d + 1
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        self._dist_cache[key] = INF
        return INF

    def __call__(self, node):
        """Compute the heuristic value for the given search node."""
        state = node.state

        # 1) If goal is reached, return 0.
        if self.goals <= state:
            return 0

        INF = 10**6

        # 2) Parse state facts.
        men_locs = {}  # man -> location
        spanner_at = {}  # spanner -> location (if on ground)
        spanner_carried_by = {}  # spanner -> man (if carried)
        usable_spanners = set()  # names of usable spanners
        loose_nuts = set()  # set of nut names that are loose
        nut_loc = {}  # nut -> location

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                # could be man, spanner, or nut
                # Types are not explicit here; we make simple assumptions
                if obj.startswith("spanner"):
                    spanner_at[obj] = loc
                elif obj.startswith("nut"):
                    nut_loc[obj] = loc
                else:
                    # assume any other 'at' refers to a man (e.g., bob)
                    # This is conservative: domain uses (at ?m - locatable ?l - location),
                    # and men are of type 'man' (names like bob), while locations are known by links.
                    # To be robust, we check against adjacency keys for location membership:
                    # If loc is a known location (appears in static links), treat obj as movable agent.
                    men_locs[obj] = loc
            elif pred == "carrying":
                man, sp = parts[1], parts[2]
                spanner_carried_by[sp] = man
            elif pred == "usable":
                usable_spanners.add(parts[1])
            elif pred == "loose":
                loose_nuts.add(parts[1])

        # If some nuts are loose but their locations are not known in state, try to be robust:
        # (this should not happen in well-formed instances, but guard against KeyError)
        nuts = list(loose_nuts)
        if not nuts:
            return 0  # nothing to do

        # Build available usable spanner records
        # Each spanner is represented as tuple (name, location, carried_by or None)
        available_spanners = {}
        for sp in usable_spanners:
            if sp in spanner_carried_by:
                available_spanners[sp] = ("carried", spanner_carried_by[sp])
            elif sp in spanner_at:
                available_spanners[sp] = ("at", spanner_at[sp])
            else:
                # usable spanner but no location info (shouldn't happen) - mark unreachable
                available_spanners[sp] = ("unknown", None)

        # Feasibility: if not enough usable spanners overall, return INF
        if len(available_spanners) < len(loose_nuts):
            return INF

        # If there are no men found in the state, the problem is unsolvable (no actor to tighten nuts)
        if not men_locs:
            return INF

        # Prepare mutable structures for greedy assignment
        remaining_nuts = set(loose_nuts)
        # Copy available spanners to a mutable dict we will update when spanners are used
        free_spanners = dict(available_spanners)
        # Man dynamic locations (we will update as we simulate greedy sequence)
        men_current_locs = dict(men_locs)

        total_cost = 0

        # Greedy loop: assign one nut at a time to the best man+spanner option
        while remaining_nuts:
            best = None  # tuple (cost, man, nut, spanner, details)
            # For each man and each nut determine best spanner for them
            for man, man_loc in men_current_locs.items():
                for nut in remaining_nuts:
                    if nut not in nut_loc:
                        # No known location for this nut: treat as unreachable
                        continue
                    nut_location = nut_loc[nut]

                    # Option A: use a usable spanner that is carried by this man
                    # Identify any spanner that is usable and carried by this man
                    carried_spanners = [sp for sp, v in free_spanners.items() if v[0] == "carried" and v[1] == man]
                    for sp in carried_spanners:
                        d = self._distance(man_loc, nut_location)
                        if d >= INF:
                            continue
                        cost = d + 1  # walk actions + tighten action
                        info = ("carried", sp, man_loc, nut_location)
                        if best is None or cost < best[0]:
                            best = (cost, man, nut, sp, info)

                    # Option B: use a usable spanner that is on the ground at some location
                    for sp, v in free_spanners.items():
                        if v[0] != "at":
                            continue
                        sp_loc = v[1]
                        d1 = self._distance(man_loc, sp_loc)
                        if d1 >= INF:
                            continue
                        d2 = self._distance(sp_loc, nut_location)
                        if d2 >= INF:
                            continue
                        cost = d1 + 1 + d2 + 1  # walk to sp + pickup + walk to nut + tighten
                        info = ("ground", sp, sp_loc, nut_location)
                        if best is None or cost < best[0]:
                            best = (cost, man, nut, sp, info)

                    # Option C: use a usable spanner carried by another man (that other man must go)
                    # Consider spanners carried by other men as candidates for that other man,
                    # not the current man. So these will be captured when iterating that other man.

            if best is None:
                # No feasible assignment found (unreachable); return INF
                return INF

            cost, chosen_man, chosen_nut, chosen_spanner, info = best

            # Apply assignment: remove nut and spanner; update man's location and total cost
            total_cost += cost
            # After tightening, chosen_man will be at the nut's location
            men_current_locs[chosen_man] = nut_loc[chosen_nut]
            # Remove spanner from free_spanners
            if chosen_spanner in free_spanners:
                del free_spanners[chosen_spanner]
            # Remove nut from remaining
            remaining_nuts.remove(chosen_nut)

            # Quick safety: if total_cost grows beyond a large threshold, bail out with INF to avoid overflow
            if total_cost >= INF:
                return INF

        return int(total_cost)