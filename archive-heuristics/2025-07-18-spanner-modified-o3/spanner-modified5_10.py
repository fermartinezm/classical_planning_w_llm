from collections import deque
from heuristics.heuristic_base import Heuristic


def _get_parts(fact: str):
    """
    Split a PDDL fact (given as a string) into its individual components
    without the surrounding parentheses.
    Example:  "(at bob shed)"  ->  ["at", "bob", "shed"]
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  The
    heuristic estimates the number of actions still required to tighten every
    loose nut.  The estimate is based on:
        •  the distance the (only) man still has to walk,
        •  whether he already carries a usable spanner or has to pick one up,
        •  one `pickup_spanner` action (if necessary) and
        •  one `tighten_nut` action per remaining nut.

    Assumptions
    -----------
    1.  There is exactly one man in every task (consistent with all benchmark
        instances of this domain).
    2.  All spanners declared *usable* in the initial state remain usable for
        the whole task (the predicate never changes).
    3.  `walk` actions are costed 1 – i.e. the length of the shortest undirected
        path over the static `link` graph equals the number of required
        `walk` actions.
    4.  A single spanner can be used to tighten any number of nuts, therefore
        only one pickup is necessary.

    Heuristic Initialization
    ------------------------
    •  Build an undirected graph from all static `(link l1 l2)` facts.
    •  Identify and remember
         – the name of the unique man,
         – all *usable* spanners,
         – an empty distance cache that will be filled lazily.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node:
        1.  Collect all currently *loose* nuts and map each object name to its
            current location via the `(at … …)` facts.
        2.  If no nuts are loose, return 0 (goal state).
        3.  Determine
              •  the man’s current location,
              •  whether the man is already *carrying* a spanner.
        4.  Compute the cost of *acquiring* a spanner and reaching the first
            nut:
              •  If the man carries a spanner, it is simply the shortest
                 distance from the man’s location to the nearest loose nut.
              •  Otherwise, consider every *usable* spanner that is lying on
                 the ground.  For each such spanner add
                     distance(man, spanner) + 1 (pickup action) +
                     distance(spanner, nearest loose nut)
                 and keep the minimum of these values.
        5.  Estimate the remaining walking effort after the first nut by the
            weight of a Minimum Spanning Tree (MST) over all nut locations.
            (The MST is computed with Prim’s algorithm using the cached
            shortest-path distances.)
        6.  Add one action (`tighten_nut`) per still loose nut.
        7.  Return
              acquire_cost + mst_cost + #loose_nuts
            as the heuristic value.

    The computation is polynomial and, thanks to distance caching, very fast
    in practice, while giving a much more informed estimate than simple
    counting heuristics.
    """

    # --------------------------------------------------------------------- #
    #                          INITIALISATION                               #
    # --------------------------------------------------------------------- #

    def __init__(self, task):
        self.goals = task.goals
        self._graph = {}          # adjacency list of the static link graph
        self._dist_cache = {}     # (loc1, loc2) -> length of shortest path

        # --- build the undirected link graph --------------------------------
        for fact in task.static:
            parts = _get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self._graph.setdefault(l1, set()).add(l2)
                self._graph.setdefault(l2, set()).add(l1)

        # --- remember all usable spanners (the predicate never changes) -----
        self._usable_spanners = {
            _get_parts(fact)[1]
            for fact in (task.initial_state | task.static)
            if _get_parts(fact)[0] == "usable"
        }

        # --- try to identify the single man ---------------------------------
        self._man = None
        # first try a carrying fact
        for fact in task.initial_state:
            parts = _get_parts(fact)
            if parts[0] == "carrying":
                self._man = parts[1]
                break
        # otherwise: the only locatable object that is neither nut nor spanner
        if self._man is None:
            for fact in task.initial_state:
                parts = _get_parts(fact)
                if parts[0] == "at":
                    obj = parts[1]
                    if not obj.startswith("spanner") and not obj.startswith("nut"):
                        self._man = obj
                        break

    # --------------------------------------------------------------------- #
    #                            DISTANCE HELPERS                           #
    # --------------------------------------------------------------------- #

    def _distance(self, src: str, dst: str) -> int:
        """Shortest number of `walk` actions between src and dst (BFS)."""
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # Breadth-first search
        visited = {src}
        queue = deque([(src, 0)])
        while queue:
            loc, depth = queue.popleft()
            for neigh in self._graph.get(loc, ()):
                if neigh == dst:
                    self._dist_cache[key] = self._dist_cache[(dst, src)] = depth + 1
                    return depth + 1
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, depth + 1))

        # Unreachable – should not happen in solvable tasks
        self._dist_cache[key] = self._dist_cache[(dst, src)] = float("inf")
        return float("inf")

    def _mst_weight(self, locations):
        """Prim’s algorithm – returns 0/∞ correctly for |locations| ≤ 1."""
        locs = list(locations)
        if len(locs) <= 1:
            return 0

        visited = {locs[0]}
        unvisited = set(locs[1:])
        weight = 0

        while unvisited:
            best_cost, best_node = float("inf"), None
            for v in visited:
                for u in unvisited:
                    c = self._distance(v, u)
                    if c < best_cost:
                        best_cost, best_node = c, u
            weight += best_cost
            visited.add(best_node)
            unvisited.remove(best_node)
        return weight

    # --------------------------------------------------------------------- #
    #                            HEURISTIC CORE                             #
    # --------------------------------------------------------------------- #

    def __call__(self, node):
        state = node.state

        # Fast goal test (needed by GBFS behaviour: h=0 only in goal states)
        if self.goals <= state:
            return 0

        # --- gather dynamic information ------------------------------------
        at_locs = {}          # object  -> location
        loose_nuts = set()    # names of still loose nuts
        for fact in state:
            parts = _get_parts(fact)
            pred = parts[0]
            if pred == "at":
                at_locs[parts[1]] = parts[2]
            elif pred == "loose":
                loose_nuts.add(parts[1])

        if not loose_nuts:            # should not occur (handled above)
            return 0

        # man location
        man_loc = at_locs[self._man]

        # is the man already carrying a spanner?
        has_spanner = any(
            _get_parts(fact)[0] == "carrying" and _get_parts(fact)[1] == self._man
            for fact in state
        )

        # locations of all loose nuts
        nut_locs = [at_locs[n] for n in loose_nuts]

        # ------------------------------------------------------------------ #
        # 1) COST OF GETTING EQUIPPED + REACHING THE FIRST NUT               #
        # ------------------------------------------------------------------ #
        if has_spanner:
            acquire_cost = min(self._distance(man_loc, nloc) for nloc in nut_locs)
        else:
            # every ground spanner that is usable is a candidate
            ground_spanners = {
                obj: loc for obj, loc in at_locs.items() if obj in self._usable_spanners
            }
            best = float("inf")
            for sp_loc in ground_spanners.values():
                to_spanner = self._distance(man_loc, sp_loc)
                pick = 1                       # one pickup_spanner action
                to_nut = min(
                    self._distance(sp_loc, nloc) for nloc in nut_locs
                )
                best = min(best, to_spanner + pick + to_nut)
            acquire_cost = best

        # ------------------------------------------------------------------ #
        # 2) WALKING BETWEEN REMAINING NUTS – APPROXIMATE BY MST             #
        # ------------------------------------------------------------------ #
        mst_cost = self._mst_weight(nut_locs)

        # ------------------------------------------------------------------ #
        # 3) ONE TIGHTEN ACTION PER REMAINING NUT                            #
        # ------------------------------------------------------------------ #
        tighten_cost = len(loose_nuts)

        return acquire_cost + mst_cost + tighten_cost
