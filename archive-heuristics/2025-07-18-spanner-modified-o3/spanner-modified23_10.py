from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Helper to split a PDDL fact into its individual symbols.

    Example
    -------
    >>> get_parts("(at bob shed)")
    ['at', 'bob', 'shed']
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimates the number of actions that are still required to reach a goal
    state in the *spanner-modified* domain.  It combines
    (1) the distance needed to obtain a usable spanner,
    (2) the travelling effort to visit every yet-untightened nut by means of a
        minimum-spanning-tree approximation, and
    (3) one `tighten_nut` action for every remaining nut.

    Assumptions
    -----------
    •  Only one man exists (true for all IPC instances of this domain).  
    •  Once a spanner is picked up it can be reused for all nuts and never has
       to be dropped.  
    •  Walking between locations incurs unit cost; links are *directed* exactly
       as given by the static `(link … …)` facts.  
    •  Every spanner that is *usable* in the initial state stays usable for the
       whole planning task (it is declared static in the PDDL).  

    Heuristic Initialization
    ------------------------
    •  Parse static `(link l1 l2)` facts and build an adjacency list that is
       used for fast breadth-first shortest-path computation.  
    •  Identify all spanner objects through `(usable ?s)` facts.  
    •  Identify all nut objects through `(loose ?n)` or `(tightened ?n)` facts
       (including goal facts).  
    •  Prepare a cache `self._dist[(src, dst)]` for memoising pairwise shortest
       distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search `node`, proceed as follows:

    1.  **Locate the man**  
        Find the single object that appears in an `(at obj loc)` fact and that
        is *neither* a spanner nor a nut – this must be the man.  Record his
        current location `L_m`.

    2.  **Check whether the man already carries a spanner**  
        If a `(carrying man ?s)` fact exists we are equipped, otherwise we have
        to fetch a spanner first.

    3.  **Collect the set `N_rem` of still loose nuts**  
        A nut is still open iff `(tightened n)` is *not* contained in the
        current state.  Record for every such nut its location.

        –  If `N_rem` is empty the goal is already satisfied → heuristic 0.

    4.  **Cost for fetching a spanner (if necessary)**  
        For every spanner that is currently *at* some location compute the
        shortest distance from `L_m`; pick the minimum `d_sp`.  
        The cost for getting equipped is  
        `cost_spanner = d_sp             # walk   
                         + 1              # pickup_spanner`
        (0 if the man already carries one).

    5.  **Travelling & tightening the nuts**  
        Let `L_start` be the man’s location *after* the potential pickup
        (`L_m` if already equipped, otherwise the chosen spanner location).

        •  `d_first`  –  shortest distance from `L_start` to the closest nut.  
        •  `mst_len`  –  weight of a Minimum Spanning Tree over all remaining
                         nut locations (edge weights are shortest path
                         distances).  A Prim-style O(k²) algorithm is used,
                         where k = |N_rem|, which is negligible for the very
                         small instances of this domain.  
        •  Tightening each nut costs one additional action,
           hence `cost_tighten = k`.

    6.  **Return the final estimate**

           h = cost_spanner + d_first + mst_len + cost_tighten

        The function is *inadmissible* on purpose (it ignores that the man
        might have to walk back when collecting the spanner etc.) but it is
        extremely informative and inexpensive to compute, thereby strongly
        reducing node expansions in greedy best-first search.
    """

    # ----------  constructor -------------------------------------------------
    def __init__(self, task):
        # --- identify spanners and nuts -------------------------------------
        self.spanners = {get_parts(f)[1] for f in task.initial_state
                         if get_parts(f)[0] == "usable"}

        self.nuts = set()
        for f in task.initial_state | task.goals:
            pred, obj = get_parts(f)[:2]
            if pred in ("loose", "tightened"):
                self.nuts.add(obj)

        # --- build directed adjacency list ----------------------------------
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "link":
                src, dst = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(dst)

        # distance cache  (src, dst) -> length
        self._dist = {}

    # ----------  helper: shortest directed distance --------------------------
    def _shortest_dist(self, src, dst):
        """
        Directed shortest-path length between locations `src` and `dst`.
        BFS with memoisation; returns float('inf') if `dst` is unreachable.
        """
        if src == dst:
            return 0
        if (src, dst) in self._dist:
            return self._dist[(src, dst)]

        visited = {src}
        queue = deque([(src, 0)])

        while queue:
            loc, d = queue.popleft()
            for nxt in self.adj.get(loc, ()):
                if nxt in visited:
                    continue
                if nxt == dst:
                    self._dist[(src, dst)] = d + 1
                    return d + 1
                visited.add(nxt)
                queue.append((nxt, d + 1))

        self._dist[(src, dst)] = float("inf")
        return float("inf")

    # ----------  helper: MST length -----------------------------------------
    def _mst_length(self, locations):
        """
        Prim algorithm on the *metric* graph induced by shortest-path
        distances.  `locations` is an *iterable* of location names.
        """
        locs = list(locations)
        if len(locs) < 2:
            return 0

        reached = {locs[0]}
        unreached = set(locs[1:])
        mst = 0

        while unreached:
            best_dist = float("inf")
            best_node = None
            for r in reached:
                for u in unreached:
                    d = self._shortest_dist(r, u)
                    if d < best_dist:
                        best_dist, best_node = d, u
            mst += best_dist
            reached.add(best_node)
            unreached.remove(best_node)

        return mst

    # ----------  main heuristic function ------------------------------------
    def __call__(self, node):
        state = node.state

        # --- locate man ------------------------------------------------------
        man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj not in self.spanners and obj not in self.nuts:
                    man_loc = loc
                    man_name = obj
                    break

        if man_loc is None:                # should never happen in valid tasks
            return 0

        # --- nuts that are still untightened --------------------------------
        remaining_nuts = [n for n in self.nuts if f"(tightened {n})" not in state]
        if not remaining_nuts:
            return 0                       # goal state

        nut_locations = []
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in remaining_nuts:
                nut_locations.append(parts[2])
        # Safety fallback: if a nut has vanished (shouldn't happen) ignore it.

        # --- does the man already carry a spanner? ---------------------------
        has_spanner = any(get_parts(fact)[:2] == ["carrying", man_name] for fact in state)

        # --- cost for picking up a spanner if needed -------------------------
        pickup_cost = 0
        start_loc = man_loc

        if not has_spanner:
            best_d = float("inf")
            best_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "at" and parts[1] in self.spanners:
                    d = self._shortest_dist(man_loc, parts[2])
                    if d < best_d:
                        best_d = d
                        best_loc = parts[2]
            pickup_cost = best_d + 1        # walk + pickup
            start_loc = best_loc            # man ends up here with spanner

        # If no spanner is reachable, still return a large finite value
        if start_loc is None or pickup_cost == float("inf") + 1:
            return 10_000

        # --- travelling to nuts ---------------------------------------------
        # distance from (possibly new) start location to the nearest nut
        d_first = min(self._shortest_dist(start_loc, l) for l in nut_locations)

        # approximate remaining travel by length of an MST over the nuts
        mst_len = self._mst_length(nut_locations)

        # tightening every nut costs one action
        tighten_cost = len(remaining_nuts)

        return pickup_cost + d_first + mst_len + tighten_cost
