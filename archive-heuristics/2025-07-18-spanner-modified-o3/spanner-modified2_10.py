from collections import deque
from fnmatch import fnmatch
from math import inf
from heuristics.heuristic_base import Heuristic


###############################################################################
# Helper functions for parsing PDDL facts (represented as strings)
###############################################################################
def get_parts(fact: str):
    """Return the list of tokens that constitute the fact."""
    return fact[1:-1].split()


def match(fact: str, *pattern) -> bool:
    """Wildcard matching of a fact against a pattern (uses fnmatch)."""
    parts = get_parts(fact)
    return all(fnmatch(p, q) for p, q in zip(parts, pattern))


###############################################################################
#                             HEURISTIC CLASS
###############################################################################
class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *spanner-modified* domain.  The value
    returned is a fast (non-admissible) estimate of the amount of work that
    still has to be done, expressed in number of actions.

    Assumptions
    -----------
    1.  Only **walk**, **pickup_spanner**, and **tighten_nut** actions exist.
    2.  There is at most one *man* that performs all actions (true for the
        benchmark set used in the course).
    3.  Object names follow the conventional prefixes:
        ``spanner``, ``nut`` – everything else is treated as a *man*.
    4.  Every spanner listed as *usable* can be employed to tighten any nut.
    5.  Links are interpreted **directionally**, exactly as they appear in the
        ``(link … …)`` static facts.

    Heuristic Initialisation
    ------------------------
    Static facts are analysed once:
        •  Every ``link`` is stored as a directed edge in an adjacency list  
           (dictionary *adj*).
        •  The set of all locations is collected (needed for graph searches).
    A cache table (*distance_cache*) is prepared so that once the shortest
    distance between two locations has been computed it never needs to be
    re-computed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a state *s* we perform the following steps:

    1.  Extract dynamic information  
        – location of all objects via ``(at … …)`` facts,  
        – list of *loose* nuts that still need work,  
        – whether the man is already ``(carrying …)`` a usable spanner.

    2.  If no loose nuts remain, return 0.

    3.  Estimate *acquisition_cost*:  
        – If the man already holds a usable spanner ⇒ cost = 0.  
        – Otherwise find the **closest** usable & free spanner, add  
          “distance(man, spanner) + 1”   (the “+1” is for the
          ``pickup_spanner`` action).

    4.  Define the *starting location* for the tightening tour  
        (man’s current place if he is carrying, otherwise the location of the
        selected spanner).

    5.  Estimate *walking_cost* needed to visit every loose nut **at least
        once**.  We treat the problem as computing the length of a *Minimum
        Spanning Tree* over the nodes  
        ``{start_location} ∪ {locations of loose nuts}``.  
        The sum of the MST edges is a cheap yet informative proxy for the
        walking effort.

    6.  Each nut still has to be tightened, so  
        ``tightening_cost = |loose_nuts|``.

    7.  The heuristic value is the sum of the three components:

            h(s) = acquisition_cost + walking_cost + tightening_cost

        The value is **0** in goal states, finite for all solvable states, and
        quick to compute.
    """

    # --------------------------------------------------------------------- #
    #                                INIT                                   #
    # --------------------------------------------------------------------- #
    def __init__(self, task):
        self.goals = task.goals  # needed for the "goal test == 0" property

        # -----------  store the directed map of links  ------------------- #
        self.adj = {}                     #  adjacency list  location -> {neigh:1}
        self.locations = set()

        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, src, dst = get_parts(fact)
                self.locations.update([src, dst])
                self.adj.setdefault(src, set()).add(dst)

        # prepare on-demand shortest-path cache
        self._distance_cache = {}         # (loc1,loc2) -> distance (int or inf)

    # --------------------------------------------------------------------- #
    #                         DISTANCE COMPUTATION                          #
    # --------------------------------------------------------------------- #
    def _bfs_distances_from(self, source):
        """Breadth-first search – return dict target→distance from *source*."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        # store in cache
        for tgt, d in dist.items():
            self._distance_cache[(source, tgt)] = d
        return dist

    def _distance(self, a, b):
        """Shortest directed path length from *a* to *b* (∞ if unreachable)."""
        if (a, b) in self._distance_cache:
            return self._distance_cache[(a, b)]
        # compute whole frontier from a once, fill cache
        self._bfs_distances_from(a)
        return self._distance_cache.get((a, b), inf)

    # --------------------------------------------------------------------- #
    #                        MINIMUM SPANNING TREE                          #
    # --------------------------------------------------------------------- #
    def _mst_length(self, nodes):
        """
        Prim’s algorithm over *nodes* (set of locations).  Uses directed
        shortest-path distances.  If some node is unreachable the length is ∞.
        """
        if len(nodes) <= 1:
            return 0

        nodes = set(nodes)
        start = next(iter(nodes))
        visited = {start}
        length = 0

        while visited != nodes:
            best_d = inf
            best_v = None
            for u in visited:
                for v in nodes - visited:
                    d = self._distance(u, v)
                    if d < best_d:
                        best_d, best_v = d, v
            if best_d is inf:          # graph not fully connected
                return inf
            length += best_d
            visited.add(best_v)
        return length

    # --------------------------------------------------------------------- #
    #                              CALL                                     #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # -------------- 1.  collect dynamic information ------------------ #
        at_locs = {}
        usable_spanners = set()
        loose_nuts = set()
        carrying = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                at_locs[obj] = loc
            elif match(fact, "usable", "*"):
                _, sp = get_parts(fact)
                usable_spanners.add(sp)
            elif match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                loose_nuts.add(nut)
            elif match(fact, "carrying", "*", "*"):
                _, man, sp = get_parts(fact)
                carrying[man] = sp

        if not loose_nuts:
            return 0  # goal reached

        # identify the unique man
        men = {obj for obj in at_locs
               if not obj.startswith("spanner") and not obj.startswith("nut")}
        if carrying:
            men.update(carrying.keys())
        man = next(iter(men))  # assume exactly one man
        man_loc = at_locs[man]

        # -------------- 2.  acquisition cost for a spanner --------------- #
        has_usable = man in carrying and carrying[man] in usable_spanners

        acquisition_cost = 0
        if not has_usable:
            # candidate spanners that are usable and standing on the floor
            candidates = [sp for sp in usable_spanners
                          if sp in at_locs]        # i.e., (at sp loc) holds
            if not candidates:
                return 0  # pathological / unsolvable – fall back to 0
            best_sp, best_d = None, inf
            for sp in candidates:
                d = self._distance(man_loc, at_locs[sp])
                if d < best_d:
                    best_sp, best_d = sp, d
            acquisition_cost = best_d + 1      # +1 for (pickup_spanner)
            man_loc = at_locs[best_sp]         # new start location afterwards

        # -------------- 3.  walking cost (MST over nuts + start) --------- #
        nut_locations = {at_locs[nut] for nut in loose_nuts if nut in at_locs}
        mst_nodes = nut_locations | {man_loc}
        walking_cost = self._mst_length(mst_nodes)

        # -------------- 4.  tightening actions --------------------------- #
        tightening_cost = len(loose_nuts)

        # -------------- 5.  final heuristic value ------------------------ #
        h_val = acquisition_cost + walking_cost + tightening_cost
        # Ensure finiteness even if graph disconnected
        return int(h_val if h_val < inf else 1000000)
