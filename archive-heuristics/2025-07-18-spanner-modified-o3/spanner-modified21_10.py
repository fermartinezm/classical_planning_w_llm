import math
from collections import deque
from fnmatch import fnmatch

from heuristics.heuristic_base import Heuristic


###############################################################################
# Helper functions
###############################################################################
def split_fact(fact):
    """
    Turn a fact like "(at bob shed)" into ['at', 'bob', 'shed'].
    """
    return fact[1:-1].split()


def matches(fact, *pattern):
    """
    Wild-card matching for a fact.
        matches("(at bob shed)", "at", "*", "shed")  -> True
    """
    parts = split_fact(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


###############################################################################
# Domain-dependent heuristic
###############################################################################
class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–specific heuristic for the *spanner-modified* domain.  It estimates
    the number of remaining actions (walk, pickup, tighten) required to tighten
    every loose nut.

    Assumptions
    -----------
    * Every nut must be tightened exactly once and never becomes loose again.
    * Any man can reuse a spanner for several nuts without an extra cost.
    * Links between locations are bidirectional (the PDDL encodes only one
      direction, but the `walk` operator can be applied in both).
    * `usable` never changes, i.e. a spanner that is usable in the initial
      state is always usable.

    Heuristic Initialization
    ------------------------
    * From the static facts we build an undirected graph of `link` relations.
    * An all-pairs shortest path table over locations is pre-computed with BFS
      (instances are tiny, so this is inexpensive).
    * The goal set is stored to enable an early exit when it is already
      satisfied.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For the current state we perform the following steps:

    1.  Extract
        • location of every man,
        • which men already carry a spanner,
        • the location of every *free* usable spanner,
        • the location of every loose nut.

    2.  For each loose nut we compute the cheapest way for *any* man to tighten
        it:
        a) If a man already carries a spanner, only walking and the final
           `tighten` action are required:
               distance(man, nut) + 1  (tighten)
        b) Otherwise the man must first fetch a spanner lying somewhere:
               distance(man, spanner)
             + 1                      (pickup)
             + distance(spanner, nut)
             + 1                      (tighten)
        The minimum over all men and (if necessary) all usable spanners is taken.

    3.  The heuristic value returned is the **sum** of those individual minima
        for all currently loose nuts.  
        Summing ignores positive interactions (e.g. one walk can serve two
        nuts), therefore the heuristic is *inadmissible* on purpose but
        produced quickly and is usually informative enough for GBFS.

    The heuristic returns 0 for goal states and a finite value for all
    reachable states.
    """

    def __init__(self, task):
        self.goals = task.goals

        # ------------------------------------------------------------------
        # Build an undirected graph from the static `link` facts
        # ------------------------------------------------------------------
        self.locations = set()
        self.graph = {}
        for fact in task.static:
            if matches(fact, "link", "*", "*"):
                _, l1, l2 = split_fact(fact)
                self.locations.update([l1, l2])
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)

        # ------------------------------------------------------------------
        # Pre-compute all-pairs shortest path distances (BFS for each node)
        # ------------------------------------------------------------------
        self.dist = {loc: self._bfs(loc) for loc in self.locations}

    # --------------------------------------------------------------------------
    # Breadth-first search for single-source shortest paths
    # --------------------------------------------------------------------------
    def _bfs(self, start):
        d = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for nxt in self.graph.get(v, []):
                if nxt not in d:
                    d[nxt] = d[v] + 1
                    q.append(nxt)
        return d

    def _distance(self, a, b):
        """
        Shortest walking distance between two locations.
        Returns +inf if no path exists (should not happen in solvable tasks).
        """
        return self.dist.get(a, {}).get(b, math.inf)

    # --------------------------------------------------------------------------
    # Heuristic evaluation
    # --------------------------------------------------------------------------
    def __call__(self, node):
        state = node.state

        # Early exit if the goal is already satisfied.
        if self.goals <= state:
            return 0

        # ----------------------------------------
        # Extract dynamic information from state
        # ----------------------------------------
        man_location    = {}           # man  -> location
        man_has_tool    = set()        # {man}
        spanner_at_loc  = {}           # spanner -> location   (only free ones)
        usable_spanners = set()        # {spanner}
        loose_nuts      = {}           # nut -> location

        for fact in state:
            parts = split_fact(fact)
            pred = parts[0]

            if pred == "at":
                obj, loc = parts[1], parts[2]
                # Distinguish by type name suffix
                if obj.startswith("nut"):
                    loose_nuts.setdefault(obj, loc)  # will be removed when tightened
                elif obj.startswith("spanner"):
                    spanner_at_loc[obj] = loc
                else:  # man
                    man_location[obj] = loc

            elif pred == "carrying":
                man, spanner = parts[1], parts[2]
                man_location.setdefault(man, None)  # filled earlier usually
                man_has_tool.add(man)

            elif pred == "usable":
                usable_spanners.add(parts[1])

            elif pred == "loose":
                nut = parts[1]
                # The *at* fact for this nut will be encountered as well
                # -> nothing needed here.

        # Keep only loose nuts (remove those already tightened)
        loose_nuts = {nut: loc for nut, loc in loose_nuts.items()
                      if f"(loose {nut})" in state}

        # Keep only usable and free spanners
        free_usable_spanners = {s: loc for s, loc in spanner_at_loc.items()
                                if s in usable_spanners}

        # ------------------------------------------------------------------
        # Compute heuristic as sum over each loose nut
        # ------------------------------------------------------------------
        total_cost = 0
        for nut, nut_loc in loose_nuts.items():
            best_for_nut = math.inf

            for man, man_loc in man_location.items():
                # ------------------------------------------------------
                # Option 1: man already carries (any) spanner
                # ------------------------------------------------------
                if man in man_has_tool:
                    dist = self._distance(man_loc, nut_loc)
                    best_for_nut = min(best_for_nut, dist + 1)  # +1 tighten
                    continue  # carrying is already optimal for this man

                # ------------------------------------------------------
                # Option 2: man must fetch a free, usable spanner
                # ------------------------------------------------------
                for spanner_loc in free_usable_spanners.values():
                    d1 = self._distance(man_loc, spanner_loc)
                    d2 = self._distance(spanner_loc, nut_loc)
                    best_for_nut = min(best_for_nut, d1 + 1 + d2 + 1)  # pickup + tighten

            # Fallback (disconnected graphs) – keeps heuristic finite
            if best_for_nut == math.inf:
                best_for_nut = 0

            total_cost += best_for_nut

        return total_cost
