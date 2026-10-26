from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


###############################################################################
# Helper functions
###############################################################################
def split_fact(fact):
    """
    Convert a PDDL fact string such as "(at bob shed)" into the list
    ['at', 'bob', 'shed'].
    """
    return fact[1:-1].split()


def matches(fact, *pattern):
    """
    Wild-card matching for (partial) facts.  Example:

        matches("(link l1 l2)", "link", "*", "l2")     ->  True
        matches("(at bob shed)",  "at",   "*", "gate") ->  False
    """
    parts = split_fact(fact)
    return all(fnmatch(p, q) for p, q in zip(parts, pattern))


###############################################################################
# Domain dependent heuristic
###############################################################################
class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimate the number of additional actions required to tighten all
    remaining loose nuts in the *spanner-modified* domain.  The heuristic
    explicitly reasons about walking distances, picking up a single spanner
    and subsequently visiting every yet-un-tightened nut.

    Assumptions
    -----------
    1. Exactly one `man` object exists in every task.
    2. A single usable spanner suffices for tightening *all* nuts – the man
       keeps carrying it afterwards.
    3. The (link … …) predicate is *static* and undirected; every walk action
       costs one step.
    4. A *tighten_nut* action always costs one step, independent of the nut.

    Heuristic Initialisation
    ------------------------
    During construction the heuristic:
    • extracts the goal set of nuts that must finally hold `(tightened ?n)`;
    • builds an undirected adjacency list from all static `(link l1 l2)` facts;
    • pre-computes on-demand shortest path distances between any two
      locations and caches the results in `self._dist_cache`.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    When invoked on a search node the heuristic proceeds as follows:

    1.  Parse the current state to obtain
        – the man’s current location,
        – the set and location of every usable spanner,
        – the set of nuts still *loose* together with their locations,
        – whether the man is already *carrying* a spanner.

    2.  If no loose nuts remain, return 0 (goal reached).

    3.  Compute the cheapest *setup* cost that brings the man to the first
        loose nut **while already carrying a spanner**:
            a)  If the man carries a spanner, this is simply the distance to
                the nearest loose nut.
            b)  Otherwise, for every spanner `s` determine
                distance(man, loc(s))          // walk to spanner
              + 1                              // pickup_spanner
              + distance(loc(s), nearest_nut)  // walk to first nut
                and keep the minimum of those alternatives.

    4.  Add      1 × (# loose nuts)            // one tighten_nut per nut.

    5.  Add the length of a Minimum Spanning Tree (MST) that connects all
        loose-nut locations.  This estimates the walking effort needed to
        visit *all* nuts after the first one has been reached.

    The resulting sum is a non-admissible yet usually close estimate of the
    remaining plan length and is inexpensive to compute.
    """

    ###########################################################################
    # Construction helpers
    ###########################################################################
    def __init__(self, task):
        # ------------------------------------------------------------------ #
        #  Goal nuts
        # ------------------------------------------------------------------ #
        self.goal_nuts = {
            split_fact(fact)[1] for fact in task.goals if matches(fact, "tightened", "*")
        }

        # ------------------------------------------------------------------ #
        #  Undirected link graph
        # ------------------------------------------------------------------ #
        self.graph = defaultdict(set)
        for fact in task.static:
            if matches(fact, "link", "*", "*"):
                _, l1, l2 = split_fact(fact)
                self.graph[l1].add(l2)
                self.graph[l2].add(l1)

        # Distance cache for lazy all-pairs shortest paths
        self._dist_cache = {}

    ###########################################################################
    #  Public interface
    ###########################################################################
    def __call__(self, node):
        state = node.state

        # ------------------------------------------------------------------ #
        #  Extract dynamic objects
        # ------------------------------------------------------------------ #
        spanners = {split_fact(f)[1] for f in state if matches(f, "usable", "*")}
        nuts     = {split_fact(f)[1] for f in state if matches(f, "loose", "*") or
                                                   matches(f, "tightened", "*")}

        # Locations --------------------------------------------------------- #
        spanner_loc = {}   # spanner -> location
        nut_loc     = {}   # nut     -> location
        man_loc     = None
        man_name    = None
        carrying_set = set()

        for fact in state:
            pred, *args = split_fact(fact)

            if pred == "at":
                obj, loc = args
                if obj in spanners:
                    spanner_loc[obj] = loc
                elif obj in nuts:
                    nut_loc[obj] = loc
                else:
                    # Object is the (single) man
                    man_name = obj
                    man_loc  = loc

            elif pred == "carrying":
                man_name, sp = args
                carrying_set.add(sp)

        # In some initial states the man might carry no spanner; we still need
        # his location which must already have been found.
        assert man_loc is not None, "Unable to determine man's current location"

        man_carries_spanner = len(carrying_set) > 0

        # ------------------------------------------------------------------ #
        #  Identify remaining loose nuts
        # ------------------------------------------------------------------ #
        loose_nuts = {
            n for n in self.goal_nuts if not matches_in_state(state, "tightened", n)
        }

        if not loose_nuts:
            return 0  # Goal already satisfied

        loose_locations = {nut_loc[n] for n in loose_nuts}

        # ------------------------------------------------------------------ #
        #  STEP 3 – Cost to reach first nut with a spanner
        # ------------------------------------------------------------------ #
        setup_cost = float("inf")

        # Helper: distance between two locations
        def dist(a, b):
            key = (a, b)
            if key not in self._dist_cache:
                self._dist_cache[key] = self._dist_cache[(b, a)] = self._shortest_path(a, b)
            return self._dist_cache[key]

        # Case (a): already carrying a spanner
        if man_carries_spanner:
            closest = min(dist(man_loc, loc) for loc in loose_locations)
            setup_cost = closest
        else:
            # Case (b): must fetch a spanner first
            for sp, s_loc in spanner_loc.items():
                # distance: man -> spanner
                d1 = dist(man_loc, s_loc)
                # choose best first nut from there
                nearest_nut_dist = min(dist(s_loc, loc) for loc in loose_locations)
                total = d1 + 1 + nearest_nut_dist   # +1 for pickup_spanner
                setup_cost = min(setup_cost, total)

        # ------------------------------------------------------------------ #
        #  STEP 4 – One tighten action per remaining nut
        # ------------------------------------------------------------------ #
        tighten_cost = len(loose_nuts)

        # ------------------------------------------------------------------ #
        #  STEP 5 – MST over loose nut locations
        # ------------------------------------------------------------------ #
        mst_cost = self._mst_weight(loose_locations, dist)

        # ------------------------------------------------------------------ #
        #  Final heuristic value
        # ------------------------------------------------------------------ #
        return setup_cost + tighten_cost + mst_cost

    ###########################################################################
    #  Internal helpers
    ###########################################################################
    def _shortest_path(self, start, goal):
        """
        Breadth-first search in the static link graph.
        Returns the length of the shortest undirected path between two
        locations (number of walk actions).
        """
        if start == goal:
            return 0
        frontier = deque([(start, 0)])
        visited = {start}

        while frontier:
            current, d = frontier.popleft()
            for nxt in self.graph[current]:
                if nxt == goal:
                    return d + 1
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append((nxt, d + 1))

        # Graph is assumed to be connected in all solvable instances
        return float("inf")

    def _mst_weight(self, locations, distance_fn):
        """
        Prim’s algorithm on an implicit complete graph whose edge weights are
        given by `distance_fn(loc1, loc2)`.
        """
        locs = list(locations)
        if len(locs) <= 1:
            return 0

        visited = {locs[0]}
        total   = 0

        while len(visited) < len(locs):
            best_dist = float("inf")
            best_loc  = None
            for v in visited:
                for u in locations:
                    if u in visited:
                        continue
                    d = distance_fn(v, u)
                    if d < best_dist:
                        best_dist, best_loc = d, u
            visited.add(best_loc)
            total += best_dist

        return total


###############################################################################
#  Small utility for state membership checks
###############################################################################
def matches_in_state(state, predicate, obj):
    """
    True iff a fact with given predicate and single argument `obj`
    exists in *state*.  Example:

        matches_in_state(state, "tightened", "nut1")
    """
    pattern = f"({predicate} {obj})"
    return pattern in state
