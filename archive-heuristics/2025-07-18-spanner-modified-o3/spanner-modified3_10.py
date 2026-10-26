from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Helper that removes the surrounding brackets of a PDDL fact
    and splits it into its individual components.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Wild-card matcher for PDDL facts.
    All elements in *pattern must match the corresponding components
    of the fact.  A '*' in the pattern is treated as a wild-card
    (see fnmatch module).

    >>> match("(at bob shed)", "at", "bob", "*")   -> True
    >>> match("(at bob shed)", "at", "*", "gate")  -> False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *spanner-modified* domain.  It tries to
    approximate the number of remaining actions (walk, pickup_spanner,
    tighten_nut) that are still required to tighten all loose nuts.

    Assumptions
    ----------
    • There is only one man in the task (as in all IPC versions of this
      domain).  
    • Every spanner that appears in the initial state with `(usable ?s)`
      stays usable forever (the predicate never changes).  
    • `walk` moves exactly one edge of the undirected location graph.  
    • Distances between locations are measured in number of `walk` actions.

    Heuristic Initialisation
    ------------------------
    During construction the heuristic
    1. stores the goal conditions (which are always of the form
       `(tightened ?n)`),  
    2. extracts the static *link* graph and builds an *undirected* adjacency
       list,  
    3. prepares a cache for shortest-path distances that will be filled on
       demand (via breadth-first search).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node the heuristic performs the following steps

    1.   Collect the dynamic information of the current state  
         • location of the man,  
         • set of loose nuts with their locations,  
         • locations of all *usable* spanners,  
         • whether the man is already **carrying** a usable spanner.  

    2.   If **all** nuts are already tightened – return 0.

    3.   Make sure the man gets a usable spanner  
         • If he carries none (or only unusable ones) choose the *closest*
           usable spanner, add  
             distance(man, spanner) `+ 1`               (pickup_spanner)
           to the heuristic value and pretend the man is now standing at
           the spanner’s location holding it.

    4.   Tighten the remaining nuts  
         • Every nut needs **one** `tighten_nut` action  → `+ len(loose_nuts)`  
         • The man has to *walk* between locations:
           – distance from the current position to the *first* nut,  
           – plus an estimate of the travel needed to visit all other nuts.
           Here we use the length of a **Minimum Spanning Tree** (MST) over
           the nut locations as a cheap approximation of an optimal tour.
           The cost added therefore is  
             distance_to_closest_nut `+ mst_length`.

    5.   Sum up all individual costs and return the total as heuristic value.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, task):
        self.goals = task.goals

        # --- build undirected location graph from static (link ...) facts ---
        self.graph = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)  # treat as undirected

        # cache for shortest path lengths:  (loc1, loc2) -> int
        self._dist_cache = {}

    # ---------------------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------------------
    def _distance(self, start, goal):
        """Shortest undirected path length between two locations (BFS)."""
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # classic BFS
        frontier = deque([(start, 0)])
        visited = {start}
        while frontier:
            current, d = frontier.popleft()
            for neigh in self.graph.get(current, ()):
                if neigh == goal:
                    self._dist_cache[key] = self._dist_cache[(goal, start)] = d + 1
                    return d + 1
                if neigh not in visited:
                    visited.add(neigh)
                    frontier.append((neigh, d + 1))

        # If no path exists (should not happen in solvable instances)
        # return a large constant so that the state still receives a finite
        # heuristic value.
        self._dist_cache[key] = self._dist_cache[(goal, start)] = 1000
        return 1000

    def _mst_length(self, locations):
        """Length of a Minimum Spanning Tree over the given locations."""
        if len(locations) <= 1:
            return 0

        remaining = set(locations)
        current = remaining.pop()          # arbitrary start
        visited = {current}
        cost = 0

        while remaining:
            # find shortest edge that connects visited <-> remaining
            best_dist, best_node = min(
                ((self._distance(v, r), r) for v in visited for r in remaining),
                key=lambda x: x[0],
            )
            cost += best_dist
            visited.add(best_node)
            remaining.remove(best_node)

        return cost

    # ---------------------------------------------------------------------
    # Main heuristic function
    # ---------------------------------------------------------------------
    def __call__(self, node):
        state = node.state

        # 1. Collect static sets for easy membership tests
        spanners = {get_parts(f)[1] for f in state if match(f, "usable", "*")}
        nuts     = {get_parts(f)[1] for f in state
                    if match(f, "loose", "*") or match(f, "tightened", "*")}

        # 2. Locate every object listed in an (at ?obj ?loc) fact
        object_location = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                object_location[obj] = loc

        # 3. Find the (single) man
        #    – easiest: the object that occurs in a carrying-fact
        man = None
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man = get_parts(fact)[1]
                break
        if man is None:
            # fall-back: choose the first object that is located somewhere
            # and is NOT a spanner or nut (works for the benchmark suite)
            for obj in object_location:
                if obj not in spanners and obj not in nuts:
                    man = obj
                    break
        man_loc = object_location.get(man, None)

        # 4. Determine which spanners are currently carried and usable
        carried_spanners = {get_parts(f)[2] for f in state
                            if match(f, "carrying", man, "*")}
        has_usable_spanner = any(s in spanners for s in carried_spanners)

        # 5. Identify all loose nuts and their locations
        loose_nuts = []
        loose_nut_locations = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loc = object_location[nut]
                loose_nuts.append(nut)
                loose_nut_locations.append(loc)

        # Goal test – if no loose nuts remain the estimate is 0
        if not loose_nuts:
            return 0

        # ---------------------------------------------------------------
        # Estimate remaining costs
        # ---------------------------------------------------------------
        cost = 0

        # --- Step A: acquire a usable spanner if necessary -----------------
        if not has_usable_spanner:
            # Find closest usable spanner on the ground
            best_dist = float("inf")
            best_loc  = None
            for sp in spanners:
                if sp in carried_spanners:
                    continue  # unusable or already counted
                sp_loc = object_location.get(sp, None)
                if sp_loc is None:
                    continue
                d = self._distance(man_loc, sp_loc)
                if d < best_dist:
                    best_dist, best_loc = d, sp_loc

            # In solvable instances at least one usable spanner must exist
            cost += best_dist + 1        # walk distance + pickup_spanner
            man_loc = best_loc           # pretend the man now stands here

        # --- Step B: tighten the remaining nuts ----------------------------
        num_nuts = len(loose_nut_locations)

        # (i)  cost for walking
        dist_to_first = min(self._distance(man_loc, loc)
                            for loc in loose_nut_locations)
        mst_len = self._mst_length(loose_nut_locations)
        cost += dist_to_first + mst_len

        # (ii) cost for tighten_nut actions (one per nut)
        cost += num_nuts

        return cost
