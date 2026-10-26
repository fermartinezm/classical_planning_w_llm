import math
from collections import deque
from heuristics.heuristic_base import Heuristic


def split_fact(fact):
    """
    Helper that converts a fact string like "(at bob shed)" into the list
    ["at", "bob", "shed"].
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *spanner-modified* domain.  The
    heuristic estimates the number of actions that are still required to reach
    a goal state (all nuts tightened) from the current state.  It does **not**
    need to be admissible – it is tuned for guiding a greedy best-first search
    while keeping the computation inexpensive.

    Assumptions
    -----------
    1.  Only one `man` object exists in every task (the benchmark set fulfils
        this requirement).
    2.  `usable` is a static predicate – once a spanner is usable it always
        remains usable.
    3.  All `link` facts are static and model an undirected, unit-cost graph
        that connects locations.
    4.  The cost of every primitive action is 1; therefore the heuristic is
        expressed directly in number of actions.

    Heuristic Initialization
    ------------------------
    During construction the heuristic
    •   extracts every `(link l1 l2)` fact from the task’s static information  
        and builds an undirected adjacency list,  
    •   pre-computes all-pairs shortest path lengths between the locations
        (using a BFS from every node – sufficient because all edges have unit
        cost).

    Step-By-Step Thinking for Computing the Heuristic
    -------------------------------------------------
    Let *d(x,y)* be the pre-computed shortest distance between locations *x*
    and *y* (number of `walk` actions):

    1.  Collect information from the current state  
        •   the location of the *man*,  
        •   which usable spanners he is currently carrying,  
        •   for every *usable* spanner that is **not** carried, its location,  
        •   the set *N* of *loose* nuts and for each of them its location.
    2.  Tightening a nut costs **one** action, so the base cost component is  
        `tighten_cost = |N|`.
    3.  If the man already carries at least one usable spanner
        3.1  build the set `V = { man_location } ∪ { locations of loose nuts }`,  
        3.2  compute the total length of a Minimum Spanning Tree (MST) over
             `V` using the distances *d* – this approximates the minimal number
             of `walk` actions that allow the man (with spanner in hand) to
             visit every nut location,  
        3.3  heuristic value = *MST-cost* + *tighten_cost*.
    4.  Otherwise the man has to **fetch** a usable spanner first.
        For every usable spanner *s* (at location *ls*) compute  
        `cost(s)  = d(man_location, ls)        # walk to spanner
                   + 1                         # pickup_spanner
                   + MST( {ls} ∪ nut_locations )
                   + tighten_cost`
        Choose the minimum `cost(s)` over all usable spanners.  (The first term
        is paid before the MST tour starts; afterwards the man stands at *ls*
        holding the spanner, so *ls* is included in the MST.)
    5.  The resulting integer is returned as heuristic value.  A state in
        which all goals hold immediately yields the heuristic value 0.

    The heuristic is deliberately optimistic but captures the two dominant
    resources of the domain: (i) obtaining a spanner and (ii) moving along the
    `link` graph to visit every still-loose nut.
    """

    # ------------------------------------------------------------------ #
    # Construction helpers                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, task):
        self.goals = task.goals

        # -------- build undirected graph of locations from (link ...) --- #
        self.locations = set()
        self.neighbours = {}  # adjacency list  {loc: set(loc, …)}
        for fact in task.static:
            pred, *args = split_fact(fact)
            if pred == "link":
                l1, l2 = args
                self.locations.update((l1, l2))
                self.neighbours.setdefault(l1, set()).add(l2)
                self.neighbours.setdefault(l2, set()).add(l1)

        # ------------- pre-compute all-pairs shortest path lengths ------- #
        self.dist = {loc: self._bfs(loc) for loc in self.locations}

    # breadth-first search delivering single-source shortest path lengths
    def _bfs(self, start):
        d = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for u in self.neighbours.get(v, ()):
                if u not in d:
                    d[u] = d[v] + 1
                    q.append(u)
        return d

    # tiny wrapper that hides “unknown location” with  +∞
    def _distance(self, l1, l2):
        if l1 == l2:
            return 0
        return self.dist.get(l1, {}).get(l2, math.inf)

    # ------------------------------------------------------------------ #
    # Heuristic evaluation                                               #
    # ------------------------------------------------------------------ #

    def __call__(self, node):
        state = node.state

        # -------- goal reached? ---------------------------------------- #
        if self.goals <= state:
            return 0

        # -------- gather dynamic information from the state ------------ #
        man_name = None
        carried_spanners = []
        man_location = None
        spanner_locs = {}
        usable = set()
        loose_nuts = []
        nut_locs = {}

        for fact in state:
            pred, *args = split_fact(fact)

            if pred == "carrying":
                man_name, sp = args
                carried_spanners.append(sp)

            elif pred == "at":
                obj, loc = args
                if obj.startswith("spanner"):
                    spanner_locs[obj] = loc
                elif obj.startswith("nut"):
                    nut_locs[obj] = loc
                else:  # assume this is the man   (only one exists)
                    if man_name is None:
                        man_name = obj
                    if obj == man_name:
                        man_location = loc

            elif pred == "usable":
                usable.add(args[0])

            elif pred == "loose":
                loose_nuts.append(args[0])

        # a final attempt to get the man's location if it was not found yet
        if man_location is None and man_name is not None:
            for fact in state:
                if fact.startswith(f"(at {man_name} "):
                    man_location = split_fact(fact)[2]
                    break

        # if still unknown (should never happen in valid tasks) fall back
        if man_location is None:
            return len(loose_nuts) + 1  # finite, non-infinite fallback

        if not loose_nuts:
            return 0

        tighten_cost = len(loose_nuts)  # one action per nut

        # -------------- helper:  MST length for a given node set -------- #
        def mst_length(nodes):
            nodes = list(nodes)
            if len(nodes) <= 1:
                return 0
            visited = {nodes[0]}
            length = 0
            while len(visited) < len(nodes):
                best_d = math.inf
                best_node = None
                for v in visited:
                    for u in nodes:
                        if u in visited:
                            continue
                        dist_vu = self._distance(v, u)
                        if dist_vu < best_d:
                            best_d = dist_vu
                            best_node = u
                length += best_d
                visited.add(best_node)
            return length

        nut_locations = {nut_locs[n] for n in loose_nuts}

        # -------------- case 1: already carrying usable spanner --------- #
        have_usable = any(s in usable for s in carried_spanners)
        if have_usable:
            nodes = nut_locations | {man_location}
            walk_cost = mst_length(nodes)
            return walk_cost + tighten_cost

        # -------------- case 2: need to fetch a usable spanner ----------- #
        best = math.inf
        for sp, sp_loc in spanner_locs.items():
            if sp not in usable:
                continue
            to_spanner = self._distance(man_location, sp_loc)
            if math.isinf(to_spanner):       # unreachable, skip
                continue
            walk_after_pick = mst_length(nut_locations | {sp_loc})
            cost = to_spanner + 1 + walk_after_pick + tighten_cost  # +1 pickup
            best = min(best, cost)

        # if for some (invalid) reason no usable spanner is reachable
        if math.isinf(best):
            best = tighten_cost + len(nut_locations)  # still finite

        return best
