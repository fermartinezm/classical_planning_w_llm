import math
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def split_fact(fact):
    """
    Return predicate and arguments of a PDDL fact that is given as a string,
    e.g. '(at bob shed)'  ->  ('at', ['bob', 'shed'])
    """
    parts = fact[1:-1].split()
    return parts[0], parts[1:]


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  It
    estimates the number of actions that are still necessary to tighten the
    remaining loose nuts.  The estimate is composed of
    1. the cheapest cost for obtaining **one** usable spanner,
    2. a lower-bound on the walking effort that is required to visit all
       locations containing loose nuts (minimal spanning tree over those
       locations plus the starting location), and
    3. one unit of cost for each still loose nut to account for the actual
       `tighten_nut` operator.

    Assumptions
    -----------
    • Only one man exists (the domain instances follow this convention).  
    • A usable spanner can tighten an arbitrary number of nuts, hence it
      suffices to pick up *one* usable spanner.  
    • `link` predicates never change and therefore form a static (possibly
      directed) graph of unit-cost edges that is the same for every state of
      the search.  
    • `usable` predicates are static as well – a spanner that is usable in the
      initial state stays usable for the whole planning task.  

    Heuristic Initialization
    ------------------------
    While constructing the heuristic object we
    1. extract the directed link graph from the static facts,
    2. remember every location that appears in a link,
    3. pre-compute all-pairs shortest path distances between the locations
       (simple breadth-first search from every location because edges are
       unweighted), and
    4. store the set of *usable* spanners for quick access in the heuristic
       call.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search **node** we perform the following steps.

    1.  Parse the state and collect the required information:  
        – current location of the man,  
        – which usable spanners he is already carrying,  
        – locations of all *usable* spanners that are still lying around,  
        – locations of all nuts that are still *loose* and therefore have to
          be tightened.

    2.  If there are no loose nuts left, we are already in a goal state and
        return 0.

    3.  Decide whether the man still needs to fetch a spanner.  
        If he is already carrying at least one **usable** spanner,
            pickup_cost = 0 and the “tour” starts at his current location.  
        Otherwise we iterate over all reachable usable spanners that are still
        lying around, compute the distance from the current location to the
        spanner, add the *pickup* action (cost 1), and later choose the
        *cheapest* of those candidates.

    4.  For the chosen starting location we have to visit every nut location
        at least once.  A lower bound on the total walking effort is the cost
        of a Minimum Spanning Tree (MST) that spans the starting location and
        all nut locations.  Because all edge costs are 1, Prim’s algorithm
        running on the pre-computed distance matrix is sufficient and fast.

    5.  Finally we add one unit of cost for every still loose nut to model the
        actual `tighten_nut` operator that has to be executed at those
        locations.

    The heuristic value is therefore  
        h = cheapest_pickup_cost + mst_cost + (#loose_nuts).

    The computation is polynomial in the number of locations/nuts and very
    light-weight in practice.
    """

    def __init__(self, task):
        # ---------- extract static information --------------------------------
        static_facts = task.static

        # directed link graph
        self.graph = defaultdict(list)
        locations = set()
        for fact in static_facts:
            pred, args = split_fact(fact)
            if pred == "link":
                src, dst = args
                self.graph[src].append(dst)
                locations.update([src, dst])

        self.locations = list(locations)

        # usable spanners
        self.usable_spanners = {
            split_fact(fact)[1][0] for fact in static_facts
            if split_fact(fact)[0] == "usable"
        }

        # ---------- pre-compute shortest path distances -----------------------
        self.dist = {loc: self._bfs_distances(loc) for loc in self.locations}

    # --------------------------------------------------------------------- #
    #  Helper: BFS distances from one start location                         #
    # --------------------------------------------------------------------- #
    def _bfs_distances(self, start):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in self.graph[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        # For convenience we return a defaultdict that yields +INF for
        # unreachable nodes (should not appear in solvable instances).
        return defaultdict(lambda: math.inf, dist)

    # --------------------------------------------------------------------- #
    #  Helper: MST cost over a given set of locations                        #
    # --------------------------------------------------------------------- #
    def _mst_cost(self, nodes):
        """Return the weight of a Minimum Spanning Tree over *nodes*."""
        if len(nodes) <= 1:
            return 0

        # Prim's algorithm – very small graphs, so a simple O(n²) version
        start = next(iter(nodes))
        visited = {start}
        cost = 0

        while visited != nodes:
            best_edge = math.inf
            best_node = None
            for u in visited:
                for v in nodes - visited:
                    d = self.dist[u][v]
                    if d < best_edge:
                        best_edge = d
                        best_node = v
            # The instance is assumed to be solvable, so best_edge < inf
            cost += best_edge
            visited.add(best_node)

        return cost

    # --------------------------------------------------------------------- #
    #  The heuristic function                                                #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ------------ gather dynamic information -----------------------------
        man_location = None
        carried_usable = set()
        loose_nuts = []
        nut_locations = {}
        spanner_at_location = []  # only usable ones lying around

        for fact in state:
            pred, args = split_fact(fact)

            if pred == "at":
                obj, loc = args
                if obj.endswith("bob") or obj == "bob":       # the man
                    man_location = loc
                elif obj in self.usable_spanners:             # a spanner
                    spanner_at_location.append((obj, loc))
                else:                                         # must be a nut
                    nut_locations[obj] = loc

            elif pred == "carrying":
                man, spanner = args
                if spanner in self.usable_spanners:
                    carried_usable.add(spanner)

            elif pred == "loose":
                nut = args[0]
                loose_nuts.append(nut)

        # --------------------------- goal test -------------------------------
        if not loose_nuts:
            return 0

        # ----------------- already carrying a usable spanner? ----------------
        best_heuristic = math.inf

        if carried_usable:
            # no need to pick up, just estimate tour
            start_loc = man_location
            nodes = set(nut_locations[n] for n in loose_nuts) | {start_loc}
            h = self._mst_cost(nodes) + len(loose_nuts)
            best_heuristic = h

        # --------------- consider picking up a (different) spanner -----------
        for spanner, loc in spanner_at_location:
            walk = self.dist[man_location][loc]
            if math.isinf(walk):
                continue  # unreachable
            pickup_cost = walk + 1  # +1 for the pickup action

            start_loc = loc  # after pickup the man is here
            nodes = set(nut_locations[n] for n in loose_nuts) | {start_loc}
            tour_cost = self._mst_cost(nodes)

            h = pickup_cost + tour_cost + len(loose_nuts)
            if h < best_heuristic:
                best_heuristic = h

        # ------------- fallback (should not happen in solvable tasks) --------
        if math.isinf(best_heuristic):
            best_heuristic = 0

        return best_heuristic
