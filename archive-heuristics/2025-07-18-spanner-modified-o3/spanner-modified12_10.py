from collections import defaultdict, deque
from fnmatch import fnmatch
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return a list with the elements of a PDDL fact (without the outer brackets)."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Compare a fact against a pattern that may contain wild-cards (*).

    Example
    -------
    match("(at bob shed)", "at", "*", "*")  ->  True
    """
    parts = get_parts(fact)
    return all(fnmatch(p, q) for p, q in zip(parts, pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the **spanner-modified** domain.  It estimates
    the number of actions still required to tighten all *loose* nuts.

    Assumptions
    -----------
    1. Exactly one `man` exists (e.g. *bob*).
    2. All `spanner` objects that appear in the problem are *usable* for the whole
       planning task.
    3. The action costs of the domain are uniform (each action costs one step).

    Heuristic Initialisation
    ------------------------
    • The static `link` relations are extracted once and transformed into a directed
      graph *G = (V,E)* with unit weights.  
    • All-pairs shortest path lengths `d(u,v)` are pre-computed via repeated
      breadth-first search; they are later re-used in O(1).  
    • No other static information is required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Let  
      L  … set of locations that still contain *loose* nuts.  
      s  … current location of the (single) man.  

    1.  If `L` is empty we are already in a goal state ⇒ return *0*.  
    2.  Determine whether the man already carries a spanner.  
    3.  If **a spanner is already carried**  
        a. The man only needs to visit all nut-locations and execute one
           `tighten_nut` action per nut.  
        b. Travelling cost is approximated by the length of a
           Minimum Spanning Tree that spans `L ∪ {s}` with the metric
           distances `d(u,v)`.  
        c. The estimated cost equals  
                 MST_cost + |L|              (|L| many tighten actions)
    4.  If **no spanner is carried**  
        Consider every *usable* spanner that currently lies on the ground
        in location `p`.  For each such spanner compute  
           pickup_cost(p)   =  d(s,p)  + 1                 (walk + pickup)  
           future_cost(p)   =  MST_cost over  L ∪ {p}  + |L|  
        The heuristic chooses the *minimum* of  
                 pickup_cost(p) + future_cost(p)   for all candidate `p`.
    5.  The returned number is a **non-admissible but informative** estimate of
        the remaining plan length.  It is zero only for states that already
        meet the goal and finite for every solvable state.

    Complexity
    ----------
    • Pre-processing: `O(|V| ⋅ (|V|+|E|))` for all-pairs BFS ( 10–20 nodes 
      in the given benchmarks).  
    • Per call:  
        – State parsing            `O(|state|)`  
        – A handful of distance look-ups and an MST via Prim’s algorithm  
          on ≤ 11 nodes            `O(|L|²)`   (negligible).
    """

    ###########################################################################
    #                               INITIALISATION                            #
    ###########################################################################
    def __init__(self, task):
        # --------- build directed graph from static (link … …) facts ----------
        self.graph = defaultdict(set)
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, u, v = get_parts(fact)
                self.graph[u].add(v)                        # directed edge

        # collect all locations that ever appear
        self.locations = set(self.graph.keys())
        for nbrs in self.graph.values():
            self.locations.update(nbrs)

        # --------- pre-compute all-pairs shortest path distances --------------
        self.dist = {loc: self._bfs_distances(loc) for loc in self.locations}

    ###########################################################################
    #                            PUBLIC INTERFACE                              #
    ###########################################################################
    def __call__(self, node):
        state = node.state

        # -------------- 1. gather dynamic information from the state ----------
        at_map = {}            # object  -> location
        loose_nuts = set()     # names of untightened nuts
        carried_spanners = []  # names of spanners already in hand
        man_name = None

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                at_map[obj] = loc

            elif match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                loose_nuts.add(nut)

            elif match(fact, "carrying", "*", "*"):
                _, man, sp = get_parts(fact)
                carried_spanners.append(sp)
                man_name = man                         # identify the man

        # infer the man's location (robust to the name that was chosen)
        if man_name is None:
            # no spanner held – fall back to the unique object that is at
            # some location but not a nut or spanner
            for obj, loc in at_map.items():
                if not (obj.startswith("spanner") or obj.startswith("nut")):
                    man_name = obj
                    break

        man_loc = at_map[man_name]

        # early exit: all nuts tightened
        if not loose_nuts:
            return 0

        # -------------- 2. prepare helper structures --------------------------
        nut_locations = {at_map[nut] for nut in loose_nuts}

        def mst_length(nodes):
            """Return the weight of a Prim-style MST on a *complete* graph whose
            edge weights are the pre-computed shortest path distances."""
            if len(nodes) <= 1:
                return 0
            visited = {next(iter(nodes))}
            length = 0
            while len(visited) < len(nodes):
                best_edge = inf
                best_node = None
                for v in visited:
                    for u in nodes - visited:
                        d = self._distance(v, u)
                        if d < best_edge:
                            best_edge, best_node = d, u
                if best_edge == inf:      # disconnected graph (unsolvable)
                    return inf
                length += best_edge
                visited.add(best_node)
            return length

        # -------------- 3. compute heuristic value ----------------------------
        tighten_cost = len(loose_nuts)       # one action per nut

        # case A: spanner already carried
        if carried_spanners:
            travel_cost = mst_length(nut_locations | {man_loc})
            return travel_cost + tighten_cost

        # case B: need to pick up a spanner first
        best_total = inf
        for fact in state:
            if match(fact, "at", "spanner*", "*"):
                _, sp, s_loc = get_parts(fact)
                # cost to reach spanner and pick it up
                pickup = self._distance(man_loc, s_loc) + 1      # +1 for pickup
                # afterwards start walking with spanner from s_loc
                travel = mst_length(nut_locations | {s_loc})
                best_total = min(best_total, pickup + travel)

        return best_total + tighten_cost

    ###########################################################################
    #                             INTERNAL METHODS                            #
    ###########################################################################
    def _bfs_distances(self, start):
        """Return a dict {loc : dist(start,loc)} (∞ if unreachable)."""
        dist = defaultdict(lambda: inf)
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if dist[v] == inf:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def _distance(self, a, b):
        """Shortest path length between two locations in the link graph."""
        return self.dist.get(a, {}).get(b, inf)
