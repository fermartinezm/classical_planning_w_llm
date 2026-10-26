from collections import defaultdict, deque
from functools import lru_cache
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a fact of the form '(pred arg1 … argN)' into its components
    without the surrounding brackets.
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimates, for the **spanner-modified** domain, how many actions are still
    required until every nut is tightened.  The estimate adds
      • walking steps that the man must still perform  
      • one action per remaining loose nut (the *tighten_nut* action)  
      •, if required, the effort for picking up the closest usable spanner.

    Assumptions
    -----------
    1. Exactly one man exists (all benchmark sets contain only “bob”).
    2. Every *usable* spanner is usable for the whole task and remains usable
       once picked up (the domain contains no “drop” action).
    3. Walking cost is the length (in *walk* actions) of the shortest path
       along *link* relations; *link* is interpreted as **undirected**,
       because all benchmarks contain pairs of links that are meant to be
       traversable in both directions.
    4. To keep the computation cheap we approximate the travelling costs for
       visiting all nuts by  
       `d(start, nearest_nut) + MST(nut_locations)`  
       where *MST* denotes a minimum-spanning-tree over the remaining nut
       locations and the edge weights are walking distances.

    Heuristic Initialization
    ------------------------
    During construction we extract and cache
      • the set of nuts occurring in the goal description,  
      • all spanners that are marked *usable* in the (static) initial state,  
      • the undirected graph induced by *link* facts, and an all-pairs
        shortest-path distance table for every location occurring in that
        graph (computed with BFS from each location).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node we perform the following steps:

    1.  Identify the man and his current location:
        the first `(at ?obj ?loc)` fact whose object is **neither** a nut
        nor a usable spanner is assumed to refer to the man.
    2.  Determine whether the man already *carries* a spanner
        (`(carrying man ?s)`).
    3.  Collect all nuts that are **not yet tightened**.  If none remain,
        return 0.
    4.  (Optional)  If the man does **not** carry a spanner
        • find the closest usable spanner that is still lying on the ground,  
        • add `distance(man_loc, spanner_loc) + 1` to the heuristic
          (1 for the *pickup_spanner* action) and
        • set the man’s “virtual” location to that spanner location.
    5.  Build the set *N* of locations of the remaining loose nuts.
        Compute
          a) `d_start = min_{l in N} distance(man_loc, l)`  
          b) `mst(N)  =` length of a minimum spanning tree over *N*
        (both costs rely on the pre-computed distance table).
    6.  The heuristic value is

            cost_pickup              (0 or distance + 1)
          + d_start                  (walk to the first nut)
          + mst(N)                   (walk among the nuts)
          + |N|                      (one tighten action per nut)

    The resulting value is not admissible but empirically provides a rather
    tight guidance while being inexpensive to compute.
    """

    # --------------------------------------------------------------------- #
    #                         INITIALISATION                                #
    # --------------------------------------------------------------------- #
    def __init__(self, task):
        self.goal_facts = task.goals

        # ----------  gather nuts and usable spanners --------------------- #
        self.nuts = {
            get_parts(g)[1] for g in self.goal_facts if get_parts(g)[0] == "tightened"
        }
        self.usable_spanners = {
            get_parts(f)[1]
            for f in task.static | task.initial_state
            if get_parts(f)[0] == "usable"
        }

        # ----------  build undirected link graph ------------------------- #
        self.locations = set()
        self.neighbours = defaultdict(set)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "link":
                a, b = parts[1], parts[2]
                self.locations.update([a, b])
                self.neighbours[a].add(b)
                self.neighbours[b].add(a)    # treat as undirected

        # ----------  pre-compute all-pairs shortest path distances ------- #
        self.dist = {loc: self._bfs_all_distances(loc) for loc in self.locations}

    # --------------------------------------------------------------------- #
    #                         PUBLIC CALL                                   #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ----------  remaining loose nuts -------------------------------- #
        remaining_nuts = [
            n for n in self.nuts if f"(tightened {n})" not in state
        ]
        if not remaining_nuts:            # goal state
            return 0

        # ----------  man and his position -------------------------------- #
        man_name, man_loc = self._locate_man(state)
        if man_loc is None:
            # Should never happen for legal instances
            return float("inf")

        # ----------  does he carry a spanner? ---------------------------- #
        carries_spanner = any(
            fact.startswith(f"(carrying {man_name} ") for fact in state
        )

        # ----------  optional spanner pick-up ---------------------------- #
        pickup_cost = 0
        if not carries_spanner:
            spanner_positions = [
                get_parts(f)[2]                     # location
                for f in state
                if (parts := get_parts(f))[0] == "at"
                and parts[1] in self.usable_spanners
            ]
            if not spanner_positions:               # unsolvable, but stay finite
                return float("inf")

            # nearest spanner
            dist_to_spanner, spanner_loc = min(
                ((self._distance(man_loc, l), l) for l in spanner_positions),
                key=lambda x: x[0],
            )
            pickup_cost = dist_to_spanner + 1       # +1 for the pickup action
            man_loc = spanner_loc                   # virtual new start

        # ----------  build list of loose nut locations ------------------- #
        nut_locs = [
            get_parts(f)[2]                         # location
            for f in state
            if (parts := get_parts(f))[0] == "at"
            and parts[1] in remaining_nuts
        ]

        # ----------  walk cost: start distance + MST --------------------- #
        start_to_first = min(self._distance(man_loc, l) for l in nut_locs)
        mst_cost = self._mst_length(nut_locs)

        walk_cost = start_to_first + mst_cost
        tighten_cost = len(remaining_nuts)          # 1 per nut

        return pickup_cost + walk_cost + tighten_cost

    # --------------------------------------------------------------------- #
    #                       AUXILIARY ROUTINES                              #
    # --------------------------------------------------------------------- #
    def _locate_man(self, state):
        """
        Return (man_name, location) assuming that every object that is
        neither a nut nor a usable spanner and occurs in an (at … …) fact
        is a man.
        """
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj not in self.nuts and obj not in self.usable_spanners:
                    return obj, loc
        return None, None

    def _distance(self, a, b):
        """Shortest-path distance between two locations (both must exist)."""
        if a == b:
            return 0
        # Guard for isolated locations (should not happen in solvable tasks)
        return self.dist.get(a, {}).get(b, float("inf"))

    def _bfs_all_distances(self, start):
        """Breadth-first search to compute distances from *start*."""
        d = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.neighbours[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    # ---------------  MST over nut locations (Prim)  --------------------- #
    def _mst_length(self, nodes):
        if len(nodes) <= 1:
            return 0

        visited = {nodes[0]}
        edges = [
            (self._distance(nodes[0], v), v)
            for v in nodes[1:]
        ]
        total = 0
        while len(visited) < len(nodes):
            # pick cheapest edge entering the growing tree
            dist, nxt = min(edges, key=lambda x: x[0])
            total += dist
            visited.add(nxt)
            edges = [
                (self._distance(nxt, v), v) if v not in visited else (d, v)
                for (d, v) in edges
            ]
            edges.extend(
                (self._distance(nxt, v), v)
                for v in nodes
                if v not in visited
            )
            # keep only edges that connect visited with unvisited nodes
            edges = [e for e in edges if e[1] not in visited]

        return total
