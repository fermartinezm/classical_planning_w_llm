from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def _parts(fact: str):
    """
    Split a fact string '(pred a b c)' into the list ['pred', 'a', 'b', 'c'].
    The surrounding brackets are removed, satisfying checklist item 1.
    """
    return fact[1:-1].split()


def _is(fact: str, *pattern) -> bool:
    """
    Small helper that allows wild-cards when matching facts, e.g.
    _is(fact, "at", "*", "shed")
    """
    return all(fnmatch(p, q) for p, q in zip(_parts(fact), pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent (inadmissible) heuristic for the *spanner-modified* domain.
    The heuristic assumes a single mechanic who has to pick up one usable spanner
    and then visit every still-loose nut to tighten it.  
    The estimate equals:
        min-distance(to a usable spanner)        +
        1   (for the pickup action if needed)    +
        MST-length over {start,+all loose nuts}  +
        |loose nuts|  (one `tighten_nut` action per nut)

    Assumptions
    -----------
    • Only one `man` exists (true for all IPC instances of the domain).  
    • `link` facts never change – they are contained in `task.static`.  
    • Once a spanner is picked up it is never dropped again.  
    • A spanner that is marked `usable` never becomes unusable.  
    • Walking over one `link` edge costs one action.

    Heuristic Initialization
    ------------------------
    During construction we
    1. extract the list of goal nuts,
    2. build an undirected graph from the static `link` facts,
    3. pre-compute all-pairs shortest path distances (BFS per start node).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node we  
    1. identify
       • the mechanic’s current location,  
       • whether he already *carries* a spanner,  
       • the set and locations of still-loose nuts,  
       • the set and locations of all *usable* spanners that are not carried.
    2. If the goal is already satisfied (`loose` set empty) → return 0.  
    3. If no spanner is carried  
       • choose the closest usable spanner,  
       • add *distance + 1* (for the obligatory `pickup_spanner`) to the cost,  
       • set the current location to that spanner’s position.
    4. Build the location set  S  =  {current location} ∪ {locations of all loose nuts}.  
       Compute the length of a Minimum Spanning Tree over S using Prim’s algorithm
       with the pre-computed shortest-path distances as edge weights.  
       This is a quick but informative approximation of the walking that has to be
       done to visit all nuts exactly once (TSP lower bound).
    5. Add one action cost per nut for the final `tighten_nut`.
    """

    def __init__(self, task):
        # -------- remember which nuts have to be tightened -----------
        self.goal_nuts = {
            _parts(g)[1] for g in task.goals if _is(g, "tightened", "*")
        }

        # -------- build undirected graph of locations ----------------
        self._adj = defaultdict(set)
        self._locations = set()
        for fact in task.static:
            if _is(fact, "link", "*", "*"):
                l1, l2 = _parts(fact)[1:3]
                self._adj[l1].add(l2)
                self._adj[l2].add(l1)
                self._locations.update((l1, l2))

        # -------- all-pairs shortest path distances ------------------
        self._dist = {loc: self._bfs(loc) for loc in self._locations}

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def _bfs(self, start):
        """Breadth-first search returning a distance dictionary from *start*."""
        dist = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for nxt in self._adj[v]:
                if nxt not in dist:
                    dist[nxt] = dist[v] + 1
                    q.append(nxt)
        return dist

    def _distance(self, a, b):
        """Constant-time lookup of the shortest walking distance."""
        if a == b:
            return 0
        return self._dist.get(a, {}).get(b, float("inf"))

    # -----------------------------------------------------------------
    # main heuristic function
    # -----------------------------------------------------------------
    def __call__(self, node):
        state = node.state

        # ------------- which nuts are still loose? -------------------
        loose_nuts = [
            n for n in self.goal_nuts if f"(tightened {n})" not in state
        ]
        if not loose_nuts:          # checklist item 2
            return 0

        # ------------- mechanic’s identity & position ----------------
        man = None
        for fact in state:
            if _is(fact, "carrying", "*", "*"):
                man = _parts(fact)[1]
                break
        if man is None:
            # fall-back: first object that is at some location and is not a nut/spanner
            spanners_static = { _parts(f)[1] for f in state if _is(f, "usable", "*") }
            for fact in state:
                if _is(fact, "at", "*", "*"):
                    obj, _loc = _parts(fact)[1:3]
                    if obj not in self.goal_nuts and obj not in spanners_static:
                        man = obj
                        break

        man_loc = None
        for fact in state:
            if _is(fact, "at", man, "*"):
                man_loc = _parts(fact)[2]
                break

        if man_loc is None:
            # extremely defensive – still return a finite value (checklist 3)
            return len(loose_nuts) * 10

        # ------------- is a spanner already carried? -----------------
        has_spanner = any(_is(f, "carrying", man, "*") for f in state)

        # ------------- positions of usable, non-carried spanners -----
        usable_spanners = {
            _parts(f)[1] for f in state if _is(f, "usable", "*")
        }
        spanner_pos = {}
        for fact in state:
            if _is(fact, "at", "*", "*"):
                obj, loc = _parts(fact)[1:3]
                if obj in usable_spanners:
                    spanner_pos[obj] = loc

        # ------------- positions of the still-loose nuts -------------
        nut_pos = {}
        for fact in state:
            if _is(fact, "at", "*", "*"):
                obj, loc = _parts(fact)[1:3]
                if obj in loose_nuts:
                    nut_pos[obj] = loc

        # ------------- cost for fetching a spanner (if needed) -------
        fetch_cost = 0
        current_loc = man_loc
        if not has_spanner:
            best = float("inf")
            best_loc = None
            for loc in spanner_pos.values():
                d = self._distance(man_loc, loc)
                if d < best:
                    best = d
                    best_loc = loc
            if best == float("inf"):          # unreachable – extremely unlikely
                return len(loose_nuts) * 10 + 100
            fetch_cost = best + 1             # +1 for the pickup action
            current_loc = best_loc

        # ------------- build node set for MST ------------------------
        node_set = {current_loc} | {nut_pos[n] for n in loose_nuts}

        # Prim’s algorithm (graph is tiny, so O(n²) is fine)
        visited = {current_loc}
        edge_cost = {
            loc: self._distance(current_loc, loc) for loc in node_set if loc != current_loc
        }
        mst_len = 0
        while edge_cost:
            nxt, cost = min(edge_cost.items(), key=lambda x: x[1])
            mst_len += cost
            visited.add(nxt)
            edge_cost.pop(nxt)
            for loc in node_set - visited:
                new_c = self._distance(nxt, loc)
                if new_c < edge_cost.get(loc, float("inf")):
                    edge_cost[loc] = new_c

        # ------------- finally: add one action per nut ---------------
        tighten_cost = len(loose_nuts)

        return fetch_cost + mst_len + tighten_cost
