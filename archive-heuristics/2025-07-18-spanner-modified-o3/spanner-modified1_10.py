from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a PDDL fact into its components.
    Example: '(at bob shed)' -> ['at', 'bob', 'shed']
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  It
    estimates how many actions are still necessary to tighten every nut
    that is still loose.

    Assumptions
    -----------
    1. Exactly one *man* is present in every problem instance
       (this is true for all benchmarks that accompany the domain).
    2. A spanner, once picked up, remains usable for the whole plan;
       therefore only one usable spanner is required.
    3. A walk action moves the man across one linked pair of locations
       and costs one step.
    4. Tightening a nut costs one step and can be executed whenever the
       man is at the nut’s location and is carrying a usable spanner.
    5. The predicate `link` is static and undirected,
       so every `(link l1 l2)` can be traversed in both directions.

    Heuristic Initialization
    ------------------------
    • Extract the undirected graph that is induced by all `link`
      predicates appearing in the static facts.  
    • Compute the set of *usable* spanners (facts
      `(usable <spanner>)`) – they never change, hence they can be
      treated as static.  
    • Prepare a cache that stores shortest-path distances between
      locations, filled on demand by breadth-first search.

    Step-By-Step Thinking for Computing the Heuristic
    -------------------------------------------------
    1.  If the goal is already reached (no `loose` facts remain),
        return 0.  
    2.  Determine  
        – the man’s current location,  
        – all still loose nuts together with their locations,  
        – the set of spanners the man is currently carrying, and  
        – every usable spanner that is still lying on the ground
          together with its location.  
    3.  Decide whether the man already possesses a usable spanner:
        • **Yes**  →  no pick-up cost.  
        • **No**   →  choose the closest *ground* spanner,
          add “distance to it” + 1 (for the `pickup_spanner`
          action) to the heuristic and from now on regard the spanner’s
          location as the starting position for traversing the nuts.  
    4.  Regard the traversal of all nut locations (plus the starting
        location from step 3) as a *Steiner tree* problem on the
        `link` graph.  A lower bound on the number of `walk` actions
        needed to visit all those locations is the length of a
        Minimum-Spanning-Tree (MST) over that set of nodes.  Compute the
        MST length with Prim’s algorithm by looking up pre-cached
        shortest path distances between locations.  
    5.  Add one step for every still loose nut (the `tighten_nut`
        action itself).  
    6.  Return  
        pick-up cost + MST length + (# of still loose nuts).

    The heuristic is not admissible (an MST may underestimate travel
    distance and we ignore the final return trip), but it is very cheap
    to compute and dramatically reduces the number of expanded nodes in
    practice.
    """

    def __init__(self, task):
        # ------------- extract static information ------------------
        self.goals = task.goals

        # Build undirected adjacency list from all link predicates
        self.adj = defaultdict(set)
        for fact in task.static:
            pred, *args = get_parts(fact)
            if pred == "link":
                l1, l2 = args
                self.adj[l1].add(l2)
                self.adj[l2].add(l1)

        # Usable spanners never change
        self.usable_spanners = {
            get_parts(fact)[1] for fact in task.static if get_parts(fact)[0] == "usable"
        }

        # Cache for shortest path lengths:  {src: {dst: distance}}
        self._dist_cache = defaultdict(dict)

    # ------------------------------------------------------------------
    #                      helper  utilities
    # ------------------------------------------------------------------
    def _bfs_distances(self, start):
        """Compute all shortest distances from <start> with BFS."""
        if start in self._dist_cache:
            return self._dist_cache[start]

        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in self.adj[current]:
                if nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    queue.append(nxt)
        self._dist_cache[start] = dist
        return dist

    def _distance(self, loc1, loc2):
        """Return shortest path length between two locations."""
        if loc1 == loc2:
            return 0
        return self._bfs_distances(loc1).get(loc2, float("inf"))

    def _mst_length(self, nodes):
        """Length of an MST over *nodes* using pre-computed distances."""
        if len(nodes) <= 1:
            return 0

        nodes = list(nodes)
        visited = {nodes[0]}
        edges = [
            (self._distance(nodes[0], v), v) for v in nodes[1:]
        ]  # (cost, node)
        import heapq

        heapq.heapify(edges)
        total = 0
        while edges:
            cost, v = heapq.heappop(edges)
            if v in visited:
                continue
            visited.add(v)
            total += cost
            for w in nodes:
                if w not in visited:
                    heapq.heappush(edges, (self._distance(v, w), w))
        return total

    # ------------------------------------------------------------------
    #                main  heuristic  evaluation
    # ------------------------------------------------------------------
    def __call__(self, node):
        state = node.state

        # ------------ 0. Goal test -------------
        if self.goals <= state:
            return 0

        # ------------ 1. classify objects -------------
        loose_nuts = set()
        nut_locations = {}
        spanner_locations = {}
        carried_spanners = set()
        man_location = None

        # First pass – collect nuts & spanners
        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "loose":
                loose_nuts.add(args[0])

            elif pred == "at":
                obj, loc = args
                # Location of nuts/spanners recorded below,
                # man location will be determined later.
                if obj.startswith("nut") or obj in loose_nuts:
                    nut_locations[obj] = loc
                elif obj in self.usable_spanners or obj.startswith("spanner"):
                    spanner_locations[obj] = loc

            elif pred == "carrying":
                man, spanner = args
                carried_spanners.add(spanner)

        # Second pass – find the man location (entity that is not a nut or spanner)
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "at":
                entity, loc = args
                if entity not in spanner_locations and entity not in nut_locations:
                    man_location = loc
                    break

        # Safety fallback
        if man_location is None:
            return float("inf")

        # ------------ 2. pick-up cost (if needed) -------------
        carrying_usable = any(s in self.usable_spanners for s in carried_spanners)
        pickup_cost = 0
        start_location = man_location

        if not carrying_usable:
            # find nearest usable spanner that is on the ground
            candidates = [
                (sp, loc)
                for sp, loc in spanner_locations.items()
                if sp in self.usable_spanners
            ]
            if not candidates:
                return float("inf")  # unsolvable (should not happen)

            best_spanner, best_loc = min(
                candidates, key=lambda x: self._distance(man_location, x[1])
            )
            pickup_cost = self._distance(man_location, best_loc) + 1  # +1 for pickup
            start_location = best_loc

        # ------------ 3. movement cost:  MST over nuts + start -------------
        remaining_nut_locs = {nut_locations[n] for n in loose_nuts}
        nodes = remaining_nut_locs | {start_location}
        walk_cost = self._mst_length(nodes)

        # ------------ 4. tighten cost -------------
        tighten_cost = len(loose_nuts)  # one action per nut

        return pickup_cost + walk_cost + tighten_cost
