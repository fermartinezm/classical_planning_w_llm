from collections import deque
from fnmatch import fnmatch
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Helper: split a fact string '(predicate arg1 arg2 ...)' into
    ['predicate', 'arg1', 'arg2', ...].
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic tailored to the *spanner-modified* domain.
    It approximates the number of actions still required to tighten all nuts
    by combining
      • the walking distance that the (single) mechanic still has to cover,
      • one action per remaining tightening,
      • and, if necessary, the effort for picking up a usable spanner.

    Assumptions
    -----------
    1. Only one `man` object exists (the examples use *bob*).
    2. `link` facts form an undirected, unit–cost graph that never changes.
    3. A single usable spanner is sufficient to tighten every nut.
    4. Action costs are uniform and counted as one per operator execution.

    Heuristic Initialization
    ------------------------
    From the *static* part of the task we build an adjacency dictionary that
    allows us to compute shortest path lengths (by breadth-first search, BFS)
    between any pair of locations.  All results are cached because the
    topology never changes.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search node, the heuristic proceeds as follows

    1.  Early exit – if every goal literal is already true, return 0.
    2.  Extract dynamic information
        a) current location of the man,
        b) whether he already *carries* a spanner,
        c) positions of all usable and still lying spanners,
        d) positions of all *loose* nuts that are still to be tightened.
    3.  Cost for acquiring a spanner
        • If the man already carries a spanner:         0
        • else:   min_distance(man, spanner)  +  1 (pickup)
    4.  Walking cost for visiting every remaining nut
        • Build the set  S  :=  { current_position }  ∪  { nut locations }.
        • Compute the length of a minimum-spanning tree (MST) over  S
          with edge weights = shortest path lengths in the link graph.
          This gives a (loose) lower bound on the walking needed to reach
          every nut once from the current position.
    5.  Tightening actions
        • Each still loose nut needs exactly one `tighten_nut` action.
    6.  Heuristic value
            h = cost_acquire_spanner + MST_length + (#loose_nuts)

    All three summands are non-negative, therefore  h is 0 iff the state is
    a goal state and finite for every solvable state.
    """

    def __init__(self, task):
        # store goal literals for the early-exit test
        self.goals = task.goals

        # ------------------------------------------------------------------
        # Build an (undirected) adjacency list from the static link facts
        # ------------------------------------------------------------------
        self.adj = {}                       # dict: location -> set(neighbours)
        for fact in task.static:
            if fnmatch(fact, "(link * *)"):
                _, l1, l2 = get_parts(fact)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)

        # Usable spanners never change → record once
        self.usable_spanners = {
            get_parts(f)[1] for f in task.static if fnmatch(f, "(usable *)")
        }

        # Distance cache:  (loc1, loc2)   ->  shortest path length
        self._dist_cache = {}

    # ------------- distance handling ----------------------------------

    def _distance(self, loc_from, loc_to):
        """
        Unit-cost shortest path length in the static link graph
        (BFS, results cached).
        """
        if loc_from == loc_to:
            return 0
        key = (loc_from, loc_to)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # BFS
        visited = {loc_from}
        queue = deque([(loc_from, 0)])
        while queue:
            current, dist = queue.popleft()
            for nxt in self.adj.get(current, ()):
                if nxt in visited:
                    continue
                if nxt == loc_to:
                    self._dist_cache[key] = self._dist_cache[(loc_to, loc_from)] = dist + 1
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))

        # If the graph is disconnected this should never happen in valid tasks
        # but return a very large number to stay finite.
        self._dist_cache[key] = self._dist_cache[(loc_to, loc_from)] = 1000000
        return 1000000

    # ------------- MST helper -----------------------------------------

    def _mst_length(self, locations):
        """
        Prim's algorithm: length of a minimum-spanning tree connecting all
        locations (undirected, edge costs = shortest path length).
        """
        if len(locations) <= 1:
            return 0

        locations = list(locations)
        reached = {locations[0]}
        unreached = set(locations[1:])
        total = 0

        while unreached:
            best_dist = inf
            best_loc = None
            for r in reached:
                for u in unreached:
                    d = self._distance(r, u)
                    if d < best_dist:
                        best_dist, best_loc = d, u
            total += best_dist
            reached.add(best_loc)
            unreached.remove(best_loc)

        return total

    # ------------- main heuristic -------------------------------------

    def __call__(self, node):
        state = node.state

        # ------------------------------------------------------------------
        # 0) goal test
        # ------------------------------------------------------------------
        if self.goals <= state:
            return 0

        # ------------------------------------------------------------------
        # 1) identify man, his location and carried spanner (if any)
        # ------------------------------------------------------------------
        man = None
        carried_spanner = None
        for fact in state:
            if fnmatch(fact, "(carrying * *)"):
                _, man, carried_spanner = get_parts(fact)
                break

        if man is None:
            # Fallback – no carrying facts → first 'at' fact with a non-spanner, non-nut object
            for fact in state:
                if fnmatch(fact, "(at * *)"):
                    _, obj, _ = get_parts(fact)
                    if not obj.startswith("spanner") and not obj.startswith("nut"):
                        man = obj
                        break

        # Locate the man
        man_loc = None
        for fact in state:
            if fnmatch(fact, f"(at {man} *)"):
                man_loc = get_parts(fact)[2]
                break

        # Sanity check
        if man is None or man_loc is None:
            return 0  # should not happen in valid instances

        # ------------------------------------------------------------------
        # 2) collect loose nuts and their locations
        # ------------------------------------------------------------------
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if fnmatch(fact, "(loose *)"):
                nut = get_parts(fact)[1]
                loose_nuts.append(nut)

        if not loose_nuts:
            # goals not yet satisfied (maybe missing tightened predicates),
            # but no nuts are loose → return 1 to keep search progressing
            return 1

        # find locations of those nuts
        for fact in state:
            if fnmatch(fact, "(at * *)"):
                _, obj, loc = get_parts(fact)
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        remaining_nut_locs = {nut_locations[n] for n in loose_nuts}

        # ------------------------------------------------------------------
        # 3) cost to acquire a usable spanner (if necessary)
        # ------------------------------------------------------------------
        cost_acquire = 0
        current_pos = man_loc

        if carried_spanner is None:
            # find reachable usable spanners that lie around
            best_extra = inf
            best_spanner_loc = None
            for fact in state:
                if fnmatch(fact, "(at * *)"):
                    _, obj, loc = get_parts(fact)
                    if obj in self.usable_spanners:
                        dist = self._distance(man_loc, loc)
                        extra = dist + 1  # walk + pickup
                        if extra < best_extra:
                            best_extra, best_spanner_loc = extra, loc

            cost_acquire = best_extra if best_extra < inf else 0
            current_pos = best_spanner_loc if best_spanner_loc else man_loc

        # ------------------------------------------------------------------
        # 4) walking cost: length of an MST through current_pos and all nuts
        # ------------------------------------------------------------------
        mst_nodes = set(remaining_nut_locs)
        mst_nodes.add(current_pos)
        walk_cost = self._mst_length(mst_nodes)

        # ------------------------------------------------------------------
        # 5) one tighten action per loose nut
        # ------------------------------------------------------------------
        tighten_cost = len(loose_nuts)

        # ------------------------------------------------------------------
        # 6) final heuristic value
        # ------------------------------------------------------------------
        return cost_acquire + walk_cost + tighten_cost
