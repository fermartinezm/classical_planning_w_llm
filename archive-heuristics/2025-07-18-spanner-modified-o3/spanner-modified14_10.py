from collections import defaultdict, deque
from math import ceil
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Utility: split a PDDL fact into its components without the surrounding
    brackets, e.g. '(at bob shed)' -> ['at', 'bob', 'shed']
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  It
    estimates the number of actions still required to tighten every nut that
    appears in the goal.  The estimate is purposely *inadmissible* (it may
    over-estimate) because it is designed for Greedy Best-First Search and
    therefore focuses on guidance quality rather than on optimality.

    Assumptions
    -----------
    1. Exactly one man exists in every task.
    2. All `link` relations are bidirectional and remain static.
    3. Every spanner that is marked `usable` is usable for the entire task.
    4. A single spanner can be used for tightening any number of nuts once it
       is in the man’s hand.
    5. All actions have unit cost.

    Heuristic Initialization
    ------------------------
    • Extract from the *static* part of the task an undirected graph of the
      locations (`link` facts).  Compute and store *all-pairs* shortest path
      lengths in an adjacency dictionary `self.dist[loc1][loc2]`.

    • Determine the sets of *locations*, *spanners* and *nuts* by scanning the
      initial state.  The name of the (single) man is the object that appears
      in an `at` fact and is **neither** a spanner **nor** a nut.

    • Store the list of nuts that must become `(tightened …)` according to the
      goal description.

    Step-By-Step Thinking for Computing the Heuristic
    -------------------------------------------------
    Given a search node:
    1.  If all goals are already satisfied, return 0.
    2.  Identify the man’s current location and whether he is *carrying* a
        spanner (and, if so, which one).
    3.  Build the list of *remaining* loose nuts (those whose `(tightened …)`
        fact is **not** in the current state) together with their locations.
    4.  Decide which spanner will be used:
        • If the man already carries a spanner, that spanner is forced.
        • Otherwise evaluate every usable spanner that is lying around:
          cost = distance(man, spanner) + 1 (pick-up action).
          Select the spanner that minimises this cost **plus** the distance
          from the spanner to the closest not-yet-tightened nut.
    5.  Walking & tightening cost after the spanner is in hand:
        • Every nut requires one `tighten_nut` action  →  `k` actions.
        • The walking effort for visiting all nut locations is approximated by
          the weight of a *Minimum Spanning Tree* (MST) that connects
          {start location} ∪ {nut locations}.  The MST is computed with Prim’s
          algorithm using `self.dist` as edge weights.  The MST length is a
          lower bound on actually having to visit every nut at least once.
    6.  The heuristic value returned is:

            pick_up_cost        (0 if already carrying)
          + mst_length          (moving between locations)
          + #nuts_to_tighten    (one action per nut)

    All arithmetic is integer; the result is finite for every reachable state
    and is zero **only** for goal states.
    """

    def __init__(self, task):
        # ---------- extract static information ----------
        self.goals = task.goals
        static_facts = task.static
        init = task.initial_state

        # 1. Build undirected graph of locations from (link l1 l2)
        graph = defaultdict(set)
        for fact in static_facts:
            pred, *args = get_parts(fact)
            if pred == "link":
                l1, l2 = args
                graph[l1].add(l2)
                graph[l2].add(l1)

        self.locations = set(graph.keys())

        # 2. Pre-compute all-pairs shortest paths (unit weights)
        self.dist = {loc: {loc: 0 for loc in self.locations} for loc in self.locations}
        for start in self.locations:
            visited = self.dist[start]
            q = deque([start])
            while q:
                cur = q.popleft()
                for nxt in graph[cur]:
                    if nxt not in visited:
                        visited[nxt] = visited[cur] + 1
                        q.append(nxt)

        # 3. Identify spanners, nuts and the man from the initial state
        self.spanners = set()
        self.nuts = set()
        for fact in init:
            pred, *args = get_parts(fact)
            if pred == "usable":          # (usable spannerX)
                self.spanners.add(args[0])
            elif pred in ("loose", "tightened"):
                self.nuts.add(args[0])

        # objects that appear in an (at obj location) but are not spanners/nuts
        at_objects = {get_parts(f)[1] for f in init if get_parts(f)[0] == "at"}
        self.man = next(iter(at_objects - self.spanners - self.nuts))

        # 4. Cache goal nuts (those that have to become tightened)
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "tightened"}

    # ---------------------------------------------------------------------- #
    #                           helper functions                              #
    # ---------------------------------------------------------------------- #
    def _mst_length(self, start_loc, target_locs):
        """
        Prim’s algorithm on the complete graph defined by `self.dist`.
        Computes the length of an MST that connects  {start_loc} ∪ target_locs
        """
        if not target_locs:
            return 0
        visited = {start_loc}
        unvisited = set(target_locs)
        total = 0
        while unvisited:
            best_dist = float("inf")
            best_node = None
            for v in visited:
                dv = self.dist[v]
                for u in unvisited:
                    if dv[u] < best_dist:
                        best_dist = dv[u]
                        best_node = u
            visited.add(best_node)
            unvisited.remove(best_node)
            total += best_dist
        return total

    # ---------------------------------------------------------------------- #
    #                                call                                     #
    # ---------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ---------- goal test ----------
        if self.goals <= state:
            return 0

        # ---------- current situation ----------
        # 1. man's location
        man_loc = next(loc for (pred, man, loc) in (get_parts(f) for f in state)
                       if pred == "at" and man == self.man)

        # 2. nuts still to tighten
        remaining_nuts = [nut for nut in self.goal_nuts if f"(tightened {nut})" not in state]
        if not remaining_nuts:
            return 0  # should not happen because of goal test above

        nut_locations = {}
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "at" and args[0] in remaining_nuts:
                nut_locations[args[0]] = args[1]

        # 3. carried spanner?
        carried_spanner = None
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "carrying" and args[0] == self.man:
                carried_spanner = args[1]
                break

        # ---------- choose the “best” spanner ----------
        pick_cost = 0
        start_loc = man_loc

        if carried_spanner is None:
            best_total = float("inf")
            best_start = None
            # evaluate every usable spanner that is lying somewhere
            spanner_locations = {get_parts(f)[1]: get_parts(f)[2]
                                 for f in state
                                 if get_parts(f)[0] == "at" and get_parts(f)[1] in self.spanners}

            for sp, sp_loc in spanner_locations.items():
                # distance man -> spanner + pickup
                cost_to_pick = self.dist[man_loc][sp_loc] + 1     # +1 for pickup_spanner
                # distance spanner -> closest nut
                closest = min(self.dist[sp_loc][nut_locations[n]] for n in remaining_nuts)
                total = cost_to_pick + closest
                if total < best_total:
                    best_total = total
                    best_start = sp_loc
            # store decision
            pick_cost = best_total - closest  # distance(man,sp) + 1
            start_loc = best_start
        # else: already carrying – no extra cost, start at man’s location

        # ---------- walk + tighten cost ----------
        mst_len = self._mst_length(start_loc, [nut_locations[n] for n in remaining_nuts])
        tighten_actions = len(remaining_nuts)

        heuristic_value = pick_cost + mst_len + tighten_actions
        return int(ceil(heuristic_value))
