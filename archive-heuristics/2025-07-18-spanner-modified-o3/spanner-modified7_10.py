from collections import deque
from heuristics.heuristic_base import Heuristic


def _get_parts(fact: str):
    """
    Split a fact string like '(at bob shed)' into ['at', 'bob', 'shed'].
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *spanner-modified* domain.  The
    heuristic returns an (inadmissible) estimate of the number of actions
    still required to reach a state in which every nut is **tightened**.

    Assumptions
    -----------
    1.  Only *walk*, *pickup-spanner* and *tighten-nut* actions exist.
    2.  All **link** facts are static; we treat every link as an undirected
        edge that can be traversed with a single *walk* action.
    3.  A spanner, once picked up, is never dropped (there is no action for
        that in the domain).  Hence a single “usable’’ spanner is sufficient
        to tighten an arbitrary number of nuts.
    4.  Most problem descriptions contain a single man, but the code handles
        several and returns the cheapest estimate over all of them.
    5.  The heuristic is intended for greedy best-first search and therefore
        focuses on reducing node expansions, not on being admissible.

    Heuristic Initialisation
    ------------------------
    •   During construction we read the *static* facts of the task and build
        an adjacency list as well as an all-pairs shortest-path distance
        table over the locations mentioned in **link** predicates.  
    •   These distances correspond to the minimum number of *walk* actions
        needed between any two locations and are reused in every call.

    Step-By-Step Thinking for Computing `h(s)`
    -----------------------------------------
    1.  Extract from the current state  
        • the position of every man,  
        • the position (or carried status) of every *usable* spanner,  
        • the set of *loose* nuts together with their locations.
    2.  For each man, independently build a relaxed plan and remember the
        smallest cost obtained:
        a.  If the man already **carries** a spanner, start from his current
            location with no extra cost.
        b.  Otherwise choose the closest *usable* spanner that is not carried
            by anybody, add  
            `distance(man, spanner) + 1` (one action for the walk sequence,
            one *pickup_spanner* action) to the cost, and continue from the
            spanner’s location.
        c.  Repeatedly tighten all remaining loose nuts by a greedy nearest-
            neighbour strategy:  
            • choose the nut with the minimum `distance(current_loc, nutLoc)`,  
            • add `distance + 1` (for the *tighten_nut* action) to the cost,  
            • move the virtual man to that nut’s location and continue until
              no loose nut is left.
    3.  Return the minimum of the individual costs.  
        (If the state is already a goal state, the set of loose nuts is
        empty and the heuristic returns 0.)

    The heuristic obviously drops many constraints (e.g. edge direction,
    global optimal ordering of nuts, possible competition for the same
    spanner) and is therefore **inadmissible**, yet experience shows it gives
    a close and cheaply computable estimate that guides the search well.
    """

    INF = 10 ** 6  # Used when no path exists (should not happen in solvable tasks)

    # ------------------------------------------------------------------ #
    #  Construction helpers                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, task):
        # Build undirected adjacency list from static link facts
        self._adj = {}
        for fact in task.static:
            pred, l1, l2 = _get_parts(fact)
            if pred != "link":
                continue
            self._adj.setdefault(l1, set()).add(l2)
            self._adj.setdefault(l2, set()).add(l1)

        # Pre-compute all-pairs shortest distances between locations
        self._dist = {}
        for loc in self._adj.keys():
            self._dist[loc] = self._breadth_first_distances(loc)

    def _breadth_first_distances(self, start):
        """
        Return a dict mapping every reachable location to the length of the
        shortest path (in *walk* actions) starting from *start*.
        """
        dists = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in self._adj.get(current, []):
                if nxt not in dists:
                    dists[nxt] = dists[current] + 1
                    queue.append(nxt)
        return dists

    def _distance(self, a, b):
        """
        Distance (number of *walk* actions) between two locations.  Returns a
        large constant if the locations are disconnected.
        """
        if a == b:
            return 0
        return self._dist.get(a, {}).get(b, self.INF)

    # ------------------------------------------------------------------ #
    #  Heuristic evaluation                                              #
    # ------------------------------------------------------------------ #
    def __call__(self, node):
        state = node.state

        # -------------------------------------------------------------- #
        #  Gather dynamic information                                   #
        # -------------------------------------------------------------- #
        loose_nuts = set()
        nut_locations = {}
        men = set()
        man_locations = {}
        usable_spanners = set()
        spanner_locations = {}
        carried_by = {}  # spanner -> man

        # First pass: classify objects
        for fact in state:
            parts = _get_parts(fact)
            pred = parts[0]
            if pred == "usable":
                usable_spanners.add(parts[1])
            elif pred == "loose":
                loose_nuts.add(parts[1])
            elif pred == "carrying":
                man, spanner = parts[1], parts[2]
                men.add(man)
                carried_by[spanner] = man
            # Nothing else to do in first pass

        # Second pass: grab locations
        for fact in state:
            parts = _get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                if obj in loose_nuts:
                    nut_locations[obj] = loc
                elif obj in usable_spanners:
                    spanner_locations[obj] = loc
                else:  # everything else that is at a location but not nut/spanner is a man
                    men.add(obj)
                    man_locations[obj] = loc

        # Carrying men have to have locations as well
        for spanner, man in carried_by.items():
            if man not in man_locations:
                # Defensive programming: the planner should always store the man's location
                # but in case it is missing we treat the man as located nowhere.
                man_locations[man] = None

        # If the goal is already reached, heuristic is 0
        if not loose_nuts:
            return 0

        # -------------------------------------------------------------- #
        #  Evaluate each man and take the best                          #
        # -------------------------------------------------------------- #
        best_cost = self.INF
        for man in men:
            current_loc = man_locations.get(man)
            if current_loc is None:
                # No information about man's whereabouts ⇒ skip
                continue

            # ---------------------------------------------------------- #
            #  Do we already carry a usable spanner?                    #
            # ---------------------------------------------------------- #
            carrying = any(sp for sp in carried_by if carried_by[sp] == man and sp in usable_spanners)

            cost = 0
            # If not carrying, choose the closest usable spanner
            if not carrying:
                best_pick_cost = self.INF
                chosen_spanner_loc = None
                for sp in usable_spanners:
                    # Ignore spanners that are already being carried
                    if sp in carried_by:
                        continue
                    sp_loc = spanner_locations.get(sp)
                    if sp_loc is None:
                        continue
                    dist = self._distance(current_loc, sp_loc)
                    if dist < best_pick_cost:
                        best_pick_cost = dist
                        chosen_spanner_loc = sp_loc

                # If no free spanner exists, this branch cannot achieve the goal
                if chosen_spanner_loc is None:
                    continue

                cost += best_pick_cost        # number of walk actions
                cost += 1                     # pickup_spanner action
                current_loc = chosen_spanner_loc

            # ---------------------------------------------------------- #
            #  Greedy tour over remaining nuts                          #
            # ---------------------------------------------------------- #
            remaining_nuts = set(loose_nuts)
            while remaining_nuts:
                # Select nearest nut
                nearest_nut, nearest_dist = None, self.INF
                for nut in remaining_nuts:
                    nut_loc = nut_locations[nut]
                    dist = self._distance(current_loc, nut_loc)
                    if dist < nearest_dist:
                        nearest_nut, nearest_dist = nut, dist

                # No path to a nut – skip this man
                if nearest_nut is None or nearest_dist == self.INF:
                    cost = self.INF
                    break

                cost += nearest_dist       # walk actions
                cost += 1                  # tighten_nut action
                current_loc = nut_locations[nearest_nut]
                remaining_nuts.remove(nearest_nut)

            best_cost = min(best_cost, cost)

        # The problem is solvable, so best_cost should be finite
        return best_cost if best_cost < self.INF else 0
