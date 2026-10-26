from collections import deque
from heuristics.heuristic_base import Heuristic


def parts(fact: str):
    """
    Helper: split a fact string '(pred arg1 arg2 …)' into ['pred', 'arg1', …].
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimate the number of actions still needed to reach the goal in the
    *spanner-modified* domain.  The heuristic tries to count

      • walk actions that move the (single) man through the `link` graph  
      • one `pickup_spanner` action if a usable spanner is not being carried  
      • one `tighten_nut` action for every still-loose nut  
      • additional walks necessary to visit all remaining nuts
        (approximated by   distance(start → closest nut) + MST of nut locations)

    Assumptions
    -----------
    1. There is exactly one man.
    2. Every usable spanner can tighten every nut.
    3. Travelling through `link` edges is possible in both directions
       (the domain usually lists the reverse edge, but we add both directions
       defensively to avoid gross overestimation).
    4. Each `tighten_nut` completes a nut in exactly **one** action.
       The heuristic therefore never under–estimates that cost.

    Heuristic Initialization
    ------------------------
    From the task’s static information we extract the undirected *link graph*
    and store it as an adjacency list.  
    A small cache (`self._dist`) memorises distances that have already been
    computed with breadth-first search.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node we

    1. Parse the state to obtain  
       – current location of every locatable object (`at …`)  
       – which spanners are **usable** (`usable …`)  
       – which nuts are still **loose** (`loose …`)  
       – which spanners are already **carried** (`carrying man spanner`)
    2. Identify the single man and his current location.
    3. If the man already carries a *usable* spanner, we can skip the pick-up
       phase.  Otherwise:  
          – find the nearest still-lying usable spanner  
          – add “walk-distance + 1” (for `pickup_spanner`) to the cost  
          – set the *new* starting location to the position of that spanner
    4. Collect the set of locations that still contain loose nuts.  
       If none exist, the heuristic returns **0**.
    5. Movement estimate to visit all remaining nuts while already carrying a
       spanner:  
          – `d_start`  = shortest path from current position to the **closest**
                         nut location  
          – `mst_cost` = weight of a minimum spanning tree that connects all
                         nut locations (computed with Prim’s algorithm and
                         unit edge costs derived from the link graph)  
          – `move_cost = d_start + mst_cost`
    6. Add `tighten_cost = #loose_nuts` (one action per nut).
    7. The heuristic value returned is  
         `extra_pickup_cost + move_cost + tighten_cost`.

    This computation is inexpensive (distances are cached) yet clearly more
    informative than a simple “number of loose nuts” counter, leading to far
    fewer node expansions in greedy best-first search.
    """

    def __init__(self, task):
        # ----- build undirected link graph ----------------------------------
        self.graph = {}
        for fact in task.static:
            if fact.startswith("(link "):
                _, a, b = parts(fact)
                self.graph.setdefault(a, set()).add(b)
                self.graph.setdefault(b, set()).add(a)
        # distance cache
        self._dist = {}
        # store goals so that goal states receive heuristic 0 quickly
        self._goals = task.goals

    # --------------------------------------------------------------------- #
    # low-level utilities                                                   #
    # --------------------------------------------------------------------- #
    def _shortest(self, src: str, dst: str) -> int:
        """
        Unweighted shortest-path length (number of `walk` actions).
        Uses BFS with memoisation.
        """
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist:
            return self._dist[key]

        visited = {src}
        q = deque([(src, 0)])
        while q:
            cur, d = q.popleft()
            for nb in self.graph.get(cur, ()):
                if nb == dst:
                    self._dist[key] = self._dist[dst, src] = d + 1
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))

        # Should not happen in solvable tasks; keep heuristic finite anyway.
        self._dist[key] = self._dist[dst, src] = 1000
        return 1000

    # --------------------------------------------------------------------- #
    # main callable                                                         #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ---------- quick goal test ----------------------------------------
        if self._goals <= state:
            return 0

        # ---------- collect information from the current state ------------
        usable_spanners = set()
        loose_nuts = set()
        at_location = {}
        carried_spanners = set()

        for fact in state:
            if fact.startswith("(usable "):
                usable_spanners.add(parts(fact)[1])
            elif fact.startswith("(loose "):
                loose_nuts.add(parts(fact)[1])
            elif fact.startswith("(at "):
                _, obj, loc = parts(fact)
                at_location[obj] = loc
            elif fact.startswith("(carrying "):
                _, man, sp = parts(fact)
                carried_spanners.add(sp)

        # no work left?
        if not loose_nuts:
            return 0

        # identify the single man (object that is *at* some location but is
        # neither a nut nor a spanner)
        man = next(
            obj
            for obj, loc in at_location.items()
            if obj not in usable_spanners and obj not in loose_nuts
        )
        man_loc = at_location[man]

        # ---------- check whether we already have a usable spanner ----------
        have_spanner = any(sp in usable_spanners for sp in carried_spanners)
        extra_cost = 0

        if not have_spanner:
            # choose the closest usable, not yet carried spanner
            remaining_spanners = [
                (sp, at_location[sp])
                for sp in usable_spanners
                if sp not in carried_spanners
            ]
            if not remaining_spanners:        # unsolvable but keep finite
                return len(loose_nuts) * 2 + 10

            dist, sp_loc = min(
                (self._shortest(man_loc, loc), loc) for _, loc in remaining_spanners
            )
            extra_cost += dist          # walks to the spanner
            extra_cost += 1             # pickup_spanner
            man_loc = sp_loc            # new start position (carrying now)

        # ---------- movement cost: start -> nuts ---------------------------
        nut_locs = {at_location[nut] for nut in loose_nuts}

        # distance to closest nut
        d_start = min(self._shortest(man_loc, loc) for loc in nut_locs)

        # MST over nut locations (Prim’s algorithm, unit edge costs)
        if len(nut_locs) == 1:
            mst_cost = 0
        else:
            visited = {next(iter(nut_locs))}
            unvisited = nut_locs - visited
            mst_cost = 0
            while unvisited:
                best = min(
                    self._shortest(v, u) for v in visited for u in unvisited
                )
                mst_cost += best
                # move one node (doesn’t matter which) from unvisited to visited
                for u in list(unvisited):
                    if any(self._shortest(v, u) == best for v in visited):
                        visited.add(u)
                        unvisited.remove(u)
                        break

        move_cost = d_start + mst_cost

        # ---------- tighten actions ---------------------------------------
        tighten_cost = len(loose_nuts)

        return extra_cost + move_cost + tighten_cost
