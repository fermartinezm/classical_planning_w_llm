from collections import deque
from typing import Dict, List, Set, Tuple, Optional

from heuristics.heuristic_base import Heuristic


def parts(fact: str) -> List[str]:
    """
    Split a PDDL fact into a list of its components, e.g.

        "(at bob shed)"           -> ["at", "bob", "shed"]
        "(carrying bob spanner1)" -> ["carrying", "bob", "spanner1"]

    The surrounding parentheses are removed before splitting.
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *spanner-modified* domain.  The
    heuristic estimates the number of actions (walk, pickup_spanner,
    tighten_nut) that are still necessary to achieve the goal state.

    Assumptions
    -----------
    1.  There is exactly one *man* in every problem instance.
    2.  All usable-spanner facts are static – a spanner that is usable at the
        beginning stays usable for the whole task (as in the example
        instances).
    3.  The `walk` action cost is 1 per traversed `link`, while
        `pickup_spanner` and `tighten_nut` both cost 1 as well.
    4.  The man never has to drop a spanner: once he carries a usable one he
        can tighten every remaining nut.

    Heuristic Initialisation
    ------------------------
    •  Build a directed *link* graph from the static facts to be able to run
       shortest-path queries (BFS).  
    •  Remember the set of all usable spanners occurring in the task.  
    •  Create a cache that stores, for every start location encountered
       during search, the shortest distance to all other locations.  This
       avoids repeated BFS calls.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Extract from the current *state*  
       – the man’s current location,  
       – whether he already carries a usable spanner,  
       – the (location → nut) mapping for every *loose* nut, and  
       – the (location → spanner) mapping for every *usable* spanner that is
         still on the ground.

    2. If *no* loose nuts exist the goal has been reached → heuristic = 0.

    3. If the man is **not** carrying a usable spanner:  
       – For every available spanner *s* and for every loose nut *n* compute  
         d₁ = distance(man_loc, loc(s))  
         d₂ = distance(loc(s), loc(n))  
         total = d₁ + 1 (pickup) + d₂ + 1 (tighten)  
       – Choose the pair (best_spanner, best_nut) with the minimal *total*
         and add that value to the heuristic.  
       – Update the current location to the nut’s location, mark the nut as
         tightened and remember that the man now carries a spanner.

    4. While loose nuts remain:  
       – Pick the nut that is *closest* (shortest walk distance) to the
         current location, say distance d.  
       – Add d + 1 (walk + tighten) to the heuristic, move the man to that
         nut’s location and mark the nut tightened.

    5. The accumulated sum is returned as heuristic value.  Every component
       is a *real* operator cost, hence the heuristic is admissible‐*enough*
       for greedy best-first search while remaining inexpensive to compute.
    """

    def __init__(self, task):
        # -------------  build static knowledge  ----------------------------
        static_facts = task.static | task.initial_state

        # adjacency list for the directed link graph
        self._adj: Dict[str, Set[str]] = {}
        for fact in static_facts:
            p = parts(fact)
            if p[0] == "link":
                frm, to = p[1], p[2]
                self._adj.setdefault(frm, set()).add(to)

        # all usable spanners appearing somewhere in the task
        self._usable_spanners: Set[str] = {
            parts(fact)[1] for fact in static_facts if parts(fact)[0] == "usable"
        }

        # cache for BFS distances: start_location -> {location: dist}
        self._dist_cache: Dict[str, Dict[str, int]] = {}

    # --------------------------------------------------------------------- #
    #                           helper functions                            #
    # --------------------------------------------------------------------- #
    def _bfs_from(self, start: str) -> Dict[str, int]:
        """
        Standard BFS on the directed link graph, returns *all* reachable
        locations together with their shortest distance (in walk actions)
        from *start*.
        """
        dists = {start: 0}
        queue = deque([start])
        while queue:
            loc = queue.popleft()
            for nxt in self._adj.get(loc, ()):
                if nxt not in dists:
                    dists[nxt] = dists[loc] + 1
                    queue.append(nxt)
        return dists

    def _distance(self, start: str, goal: str) -> Optional[int]:
        """
        Shortest number of walk actions between two locations.
        Returns *None* if there is no path.
        """
        if start == goal:
            return 0
        if start not in self._dist_cache:
            self._dist_cache[start] = self._bfs_from(start)
        return self._dist_cache[start].get(goal)

    @staticmethod
    def _identify_entities(state: Set[str]) -> Tuple[str, Set[str], Dict[str, str], Dict[str, str]]:
        """
        Deduce (man, carried_spanners, loose_nuts, spanners_on_ground) from the
        current state representation.

        Returns
        -------
        man_name
        carried_spanners        : set of spanner names
        loose_nuts              : {nut : location}
        spanners_on_ground      : {spanner : location}
        """
        carried_spanners = set()
        man_name = None
        # ---------------  carried spanners & man -----------------------
        for fact in state:
            p = parts(fact)
            if p[0] == "carrying":
                man_name = p[1]
                carried_spanners.add(p[2])
        # -------------------  all nuts & spanners ----------------------
        nuts = {parts(f)[1] for f in state if parts(f)[0] in ("loose", "tightened")}
        spanners = {parts(f)[1] for f in state if parts(f)[0] == "usable"} | carried_spanners

        # -------------------  object locations -------------------------
        obj_locs = {}
        for fact in state:
            p = parts(fact)
            if p[0] == "at":
                obj_locs[p[1]] = p[2]

        # if the man could not be found from "carrying", deduce him now
        if man_name is None:
            # the man is the only object in 'at' facts that is neither a nut
            # nor a spanner
            for obj, loc in obj_locs.items():
                if obj not in nuts and obj not in spanners:
                    man_name = obj
                    break  # there is only one man

        # loose nuts & spanners on the ground
        loose_nuts = {parts(f)[1]: obj_locs[parts(f)[1]]
                      for f in state if parts(f)[0] == "loose"}
        spanners_on_ground = {sp: loc for sp, loc in obj_locs.items()
                              if sp in spanners and sp not in carried_spanners}

        return man_name, carried_spanners, loose_nuts, spanners_on_ground

    # ------------------------------------------------------------------ #
    #                            main heuristic                          #
    # ------------------------------------------------------------------ #
    def __call__(self, node) -> int:
        state: Set[str] = node.state

        # ----------------------  extract facts ----------------------------
        man, carried, loose_nuts, ground_spanners = self._identify_entities(state)

        if not loose_nuts:            # goal state
            return 0

        current_loc = parts(next(f for f in state if f.startswith(f"(at {man}")))[2])[0] \
            if f"(at {man} " in next(iter(state)) else None
        # safer extraction:
        for fact in state:
            p = parts(fact)
            if p[0] == "at" and p[1] == man:
                current_loc = p[2]
                break

        assert current_loc is not None, "Could not determine man's location"

        heuristic_cost = 0

        # ------------------  step 1: get a usable spanner -----------------
        carrying_usable = bool(carried & self._usable_spanners)

        if not carrying_usable:
            best_plan: Optional[Tuple[int, str]] = None  # (cost, nut_name)

            # no carried spanner → pick a spanner & immediately tighten a nut
            for sp, sp_loc in ground_spanners.items():
                if sp not in self._usable_spanners:
                    continue
                d_man_to_sp = self._distance(current_loc, sp_loc)
                if d_man_to_sp is None:
                    continue
                for nut, nut_loc in loose_nuts.items():
                    d_sp_to_nut = self._distance(sp_loc, nut_loc)
                    if d_sp_to_nut is None:
                        continue
                    total = d_man_to_sp + 1 + d_sp_to_nut + 1  # walk, pickup, walk, tighten
                    if best_plan is None or total < best_plan[0]:
                        best_plan = (total, nut)

            # fall-back: if no path found (should not happen in solvable tasks)
            if best_plan is None:
                return len(loose_nuts) * 3

            # perform the chosen plan in the abstract sense
            cost_to_first, first_nut = best_plan
            heuristic_cost += cost_to_first
            current_loc = loose_nuts[first_nut]
            carrying_usable = True
            del loose_nuts[first_nut]

        # --------------  step 2: tighten remaining loose nuts -------------
        while loose_nuts:
            # choose the nearest remaining nut
            best_nut, best_dist = None, None
            for nut, loc in loose_nuts.items():
                d = self._distance(current_loc, loc)
                if d is None:
                    continue
                if best_dist is None or d < best_dist:
                    best_nut, best_dist = nut, d

            # again, unreachable should not happen in solvable instances
            if best_nut is None:
                heuristic_cost += len(loose_nuts) * 3
                break

            heuristic_cost += best_dist + 1          # walk + tighten
            current_loc = loose_nuts[best_nut]
            del loose_nuts[best_nut]

        return heuristic_cost


# The heuristic must be referenced by exactly this identifier in the planner
spanner_modifiedHeuristic = SpannerModifiedHeuristic