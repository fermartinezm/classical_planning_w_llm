from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Split a PDDL fact '(at bob shed)' into its components ['at', 'bob', 'shed'].
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether FACT (string) matches a PATTERN (tuple of strings with '*' wild-cards).
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  It estimates
    the number of actions still required to tighten every loose nut by combining
    (1) the travelling effort of the mechanic, (2) one action per tightening,
    and (3) a possible detour for picking up a usable spanner.

    Assumptions
    -----------
    1. There is exactly one mechanic (type *man*) in every task.
    2. (link A B) is interpreted as an *undirected* edge; walking costs one
       action per edge.
    3. Carrying **one** usable spanner is sufficient to tighten an arbitrary
       number of nuts (the mechanic may actually carry many – we ignore the
       difference).
    4. The heuristic is *inadmissible* on purpose – it tries to be informative,
       not necessarily optimal.

    Heuristic Initialization
    ------------------------
    • Extract all link facts from the static part and build an undirected graph.  
    • Pre-compute and cache shortest path lengths between locations with BFS on
      demand.  
    • Deduce the sets of *spanners* and *nuts* from the presence of `(usable …)`
      and `(loose …)/(tightened …)` facts in the *initial* state.  
    • Identify the single mechanic as the object that does **not** occur in any
      of the above sets.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search *state*:
    1. Collect  
       • mechanic location `M`,  
       • set `L` of loose nuts and their locations,  
       • spanners currently carried by the mechanic,  
       • locations of the remaining (usable) spanners.
    2. If no loose nuts remain, return 0.
    3. Decide whether the mechanic first needs to *pick up* a spanner.  
       • If already carrying one, start directly from `M`.  
       • Otherwise evaluate every available spanner `S`:  
         `cost(M,S) + 1(pick-up) + min_N cost(S,N)` where `N` is a loose nut.  
         Choose the cheapest spanner and remember the corresponding first nut.
    4. After reaching the first nut we still have to visit all other nuts.  
       We approximate this by the length of a *minimum spanning tree* (MST)
       connecting the current position and all remaining nut locations, using
       the graph distances as edge weights.
    5. Add `|L|` for the future `tighten_nut` actions.
    6. Return  
       `distance_to_first_target + (1 if pick-up used) + MST_length + |L|`
       as the heuristic value.  The estimate is finite for every solvable state
       and 0 in goal states.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial = task.initial_state

        # ----------  build undirected link graph ----------
        self.graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, a, b = get_parts(fact)
                self.graph.setdefault(a, set()).add(b)
                self.graph.setdefault(b, set()).add(a)

        # ----------  distance cache ----------
        self._dist_cache = {}  # (loc1, loc2) -> int

        # ----------  classify objects ----------
        self.spanners = {get_parts(f)[1] for f in self.initial
                         if match(f, "usable", "*")}
        self.nuts = {get_parts(f)[1] for f in self.initial
                     if match(f, "loose", "*") or match(f, "tightened", "*")}

        # mechanic = object that is not a spanner/nut and appears in an (at …) fact
        self.mechanic = None
        for fact in self.initial:
            if match(fact, "at", "*", "*"):
                _, obj, _ = get_parts(fact)
                if obj not in self.spanners and obj not in self.nuts:
                    self.mechanic = obj
                    break
        assert self.mechanic is not None, "No mechanic found!"

    # --------------------------------------------------------------------------
    #  Shortest path length between two locations (BFS, cached)
    # --------------------------------------------------------------------------
    def _distance(self, src, dst):
        if src == dst:
            return 0
        if (src, dst) in self._dist_cache:
            return self._dist_cache[(src, dst)]

        seen = {src}
        queue = deque([(src, 0)])
        while queue:
            node, d = queue.popleft()
            for nbr in self.graph.get(node, []):
                if nbr == dst:
                    self._dist_cache[(src, dst)] = self._dist_cache[(dst, src)] = d + 1
                    return d + 1
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append((nbr, d + 1))

        # If no path exists we store +inf (task would actually be unsolvable)
        self._dist_cache[(src, dst)] = self._dist_cache[(dst, src)] = float("inf")
        return float("inf")

    # --------------------------------------------------------------------------
    #  Prim's algorithm for MST length over a set of locations
    # --------------------------------------------------------------------------
    def _mst_length(self, locations):
        if len(locations) <= 1:
            return 0
        visited = {next(iter(locations))}
        remaining = set(locations) - visited
        length = 0
        while remaining:
            best_edge = float("inf")
            best_node = None
            for v in visited:
                for u in remaining:
                    d = self._distance(v, u)
                    if d < best_edge:
                        best_edge, best_node = d, u
            length += best_edge
            visited.add(best_node)
            remaining.remove(best_node)
        return length

    # --------------------------------------------------------------------------
    #  Main heuristic interface
    # --------------------------------------------------------------------------
    def __call__(self, node):
        state = node.state

        # -----------------  collect dynamic information -----------------
        man_location = None
        for fact in state:
            if match(fact, "at", self.mechanic, "*"):
                man_location = get_parts(fact)[2]
                break
        assert man_location is not None, "Mechanic location unknown."

        # loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts.append(nut)
        for nut in loose_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    break

        if not loose_nuts:          # goal state
            return 0

        # spanners carried
        carried_spanners = {get_parts(fact)[2]
                            for fact in state
                            if match(fact, "carrying", self.mechanic, "*")}

        has_spanner = bool(carried_spanners)

        # locations of free, usable spanners
        free_spanner_locations = []
        for sp in self.spanners:
            if sp in carried_spanners:
                continue
            # find its location if it is 'usable'
            is_usable = any(match(f, "usable", sp) for f in self.initial)
            if not is_usable:
                continue
            for fact in state:
                if match(fact, "at", sp, "*"):
                    free_spanner_locations.append((sp, get_parts(fact)[2]))
                    break

        # -------------  estimate movement + pick-up cost ----------------
        tighten_cost = len(loose_nuts)          # one action per nut
        move_cost = 0                           # to be accumulated

        if has_spanner:
            # Already equipped – compute MST starting from current location
            targets = set(nut_locations.values()) | {man_location}
            move_cost += self._mst_length(targets)
        else:
            # Need to pick up a spanner first – try every candidate
            best_total = float("inf")
            best_first_nut_loc = None
            for sp, sp_loc in free_spanner_locations:
                # distance: man -> spanner (walk) + 1 (pick) + spanner -> first nut
                best_d_to_nut = float("inf")
                best_nut = None
                for nut, n_loc in nut_locations.items():
                    d = self._distance(sp_loc, n_loc)
                    if d < best_d_to_nut:
                        best_d_to_nut, best_nut = d, n_loc
                total_first_part = (self._distance(man_location, sp_loc) +
                                    1 +                       # pick-up action
                                    best_d_to_nut)
                if total_first_part < best_total:
                    best_total = total_first_part
                    best_first_nut_loc = best_nut

            move_cost += best_total

            # Remaining nuts (after the first tightened one)
            remaining_nut_locs = set(nut_locations.values())
            remaining_nut_locs.discard(best_first_nut_loc)
            remaining_nut_locs.add(best_first_nut_loc)    # current position
            move_cost += self._mst_length(remaining_nut_locs)

        return move_cost + tighten_cost
