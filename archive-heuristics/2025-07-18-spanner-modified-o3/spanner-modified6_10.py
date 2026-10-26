from collections import defaultdict
from fnmatch import fnmatch
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a fact-string '(predicate arg1 arg2 ...)' into ['predicate','arg1',...]
    """
    return fact[1:-1].split()


def match(fact: str, *pattern) -> bool:
    """
    fnmatch-based pattern matching for (partial) PDDL facts.
    Wildcards ('*') may be used inside *pattern*.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class spanner_modifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the **spanner-modified** domain.  The
    heuristic estimates the number of remaining actions (walk, pickup and
    tighten) that are still needed to reach a goal state.

    Assumptions
    -----------
    1.  Only one man exists (as in all IPC instances of the domain).
    2.  A *usable* spanner never becomes unusable and can tighten an unlimited
        number of nuts once it is in the man’s possession.
    3.  Walking costs are modelled with the shortest undirected path over the
        static `link` graph.
    4.  The walking costs for visiting several loose nuts are approximated by
        `distance(start, closest nut) + MST(loose nuts)`.  
        (MST = weight of a minimum spanning tree over all loose-nut locations.)

    Heuristic Initialization
    ------------------------
    •  Extract the static `link` relations and build an undirected adjacency
       list as well as an **all-pairs shortest-path** distance table by BFS
       from every location appearing in the task.  
    •  Store the set of *usable spanners* (these facts are static).  
    •  Store the goal facts for quick “goal reached” checks.

    Step-By-Step Thinking for Computing Heuristic h(s)
    --------------------------------------------------
    1. **Goal test** – return `0` if all goal facts hold in *s*.  
    2. **Locate the man** – find the single object *m* that satisfies
       `(at m loc)` and is either
         • mentioned as first argument of a `carrying` fact, or
         • not a location, spanner or nut name.
    3. **Check spanner possession** – determine whether the man already carries
       at least one *usable* spanner.
    4. **If no usable spanner is carried**  
       • compute the distance to the nearest *usable* spanner lying on the
         ground,  
       • add `walk_distance + 1` (the *pickup* action), and  
       • set the current position to the spanner’s location.
    5. **Identify all loose nuts** and their locations.  If none are loose,
       the heuristic cost accumulated so far is returned.
    6. **Walking cost to visit all loose nuts**  
       • `d_start` = distance from the current position to the *closest* loose
         nut,  
       • `mst_weight` = weight of an MST over the graph induced by the loose
         nut locations (Prim’s algorithm with the pre-computed distances).  
       The estimated walking effort is `d_start + mst_weight`.
    7. **Tightening cost** – each loose nut needs exactly one `tighten_nut`
       action, therefore add `|loose_nuts|` to the cost.
    8. **Return** the sum of the components assembled above.  The value is
       always finite for solvable states and equals 0 for goal states, yet it
       is intentionally *inadmissible* to obtain stronger guidance for the
       greedy best-first search.
    """

    def __init__(self, task):
        # Store goal set for quick test
        self.goals = task.goals

        # ------------------------------------------------------------------ #
        #  Extract static information: locations, usable spanners, link graph #
        # ------------------------------------------------------------------ #
        self.locations = set()
        self.adj = defaultdict(set)          # undirected link graph
        self.usable_spanners = set()

        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.update((l1, l2))
                self.adj[l1].add(l2)
                self.adj[l2].add(l1)

        # 'usable' predicates are also static
        for fact in task.static:
            if match(fact, "usable", "*"):
                self.usable_spanners.add(get_parts(fact)[1])

        # ------------------------------------------------------------- #
        #  Pre-compute all-pairs shortest path distances between        #
        #  locations using repeated breadth-first search.               #
        # ------------------------------------------------------------- #
        self.distances = {loc: self._bfs_distances(loc) for loc in self.locations}

    # ---------------------------------------------------------------------- #
    #  Helper: BFS distances from one start node to all other locations      #
    # ---------------------------------------------------------------------- #
    def _bfs_distances(self, start):
        dist = {start: 0}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            for nxt in self.adj[current]:
                if nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    frontier.append(nxt)
        # Fill unreachable nodes with +inf
        for loc in self.locations:
            dist.setdefault(loc, inf)
        return dist

    # ---------------------------------------------------------------------- #
    #  Public interface – compute heuristic value                            #
    # ---------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # 1. goal reached?
        if self.goals <= state:
            return 0

        # ---------------------------------------------------------- #
        # 2. Locate man and current position                         #
        # ---------------------------------------------------------- #
        man_name = None
        man_location = None

        # candidates that surely are men because they occur in 'carrying'
        men_from_carrying = {get_parts(f)[1] for f in state if match(f, "carrying", "*", "*")}

        # find the unique (at man location) fact
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                if obj in men_from_carrying:
                    man_name, man_location = obj, loc
                    break

        # Fallback: heuristically guess the man (object that is at a location
        # and is not itself a location, spanner or nut)
        if man_location is None:
            for fact in state:
                if match(fact, "at", "*", "*"):
                    _, obj, loc = get_parts(fact)
                    if obj not in self.locations and not obj.startswith("spanner") and not obj.startswith("nut"):
                        man_name, man_location = obj, loc
                        break
        # Safety: if still None, return harmless heuristic
        if man_location is None:
            return 0

        # ---------------------------------------------------------- #
        # 3. Does the man already carry a usable spanner?            #
        # ---------------------------------------------------------- #
        carries_usable = any(
            match(fact, "carrying", man_name, sp) and sp in self.usable_spanners
            for fact in state
            for sp in [get_parts(fact)[2]] if match(fact, "carrying", man_name, "*")
        )

        cost = 0

        # ---------------------------------------------------------- #
        # 4. Acquire a spanner if necessary                          #
        # ---------------------------------------------------------- #
        if not carries_usable:
            # find positions of all usable spanners that are not already carried
            ground_spanners = {}
            for fact in state:
                if match(fact, "at", "*", "*"):
                    _, obj, loc = get_parts(fact)
                    if obj in self.usable_spanners:
                        ground_spanners[obj] = loc

            # if none on ground (should not happen in solvable tasks) fallback
            if not ground_spanners:
                return 0

            # choose nearest one
            best_dist, best_loc = inf, None
            for loc in ground_spanners.values():
                d = self.distances[man_location].get(loc, inf)
                if d < best_dist:
                    best_dist, best_loc = d, loc

            cost += best_dist            # walk to spanner
            cost += 1                    # pickup action
            man_location = best_loc      # current position updated

        # ---------------------------------------------------------- #
        # 5. Collect all loose nuts and their locations              #
        # ---------------------------------------------------------- #
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # locate this nut
                for f2 in state:
                    if match(f2, "at", nut, "*"):
                        loc = get_parts(f2)[2]
                        loose_nuts.append(loc)
                        break

        if not loose_nuts:
            return cost  # nothing left to tighten (should be goal, but safe)

        # ---------------------------------------------------------- #
        # 6. Walking cost estimate: nearest nut + MST over nuts      #
        # ---------------------------------------------------------- #
        # distance from current position to the closest loose nut
        start_to_nut = min(self.distances[man_location][loc] for loc in loose_nuts)

        # build MST over loose nut locations (Prim)
        remaining = set(loose_nuts)
        first = remaining.pop()
        mst_weight = 0
        reached = {first}

        while remaining:
            best_edge = inf
            best_node = None
            for u in reached:
                for v in remaining:
                    d = self.distances[u][v]
                    if d < best_edge:
                        best_edge, best_node = d, v
            mst_weight += best_edge
            reached.add(best_node)
            remaining.remove(best_node)

        walk_cost = start_to_nut + mst_weight
        cost += walk_cost

        # ---------------------------------------------------------- #
        # 7. Tighten action cost – one per loose nut                 #
        # ---------------------------------------------------------- #
        cost += len(loose_nuts)

        return cost


__all__ = ["spanner_modifiedHeuristic"]
