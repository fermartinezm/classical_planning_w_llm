from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Split a PDDL fact into its components without the surrounding brackets.

    Example
    -------
    "(at bob shed)" → ["at", "bob", "shed"]
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Greedy distance–based heuristic for the *spanner-modified* domain.  It
    returns an estimate of the number of actions that are still necessary to
    reach a state in which every nut appearing in the goal has been tightened.

    Assumptions
    -----------
    1. Exactly one man exists in every planning instance.
    2. `usable` never changes (it is given only in the initial state).
    3. Once a spanner is picked up it stays in the man's possession for the
       rest of the plan (the domain provides no *drop* operator).
    4. All *link* facts are static and describe an undirected connectivity
       graph between locations.  Every walk action costs one step.

    Heuristic Initialization
    ------------------------
    • From the task’s *static* facts all `link` predicates are collected and
      transformed into an undirected adjacency list `self.graph`.

    • A very small *on-demand* all–pairs–shortest-path cache
      (`self.distance_cache`) is created.  Breadth-first-search is used the
      first time a distance between two locations is requested and afterwards
      stored for re-use.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a state *s*:

    1.  If *s* already satisfies every goal fact the heuristic returns `0`.

    2.  Extract
        - the current man location *l₀*,
        - the set **N** of *untightened* nuts together with their locations,
        - whether the man is already carrying at least one *usable* spanner,
        - the set **S** of *usable* spanners lying on the ground
          (`(at spannerX loc)` and `(usable spannerX)`).

    3.  If the man is **not** carrying a spanner, add  
        `min_{s∈S} distance(l₀,s)      # walk to nearest spanner  
         + 1                           # pickup_spanner`
        to the heuristic value and replace *l₀* by that spanner’s location –
        afterwards the man is assumed to be equipped.

    4.  Visit all yet loose nuts in a nearest-neighbour fashion:

        current_loc ← l₀  
        while **N** ≠ ∅:  
            • choose n* ∈ **N** that minimises distance(current_loc, loc(n*))  
            • h += distance(current_loc, loc(n*))   # walk  
            • h += 1                                # tighten_nut  
            • current_loc ← loc(n*)  
            • **N** ← **N** \\ {n*}

        This greedy tour length is a cheap approximation of the real travelling
        costs and monotonically decreases while the search progresses.

    5.  The accumulated value *h* is returned.  It is `0` for goal states and
        finite for every solvable state because all distances inside a finite
        connectivity graph are finite and every remaining nut contributes at
        least the cost of its tighten action.
    """

    # --------------------------------------------------------------------- #
    #  Construction helpers                                                 #
    # --------------------------------------------------------------------- #
    def __init__(self, task):
        self.goals = task.goals                      # used to detect goal states
        self.graph = defaultdict(set)                # undirected link graph
        self.distance_cache = defaultdict(dict)      # on-demand APSP cache

        # Build undirected adjacency list from the static link facts
        for fact in task.static:
            p, *args = get_parts(fact)
            if p == "link":
                l1, l2 = args
                self.graph[l1].add(l2)
                self.graph[l2].add(l1)

    # --------------------------------------------------------------------- #
    #  Public call operator                                                 #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # --- 1) early exit ------------------------------------------------ #
        if self.goals <= state:
            return 0

        # --- 2) gather dynamic information -------------------------------- #
        man, man_loc = self._locate_man(state)
        carrying_spanner = any(
            get_parts(fact)[0] == "carrying" and get_parts(fact)[1] == man
            for fact in state
        )

        # map every locatable object to its location
        object_location = {
            parts[1]: parts[2] for fact in state for parts in (get_parts(fact),)
            if parts[0] == "at"
        }

        #       loose nuts that still have to be tightened
        loose_nuts = {
            get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "loose"
        }
        nut_locations = {nut: object_location[nut] for nut in loose_nuts}

        #       usable spanners lying on the ground
        usable_spanners = {
            get_parts(fact)[1] for fact in state if get_parts(fact)[0] == "usable"
        }
        ground_spanner_locations = [
            object_location[s] for s in usable_spanners if s in object_location
        ]

        # --- 3) possibly fetch a spanner ---------------------------------- #
        cost = 0
        current_loc = man_loc

        if not carrying_spanner:
            if ground_spanner_locations:
                dist, loc = self._nearest(current_loc, ground_spanner_locations)
                cost += dist          # walking
                cost += 1             # pickup_spanner
                current_loc = loc     # new position after pickup
            else:
                # no spanner on ground (very unlikely) – still return a
                # heuristic value to stay finite
                carrying_spanner = True

        # --- 4) visit every remaining nut --------------------------------- #
        remaining_locations = set(nut_locations.values())
        while remaining_locations:
            dist, loc = self._nearest(current_loc, remaining_locations)
            cost += dist      # walk
            cost += 1         # tighten_nut
            current_loc = loc
            remaining_locations.remove(loc)

        return cost

    # --------------------------------------------------------------------- #
    #  Auxiliary functions                                                  #
    # --------------------------------------------------------------------- #
    def _nearest(self, source, targets):
        """
        Return (distance, target) of the target in *targets* that is closest to
        *source*.  The graph distance is looked-up / computed via BFS.
        """
        best_dist = float("inf")
        best_target = None
        for t in targets:
            d = self._distance(source, t)
            if d < best_dist:
                best_dist, best_target = d, t
        return best_dist, best_target

    def _distance(self, start, goal):
        """
        Shortest path length between two locations in the undirected link
        graph.  Results are cached in `self.distance_cache`.
        """
        if start == goal:
            return 0
        if goal in self.distance_cache[start]:
            return self.distance_cache[start][goal]

        # Breadth-first search
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.graph[node]:
                if nbr == goal:
                    self.distance_cache[start][goal] = dist + 1
                    self.distance_cache[goal][start] = dist + 1
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))

        # Not connected – should not happen in a solvable task
        large = 1_000_000
        self.distance_cache[start][goal] = large
        self.distance_cache[goal][start] = large
        return large

    def _locate_man(self, state):
        """
        Return (man_name, man_location).

        Tries the most reliable cues first:
        1.  the first argument of a `carrying` fact
        2.  an object that appears in exactly one `(at … loc)`,
            never in `(loose …)` / `(tightened …)` / `(usable …)`.
        """
        # 1) by `carrying`
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying":
                man = parts[1]
                # look-up his position
                for f2 in state:
                    p2 = get_parts(f2)
                    if p2[0] == "at" and p2[1] == man:
                        return man, p2[2]

        # 2) fallback – heuristic guess
        at_count = defaultdict(int)
        for fact in state:
            p = get_parts(fact)
            if p[0] == "at":
                at_count[p[1]] += 1

        for fact in state:
            p = get_parts(fact)
            if p[0] == "at":
                obj, loc = p[1], p[2]
                if at_count[obj] == 1:  # appears only once in 'at'
                    if (f"(loose {obj})" not in state and
                            f"(tightened {obj})" not in state and
                            f"(usable {obj})" not in state):
                        return obj, loc

        # As a last resort return the first 'at' fact found
        p = get_parts(next(f for f in state if get_parts(f)[0] == "at"))
        return p[1], p[2]
