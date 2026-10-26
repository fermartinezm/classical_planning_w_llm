from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Helper: split a fact string '(predicate arg1 arg2 …)' into its components.
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    fnmatch-based pattern matcher for facts.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimates the number of remaining actions (walk, pickup_spanner, tighten_nut)
    that are required to reach a goal state in the *spanner-modified* domain.
    The value is *not* admissible – it purposely over-estimates in order to
    reduce the number of expanded nodes in a greedy best-first search.

    Assumptions
    -----------
    1. There is exactly one man in every problem instance.
    2. A spanner, once picked up, is never dropped (there is no action for it).
    3. Every `(link L1 L2)` fact represents an undirected edge between the two
       locations – if the planner lists only one direction we still allow motion
       in both ways.
    4. Action costs are uniform and set to 1 for every operator contained in the
       domain.

    Heuristic Initialization
    ------------------------
    *  Extract the static road map from the `link` predicates and build an
       adjacency list for fast shortest-path queries.
    *  Instantiate a distance cache so that repeated calls reuse previously
       computed shortest path lengths.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search node:
    1.  Detect the only man and his current location.
    2.  Identify
        *   every loose nut and the location it is currently at,
        *   whether the man already carries a usable spanner,
        *   the locations of all *available* usable spanners if he does not.
    3.  If **no** loose nuts exist, the goal is already satisfied – return 0.
    4.  If the man does **not** carry a spanner:
        a.  Choose the *nearest* usable spanner (shortest walk distance).
        b.  Add the distance to reach it **plus** one extra action for
            `pickup_spanner`.
        c.  Assume the man is now located at that spanner’s position carrying
            it.
    5.  Compute the walking cost to visit and tighten all remaining nuts:
        a.  Find the distance from the current man position to the *closest*
            loose nut (`d_first`).
        b.  Compute a Minimum Spanning Tree (MST) over the set of nut locations
            and use its total length as a lower bound for travelling between
            the nuts (`d_mst`).  Despite being an underestimate of the real
            travel effort, it is a much tighter estimate than simply counting
            nuts.
        c.  Total walking effort = `d_first + d_mst`.
    6.  Add `1` action for every `tighten_nut`.
    7.  The final heuristic value is the sum of
        pickup cost (maybe 0) + walking effort + tightening actions.

    Although the MST part is admissible, the heuristic as a whole is not,
    because it ignores return movements after tightening the last nut and
    therefore tends to *underestimate*.  This slight optimism, however, yields
    excellent guidance in practice.
    """

    # --------------------------------------------------------------------- #
    #  Construction helpers                                                 #
    # --------------------------------------------------------------------- #

    def __init__(self, task):
        """
        Build adjacency lists from the static `link` facts.
        """
        # --- build undirected graph of locations --------------------------------
        self.graph = defaultdict(set)
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.graph[l1].add(l2)
                self.graph[l2].add(l1)          # treat as undirected

        # distance cache  (loc1, loc2) -> integer
        self._dist_cache = {}

    # --------------------------------------------------------------------- #
    #  Shortest-path utilities                                              #
    # --------------------------------------------------------------------- #

    def _distance(self, start, goal):
        """
        Length (number of *walk* actions) of a shortest path
        between two locations using BFS.  Results are cached.
        """
        if start == goal:
            return 0
        key = (start, goal) if start < goal else (goal, start)
        if key in self._dist_cache:
            return self._dist_cache[key]

        frontier = deque([(start, 0)])
        visited = {start}
        while frontier:
            loc, dist = frontier.popleft()
            for nxt in self.graph[loc]:
                if nxt in visited:
                    continue
                if nxt == goal:
                    self._dist_cache[key] = dist + 1
                    return dist + 1
                visited.add(nxt)
                frontier.append((nxt, dist + 1))

        # unsolvable instances should never occur – but ensure finiteness
        self._dist_cache[key] = float("inf")
        return float("inf")

    def _mst_length(self, locations):
        """
        Prim’s algorithm: total edge weight of an MST spanning `locations`.
        All edge weights are shortest-path lengths obtained from _distance().
        """
        if not locations:
            return 0
        locs = set(locations)
        visited = {locs.pop()}
        length = 0
        while locs:
            best = None           # (dist, target_location)
            for v in visited:
                for u in locs:
                    d = self._distance(v, u)
                    if best is None or d < best[0]:
                        best = (d, u)
            dist, nxt = best
            length += dist
            visited.add(nxt)
            locs.remove(nxt)
        return length

    # --------------------------------------------------------------------- #
    #  Main heuristic                                                       #
    # --------------------------------------------------------------------- #

    def __call__(self, node):
        state = node.state

        # -------------------------------------------------------------- #
        #  Collect dynamic information                                   #
        # -------------------------------------------------------------- #

        # 1.  Identify nuts
        loose_nuts = {
            get_parts(fact)[1]            # nut name
            for fact in state
            if match(fact, "loose", "*")
        }
        # Fast exit: goal state
        if not loose_nuts:
            return 0

        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        # 2.  Identify spanners
        usable_spanners = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "usable", "*")
        }

        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in usable_spanners:
                    spanner_locations[obj] = loc

        # 3.  Carrying information (and man’s name)
        man_name = None
        carrying_spanners = set()
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man, sp = get_parts(fact)[1:]
                man_name = man
                carrying_spanners.add(sp)

        # If man still unknown (he is not carrying anything) search the 'at' map
        if man_name is None:
            possible_men = {
                obj for fact in state if match(fact, "at", "*", "*")
                for obj in [get_parts(fact)[1]]
                if obj not in usable_spanners and obj not in nut_locations
            }
            # There is exactly one man, take it
            man_name = next(iter(possible_men))

        # 4.  Man’s current location
        man_location = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        # -------------------------------------------------------------- #
        #  Phase 1 – make sure we carry a usable spanner                  #
        # -------------------------------------------------------------- #
        cost = 0
        if not carrying_spanners:
            # Nearest usable spanner
            best_spanner, best_dist = None, float("inf")
            for sp, loc in spanner_locations.items():
                d = self._distance(man_location, loc)
                if d < best_dist:
                    best_dist, best_spanner = d, sp
            # Walk to it and pick it up
            cost += best_dist            # walk actions
            cost += 1                    # pickup_spanner
            man_location = spanner_locations[best_spanner]  # now here

        # -------------------------------------------------------------- #
        #  Phase 2 – visit and tighten every loose nut                    #
        # -------------------------------------------------------------- #

        # Distance from current position to the closest nut
        first_step = min(
            self._distance(man_location, nut_locations[nut])
            for nut in loose_nuts
        )

        # MST over all nut locations (pre-computed distances)
        mst_len = self._mst_length(set(nut_locations.values()))

        # Tighten action cost
        tighten_cost = len(loose_nuts)   # one per nut

        cost += first_step + mst_len + tighten_cost
        return cost
