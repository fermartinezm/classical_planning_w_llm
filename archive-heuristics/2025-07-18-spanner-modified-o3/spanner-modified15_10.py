from collections import deque
from fnmatch import fnmatch
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """Return the individual components of a PDDL fact."""
    return fact[1:-1].split()


def match(fact: str, *pattern):
    """Return True iff a fact matches the (possibly wildcard) pattern."""
    return all(fnmatch(p, q) for p, q in zip(get_parts(fact), pattern))


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *spanner-modified* domain.  It
    estimates the number of actions still required to tighten all remaining
    loose nuts.  The estimate is *not* admissible – it is intentionally biased
    towards being informative rather than safe – and is tailored for greedy
    best-first search.

    Assumptions
    -----------
    • Exactly one *man* exists in every task (true for all benchmark problems).  
    • Every spanner that is declared `usable` remains usable forever.  
    • The graph induced by the `link` predicate is treated as *undirected* for
      distance estimation (walking backwards over a link in the real world
      might require a different operator, but relaxing this increases guidance
      quality and never overestimates the true cost).  
    • Each action has unit cost.  
    • A nut can be tightened as soon as the man, a usable spanner he is
      carrying, and the nut share the same location.

    Heuristic Initialization
    ------------------------
    During construction the static `link` facts are converted into a bidirected
    adjacency list so that shortest path queries between locations can be
    answered with simple breadth-first search.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node we perform the following steps:

    1.  **Classify objects.**
        Objects that appear in facts `(usable X)` are spanners, objects in
        `(loose X)` or `(tightened X)` are nuts, and every remaining object
        occurring in an `at` fact is considered a man.  (There is exactly one
        such man.)

    2.  **Extract current state information.**
        • Location of the man.  
        • Whether the man is already *carrying* a usable spanner.  
        • Locations of all currently loose nuts.  
        • Locations of all usable spanners that are *not* being carried.

    3.  **Early termination.**
        If no loose nuts remain the goal is reached → heuristic = 0.

    4.  **Cost of obtaining a spanner (if necessary).**
        If the man is not carrying a spanner we consider every still-available
        usable spanner `s`, compute  
           `dist(man, loc(s)) + 1(pick-up) + dist(loc(s), nearest loose nut)`  
        and retain the minimum such value.  This optimistic estimate says:
        “walk to the closest spanner, pick it up, and walk straight to the
        closest nut”.

    5.  **Cost of travelling between nuts.**
        After the first nut is reached the man must continue to visit all
        remaining loose nuts.  We approximate this with  

            MST_cost(loose-nut locations)

        i.e. the total weight of a minimum spanning tree over the locations,
        using shortest path lengths as edge weights.  This is a common,
        informative relaxation of the travelling salesman problem.

    6.  **Cost of the tightening actions themselves.**
        Tightening a nut is a single action, hence we add
        `len(loose_nuts)`.

    7.  **Put everything together.**
        If the man already has a spanner we skip step 4.  Otherwise we add the
        cheapest “obtain spanner” figure.  The resulting sum is returned as the
        heuristic value.

    The computation is polynomial and extremely fast in practice because
    instances are small (≤ 20 locations, ≤ 10 nuts).
    """

    def __init__(self, task):
        # Build an undirected adjacency list from the static link facts.
        self.adj = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)  # treat as undirected

        # Store goal facts to recognise when the search state is already solved.
        self.goals = task.goals

    # --------------------------------------------------------------------- #
    # Helper functions                                                      #
    # --------------------------------------------------------------------- #
    def _shortest_distance(self, start, goal):
        """Breadth-first distance between two locations; INF if unreachable."""
        if start == goal:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for nb in self.adj.get(loc, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        return inf  # disconnected – shouldn't happen in solvable tasks

    def _dist_to_set(self, start, targets):
        """Shortest distance from *start* to any location in *targets*."""
        if start in targets:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            loc, d = queue.popleft()
            for nb in self.adj.get(loc, ()):
                if nb in targets:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        return inf

    # --------------------------------------------------------------------- #
    # Main heuristic call                                                   #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # --------------------------------------------------------------- #
        # 1.  Collect facts from the current state                        #
        # --------------------------------------------------------------- #
        spanners_usable = {get_parts(f)[1] for f in state if match(f, "usable", "*")}
        loose_nuts = {get_parts(f)[1] for f in state if match(f, "loose", "*")}
        tightened_nuts = {get_parts(f)[1] for f in state if match(f, "tightened", "*")}

        # Goal test (required for heuristic correctness).
        if not loose_nuts:
            return 0

        # Identify man                                              #
        man_names_from_carry = {get_parts(f)[1] for f in state if match(f, "carrying", "*", "*")}
        if man_names_from_carry:
            man = next(iter(man_names_from_carry))
        else:
            # Deduce from 'at' facts: those objects that are not nuts or spanners
            mans = set()
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj, _ = get_parts(fact)[1:]
                    if obj in spanners_usable or obj in loose_nuts or obj in tightened_nuts:
                        continue
                    mans.add(obj)
            man = next(iter(mans))  # exactly one according to assumptions

        # Current location of man.
        man_loc = next(
            get_parts(f)[2] for f in state if match(f, "at", man, "*")
        )

        # Is the man already carrying a usable spanner?
        carrying = {
            get_parts(f)[2] for f in state if match(f, "carrying", man, "*")
        }
        has_spanner = any(s in spanners_usable for s in carrying)

        # --------------------------------------------------------------- #
        # 2.  Distance to obtain a spanner (if not already carrying one)  #
        # --------------------------------------------------------------- #
        obtain_spanner_cost = 0
        if not has_spanner:
            best = inf
            # Locations of every still available spanner.
            for sp in spanners_usable:
                # Skip if already carried by someone else (very rare, but safe).
                if sp in carrying:
                    best = 0
                    break
                # Find spanner's location in the state.
                locs = [
                    get_parts(f)[2]
                    for f in state
                    if match(f, "at", sp, "*")
                ]
                if not locs:
                    continue  # spanner not lying around
                sp_loc = locs[0]

                d1 = self._shortest_distance(man_loc, sp_loc)
                d2 = self._dist_to_set(sp_loc, {get_parts(f)[2] for f in state if match(f, "at", "*", "*") and get_parts(f)[1] in loose_nuts})
                cand = d1 + 1 + d2  # walk to sp, pick it up, walk to closest nut
                best = min(best, cand)
            obtain_spanner_cost = best

        # --------------------------------------------------------------- #
        # 3.  Cost of travelling between nuts (MST approximation)         #
        # --------------------------------------------------------------- #
        nut_locs = [
            get_parts(f)[2] for f in state if match(f, "at", "*", "*") and get_parts(f)[1] in loose_nuts
        ]

        # Distance from man's *current* (or post-pickup) position to the first nut
        if has_spanner:
            first_leg = self._dist_to_set(man_loc, set(nut_locs))
        else:
            # We already counted distance to nearest nut in obtain_spanner_cost,
            # so the tour starts there.
            first_leg = 0

        # Build complete graph between nuts and run Prim’s algorithm.
        if len(nut_locs) == 1:
            mst_cost = 0
        else:
            remaining = set(range(len(nut_locs)))
            visited = {remaining.pop()}  # arbitrary start nut
            mst_cost = 0
            while remaining:
                best_edge = inf
                best_j = None
                for i in visited:
                    for j in remaining:
                        d = self._shortest_distance(nut_locs[i], nut_locs[j])
                        if d < best_edge:
                            best_edge = d
                            best_j = j
                visited.add(best_j)
                remaining.remove(best_j)
                mst_cost += best_edge

        # --------------------------------------------------------------- #
        # 4.  Tighten actions themselves                                  #
        # --------------------------------------------------------------- #
        tighten_actions = len(loose_nuts)

        total = obtain_spanner_cost + first_leg + mst_cost + tighten_actions

        # Sanity: cost must always be finite for solvable instances
        if total == inf:
            total = 0  # fallback, should not happen in practice

        return total
