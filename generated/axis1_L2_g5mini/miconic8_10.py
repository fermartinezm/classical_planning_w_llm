from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the predicate and arguments of a PDDL fact string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l) domain.
    The domain's goal atoms are of the form (guby8wxu <obj>). To achieve
    (guby8wxu o) two steps are required in general:
      1) Create an intermediate unary fact (eacj8ftl o) by applying
         makpz2tc when the moving token v0yy74n7 is located at a source
         location L where (pxch7pi8 o L) holds.
      2) Convert (eacj8ftl o) into (guby8wxu o) by applying jg64qouf when
         the token is located at a target location T where (pahbmn6t o T)
         holds.
    Movement of the single token v0yy74n7 along locations is possible
    using the two move actions (which together allow traversal along any
    declared pvhp8m0p edge in either direction). Each token move costs 1.
    Both makpz2tc and jg64qouf cost 1 each.

    Assumptions
    -----------
    - The planner's state representation uses PDDL fact strings like
      "(pred arg1 arg2)". get_parts() extracts parts accordingly.
    - Static facts (task.static) contain the pvhp8m0p edges and the
      pahbmn6t mapping (object -> goal locations).
    - pxch7pi8 facts (object -> source locations for creating eacj8ftl)
      are dynamic and read from the current state.
    - There is exactly one token fact (v0yy74n7 <loc>); if none is found,
      the object goals that require movement are considered unreachable.
    - The heuristic treats goal subproblems independently and sums the
      individual estimated costs. This is not necessarily admissible but
      produces informative guidance for greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal objects for which (guby8wxu <obj>) must hold.
    - From static facts:
       * Build an undirected adjacency graph of locations from pvhp8m0p.
         (Because domain provides two actions that together allow movement
          between nodes if an edge is present in either direction.)
       * Build a mapping from each object to its set of target locations
         (locations T with (pahbmn6t obj T)).
    - Do not assume anything about pxch7pi8 facts; they are read from states.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planning node/state:
    1. If the overall goal is already reached (all goal atoms are present
       in the state), return 0 (the heuristic is zero iff the goal is reached).
    2. Locate the token's current location s by finding the v0yy74n7 fact.
       If the token is missing, the heuristic will return a large cost for
       goals that require movement (treated as unreachable).
    3. Parse dynamic pxch7pi8 facts in the current state to know where each
       object can generate eacj8ftl.
    4. For each object o that must satisfy (guby8wxu o):
       - If (guby8wxu o) is already true in the state: cost 0.
       - Else if (eacj8ftl o) is true in the state:
           * Find the nearest target location T in the static pahbmn6t set
             that is reachable from s in the movement graph. Cost is
             distance(s,T) [token moves] + 1 (jg64qouf).
       - Else (no eacj8ftl yet):
           * Consider all source locations L where (pxch7pi8 o L) holds in
             the current state and all target locations T (from static).
           * For each pair (L,T) compute:
               distance(s,L) + 1 (makpz2tc) + distance(L,T) + 1 (jg64qouf)
             and take the minimum over all pairs. If L == T the formula
             naturally reduces to distance(s,T) + 2.
       - If no valid path exists for an object (no source and no eacj8ftl,
         or targets unreachable), that object is considered unreachable
         and contributes a large penalty to the heuristic.
    5. Sum the estimated costs for all goal objects and return that sum
       as the heuristic value. The heuristic deliberately overestimates
       interactions between objects (token movement is counted per-object),
       but it's efficient to compute and discriminative for greedy search.
    """

    def __init__(self, task):
        """Extract static information and the goals from the task."""
        self.goals = set(task.goals)  # set of goal fact strings
        static_facts = task.static

        # Movement graph: undirected adjacency between location names.
        # pvhp8m0p facts have form (pvhp8m0p loc1 loc2)
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)
                # edges are effectively traversable in both directions (two move actions)
                self.adjacency.setdefault(b, set()).add(a)

        # For each object (jxhpkqr5) collect its target locations from pahbmn6t
        # facts: (pahbmn6t obj loc)
        self.obj_to_targets = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.obj_to_targets.setdefault(obj, set()).add(loc)

        # Prepare set of goal objects (objects that must have guby8wxu)
        self.goal_objects = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "guby8wxu" and len(parts) >= 2:
                self.goal_objects.append(parts[1])

        # Cache BFS distance maps: start_location -> {loc: dist}
        self._dist_cache = {}

        # Large penalty for unreachable components
        self._INF_COST = 100000

    def _bfs_distances_from(self, start):
        """Return dict mapping location -> distance from start (BFS). Cache results."""
        if start in self._dist_cache:
            return self._dist_cache[start]
        distances = {}
        if start not in self.adjacency:
            # still cache empty map (no neighbors), but include start->0
            distances[start] = 0
            self._dist_cache[start] = distances
            return distances
        dq = deque([start])
        distances[start] = 0
        while dq:
            u = dq.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in distances:
                    distances[v] = distances[u] + 1
                    dq.append(v)
        self._dist_cache[start] = distances
        return distances

    def __call__(self, node):
        """
        Compute heuristic estimate of number of actions to reach the goal from node.

        Returns an integer >= 0. Returns 0 iff the state's goals are all satisfied.
        """
        state = node.state

        # If goal already reached, heuristic must be 0
        # (ensures condition 2 of the checklist).
        # The Task object normally would be available to check goal_reached,
        # but we only have the node/state here. So check whether every goal fact is in state.
        if self.goals <= state:
            return 0

        # Find token location s from state: (v0yy74n7 <loc>)
        token_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7" and len(parts) >= 2:
                token_loc = parts[1]
                break

        # Collect current pxch7pi8 facts from state: (pxch7pi8 obj loc)
        obj_to_sources = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_to_sources.setdefault(obj, set()).add(loc)

        # Collect which objects currently have eacj8ftl in the state
        eac_objects = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "eacj8ftl" and len(parts) >= 2:
                eac_objects.add(parts[1])

        total_cost = 0

        # If token is missing, treat movement as impossible -> large penalty for any object needing movement
        for obj in self.goal_objects:
            goal_fact = f"(guby8wxu {obj})"
            if goal_fact in state:
                # already satisfied
                continue

            # If object already has eacj8ftl in state, we only need jg64qouf
            if obj in eac_objects:
                # need to bring token to any target location T in obj_to_targets[obj]
                targets = self.obj_to_targets.get(obj, set())
                if not targets:
                    # No declared target in static info: we cannot apply jg64qouf -> unreachable
                    total_cost += self._INF_COST
                    continue
                if token_loc is None:
                    total_cost += self._INF_COST
                    continue
                dist_map = self._bfs_distances_from(token_loc)
                best = None
                for t in targets:
                    if t in dist_map:
                        cost = dist_map[t] + 1  # moves + jg64qouf
                        if best is None or cost < best:
                            best = cost
                if best is None:
                    total_cost += self._INF_COST
                else:
                    total_cost += best
                continue

            # Otherwise, we need to create eacj8ftl first using makpz2tc at some source L
            sources = obj_to_sources.get(obj, set())
            targets = self.obj_to_targets.get(obj, set())

            if not sources:
                # No source in current state to create eacj8ftl -> cannot achieve unless eac was present (it isn't).
                total_cost += self._INF_COST
                continue
            if not targets:
                total_cost += self._INF_COST
                continue
            if token_loc is None:
                total_cost += self._INF_COST
                continue

            # For each pair (L, T) compute cost: distance(token_loc, L) + 1 (makpz2tc)
            # + distance(L, T) + 1 (jg64qouf). Take minimum.
            best_cost = None
            dist_from_token = self._bfs_distances_from(token_loc)
            # pre-cache BFS distances from each source L
            for L in sources:
                if L not in dist_from_token:
                    # If token cannot reach L at all, skip this L
                    # (but maybe we could reach T directly and L not required? No:
                    # must create eac via makpz2tc at some source L)
                    continue
                d1 = dist_from_token[L]
                dist_from_L = self._bfs_distances_from(L)
                for T in targets:
                    if T not in dist_from_L:
                        continue
                    d2 = dist_from_L[T]
                    cost = d1 + 1 + d2 + 1  # moves + makpz2tc + moves + jg64qouf
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
            if best_cost is None:
                total_cost += self._INF_COST
            else:
                total_cost += best_cost

        # Ensure the heuristic is strictly positive if goal not reached.
        if total_cost == 0:
            return 1
        return total_cost