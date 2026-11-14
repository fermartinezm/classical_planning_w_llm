from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    This heuristic estimates the number of actions required to achieve all
    goals of the form (ne6sl557 ?ris) (one per object of type wqid8mog).
    In this domain a goal for a particular wqid8mog `ris` is achieved by
    a single f7y6nzq3 action. Applying f7y6nzq3 for `ris` requires:
      - of6wn8m4 ris  (must be true before f7y6nzq3)
      - qkdx0xe2 ris ts   (ris located at some location ts)
      - a du770mtw object cz such that ndg963qz cz is true
      - a pt4b08k4 object n such that fqyrzl30 n cz is true
      - qkdx0xe2 n ts   (the pt4 object n must be at the same location as ris)

    If fqyrzl30 n cz does not yet exist, it can be created by a single
    jql8txb3 action, but jql8txb3 requires that at the time of application
    both the du770mtw `cz` and the pt4 `n` are at the same location `ts`.
    A pt4 can be moved between locations using nm6lzp2x, which follows the
    directed edges given by the static facts lpmls56r (these are interpreted
    as directed adjacency edges between locations).

    Assumptions
    -----------
    - of6wn8m4 facts (for wqid8mog objects) are static in the sense that
      there are no operators that create of6wn8m4; if of6wn8m4 for a goal
      object is missing in the current state, that goal is treated as
      unreachable and the heuristic returns a large finite value.
    - du770mtw objects (cz candidates) that can be used must currently
      have a qkdx0xe2 (location) fact; cz objects cannot be moved by any
      operator in the domain, so their locations are fixed in the state.
    - pt4b08k4 candidates are inferred from the qkdx0xe2 facts: any
      object that appears as the first argument of qkdx0xe2 but is neither
      a du770mtw (ndg963qz) nor a wqid8mog (of6wn8m4 / goal objects) is
      considered a pt4 candidate. This identification is conservative
      (might include extra objects) but keeps the heuristic efficient.
    - The heuristic is allowed to be inadmissible; to remain informative
      it models the consumption of ndg963qz resources (each cz can be
      used at most once). The heuristic does a greedy assignment of goals
      to distinct cz objects to reflect that ndg963qz is deleted by the
      goal-achieving action.

    Heuristic Initialization
    ------------------------
    - Extract the static directed graph of locations from the lpmls56r
      static facts (edges of the form "(lpmls56r A B)").
    - Store the task goals for quick checking.
    - No other complex preprocessing is needed; location distance
      computations are done on-demand with BFS and cached.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (with its state), compute the heuristic value as
    follows:

    1. If all goals are already satisfied in the state's fact set, return 0.

    2. Parse the current state to extract:
       - location_of[obj] from all (qkdx0xe2 obj loc) facts.
       - existing_links set from (fqyrzl30 n cz) facts.
       - available_cz set from (ndg963qz cz) facts (these are the du770mtw
         objects that can be used; note that f7y6nzq3 deletes ndg963qz,
         so each cz is a one-time resource).
       - available_of set from (of6wn8m4 ris) facts.
       - which goals are already achieved (ne6sl557 facts).

    3. Build the directed adjacency graph of locations from the static
       lpmls56r facts so we can compute shortest move distances for pt4
       objects using nm6lzp2x. Distances are computed with BFS and cached.

    4. For each goal ris that is not yet achieved:
       - If of6wn8m4 ris is missing in the state, treat the goal as
         unreachable and return a large finite heuristic (large_cost).
       - Find ris_loc = location_of[ris]; if missing, treat as unreachable.
       - For each cz in available_cz that also has a current location
         (qkdx0xe2 cz cz_loc must exist), compute the minimal cost to
         achieve ris using any pt4 candidate `n`. Pt4 candidates are
         objects appearing in the state's qkdx0xe2 facts that are
         neither in available_cz nor in available_of (heuristic
         inference of pt4 type).
         * If there already exists (fqyrzl30 n cz) and n is at ris_loc,
           then cost is 1 (just apply f7y6nzq3).
         * If (fqyrzl30 n cz) exists but n is elsewhere, need moves:
           cost = dist(n_loc, ris_loc) + 1 (f7y6nzq3).
         * If no fqyrzl30 exists for (n,cz), we must create one:
           - Move n to cz_loc (cost dist(n_loc, cz_loc) via nm6lzp2x moves),
           - Apply jql8txb3 (cost +1),
           - Move n from cz_loc to ris_loc (cost dist(cz_loc, ris_loc)),
           - Apply f7y6nzq3 (cost +1).
           Total cost = dist(n,cz) + 1 + dist(cz,ris) + 1.
       - For the given cz take the minimal cost over pt4 candidates.
       - For the given ris take the minimal cost across cz candidates.

    5. Because ndg963qz on a cz is consumed when achieving one goal, we
       prefer distinct cz objects for different goals. We therefore
       perform a greedy assignment: gather (ris, cz, cost) triplets and
       repeatedly pick the smallest-cost pair, assign that cz to the
       corresponding ris, remove both from consideration, and continue.
       If some ris has no feasible cz (no finite cost), return large_cost.

    6. Sum the assigned costs over all remaining (not yet achieved) goals
       and return this sum as the heuristic value.

    Notes on completeness and finiteness
    -----------------------------------
    - The heuristic returns 0 for goal states.
    - If all necessary static prerequisites are present and there are
      finite paths in the lpmls56r graph, the heuristic will return a
      finite integer estimate. If a goal is provably unreachable due to
      missing static preconditions (e.g., of6wn8m4 missing or no usable
      cz), the heuristic returns a large finite number (large_cost).
    """

    def __init__(self, task):
        """Extract static adjacency information and store goals."""
        self.goals = task.goals  # set of goal fact strings
        static_facts = task.static  # frozenset of static facts

        # Build directed adjacency graph for locations from lpmls56r static facts.
        # lpmls56r A B means there is a directed edge A -> B (nm6lzp2x can
        # move a pt4 from A to B).
        self.adj = defaultdict(list)  # loc -> list of neighbor locs
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj[a].append(b)

        # Cache for BFS distances between locations: (src, dst) -> distance or None.
        self._dist_cache = {}

        # A big finite number to indicate (practically) unreachable goals.
        self.large_cost = 100000

    def _bfs_distance(self, src, dst):
        """Return shortest directed distance from src to dst in adjacency graph or None."""
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]
        # Standard BFS
        q = deque([(src, 0)])
        seen = {src}
        dist = None
        while q:
            cur, d = q.popleft()
            for nb in self.adj.get(cur, ()):
                if nb == dst:
                    dist = d + 1
                    q.clear()
                    break
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        self._dist_cache[key] = dist
        return dist

    def __call__(self, node):
        """Compute heuristic estimate (sum of estimated actions to satisfy remaining goals)."""
        state = node.state  # frozenset of fact strings

        # Quick goal check: return 0 if already solved.
        if self.goals <= state:
            return 0

        # Parse current state for relevant facts.
        location_of = {}  # object -> location from qkdx0xe2 facts
        existing_links = set()  # set of (n, cz) pairs for fqyrzl30 facts
        available_cz = set()  # objects cz with ndg963qz cz
        available_of = set()  # objects ris with of6wn8m4 ris
        achieved_ne = set()  # objects that already have ne6sl557

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                location_of[obj] = loc
            elif pred == "fqyrzl30" and len(parts) >= 3:
                n, cz = parts[1], parts[2]
                existing_links.add((n, cz))
            elif pred == "ndg963qz" and len(parts) >= 2:
                available_cz.add(parts[1])
            elif pred == "of6wn8m4" and len(parts) >= 2:
                available_of.add(parts[1])
            elif pred == "ne6sl557" and len(parts) >= 2:
                achieved_ne.add(parts[1])

        # Identify goals (the objects involved in ne6sl557) that are still unmet.
        goal_objs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ne6sl557" and len(parts) >= 2:
                obj = parts[1]
                if obj not in achieved_ne:
                    goal_objs.append(obj)

        # If no remaining goals, we're done.
        if not goal_objs:
            return 0

        # Identify pt4 candidates: objects that appear in location_of but are not cz or of objects.
        pt4_candidates = [obj for obj in location_of.keys() if obj not in available_cz and obj not in available_of]

        # Precompute cz locations (cz must have a qkdx0xe2 location to be usable for jql8txb3).
        usable_cz_with_loc = []
        for cz in available_cz:
            cz_loc = location_of.get(cz)
            if cz_loc is not None:
                usable_cz_with_loc.append((cz, cz_loc))

        # If there are no usable cz objects but goals remain, unreachable.
        if not usable_cz_with_loc:
            return self.large_cost

        # For each goal, compute minimal cost per cz (and minimal overall).
        # costs_per_goal: goal -> dict(cz -> min_cost)
        costs_per_goal = {}
        for ris in goal_objs:
            # Check of6wn8m4 prerequisite.
            if ris not in available_of:
                return self.large_cost  # missing required of6wn8m4 -> treat as unreachable

            ris_loc = location_of.get(ris)
            if ris_loc is None:
                return self.large_cost  # ris has no location -> unreachable

            cz_costs = {}
            for cz, cz_loc in usable_cz_with_loc:
                best_cost_for_cz = None
                # Try existing links first: any (n,cz) in existing_links
                for n, _cz in list(existing_links):
                    if _cz != cz:
                        continue
                    n_loc = location_of.get(n)
                    if n_loc is None:
                        continue
                    # If n already at ris_loc, only f7y6nzq3 is needed.
                    if n_loc == ris_loc:
                        cost = 1  # f7y6nzq3
                    else:
                        d = self._bfs_distance(n_loc, ris_loc)
                        if d is None:
                            continue
                        cost = d + 1  # moves + f7y6nzq3
                    if best_cost_for_cz is None or cost < best_cost_for_cz:
                        best_cost_for_cz = cost

                # If no existing link yields a plan, consider creating a link via jql8txb3.
                for n in pt4_candidates:
                    n_loc = location_of.get(n)
                    if n_loc is None:
                        continue
                    # Need to move n to cz_loc (distance n_loc->cz_loc).
                    d1 = self._bfs_distance(n_loc, cz_loc)
                    if d1 is None:
                        continue
                    # After jql8txb3, n will be at cz_loc; need to move n from cz_loc to ris_loc.
                    d2 = self._bfs_distance(cz_loc, ris_loc)
                    if d2 is None:
                        continue
                    cost = d1 + 1 + d2 + 1  # moves to cz + jql8txb3 + moves to ris + f7y6nzq3
                    if best_cost_for_cz is None or cost < best_cost_for_cz:
                        best_cost_for_cz = cost

                if best_cost_for_cz is not None:
                    cz_costs[cz] = best_cost_for_cz

            # If no cz can achieve this goal, the goal is unreachable.
            if not cz_costs:
                return self.large_cost
            costs_per_goal[ris] = cz_costs

        # Now assign goals to cz objects (each cz can be used at most once).
        # Greedy assignment: repeatedly pick (ris, cz) with smallest cost.
        total_cost = 0
        remaining_goals = set(costs_per_goal.keys())
        remaining_cz = set(cz for cz, _ in usable_cz_with_loc)

        # Build a list of tuples (cost, ris, cz) updated each iteration.
        while remaining_goals:
            best_triplet = None  # (cost, ris, cz)
            for ris in remaining_goals:
                for cz, cost in costs_per_goal[ris].items():
                    if cz not in remaining_cz:
                        continue
                    if best_triplet is None or cost < best_triplet[0]:
                        best_triplet = (cost, ris, cz)
            if best_triplet is None:
                # No feasible assignment remaining
                return self.large_cost
            cost, ris_selected, cz_selected = best_triplet
            total_cost += cost
            # Remove assigned goal and cz from further consideration.
            remaining_goals.remove(ris_selected)
            remaining_cz.remove(cz_selected)

            # (Optional improvement) If we wanted to be stricter we could update
            # existing_links and locations to reflect the assignment; for
            # efficiency and simplicity we do not simulate the actual effects.

            # Early cutoff if total_cost becomes too big.
            if total_cost >= self.large_cost:
                return self.large_cost

        return int(total_cost)