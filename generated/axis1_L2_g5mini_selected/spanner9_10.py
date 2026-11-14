from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract fact parts by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw). The domain's goals
    are to make predicates of the form (ne6sl557 <wqid8mog>) true. Each such goal
    is achieved only by the action 'f7y6nzq3' which requires:
      - an object <cz> of type du770mtw with (ndg963qz <cz>) and (qkdx0xe2 <cz> <mz>),
      - a pt4 object <n54xx> with (fqyrzl30 <n54xx> <cz>) and (qkdx0xe2 <n54xx> <mz>),
      - the wqid8mog object <ris> must satisfy (of6wn8m4 <ris>) and (qkdx0xe2 <ris> <mz>).
    The domain offers two auxiliary actions relevant to the heuristic:
      - jql8txb3: creates (fqyrzl30 <pt4> <cz>) if both <pt4> and <cz> share the same
        qkdx0xe2 relation to a particular mz. It removes the (qkdx0xe2 <cz> <mz>).
      - nm6lzp2x: moves a pt4's qkdx0xe2 relation along lpmls56r graph edges (one hop).
    Thus, to make ne6sl557(<ris>) true, the planner must (in general):
      1) have a du770mtw <cz> with ndg963qz and qkdx0xe2 for the same mz as <ris>,
      2) obtain or create (fqyrzl30 <pt4> <cz>) for some pt4 that also has qkdx0xe2 to that mz
         (moving a pt4 along lpmls56r edges may be needed),
      3) execute f7y6nzq3.
    The heuristic estimates the number of actions needed by computing, for each goal
    wqid8mog <ris>, the cheapest combination of:
      - moving a pt4 along the static lpmls56r graph (each edge costs 1, nm6lzp2x),
      - optionally creating an fqyrzl30 with jql8txb3 (cost 1),
      - performing f7y6nzq3 (cost 1).
    It also recognizes existing fqyrzl30 facts and existing qkdx0xe2 facts and uses them
    to reduce the estimated cost. Since f7y6nzq3 consumes ndg963qz on a chosen <cz>,
    the heuristic accounts for limited availability of suitable <cz> objects by doing
    a greedy minimal-cost assignment from goals to cz objects (each cz used at most once).

    Assumptions
    -----------
    - lpmls56r facts are static and define a directed graph of mz nodes; moving a pt4
      along one lpmls56r edge corresponds to one nm6lzp2x action.
    - The planner's task.initial_state contains representatives of pt4 objects (fqyrzl30
      facts in the initial state are used to identify pt4 objects). Current-state fqyrzl30
      facts are also considered.
    - Only pt4 objects that currently have a qkdx0xe2 relation can be moved (nm6lzp2x precondition).
    - Each du770mtw object <cz> that has both ndg963qz and qkdx0xe2 to the required mz can be
      used once (since f7y6nzq3 deletes ndg963qz and jql8txb3 deletes qkdx0xe2 for <cz>).
    - If (of6wn8m4 <ris>) is not true in the current state, the heuristic treats the goal as
      unreachable and returns a large finite value.

    Heuristic Initialization
    ------------------------
    - Extract the static lpmls56r edges from task.static and build a directed adjacency map.
    - Record initial pt4 objects by collecting the first argument of any fqyrzl30 fact from
      task.initial_state (these are the domain's pt4 objects as per the domain declaration).
    - Precompute no all-pairs closure; distances are computed on demand with BFS because the
      lpmls56r graph is typically small and BFS is efficient.

    Step-By-Step Thinking for Computing Heuristic (for a given node)
    ----------------------------------------------------------------
    1. If the state's facts already contain all goal literals (task.goals subset of state),
       return 0.

    2. Parse the state's facts to obtain:
       - qkdx_relations: mapping object -> set of mz objects for which (qkdx0xe2 obj mz) holds.
       - fqyrzl_relations: set of (pt4, cz) pairs for existing fqyrzl30 facts.
       - ndg_set: set of cz objects with ndg963qz.
       - of6_set: set of ris objects with of6wn8m4.
       - achieved_ne6_set: set of already true ne6sl557 facts.

    3. For each goal atom (ne6sl557 ris):
       - If already true in state, cost = 0 for that goal.
       - Else, verify (of6wn8m4 ris) is true in the state (otherwise unreachable -> large).
       - Determine ris_mz by reading qkdx_relations[ris]. If no qkdx0xe2 for ris, unreachable.
       - Collect candidate cz objects: those in ndg_set that also have qkdx0xe2 to ris_mz.
       - For each candidate cz:
           a) Check if any existing fqyrzl30 (pt4, cz) exists and whether the referenced pt4
              can be moved to ris_mz (distance finite). If so, cost_via_existing_fqyrzl = dist + 1
              (dist for nm6lzp2x actions + 1 for the f7y6nzq3 action).
           b) Otherwise, find the minimal distance dist_min to move any pt4 (from the set of known
              pt4s that currently have qkdx relations) to ris_mz. If such movement is possible,
              cost_via_create_fqyrzl = dist_min + 2 (nm6lzp2x moves + jql8txb3 + f7y6nzq3).
           c) Choose the smaller cost for that cz.
       - If no cz candidates yield finite cost, the goal is unreachable.
       - Otherwise record the minimal cost for this goal for every available cz.

    4. Goals must compete for cz resources (each cz can be used once). To approximate optimal
       use of cz resources, we greedily pick the lowest cost (goal, cz) pair, assign that cz to
       that goal, and remove both from consideration. Repeat until all goals are assigned or
       some goal has no viable cz (unreachable).

    5. Sum the chosen costs and return the total. For unreachable goals, return a large finite
       value (100000 scaled by number of goals) to indicate infeasibility while keeping the
       heuristic finite.

    Notes
    -----
    - The heuristic is intentionally inadmissible in general (it is a domain-dependent
      approximation) but robust: 0 is returned exactly for goal states and finite values
      for reachable configurations in solvable instances.
    """

    def __init__(self, task):
        # Store goal literals
        self.goals = task.goals
        # Static facts (only lpmls56r edges are relevant here)
        static_facts = task.static

        # Build lpmls56r adjacency: mz_node -> set(successor_mz_nodes)
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r":
                # lpmls56r src dst
                _, src, dst = parts
                self.adj.setdefault(src, set()).add(dst)

        # Identify pt4 objects from the initial state (fqyrzl30 first argument)
        self.known_pt4s = set()
        try:
            init = task.initial_state
        except Exception:
            init = set()
        for fact in init:
            parts = get_parts(fact)
            if parts and parts[0] == "fqyrzl30":
                _, pt4_obj, _ = parts
                self.known_pt4s.add(pt4_obj)

        # A large finite value for unreachable situations
        self.INF = 100000

    def _bfs_distance(self, start, goal):
        """Return shortest distance (number of lpmls56r edges) from start to goal, or INF."""
        if start == goal:
            return 0
        if start not in self.adj:
            return self.INF
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nb in self.adj.get(node, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return self.INF

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals already in state return 0
        if self.goals <= state:
            return 0

        # Parse state facts into useful structures
        qkdx = {}  # object -> set of mz nodes
        fqyrzl = set()  # set of (pt4, cz)
        ndg = set()  # set of cz with ndg963qz
        of6 = set()  # set of ris with of6wn8m4
        achieved_ne6 = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, mz = parts[1], parts[2]
                qkdx.setdefault(obj, set()).add(mz)
            elif pred == "fqyrzl30" and len(parts) >= 3:
                pt4, cz = parts[1], parts[2]
                fqyrzl.add((pt4, cz))
                # Also record pt4 as known
                self.known_pt4s.add(pt4)
            elif pred == "ndg963qz" and len(parts) >= 2:
                cz = parts[1]
                ndg.add(cz)
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of6.add(parts[1])
            elif pred == "ne6sl557" and len(parts) >= 2:
                achieved_ne6.add(parts[1])

        # Build list of goal wqid8mog objects to achieve (exclude already achieved)
        goals_to_do = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "ne6sl557":
                ris = parts[1]
                if ris in achieved_ne6:
                    continue
                goals_to_do.append(ris)

        # If none left, return 0
        if not goals_to_do:
            return 0

        # Prepare list of candidate pt4 objects that can be moved:
        # Known pt4s and must currently have at least one qkdx mapping (required for nm6lzp2x).
        movable_pt4s = []
        for pt in self.known_pt4s:
            if pt in qkdx:
                # use first seen mz for current location; qkdx can map to multiple mz, pick any
                # but treat each as possible start positions; we will take min across them
                movable_pt4s.append(pt)
        # As a fallback, allow any object that currently has qkdx and also appears as first
        # argument in any fqyrzl in the current state (helps if known_pt4s was empty earlier).
        for (pt, cz) in fqyrzl:
            if pt in qkdx and pt not in movable_pt4s:
                movable_pt4s.append(pt)

        # If there are no candidate pt4s but some goals might already be achievable via existing
        # fqyrzl facts (if fqyrzl exists and the referenced pt4 already has qkdx to the right mz),
        # those will be discovered below. Otherwise, moving new pt4s is impossible.
        # Precompute minimal distances for pt4s from their current mzs to any target mz on demand.

        # For each goal and each usable cz, compute cost; store list of (cost, goal, cz)
        candidate_pairs = []  # (cost, goal, cz)

        for ris in goals_to_do:
            # Precondition: of6wn8m4 ris must be true in state, else unreachable
            if ris not in of6:
                return len(goals_to_do) * self.INF

            # Need ris's mz: qkdx0xe2 ris mz
            if ris not in qkdx or not qkdx[ris]:
                # No qkdx for ris: cannot match ts6t4 -> unreachable
                return len(goals_to_do) * self.INF
            # In domain ris likely has exactly one mz, but consider all possibilities: pick any
            # We'll compute per-mz but in practice only one mz exists.
            best_cost_for_ris = None
            per_cz_costs = []
            # Consider all mzs ris is linked to (usually one)
            for ris_mz in qkdx[ris]:
                # Candidate czs are those with ndg and qkdx cz ris_mz
                cz_candidates = [cz for cz in ndg if cz in qkdx and ris_mz in qkdx[cz]]
                # If no cz candidate for this ris_mz, try next ris_mz; if none for all, unreachable.
                if not cz_candidates:
                    continue

                # For each cz compute minimal cost to produce preconditions and apply f7
                for cz in cz_candidates:
                    best_cost_for_cz = self.INF

                    # 1) If there's an existing fqyrzl (pt4, cz), check whether that pt4 can be moved
                    #    to ris_mz (might already be there).
                    for (pt, cz2) in fqyrzl:
                        if cz2 != cz:
                            continue
                        # pt must have a qkdx mapping to be movable or already at a mz
                        if pt not in qkdx:
                            # This pt4 cannot be used because nm6lzp2x requires qkdx precondition.
                            continue
                        # consider all current mzs of pt (choose minimal distance)
                        min_dist_for_pt = min(
                            (self._bfs_distance(cur_mz, ris_mz) for cur_mz in qkdx[pt]),
                            default=self.INF,
                        )
                        if min_dist_for_pt >= self.INF:
                            continue
                        # If pt already at ris_mz (min_dist==0) then cost = 1 (f7).
                        cost_via_existing_fq = min_dist_for_pt + 1
                        if cost_via_existing_fq < best_cost_for_cz:
                            best_cost_for_cz = cost_via_existing_fq

                    # 2) Alternatively, create fqyrzl30 by using jql8txb3 with some pt4 that has
                    #    or can be moved to ris_mz: pick pt4 with minimal dist
                    min_dist_any_pt = self.INF
                    for pt in movable_pt4s:
                        # pt must currently have qkdx mapping(s)
                        if pt not in qkdx:
                            continue
                        # consider all possible current mzs for this pt
                        dist = min((self._bfs_distance(cur_mz, ris_mz) for cur_mz in qkdx[pt]), default=self.INF)
                        if dist < min_dist_any_pt:
                            min_dist_any_pt = dist

                    # If some pt can be moved (or already at) ris_mz, we can jql and then f7:
                    # cost = min_dist_any_pt (moves) + 1 (jql) + 1 (f7)
                    if min_dist_any_pt < self.INF:
                        cost_via_create = min_dist_any_pt + 2
                        if cost_via_create < best_cost_for_cz:
                            best_cost_for_cz = cost_via_create

                    # If best_cost_for_cz remains INF it means this cz is not usable (no pt4 path).
                    if best_cost_for_cz < self.INF:
                        per_cz_costs.append((best_cost_for_cz, ris, cz))

            if not per_cz_costs:
                # No viable cz for any ris_mz -> unreachable
                return len(goals_to_do) * self.INF

            # For now append all per_cz options for greedy assignment later
            candidate_pairs.extend(per_cz_costs)

        # Greedy assignment of cz -> goals: pick lowest-cost (cost, goal, cz) pair,
        # assign cz once and mark goal satisfied, repeat.
        total_cost = 0
        assigned_goals = set()
        used_cz = set()
        # Build priority list sorted by cost
        candidate_pairs.sort(key=lambda x: x[0])

        # For faster lookup, build mapping goal -> list of (cost, cz) sorted
        from collections import defaultdict

        goal_to_options = defaultdict(list)
        for cost, goal, cz in candidate_pairs:
            goal_to_options[goal].append((cost, cz))

        # Greedy: iterate goals until assigned
        remaining_goals = set(goal_to_options.keys())
        while remaining_goals:
            # Find minimal available option among all remaining goals
            best_entry = None  # (cost, goal, cz)
            for goal in list(remaining_goals):
                options = goal_to_options.get(goal, [])
                # find first option whose cz is not used
                found = None
                for cost, cz in options:
                    if cz not in used_cz:
                        found = (cost, cz)
                        break
                if found is None:
                    # This particular goal has no unused cz, so skip for now (may become impossible)
                    continue
                cost, cz = found
                if best_entry is None or cost < best_entry[0]:
                    best_entry = (cost, goal, cz)
            if best_entry is None:
                # No remaining assignment possible (some goals can't be assigned a free cz)
                return len(goals_to_do) * self.INF
            cost, goal, cz = best_entry
            total_cost += cost
            used_cz.add(cz)
            remaining_goals.remove(goal)

        return int(total_cost)