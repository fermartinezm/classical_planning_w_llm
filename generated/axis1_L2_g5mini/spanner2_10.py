from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a fact string like '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (du770mtw) domain.

    This heuristic estimates the number of actions required to achieve all
    goal predicates of the form (ne6sl557 <w>) for w of type wqid8mog.
    In this domain a goal (ne6sl557 w) is produced by the action f7y6nzq3,
    which requires:
      - (of6wn8m4 w) to be true (an input precondition),
      - w to be associated via (qkdx0xe2 w <ts>) with some mz6r1d30 node ts,
      - some pt4b08k4 object n to be associated via (qkdx0xe2 n <ts>) (same ts),
      - some du770mtw object cz with (qkdx0xe2 cz <ts>) and (ndg963qz cz),
      - and additionally (fqyrzl30 n cz) must hold before f7y6nzq3 can be executed.
    The operator jql8txb3 can create (fqyrzl30 n cz) when both n and cz share
    the same ts (it requires (qkdx0xe2 cz ts) and (qkdx0xe2 n ts)).
    The operator nm6lzp2x allows a pt4b08k4 object n to move its qkdx0xe2
    association along directed edges given by the static predicate (lpmls56r a b).
    Thus, to enable f7y6nzq3 for a goal object w located at ts_w, we typically:
      1) ensure there is a du770mtw cz with ndg963qz and qkdx0xe2 cz ts_w,
      2) bring some pt4b08k4 n to ts_w (using nm6lzp2x along lpmls56r edges),
      3) use jql8txb3 to create fqyrzl30 n cz (1 action),
      4) use f7y6nzq3 to convert (of6wn8m4 w) to (ne6sl557 w) (1 action).
    The heuristic therefore approximates the required actions by summing,
    for each outstanding goal w, the minimal number of nm6lzp2x moves required
    to bring any pt4b08k4 to ts_w plus the fixed costs for jql8txb3 and f7y6nzq3.
    If an appropriate (fqyrzl30 n cz) already exists for w (and the cz still has
    ndg963qz), we count only the single f7y6nzq3 action.

    Assumptions
    -----------
    - Objects appearing in (ndg963qz X) are du770mtw objects ("cz").
    - Objects appearing in (of6wn8m4 X) or (ne6sl557 X) are wqid8mog objects ("w").
    - Any other objects that appear as the first argument of (qkdx0xe2 ...) and
      are not in the ndg set or the w set are treated as pt4b08k4 candidates ("n").
      This is a conservative, practical inference of object roles from facts.
    - The lpmls56r relation in static facts defines directed edges used by nm6lzp2x.
    - A du770mtw cz with ndg963qz can be used at most once because f7y6nzq3
      deletes ndg963qz on use. The heuristic accounts for this by consuming
      available cz resources greedily.
    - If a required resource (of6wn8m4 for w, or a cz at the same ts) is missing
      the heuristic returns a large finite value to indicate (practically)
      unsolvable/very costly state.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract the set of goal objects (all terms x from goals (ne6sl557 x)).
    - Build a directed adjacency map from static facts (lpmls56r a b) for BFS
      distance computations along mz6r1d30 nodes.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If all goals are already satisfied in the given state, return 0.
    2. Parse the state to obtain:
       - qkdx_map: mapping object -> set of mz6r1d30 nodes it is currently linked to
         via (qkdx0xe2 obj ts).
       - of_set: set of wqid8mog objects that currently have (of6wn8m4 ...).
       - ndg_set: set of du770mtw objects that currently have (ndg963qz ...).
       - fq_pairs: mapping pt -> set of cz for which (fqyrzl30 pt cz) holds.
    3. Group outstanding goals (those (ne6sl557 w) not yet in the state) by
       their current ts_w (extracted from qkdx_map). If a goal w has no ts_w
       or lacks (of6wn8m4 w), the heuristic returns a large value (practically
       unsolvable).
    4. For each ts (mz node) compute how many available cz (du objects) exist
       that (a) have ndg963qz and (b) have qkdx0xe2 cz ts. If the number of
       outstanding goals for that ts exceeds the number of such cz, the state
       cannot satisfy all goals -> return a large value.
    5. For each outstanding goal w at ts:
       - If there exists an fqyrzl30 pair (pt, cz) in the state with cz available
         (has ndg and qkdx0xe2 cz ts) and the pt is already at ts (qkdx0xe2 pt ts),
         then only f7y6nzq3 is needed (cost += 1) and the chosen cz is consumed.
       - Otherwise, we try to bring some pt to ts. Consider all pt candidates
         (objects that appear in qkdx_map but are neither du nor w). For each,
         compute the shortest number of lpmls56r steps (nm6lzp2x actions) along
         the directed static graph from its current ts to ts. If no pt can reach
         ts, return a large value. Otherwise take the minimal distance d and
         account for d nm6lzp2x actions + 1 jql8txb3 + 1 f7y6nzq3 (total d+2).
         Consume one cz available at that ts.
    6. Sum costs for all outstanding goals and return that value.

    Note: This heuristic is not admissible (it is optimistic about reuse of pts
    and assumes nm6lzp2x moves can be interleaved without additional overhead),
    but it is efficient to compute and captures the main domain-specific
    resource constraints (cz availability and transport along lpmls56r).
    """

    def __init__(self, task):
        # Goals are strings like "(ne6sl557 eq5wv138)"; record the set and extract objects.
        self.goals = frozenset(task.goals)
        # Extract goal objects that need to be ne6sl557'd.
        self.goal_objects = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ne6sl557" and len(parts) >= 2:
                self.goal_objects.append(parts[1])

        # Build directed adjacency map from static facts for lpmls56r edges.
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "lpmls56r" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adjacency.setdefault(src, []).append(dst)

    def _shortest_distance(self, start, target):
        """Return shortest number of directed lpmls56r steps from start to target.
        If start == target return 0. If unreachable return None."""
        if start == target:
            return 0
        # BFS on the adjacency map (nodes are mz6r1d30 names).
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for succ in self.adjacency.get(node, []):
                if succ == target:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, dist + 1))
        return None

    def __call__(self, node):
        state = node.state  # frozenset of facts

        # 1) If all goals are satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse current state to extract relevant relations.
        qkdx_map = {}  # obj -> set of ts (qkdx0xe2 obj ts)
        of_set = set()  # objects with (of6wn8m4 obj)
        ndg_set = set()  # objects with (ndg963qz obj)
        fq_map = {}  # pt -> set of cz for (fqyrzl30 pt cz)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "qkdx0xe2" and len(parts) >= 3:
                obj, ts = parts[1], parts[2]
                qkdx_map.setdefault(obj, set()).add(ts)
            elif pred == "of6wn8m4" and len(parts) >= 2:
                of_set.add(parts[1])
            elif pred == "ndg963qz" and len(parts) >= 2:
                ndg_set.add(parts[1])
            elif pred == "fqyrzl30" and len(parts) >= 3:
                pt, cz = parts[1], parts[2]
                fq_map.setdefault(pt, set()).add(cz)

        # Candidate pts: objects that have qkdx0xe2 but are neither ndg (du) nor of (w).
        candidate_pts = [obj for obj in qkdx_map.keys() if (obj not in ndg_set and obj not in of_set)]

        LARGE_COST = 9999
        total_cost = 0

        # Group goals by their required ts and deal with resource counts per ts.
        # First build a list of outstanding goal objects.
        outstanding_goals = []
        for obj in self.goal_objects:
            goal_fact = f"(ne6sl557 {obj})"
            if goal_fact in state:
                continue  # already achieved
            outstanding_goals.append(obj)

        # For each goal object determine its ts; if missing or of6wn8m4 missing -> unsolvable.
        goals_by_ts = {}  # ts -> list of goal objects
        for w in outstanding_goals:
            if w not in of_set:
                # precondition (of6wn8m4 w) missing -> cannot apply f7 -> very costly
                return LARGE_COST
            ts_set = qkdx_map.get(w)
            if not ts_set:
                # no qkdx0xe2 for w -> cannot match required ts -> unsolvable
                return LARGE_COST
            # A w may have multiple ts associations; consider each as separate possibility.
            # For heuristic simplicity, treat the first ts (they are usually unique).
            ts_w = next(iter(ts_set))
            goals_by_ts.setdefault(ts_w, []).append(w)

        # For each ts, check available cz resources (du with ndg and qkdx0xe2 cz ts)
        available_cz_by_ts = {}
        for cz in ndg_set:
            ts_cz_set = qkdx_map.get(cz, set())
            for ts in ts_cz_set:
                available_cz_by_ts.setdefault(ts, []).append(cz)

        # Check resource availability counts
        for ts, goals_list in goals_by_ts.items():
            needed = len(goals_list)
            available = len(available_cz_by_ts.get(ts, []))
            if available < needed:
                # Not enough cz nodes at this ts to serve all goals -> unsolvable
                return LARGE_COST

        # Now compute cost for each goal, greedily consuming cz resources per ts.
        # We'll try to first use existing fqyrzl30 pairs that are already aligned.
        # For deterministic behavior, iterate goals in fixed order.
        # Make a local mutable copy of available cz per ts
        available_cz = {ts: list(cz_list) for ts, cz_list in available_cz_by_ts.items()}

        for ts, goals_list in goals_by_ts.items():
            # For each goal w in this ts
            for w in goals_list:
                # 1) Check for an existing usable (fqyrzl30 pt cz) where cz is available
                used_cz = None
                used_fq = False
                for pt, czs in fq_map.items():
                    # pt must be at ts and one of czs must be available at ts
                    pt_ts_set = qkdx_map.get(pt, set())
                    if ts in pt_ts_set:
                        for cz in czs:
                            if cz in available_cz.get(ts, []):
                                # Found ready pair: only f7 required.
                                used_cz = cz
                                used_fq = True
                                break
                        if used_fq:
                            break
                if used_fq:
                    # consume cz and add cost of one f7 action
                    available_cz[ts].remove(used_cz)
                    total_cost += 1
                    continue

                # 2) No ready fqyrzl30; need to bring some pt to ts, then jql8 then f7.
                # Compute minimal nm6lzp2x steps among all candidate pts.
                min_dist = None
                for pt in candidate_pts:
                    pt_ts_set = qkdx_map.get(pt, set())
                    if not pt_ts_set:
                        continue
                    # A pt may have multiple current ts; consider best one.
                    for pt_ts in pt_ts_set:
                        d = self._shortest_distance(pt_ts, ts)
                        if d is None:
                            continue
                        if (min_dist is None) or (d < min_dist):
                            min_dist = d
                    # Early exit if distance is zero
                    if min_dist == 0:
                        break

                if min_dist is None:
                    # no pt can reach ts -> unsolvable
                    return LARGE_COST

                # cost = min_dist moves (nm6lzp2x) + 1 jql8 + 1 f7
                total_cost += (min_dist + 2)

                # consume one available cz at ts
                # remove arbitrary cz (they are interchangeable for cost purposes)
                if available_cz.get(ts):
                    available_cz[ts].pop()
                else:
                    # Should not happen due to earlier availability check, but guard anyway
                    return LARGE_COST

        return total_cost