from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a fact string like '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5) domain.
    The domain manipulates three predicate types:
      - wuqjn5r5 <kukxix7x> <z1mgqxkc>  (a binary relation whose second argument is usually moved)
      - odqd6scj <z1mgqxkc>              (a marker that can be moved over the k21e3pdq graph)
      - ajcpzwcf <z1mgqxkc>              (a boolean flag on nodes used by operators)
    Static facts (k21e3pdq) define labeled directed edges among z1mgqxkc nodes.
    Two operators interact to change wuqjn5r5-second-arguments:
      - gww2bqr6: moves the odqd6scj marker along an edge (u -> v) but only if ajcpzwcf(v) is true.
      - yaz3jy7e: given a two-edge pattern p -> m -> s with the same label and odqd6scj at p,
                 it moves wuqjn5r5(X,m) to wuqjn5r5(X,s), also moving odqd6scj from p to m
                 and toggling ajcpzwcf flags on the involved nodes.
    The heuristic estimates the number of actions needed to achieve the goal wuqjn5r5 facts
    by reasoning about the minimal number of yaz3jy7e steps to move each required second-argument
    from its current node to the goal node, plus the necessary movements of the odqd6scj marker
    (gww2bqr6) to enable these yaz3jy7e applications.

    Assumptions
    -----------
    - All goal facts in this domain are of the form (wuqjn5r5 SUBJECT TARGET).
    - A wuqjn5r5 fact can only be moved by successive applications of yaz3jy7e (it is never
      generated from thin air). Therefore, the heuristic looks for an existing wuqjn5r5(S, cur)
      for the same SUBJECT S and estimates how many yaz3jy7e steps are required to reach the
      goal TARGET.
    - odqd6scj is unique in the state (at most one node has odqd6scj true). If it is missing,
      the heuristic will return a large finite penalty.
    - The heuristic ignores the dynamic toggling of ajcpzwcf caused by yaz3jy7e (it uses the
      current state's ajcpzwcf truth-values to determine where odqd6scj can move). This
      approximation keeps the estimate cheap while still being informative.
    - Each application of yaz3jy7e costs 1, and each application of gww2bqr6 costs 1.

    Heuristic Initialization
    ------------------------
    In the constructor we:
      - Extract all static k21e3pdq triples (u, v, label) describing labeled directed edges.
      - Build:
          * edge_triples: list of (u, v, label)
          * next_by_middle: for every middle node m, the set of possible successors s reachable
            via a yaz3jy7e step together with the required predecessor nodes p (i.e., all
            p such that (p,m,label) and (m,s,label) exist).
      - Store the planner goal facts for use in the heuristic call.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goals are already satisfied in this state, return 0.
    2. Parse the state's facts to obtain:
       - current ajcpzwcf-true nodes (used to build the odqd6scj movement graph),
       - the current location of odqd6scj (if any),
       - the current wuqjn5r5 facts mapping SUBJECT -> current node.
    3. Build an odqd movement graph: a directed edge u->v exists if there is a static
       k21e3pdq(u,v,any_label) and ajcpzwcf(v) is true in the current state (the precondition
       of gww2bqr6 requires ajcpzwcf on the target).
    4. Build the abstract "wuq-move" graph: m -> s is possible if there exists a label L and
       a predecessor p such that (p,m,L) and (m,s,L) are static. Applying yaz3jy7e moves
       from m to s but requires odqd6scj to be at some corresponding p.
    5. For each goal fact (wuqjn5r5 subj goal_node):
       a) Find the current node cur where wuqjn5r5(subj, cur) holds. If none is present,
          we add a large penalty because a wuqjn5r5 for that subject cannot be created.
       b) If cur == goal_node, cost for that goal is 0.
       c) Otherwise, find a (shortest) path cur -> ... -> goal_node in the abstract wuq-move graph
          (BFS). If no path exists, the goal is unreachable by rewriting the second-argument;
          add a large penalty.
       d) Otherwise, simulate walking along that path step by step:
          - For a step m -> s we must pick some predecessor p that realizes that labeled
            pattern (we precomputed candidates). The cost to enable the yaz3jy7e here is:
              dist_odqd(current_odqd_pos, p) + 1  (moving odqd to p using gww2bqr6 repeatedly,
             then one yaz3jy7e)
            After applying yaz3jy7e, the heuristic updates current_odqd_pos := m (because
            yaz3jy7e moves odqd from p to m) and proceeds to the next step.
          - If moving odqd to p is impossible (p unreachable in odqd graph), the path is
            considered infeasible; try alternate p candidates or fallback to a large penalty.
       e) Sum costs of odqd moves and yaz3jy7e applications along the path; that sum is the
          estimate for this specific goal.
    6. Combine the individual goal estimates by summing them. This assumes goals must be
       achieved (possibly sequentially). The sum gives a heuristic estimate of the total
       number of actions required.
    7. Return the total estimate (or 0 if goal reached).

    Note on admissibility and conservativeness:
    - This heuristic is not guaranteed admissible; we purposely allow optimistic assumptions
      (for example, ignoring ajcpzwcf toggles) to produce stronger guidance in greedy
      best-first search. It remains finite and cheap to compute for practical instances.
    """

    def __init__(self, task):
        # store goals (a set of fact strings)
        self.goals = task.goals
        # extract static k21e3pdq triples
        self.edge_triples = []
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) == 4:
                u, v, label = parts[1], parts[2], parts[3]
                self.edge_triples.append((u, v, label))

        # Build a mapping: for each middle node m, for each successor s reachable via some
        # label L, list the possible predecessor p nodes that realize p -> m -> s with same L.
        # next_by_middle[m][s] = set(p1, p2, ...)
        self.next_by_middle = {}
        # Also build simple m->s adjacency ignoring predecessors (used for BFS path search)
        self.wuq_adj = {}

        # First, index edges by (u->v) keyed and by node/label for quick lookup
        edges_by_src = {}
        edges_by_mid_label = {}  # key (mid,label) -> set of successors
        edges_by_dst_label = {}  # key (dst,label) -> set of predecessors
        for u, v, label in self.edge_triples:
            edges_by_src.setdefault(u, []).append((v, label))
            edges_by_mid_label.setdefault((u, label), []).append(v)
            edges_by_dst_label.setdefault((v, label), []).append(u)

        # For each triple (p,m,L) and (m,s,L) produce m->s edge and record p as a candidate predecessor
        # Loop over all (m,s,L) by scanning triples
        for p, m, L in self.edge_triples:
            # Find successors s for m with same label L
            successors = edges_by_mid_label.get((m, L), [])
            if not successors:
                continue
            for s in successors:
                # record successor s for middle m
                self.wuq_adj.setdefault(m, set()).add(s)
                self.next_by_middle.setdefault(m, {}).setdefault(s, set()).add(p)

        # convert sets to lists for deterministic iteration
        for m in list(self.wuq_adj.keys()):
            self.wuq_adj[m] = list(self.wuq_adj[m])
        for m in list(self.next_by_middle.keys()):
            for s in list(self.next_by_middle[m].keys()):
                self.next_by_middle[m][s] = list(self.next_by_middle[m][s])

        # A large but finite penalty (used for unreachable or missing resources)
        self.large_penalty = max(500, len(self.edge_triples) * 10)

    def _bfs_distances(self, start, adjacency):
        """
        Generic BFS from start in adjacency dict {node: [neighbors...]}.
        Returns dict node->distance (0 for start). Unreachable nodes are absent.
        """
        dist = {}
        if start is None:
            return dist
        q = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # 1) If goal already reached, heuristic is 0
        if self.goals <= state:
            return 0

        # 2) extract ajcpzwcf-true nodes and odqd6scj location and current wuqjn5r5 facts
        ajcp_true = set()
        odqd_pos = None
        wuq_map = {}  # subject -> node (assumes at most one wuqjn5r5 per subject; if many, pick any)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "ajcpzwcf" and len(parts) == 2:
                ajcp_true.add(parts[1])
            elif pred == "odqd6scj" and len(parts) == 2:
                odqd_pos = parts[1]
            elif pred == "wuqjn5r5" and len(parts) == 3:
                subj, second = parts[1], parts[2]
                # keep the first encountered mapping; typically there will be one
                if subj not in wuq_map:
                    wuq_map[subj] = second

        # If no odqd marker present, we still return finite heuristic but penalize strongly
        if odqd_pos is None:
            odqd_pos = None  # _bfs_distances will handle None by returning empty dict

        # 3) Build odqd movement adjacency using current ajcp_true:
        #    there is an odqd edge u->v if static has (u,v,label) and ajcpzwcf(v) is true now
        odqd_adj = {}
        for u, v, label in self.edge_triples:
            if v in ajcp_true:
                odqd_adj.setdefault(u, []).append(v)

        # Precompute BFS distances from any relevant node on-demand. We'll compute when needed:
        # helper to get distances from a given source in odqd_adj
        def odqd_dist_from(src):
            return self._bfs_distances(src, odqd_adj)

        total_estimate = 0

        # Only consider goals that are wuqjn5r5 facts (others would be handled by fallback)
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] != "wuqjn5r5" or len(parts) != 3:
                # Not a wuqjn5r5 goal: we don't have specific domain knowledge; add small penalty
                total_estimate += 1
                continue

            subj, goal_node = parts[1], parts[2]

            # If goal already satisfied, cost 0
            goal_string = f"(wuqjn5r5 {subj} {goal_node})"
            if goal_string in state:
                continue

            # Find current node where subject is attached
            if subj not in wuq_map:
                # No source wuqjn5r5 for this subject present -> likely impossible by domain ops
                total_estimate += self.large_penalty
                continue
            cur = wuq_map[subj]

            # If current equals goal, continue (should have been caught above)
            if cur == goal_node:
                continue

            # BFS on wuq-move graph (m->s) to find shortest path from cur to goal_node
            # adjacency is self.wuq_adj
            wuq_prev = {}  # to reconstruct path
            q = deque([cur])
            visited = {cur}
            found = False
            while q and not found:
                u = q.popleft()
                for v in self.wuq_adj.get(u, ()):
                    if v not in visited:
                        visited.add(v)
                        wuq_prev[v] = u
                        if v == goal_node:
                            found = True
                            break
                        q.append(v)

            if not found:
                # No sequence of yaz3jy7e moves can move cur to goal_node (using static labels)
                total_estimate += self.large_penalty
                continue

            # Reconstruct path cur -> ... -> goal_node
            path = []
            node_v = goal_node
            while node_v != cur:
                prev = wuq_prev[node_v]
                path.append((prev, node_v))  # step prev -> node_v
                node_v = prev
            path.reverse()

            # Simulate applying steps along path, tracking odqd position and summing costs
            current_odqd = odqd_pos
            path_cost = 0
            # Precompute an odqd distance map from current_odqd; we'll recompute after odqd moves
            odqd_dist_map = odqd_dist_from(current_odqd) if current_odqd is not None else {}

            feasible = True
            for m, s in path:
                # For this m->s step get candidate predecessor p nodes
                candidates_p = self.next_by_middle.get(m, {}).get(s, [])
                if not candidates_p:
                    # Should not happen because m->s in wuq_adj implies existence of p,
                    # but if absent due to initialization mismatch, treat as infeasible.
                    feasible = False
                    break

                # Choose p minimizing distance from current_odqd to p in odqd graph
                best_dist = None
                best_p = None
                for p in candidates_p:
                    d = odqd_dist_map.get(p)
                    if d is None:
                        # unreachable from current odqd position under current ajcp configuration
                        continue
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_p = p

                if best_p is None:
                    # None of the candidate predecessors is reachable by odqd -> infeasible under current
                    # ajcp approximation. We treat this path as infeasible.
                    feasible = False
                    break

                # Move odqd from current_odqd to best_p using gww2bqr6 edges: cost = best_dist
                path_cost += best_dist
                # Apply yaz3jy7e: cost 1
                path_cost += 1

                # After yaz3jy7e, odqd moves from p to m (effect of yaz3jy7e)
                current_odqd = m
                # Update odqd distance map from new odqd position
                odqd_dist_map = odqd_dist_from(current_odqd)

            if not feasible:
                total_estimate += self.large_penalty
            else:
                total_estimate += path_cost

        # ensure finite cost
        if total_estimate is None:
            return self.large_penalty
        return int(total_estimate)