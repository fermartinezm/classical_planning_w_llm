from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract tokens from a PDDL fact string like "(pred a b c)" -> ["pred", "a", "b", "c"].
    This function deliberately strips the surrounding parentheses before splitting.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5). The domain models
    moving a "association" (predicate wuqjn5r5 <obj> <node>) across nodes of a
    directed graph described by k21e3pdq facts. Movement is enabled by two operators:
    - gww2bqr6: moves the special token odqd6scj along a k21e3pdq edge (when the
      target node has ajcpzwcf).
    - yaz3jy7e: moves both the association (wuqjn5r5) forward by one edge and the
      token odqd6scj forward as part of the same action if the token is already in
      the right position and the target has ajcpzwcf.

    The heuristic estimates how many actions are required to reach the goal(s) that
    specify target wuqjn5r5 facts (i.e., to relocate specific objects to target nodes).

    Assumptions
    -----------
    - The relevant static information is the directed graph given by k21e3pdq facts:
      each such fact is interpreted as an edge (source -> target). The third argument
      (a typed label) is ignored for this heuristic (we approximate by ignoring label
      constraints).
    - There is (typically) a single odqd6scj token in the world; if there are multiple,
      the heuristic uses the closest one to reduce the estimate.
    - We approximate costs by assuming:
      * A single yaz3jy7e action can move an association one graph edge forward.
      * A gww2bqr6 action can move the token one graph edge forward.
      * We do not model ajcpzwcf precondition intricacies or label matching exactly,
        but assume movement along k21e3pdq edges is possible (this keeps the heuristic
        inexpensive while remaining informative).
    - The heuristic may over- or under-estimate (it is not required to be admissible).

    Heuristic Initialization
    ------------------------
    - Parse the static facts (task.static) and build a directed adjacency map
      adjacency[u] = list of v such that (k21e3pdq u v <label>) exists.
    - Extract the goal wuqjn5r5 conditions and remember which objects must be
      associated with which target nodes. Only wuqjn5r5 goals are considered;
      other goal types (if any) are ignored by this heuristic.
    - No other static information is required for the estimation.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state (node.state), the heuristic computes an estimate as follows:

    1. If all goals are already satisfied (task.goals subset of state), return 0.

    2. Build quick lookup structures from the current state:
       - current_assoc[obj] = node where (wuqjn5r5 obj node) currently holds.
       - token_nodes = set of nodes where (odqd6scj node) holds (often a singleton).

    3. For each goal of the form (wuqjn5r5 obj goal_node):
       a. If the goal fact already holds in the state, no cost is added.
       b. Otherwise, find the current node cur where the object is associated:
          - If there is no current association for the object, this goal is treated
            as expensive/unreachable (we add a large penalty).
       c. Compute the shortest directed path length L from cur to goal_node using
          the adjacency graph (each edge corresponds to one association move step).
          - If no such path exists, treat the goal as very expensive (large penalty).
       d. Compute the shortest directed path length D from any token_node to cur.
          This approximates how many token-move (gww2bqr6) actions are required
          to bring the token into position so that the first yaz3jy7e can be applied.
          (We intentionally ignore label/ajcpzwcf constraints when computing D.)
       e. The estimated cost for this goal is D + L:
          - D steps to position the token (gww2bqr6 actions),
          - L steps to advance the association along the path (one yaz3jy7e per edge).

    4. Sum the per-goal estimates to obtain the final heuristic estimate. When multiple
       goals may interact (share token movements), this heuristic does not attempt to
       reason about such synergies and therefore may overcount; this is acceptable
       for a greedy best-first search guiding function.

    Notes
    -----
    - The heuristic is cheap to compute: it uses BFS on the (static) adjacency graph.
    - It returns 0 only for states that satisfy the full goal set (check performed).
    - If a goal appears unreachable in the approximated graph, a large finite penalty is
      returned (so the heuristic value is always a finite integer).
    """

    def __init__(self, task):
        # Store goal facts and static adjacency information
        self.goals = task.goals  # set/frozenset of goal fact strings
        static_facts = task.static

        # Build adjacency map from k21e3pdq facts: treat k21e3pdq u v label as edge u -> v.
        self.adjacency = {}
        self.nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "k21e3pdq" and len(parts) >= 3:
                src, tgt = parts[1], parts[2]
                self.nodes.add(src)
                self.nodes.add(tgt)
                self.adjacency.setdefault(src, []).append(tgt)

        # Extract wuqjn5r5 goals (we only use these in the heuristic).
        self.wuq_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj, tgt = parts[1], parts[2]
                self.wuq_goals.append((obj, tgt))

    def _shortest_path_length(self, start, goal):
        """
        Return shortest directed path length (number of edges) from start to goal
        using the adjacency graph computed from static facts.
        If start == goal, returns 0.
        If no path exists, returns None.
        """
        if start == goal:
            return 0
        if start not in self.nodes or goal not in self.nodes:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neigh in self.adjacency.get(node, []):
                if neigh == goal:
                    return dist + 1
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If all goals already satisfied, heuristic is 0 (requirement 2).
        if self.goals <= state:
            return 0

        # Build current association map: obj -> node where (wuqjn5r5 obj node) holds.
        current_assoc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # If multiple associations exist (unexpected), keep the first seen.
                if obj not in current_assoc:
                    current_assoc[obj] = loc

        # Identify all token positions (odqd6scj facts). (Often a singleton.)
        token_positions = []
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "odqd6scj" and len(parts) >= 2:
                token_positions.append(parts[1])

        # If no token is present, we assume token positioning cost is 0 (optimistic).
        # (Domain usually has exactly one token.)
        if not token_positions:
            token_positions = []

        total_cost = 0
        # large penalty for unreachable / missing information (finite as required).
        INF_PENALTY = max(1000, len(self.nodes) * 10 + 100)

        for (obj, goal_loc) in self.wuq_goals:
            # If the specific goal fact already true in the state, skip.
            goal_fact = f"(wuqjn5r5 {obj} {goal_loc})"
            if goal_fact in state:
                continue

            # Find current association location for the object.
            cur = current_assoc.get(obj)
            if cur is None:
                # Object has no current association: treat as expensive/unreachable.
                total_cost += INF_PENALTY
                continue

            # Compute minimal number L of association-move steps from cur -> goal_loc.
            L = self._shortest_path_length(cur, goal_loc)
            if L is None:
                total_cost += INF_PENALTY
                continue

            # Compute minimal token distance D from any token position to cur.
            if token_positions:
                D_vals = []
                for tpos in token_positions:
                    d = self._shortest_path_length(tpos, cur)
                    if d is None:
                        # If no directed path from token to cur, treat as large but consider other tokens.
                        D_vals.append(INF_PENALTY)
                    else:
                        D_vals.append(d)
                D = min(D_vals)
            else:
                D = 0

            # Estimated cost: D token moves + L association moves.
            total_cost += (D + L)

        # Return integer heuristic estimate (finite).
        return int(total_cost)