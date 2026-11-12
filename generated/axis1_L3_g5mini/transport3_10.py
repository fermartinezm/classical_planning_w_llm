from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a fact string '(pred a b ...)' into ['pred', 'a', 'b', ...]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the (secret) blocksworld-like domain described
    in the task. The domain's goals are expressed as a set of arm-empty facts
    of the form "(arm-empty <arm> <location>)". This heuristic estimates the
    number of operator applications required to obtain all goal arm-empty facts
    from the current state.

    Assumptions
    -----------
    - The relevant goal predicates are "arm-empty". Other goal predicates are ignored.
    - Each "arm-empty" fact associates an "arm" (first argument) with a "location"
      (second argument). Operators in the domain (pickup/stack/unstack) move or
      toggle those arm-empty associations.
    - Moving an arm-empty association along an "on" relation in the (static/dynamic)
      graph typically requires one action (e.g., a pickup that uses an (on X Y)
      precondition changes an arm-empty association from X to Y). Thus the
      shortest-path distance (in edges) is used as an estimate of needed actions.
    - If an arm has no current "arm-empty" fact in the state, it needs at least one
      action to create/establish such a fact (we conservatively count that as cost 1).
    - The static facts contain many binary "on" links; we treat "on" and also any
      "on-table" relation present in the current state as undirected edges for the
      connectivity graph (this keeps the heuristic robust when reversible moves are
      required).
    - The heuristic is not required to be admissible; it aims to be informative and
      cheap to compute.

    Heuristic Initialization
    ------------------------
    - Extract all goal arm-empty facts from task.goals and store a mapping
      self.goal_arm_to_loc: arm -> desired_location.
    - Extract static "on" relations from task.static and build an undirected
      adjacency map self.static_adj to be used as the baseline graph for computing
      shortest-path distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state), the heuristic does the following:
    1. If the state's facts already satisfy all goal facts, the heuristic returns 0.
       (This ensures the heuristic is 0 only for goal states.)
    2. Parse the current state's "arm-empty" facts to obtain a mapping
       current_arm_to_loc: arm -> current_location.
    3. Build a connectivity graph (adjacency) by starting from the static adjacency
       created at initialization and adding dynamic edges coming from the current
       state (facts like "(on-table A B)" are included as undirected edges).
       - We treat both "on" and "on-table" edges as undirected for robustness,
         since the domain's stack/unstack semantics frequently allow moving in both
         directions (even if via different operator sequences).
    4. For every goal arm-empty (arm -> goal_loc):
       - If the current state already contains exactly that arm-empty fact, cost = 0.
       - Else, if we have a current location cur_loc for that arm, compute the
         shortest-path length (in number of edges) from cur_loc to goal_loc in the
         adjacency graph via BFS. That length is used directly as the estimated
         number of actions needed to move the arm-empty association there (each
         edge roughly corresponds to a pickup/stack/unstack-like operation).
         If no path exists in the graph, we conservatively fall back to a small
         positive cost (here 2) to keep the heuristic finite while acknowledging
         additional manipulations may be needed.
       - Else (the arm has no arm-empty fact in the state), assign a conservative
         cost of 1 (one action to establish an arm-empty fact) plus an optional
         small penalty if the goal location is unreachable (handled above).
    5. Sum the per-goal-arm costs and return the total as the heuristic value.

    This heuristic therefore captures the idea that each misplaced (or missing)
    arm-empty fact generally requires a small number of local manipulations which
    can be estimated as distances in a graph formed by "on" and "on-table" relations.
    Computation is efficient: building the graph is linear in the number of facts
    and each BFS is linear in the number of graph nodes/edges.
    """

    def __init__(self, task):
        """
        Extract goal arm-empty facts and static adjacency (from static "on" facts).

        Parameters
        ----------
        task : Task
            The planning task object. We use task.goals and task.static.
        """
        self.goals = task.goals
        self.static = task.static

        # Map goal arms to their desired locations, e.g., {'item2': 'item20', ...}
        self.goal_arm_to_loc = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm = parts[1]
                loc = parts[2]
                self.goal_arm_to_loc[arm] = loc

        # Build static adjacency from static "on" facts (undirected for robustness).
        self.static_adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.static_adj.setdefault(a, set()).add(b)
                self.static_adj.setdefault(b, set()).add(a)
        # Note: we intentionally do not include static "on-table" here because those
        # are typically dynamic; they will be added per-state when calling the heuristic.

    def _bfs_distance(self, adj, start, goal):
        """
        Compute shortest-path distance (#edges) between start and goal in adjacency map.
        If start == goal return 0. If no path, return None.
        """
        if start == goal:
            return 0
        if start not in adj or goal not in adj:
            # If one of the nodes is unknown in the adjacency map, we still attempt
            # to search if the other is present. If neither present, no path.
            # Return None to indicate unreachable in current graph.
            return None
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, dist = dq.popleft()
            for nb in adj.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    dq.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic value for the given search node.

        Parameters
        ----------
        node : SearchNode
            The search node with attribute .state (a frozenset of fact strings).

        Returns
        -------
        int
            Estimated number of actions to reach a goal state from node.state.
        """
        state = node.state

        # Quick goal check: 0 if all goal facts already satisfied.
        if self.goals <= state:
            return 0

        # Parse current arm-empty facts: arm -> location
        current_arm_to_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm = parts[1]
                loc = parts[2]
                current_arm_to_loc[arm] = loc

        # Build adjacency for this state: start from static adjacency and add dynamic edges
        # for "on-table" facts present in the current state (treat undirected).
        adj = {k: set(v) for k, v in self.static_adj.items()}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "on-table" and len(parts) >= 3:
                    a, b = parts[1], parts[2]
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
                # Also allow dynamic "(on ...)" relations if present in the state
                # (some problems might have them in state rather than static).
                if parts[0] == "on" and len(parts) >= 3:
                    a, b = parts[1], parts[2]
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)

        total_cost = 0

        # For each arm required by the goal, estimate cost to achieve the desired location.
        for arm, goal_loc in self.goal_arm_to_loc.items():
            # If the exact goal fact is present, no cost.
            goal_fact = f"(arm-empty {arm} {goal_loc})"
            if goal_fact in state:
                continue

            # If this arm has a current location, compute distance in the adjacency graph.
            if arm in current_arm_to_loc:
                cur_loc = current_arm_to_loc[arm]
                dist = self._bfs_distance(adj, cur_loc, goal_loc)
                if dist is None:
                    # No path found in the current connectivity graph. Fall back to a
                    # small conservative estimate (domain allows creating paths by other ops).
                    fallback_cost = 2
                    total_cost += fallback_cost
                else:
                    # Each edge roughly corresponds to a single pickup/stack/unstack-like op.
                    total_cost += dist
            else:
                # Arm has no arm-empty fact in the state; need at least one action to create it.
                # Use cost 1, but if the goal location is completely unknown in the graph, add 1 more.
                if goal_loc not in adj:
                    total_cost += 2
                else:
                    total_cost += 1

        # Return integer heuristic (small and efficient to compute).
        return int(total_cost)