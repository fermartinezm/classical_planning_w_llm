from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'miconic' (secret_domain) domain.

    This heuristic estimates the number of actions required to reach a goal
    by measuring how far the current 'origin' pointers (one per passenger-2
    object) are from their goal passenger nodes along the directed graph
    induced by the static 'served' relations. Each directed edge (served A B
    <floor>) is treated as one logical step that the origin pointer needs to
    move forward along. Because moving an origin pointer in the real domain
    typically requires a small sequence of actions (e.g. shifting boarded
    status, applying 'depart'/'up' with lift positioning), each graph-step
    is weighted by a small constant factor > 1 to better reflect realistic
    action counts.

    Assumptions
    -----------
    - The static facts of the task include a set of 'served' relations that
      form directed edges between passenger nodes. A fact "(served X Y F)"
      is interpreted as a directed edge X -> Y (the floor F is treated as
      metadata but edges across different floors all count as steps).
    - Every passenger-2 object appears in the state as some "(origin P2 Px)"
      fact (i.e., the origin mapping exists in the state). If it does not,
      the heuristic will treat the mapping as missing and assign a large but
      finite penalty.
    - The heuristic does not try to precisely simulate the required boarded
      and lift-at manipulations; instead it uses a constant multiplier per
      graph step to approximate the extra actions typically required.
    - When multiple goal origin facts must be achieved (there can be several
      origin goals), the heuristic sums the individual estimates (a standard
      relaxation used for multi-goal planning heuristics).

    Heuristic Initialization
    ------------------------
    In the constructor we extract from the static facts:
    - An adjacency dictionary mapping each passenger node to the set of its
      direct successors according to 'served' facts.
    - The set of all passenger nodes that appear in served facts.
    - The list of goal origin pairs (passenger-2, goal-passenger) derived
      from the task's goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state node:
    1. Parse the state's facts to obtain the current origin mapping(s):
       map each passenger-2 object to the passenger node it currently
       references (from facts of the form "(origin P2 Px)").
    2. For each goal origin requirement "(origin P2 Pgoal)":
       a. Find the current passenger Px that P2 points to in the state.
       b. If Px == Pgoal, this goal contributes 0 to the heuristic.
       c. Otherwise, perform a BFS on the directed graph built from
          static 'served' facts to compute the shortest number of graph
          edges (d) to get from Px to Pgoal.
          - If no path exists, return a large finite penalty (heuristic
            detects that reaching that specific goal via served edges is
            impossible in the relaxed model).
       d. Convert the graph distance d into an action estimate by
          multiplying by a small constant (STEP_COST). This constant
          approximates the extra 'depart'/'up' actions and lift manipulations
          typically required for each pointer step. A conservative value
          is used (e.g. STEP_COST = 2).
    3. Sum the estimated costs for all goal origin mappings and return
       the total as the heuristic value.
    """

    def __init__(self, task):
        """
        Extract static information and goal origin requirements.

        Parameters
        ----------
        task : Task
            The planning task object that contains static facts and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency list from static 'served' facts:
        # treat "(served A B FLOOR)" as a directed edge A -> B.
        self.adj = {}  # maps node -> set(successor nodes)
        self.nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                src = parts[1]
                dst = parts[2]
                self.nodes.add(src)
                self.nodes.add(dst)
                if src not in self.adj:
                    self.adj[src] = set()
                self.adj[src].add(dst)

        # Extract goal origin requirements as a list of (p2, goal_passenger)
        self.goal_origin_pairs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                goal_passenger = parts[2]
                self.goal_origin_pairs.append((p2, goal_passenger))

        # Cost per graph-step (one served-edge). Chosen > 1 to reflect
        # additional operations typically needed in the true domain.
        self.STEP_COST = 2

        # If BFS does not find a path, return this finite large penalty.
        # It should be finite to satisfy the planner's expectations.
        self.INF_PENALTY = max(1000, len(self.nodes) * self.STEP_COST + 10)

    def _bfs_distance(self, start, target):
        """Compute shortest directed distance (in edges) from start to target.
        Returns None if target is unreachable from start."""
        if start == target:
            return 0
        if start not in self.nodes:
            # If start not known from static served graph, there is no path.
            return None
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for succ in self.adj.get(node, ()):
                if succ == target:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : Node
            Planner search node with attribute `state` that is a frozenset of facts.

        Returns
        -------
        int
            Estimated number of actions to reach the goal from this node.
        """
        state = node.state

        # Build a mapping of passenger-2 -> current passenger from state origin facts.
        current_origin = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                passenger = parts[2]
                current_origin[p2] = passenger

        total_cost = 0

        # For each required goal origin, compute cost from current to goal.
        for (p2, goal_passenger) in self.goal_origin_pairs:
            cur = current_origin.get(p2, None)
            if cur is None:
                # Missing origin in the current state; assign penalty.
                return self.INF_PENALTY

            if cur == goal_passenger:
                continue  # already satisfied

            d = self._bfs_distance(cur, goal_passenger)
            if d is None:
                # No path found in the relaxed graph model; large finite penalty.
                return self.INF_PENALTY
            # Convert graph edge distance to action estimate.
            total_cost += d * self.STEP_COST

        # Ensure heuristic is zero for goal states.
        if total_cost == 0:
            # Double-check that all goal origin facts are present in the state.
            # This guarantees heuristic==0 only for actual goal states.
            if all(("(origin %s %s)" % (p2, goal)) in state for (p2, goal) in self.goal_origin_pairs):
                return 0
            # Otherwise, small positive value to reflect mismatch (rare).
            return 1

        return int(total_cost)