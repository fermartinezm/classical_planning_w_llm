from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the (secret_domain) blocksworld-like domain. The planner's goals
    are exclusively (arm-empty <arm> <obj>) facts in the provided instances. The
    actions that directly modify arm-empty facts (pickup, stack, unstack) move the
    second argument of an (arm-empty <arm> X) fact along object relationships
    (primarily the binary relation `on`). This heuristic models such movements as
    shortest-path transitions in a graph of objects and estimates how many such
    transitions are needed to reach each goal arm-empty fact.

    Assumptions
    -----------
    - Goals are a set of (arm-empty <arm> <target>) facts (as in the instances).
    - The primary mechanism for changing the second argument of an `arm-empty`
      fact is via relations of the form (on A B) (and, to be conservative, also
      (on-table A B)). Those relations define edges in a graph over objects.
    - A single application of pickup/stack/unstack can be modeled as a single
      step along an `on`/`on-table` edge: if currently (arm-empty arm X)
      and (on X Y) holds, then one appropriate action can make
      (arm-empty arm Y) true. Therefore each graph edge corresponds to roughly
      one action that moves the second parameter of arm-empty by one hop.
    - Interactions between different arms and conflicts for objects are ignored;
      the heuristic is therefore optimistic about parallelizability and may be
      inadmissible (which is acceptable for greedy best-first search).

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts.
    - Build an undirected adjacency graph over object nodes using all static
      facts and any state facts that encode `on` or `on-table` relationships.
      (Using both ensures the graph used for planning includes reachable links
      even if some `on` facts appear in the initial state rather than as static.)
    - The adjacency graph is stored as a dict: node -> set(neighbors).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's facts already contain all goal facts, return 0.
    2. Parse the state's (arm-empty ...) facts to determine, for each arm
       mentioned in the goals, the current object paired with that arm
       (i.e., the current second argument X for which (arm-empty arm X) is true).
    3. For each goal (arm-empty arm target):
       a. If the exact goal fact is already present in the state, the cost for
          this goal is 0.
       b. Else, if the state contains (arm-empty arm cur), compute the
          shortest-path distance d from cur to target in the adjacency graph.
          Each hop is considered to cost 1 (a single pickup/stack/unstack-like
          action that moves the arm-empty pairing by one object). Add d to the
          heuristic estimate.
       c. Else (no (arm-empty arm X) exists in the state), assume the arm will
          be brought into play with a small fixed overhead plus the distance
          from some object to the target. Concretely, compute the minimum
          distance from any node in the graph to target (if reachable) and add
          that distance + 1 (extra +1 to account for bringing the arm into a
          relevant relation). If the target is isolated, use a moderate finite
          penalty so the heuristic remains finite.
    4. Sum the per-goal estimates. Return that sum as the heuristic value.
    """

    def __init__(self, task):
        # Goals we must achieve
        self.goals = task.goals

        # Build adjacency graph from static facts (and include any `on` / `on-table`
        # facts that may appear in static). We store edges undirected because
        # moving the second argument of arm-empty can be achieved by actions
        # that effectively traverse the object relation in either direction.
        self.adj = {}
        static_facts = task.static if hasattr(task, "static") else frozenset()

        def add_edge(a, b):
            if a not in self.adj:
                self.adj[a] = set()
            if b not in self.adj:
                self.adj[b] = set()
            self.adj[a].add(b)
            self.adj[b].add(a)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on" and len(parts) >= 3:
                add_edge(parts[1], parts[2])
            elif pred == "on-table" and len(parts) >= 3:
                # Treat on-table as an edge between the object and the table/base.
                add_edge(parts[1], parts[2])

        # If there are no adjacency entries for some objects referenced in goals,
        # they will be implicitly handled at runtime when we combine static and
        # state facts to compute paths.
        # Precompute set of goal arms and targets for quick access.
        self.goal_pairs = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, target = parts[1], parts[2]
                self.goal_pairs.append((arm, target))

    def _bfs_distance(self, start, goal, adj):
        """Return shortest path length between start and goal in graph `adj` (undirected).
        If start == goal return 0. If unreachable, return None."""
        if start == goal:
            return 0
        if start not in adj or goal not in adj:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in adj.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic value for the provided search node.

        The heuristic is the sum of estimated numbers of single-object moves
        (modeled as unit-cost hops in the on/on-table graph) required to turn
        each currently-unsatisfied goal (arm-empty arm target) into a fact that
        is true. The heuristic is 0 for goal states and always finite for states
        where the target is reachable through the adjacency graph; for isolated
        targets a finite penalty is used.
        """
        state = node.state

        # 1) If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Build an adjacency graph augmented with `on` and `on-table` facts
        #    present in the current state (these can change during planning).
        adj = {k: set(v) for k, v in self.adj.items()}  # copy static adjacency
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "on" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if a not in adj:
                    adj[a] = set()
                if b not in adj:
                    adj[b] = set()
                adj[a].add(b)
                adj[b].add(a)
            elif pred == "on-table" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if a not in adj:
                    adj[a] = set()
                if b not in adj:
                    adj[b] = set()
                adj[a].add(b)
                adj[b].add(a)

        # 3) Parse current arm-empty facts in the state.
        current_arm_empty = {}  # arm -> object
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, obj = parts[1], parts[2]
                current_arm_empty[arm] = obj

        # 4) For each goal arm-empty, estimate cost.
        total_cost = 0
        # For unreachable targets use this finite penalty
        unreachable_penalty = max(5, len(adj) + 1)

        # Precompute list of nodes (keys) for the fallback min-distance computation
        all_nodes = set(adj.keys())

        for arm, target in self.goal_pairs:
            goal_fact = f"(arm-empty {arm} {target})"
            # If goal already satisfied in this state, cost 0 for this goal
            if goal_fact in state:
                continue

            # If we have a current partner for this arm, compute distance from it to target
            if arm in current_arm_empty:
                cur = current_arm_empty[arm]
                d = self._bfs_distance(cur, target, adj)
                if d is None:
                    # unreachable -> penalize but finite
                    total_cost += unreachable_penalty
                else:
                    total_cost += d
            else:
                # No current arm-empty for this arm in the state; estimate cost as:
                # 1 (to bring the arm into play) + min distance from any node to target.
                if not all_nodes:
                    # no graph nodes known at all (should be rare) -> use small finite cost
                    total_cost += 2
                else:
                    min_d = None
                    for node_name in all_nodes:
                        d = self._bfs_distance(node_name, target, adj)
                        if d is not None:
                            if min_d is None or d < min_d:
                                min_d = d
                    if min_d is None:
                        total_cost += unreachable_penalty
                    else:
                        total_cost += 1 + min_d

        return total_cost