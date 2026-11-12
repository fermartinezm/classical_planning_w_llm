from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain ("blocksworld"-like) domain.
    The goals in this domain are primarily arm-empty facts of the form
    (arm-empty <arm> <pos>). Actions in the domain move the second argument of
    arm-empty along directed "on" edges, and other actions (stack/unstack)
    interact with on-table / clear / holding facts. This heuristic focuses on
    the movement of the second argument of arm-empty facts along the static
    directed "on" graph and estimates the number of operator applications
    necessary to make each goal arm-empty fact true.

    Assumptions
    -----------
    - The primary goal facts are of the form (arm-empty A B). If other goal
      predicates appear, they are handled conservatively (1 if missing).
    - The static "on" relations define a directed graph that governs how the
      second argument of arm-empty can be propagated by pickup/unstack-like
      operators. We extract these "on" edges from the static facts available
      in the task (task.static).
    - We assume that each single step along an "on" edge can be achieved with
      roughly one domain action in the optimistic case (admissibility is not
      required). Additional required preconditions (holding/clear/on-table)
      may increase the true cost; to compensate we may add small fixed offsets
      when a direct arm-empty source for an arm is missing.
    - Heuristic may be inadmissible but tries to be informative and cheap to
      compute.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals).
    - Extract the static directed "on" graph from task.static: for each
      static fact "(on X Y)" add a directed edge X -> Y.
    - Collect the set of nodes present in that graph to bound BFS search.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state (set of PDDL fact strings):
    1. If all goal facts are already true in the state, return heuristic 0.
    2. Extract current arm-empty facts from the state. These have the form
       "(arm-empty <arm> <pos>)". For each arm we record the set of positions
       currently associated with that arm (usually one).
    3. For each goal fact:
       - If the goal is already true in the state -> cost 0 for that goal.
       - If the goal is of the form (arm-empty A target_pos) and there exists
         some current position start_pos for the same arm A, compute the
         shortest directed path length from start_pos to target_pos in the
         static "on" graph using BFS. The path length is taken as the
         heuristic cost for that goal (0 if start_pos == target_pos).
       - If no arm-empty fact for the same arm exists in the current state,
         we estimate a small additional overhead because creating an
         arm-empty with that arm may require extra actions (unstack/stack).
         Concretely, we:
           * Find the shortest distance from any currently-held arm-empty
             position (any arm's second-argument) to the goal target_pos.
           * The estimated cost is 1 (to create/redirect the missing arm-empty
             to some source) + that shortest distance. If no arm-empty facts
             exist at all, we instead return a conservative finite cost:
             3 + shortest path from any graph node to target_pos (or a small
             finite fallback if target_pos is unreachable).
       - If the goal predicate is not arm-empty, but is missing, treat its
         cost as 1 (conservative).
    4. Sum the individual estimated costs for all missing goals and return it.
    5. To keep the heuristic finite for states that might otherwise not have
       a path in the static graph, unreachable distances are mapped to a
       finite high value (based on the number of known nodes).

    Notes
    -----
    - The heuristic is intentionally optimistic on per-edge cost (1 per move)
      but adds small offsets when a required arm-empty binding is missing to
      reflect the need to create/redirect arm-empty facts. It is designed to
      be cheap and to reduce search expansions by being informative about how
      far each arm-empty goal is along the static "on" graph.
    """

    def __init__(self, task):
        # store goals and static facts
        self.goals = task.goals
        static_facts = task.static

        # Build directed graph from static "on" facts: X -> Y for "(on X Y)"
        self.edges = {}  # node -> set(successors)
        nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                nodes.add(a)
                nodes.add(b)
                if a not in self.edges:
                    self.edges[a] = set()
                self.edges[a].add(b)

        # Ensure every node appears in edges dict (even with empty successors)
        for n in nodes:
            self.edges.setdefault(n, set())

        self.nodes = set(self.edges.keys())

    def _bfs_distance(self, start, target):
        """
        BFS in the static directed on-graph to compute the shortest number of edges
        to reach target from start. If start == target returns 0. If target is not
        reachable, returns INF (a finite big number).
        """
        if start == target:
            return 0
        if start not in self.nodes or target not in self.nodes:
            # If either node is unknown in the extracted static graph,
            # consider unreachable (but finite).
            return len(self.nodes) + 5

        INF = len(self.nodes) + 5
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for succ in self.edges.get(node, ()):
                if succ == target:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, dist + 1))
        return INF

    def __call__(self, node):
        state = node.state

        # If already goal state -> heuristic 0 (check 2 from checklist)
        if self.goals <= state:
            return 0

        # Collect current arm-empty facts: map arm -> set of positions
        current_arm_empty = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, pos = parts[1], parts[2]
                current_arm_empty.setdefault(arm, set()).add(pos)

        # Also collect all available arm-empty positions (used if a specific arm is missing)
        all_arm_positions = set()
        for poss in current_arm_empty.values():
            all_arm_positions.update(poss)

        total_cost = 0
        # For each goal, estimate its individual cost and sum
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "arm-empty":
                # Conservative cost for non-arm-empty goals: 1 if missing.
                if goal in state:
                    continue
                total_cost += 1
                continue

            # arm-empty goal: (arm-empty arm goal_pos)
            if len(parts) < 3:
                # malformed goal; give a small finite cost
                total_cost += 2
                continue
            arm, goal_pos = parts[1], parts[2]

            # If this goal is already true, cost 0
            if goal in state:
                continue

            # If we have a current arm-empty for the same arm, use it as source
            if arm in current_arm_empty and current_arm_empty[arm]:
                # choose the minimal distance among possible sources for this arm
                best = None
                for src in current_arm_empty[arm]:
                    d = self._bfs_distance(src, goal_pos)
                    if best is None or d < best:
                        best = d
                # best is finite (or INF)
                total_cost += best
            else:
                # No current arm-empty for this arm in the state.
                # Estimate cost as: 1 (to create/redirect an arm-empty for this arm)
                # + minimal distance from any existing arm-empty position to goal.
                if all_arm_positions:
                    best = None
                    for src in all_arm_positions:
                        d = self._bfs_distance(src, goal_pos)
                        if best is None or d < best:
                            best = d
                    total_cost += 1 + best
                else:
                    # No arm-empty facts exist at all in the state; fall back to:
                    # small constant overhead + shortest distance from any node to goal
                    # (or INF fallback if goal_pos not in known nodes)
                    # choose the best distance from any known node
                    if goal_pos in self.nodes:
                        # compute minimal distance from any node to goal_pos
                        best = None
                        for n in self.nodes:
                            d = self._bfs_distance(n, goal_pos)
                            if best is None or d < best:
                                best = d
                        total_cost += 3 + (best if best is not None else len(self.nodes) + 5)
                    else:
                        # goal_pos unknown; give a finite penalty
                        total_cost += len(self.nodes) + 10

        # Ensure we return an int and a finite number
        if total_cost is None:
            return 0
        return int(total_cost)