from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()

def match(fact, *pattern):
    """
    Return True if `fact` matches the provided pattern.
    Wildcard '*' allowed in pattern elements.
    """
    parts = get_parts(fact)
    # Only compare as many parts as provided in pattern
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (a variant of Blocksworld).
    The domain represents manipulators/arms modeled by facts (arm-empty <arm> <pos>)
    and a directed "on" relation between positions/items. The pickup operator
    moves the "arm-empty token" along an "on" edge:
      pre: (arm-empty A X) (on X Y)  -> effect: (arm-empty A Y)
    Thus, the primary way to make (arm-empty A goal_pos) true is to move the
    arm-empty token along a path in the directed graph induced by "on" facts.

    This heuristic estimates the number of operator applications required to
    satisfy all arm-empty goals by computing shortest-path distances in the
    directed graph of "on" relations from the current position of the relevant
    arm to the desired goal position. If the specific arm is not currently
    empty, the heuristic uses a small additional penalty and attempts to reuse
    existing arm-empty tokens (from other arms) as stepping stones.

    Assumptions
    -----------
    - "on" facts (either in static or current state) define directed edges X -> Y
      where a single pickup-like action can move an arm-empty marker from X to Y.
    - Each goal is of the form (arm-empty <arm> <pos>).
    - Pickup/unstack/stack are approximated by moves along the "on" graph. The
      heuristic ignores some ancillary preconditions (e.g., holding, clear,
      on-table) and therefore is not guaranteed to be admissible, but is
      designed to be informative and fast to compute.
    - If the arm is currently empty, the number of required actions is roughly
      the shortest-path length from the arm's current position to the goal
      position (each edge roughly corresponds to one pickup/unstack step).
    - If the arm is not empty (no (arm-empty <arm> X) fact present), an
      additional small fixed penalty is added to account for the effort to
      free the arm or transfer the token to that arm.

    Heuristic Initialization
    ------------------------
    - Save the goal conditions (task.goals).
    - Build a directed adjacency list from static facts that match "(on X Y)".
      Also, adjacency will be extended at heuristic-evaluation time with any
      dynamic "on" facts present in the current state.
    - Precompute the list of all node names seen in static "on" facts so that
      fallback distance computations remain finite.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's facts already satisfy all goals, return 0 (heuristic=0
       iff goal-state).
    2. Construct the directed graph of "on" relations by combining:
       - the adjacency from static facts (built once at initialization),
       - plus any "on" facts that happen to be present in the current state.
    3. For every goal G of the form (arm-empty Arm GoalPos):
       a. If G is already true in the state, cost contribution is 0.
       b. Else attempt to find a current fact (arm-empty Arm CurrPos) in state.
          - If found, run a BFS from CurrPos to GoalPos in the directed graph.
            The BFS depth (number of edges) is taken as the estimated number of
            actions required (each edge ~ one pickup-like move). If unreachable
            by BFS, use a finite fallback cost.
          - If not found (the arm is currently not empty), attempt to reuse
            other arms' empty tokens: for each (arm-empty OtherArm CurrPos2),
            compute BFS distance CurrPos2 -> GoalPos and take the minimum
            distance among those, then add a small penalty (1) to model the
            extra step(s) needed to transfer or free the desired arm.
            If there are no arm-empty tokens at all, fall back to a finite
            estimate based on graph size.
    4. Sum contributions of all unsatisfied goals and return the total as the
       heuristic. The heuristic is always finite (we fall back to a finite
       upper bound if BFS returns no path).

    Implementation notes
    --------------------
    - BFS is used to compute shortest directed-path lengths (efficient for the
      relatively small graphs encountered here).
    - A small constant penalty is used when the exact arm-empty fact for the
      goal arm is missing; this keeps heuristic finite while distinguishing
      busy arms from empty arms.
    """

    def __init__(self, task):
        # Save goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Build adjacency list from static "on" facts (directed edges X -> Y).
        self.static_adj = {}
        self.nodes = set()
        for fact in self.static:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                src, dst = parts[1], parts[2]
                self.nodes.add(src)
                self.nodes.add(dst)
                self.static_adj.setdefault(src, set()).add(dst)

        # If no static on facts, keep static_adj empty but nodes may be empty;
        # we'll handle this at runtime with fallback values.
        # Precompute list of arm-empty goal predicates for faster evaluation.
        self.arm_empty_goals = [
            get_parts(goal) for goal in self.goals if get_parts(goal)[0] == "arm-empty"
        ]

    def shortest_distance(self, start, goal, adj):
        """
        BFS to compute shortest directed-path length from start to goal using
        adjacency dictionary `adj` where adj[u] is an iterable of neighbors.
        Returns an integer distance (>= 0) or None if unreachable.
        """
        if start == goal:
            return 0
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
        state = node.state

        # If goals satisfied, heuristic must be 0 (and only 0 in that case).
        if self.goals <= state:
            return 0

        # Build adjacency for this state by combining static adjacency with
        # dynamic "on" facts present in the current state.
        adj = {k: set(v) for k, v in self.static_adj.items()}
        nodes = set(self.nodes)
        for fact in state:
            if match(fact, "on", "*", "*"):
                parts = get_parts(fact)
                src, dst = parts[1], parts[2]
                nodes.add(src)
                nodes.add(dst)
                adj.setdefault(src, set()).add(dst)

        # Collect current arm-empty facts in the state: map arm -> position
        arm_positions = {}
        for fact in state:
            if match(fact, "arm-empty", "*", "*"):
                parts = get_parts(fact)
                _, arm, pos = parts
                arm_positions[arm] = pos

        # Also collect all arm-empty positions (for reuse if specific arm not empty)
        all_arm_empty_positions = list(arm_positions.values())

        total_cost = 0
        # A finite fallback cost used when BFS finds no path: use number of nodes + small constant
        fallback_cost = max(1, len(nodes)) + 5

        for goal_parts in self.arm_empty_goals:
            # goal_parts = ['arm-empty', arm, goalpos]
            _, arm, goalpos = goal_parts
            goal_fact = f"(arm-empty {arm} {goalpos})"
            if goal_fact in state:
                continue  # already satisfied

            # If this arm is currently empty, compute distance from its current pos.
            if arm in arm_positions:
                start_pos = arm_positions[arm]
                dist = self.shortest_distance(start_pos, goalpos, adj)
                if dist is None:
                    # unreachable via on-graph: use fallback
                    cost = fallback_cost
                else:
                    cost = dist
                total_cost += cost
            else:
                # Arm not currently empty. Attempt to reuse other arms' empty tokens.
                if all_arm_empty_positions:
                    min_dist = None
                    for pos in all_arm_empty_positions:
                        d = self.shortest_distance(pos, goalpos, adj)
                        if d is not None:
                            if min_dist is None or d < min_dist:
                                min_dist = d
                    if min_dist is None:
                        # No path found from any existing empty token -> fallback
                        cost = fallback_cost + 1  # small extra to free the arm
                    else:
                        # Add one extra step to model the effort to transfer/free the arm
                        cost = min_dist + 1
                else:
                    # No arm-empty tokens exist; rely on fallback
                    cost = fallback_cost + 2
                total_cost += cost

        # Ensure heuristic is a finite non-negative integer
        return int(total_cost)