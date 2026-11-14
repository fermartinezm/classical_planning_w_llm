from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding
    parentheses and splitting on whitespace.
    Example: "(arm-empty item2 item20)" -> ["arm-empty", "item2", "item20"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the (obfuscated) blocksworld-like secret_domain.
    The goals in this domain are expressed as arm-empty predicates,
    i.e. facts of the form (arm-empty <arm> <position>). Operators in the
    domain can change the second argument of arm-empty by following 'on'
    relationships between positions. This heuristic estimates the number of
    operator applications required to place each arm at its goal position.

    Assumptions
    -----------
    - Goal facts are only (arm-empty ...) facts (as observed in examples).
    - Each arm in the domain appears in the state as exactly one (arm-empty arm pos)
      fact (if an arm is missing, we assume a small non-zero cost to establish it).
    - The 'on' predicate (either in static facts or in the current state) defines
      directed edges between intermediate positions: (on A B) is interpreted as
      an edge A -> B, and one operator application can move an arm one step along
      such an edge (this matches the pickup-like operator patterns).
    - Operators that change arm-empty effectively move the arm along these 'on'
      edges; therefore shortest-path distance in the directed graph approximates
      the minimal number of operator applications required.
    - We ignore interactions between arms (contention) and other predicate constraints
      (clear, on-table, holding) to keep the heuristic cheap to compute.

    Heuristic Initialization
    ------------------------
    - Store goal arm-empty facts (self.goals).
    - Extract the directed adjacency graph from static facts containing (on X Y).
      The graph maps X -> set(Y) for all static 'on' facts. This is stored in
      self.adj_static for fast reuse.
    - The constructor does not assume dynamic 'on' relations are impossible; the
      __call__ method will add any (on ...) facts that happen to be present in
      the current state to the adjacency used for path search.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (its current state):
    1. If all goal facts are already present in the state, return 0.
    2. Build the adjacency graph used for estimating movement:
       - Start from the adjacency computed from static facts (fast, precomputed).
       - Add any 'on' facts that are present in the current state (they may
         reflect dynamic connectivity).
    3. Construct a mapping of current arm locations:
       - Parse all (arm-empty ARM POS) facts from the current state, yielding
         arm -> POS mappings.
       - If an arm in the goals has no current arm-empty fact in the state,
         we assume a small positive base cost to create/position it.
    4. For each goal (arm-empty ARM GOAL_POS):
       - If the exact fact exists in the state, cost contribution is 0.
       - Otherwise, find the current position CUR_POS for this ARM (if any).
         * If CUR_POS is present, compute the shortest directed path length
           from CUR_POS to GOAL_POS in the adjacency graph. The intuition is
           that each edge corresponds roughly to one operator application that
           updates the second argument of arm-empty. Use BFS for this.
         * If no path exists, use a finite fallback cost equal to
           (number_of_nodes + 1) to keep the heuristic finite yet penalize
           unreachable targets under the current connectivity assumptions.
         * If the arm has no current arm-empty fact in the state, use a small
           constant penalty (2) as the estimated cost to establish/position it.
    5. Sum the individual costs for all goal arm-empty facts and return the
       sum. This sum is not guaranteed to be admissible but is efficient to compute
       and provides informative guidance by using the domain's 'on' connectivity.
    """

    def __init__(self, task):
        """Extract and store goals and static adjacency from the task."""
        # The set of goal facts (PDDL fact strings)
        self.goals = task.goals

        # Store static facts (for debugging or extensions if needed)
        self.static = task.static

        # Build adjacency from static 'on' facts: node -> set(neighbors)
        adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                adj.setdefault(src, set()).add(dst)
        # convert sets to frozenset for safety; will copy in __call__
        self.adj_static = {k: frozenset(v) for k, v in adj.items()}

        # Precompute number of nodes in static adjacency (used for fallback finite cost)
        self.static_nodes = set(self.adj_static.keys())
        for neighs in self.adj_static.values():
            self.static_nodes.update(neighs)
        self.static_node_count = max(1, len(self.static_nodes))

    def _bfs_distance(self, src, dst, adjacency):
        """
        Compute shortest directed distance (number of edges) from src to dst
        using BFS on adjacency (dict: node -> iterable(neighbors)).
        Returns an integer distance >= 0 or None if unreachable.
        """
        if src == dst:
            return 0
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, dist = q.popleft()
            for neigh in adjacency.get(node, ()):
                if neigh in visited:
                    continue
                if neigh == dst:
                    return dist + 1
                visited.add(neigh)
                q.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """Return estimated number of operator applications to satisfy arm-empty goals."""
        state = node.state

        # 1) If already a goal state, return 0.
        if self.goals <= state:
            return 0

        # 2) Build adjacency used for path search:
        # Start with static adjacency (convert frozenset -> set for extension)
        adjacency = {k: set(v) for k, v in self.adj_static.items()}

        # Add any 'on' facts present in the current state (they may reflect dynamic links).
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                adjacency.setdefault(src, set()).add(dst)

        # Ensure adjacency neighbors are iterables (if a node missing, adjacency.get returns None -> handled in BFS)
        # 3) Map current arm positions from state: arm -> pos
        arm_pos = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, pos = parts[1], parts[2]
                arm_pos[arm] = pos

        total_cost = 0

        # For each goal arm-empty fact, compute cost contribution
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts or parts[0] != "arm-empty" or len(parts) < 3:
                # Skip non arm-empty goals (defensive)
                continue
            arm, goal_pos = parts[1], parts[2]
            # If the exact goal fact already holds in the current state, cost 0
            if goal_fact in state:
                continue

            # If we have the arm's current position, compute BFS distance; else add small penalty.
            if arm in arm_pos:
                curr_pos = arm_pos[arm]
                dist = self._bfs_distance(curr_pos, goal_pos, adjacency)
                if dist is None:
                    # unreachable under current connectivity assumptions -> large but finite penalty
                    # Use static_node_count + 1 as finite penalty
                    total_cost += (self.static_node_count + 1)
                else:
                    # We interpret each edge traversal as roughly one operator application.
                    total_cost += dist
            else:
                # Arm is not present in current state (unexpected): give a small finite penalty
                total_cost += 2

        # Ensure heuristic is at least 1 for non-goal states (unless sum is zero due to oddities)
        if total_cost == 0:
            return 1
        return total_cost