from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokenized components of a PDDL fact string, e.g. "(p a b)" -> ["p","a","b"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (u2gh8ts5) domain.
    The domain models movement of "wuqjn5r5" relations (associating a kukxix7x
    object with a z1mgqxkc node) across a graph of z1mgqxkc nodes connected by
    static edges `k21e3pdq`. Actions that move the `wuqjn5r5` facts require
    destination nodes to have `ajcpzwcf` and often involve manipulating an
    `odqd6scj` token. The heuristic estimates the number of operator
    applications required to place each goal kuk at its goal node.

    Assumptions
    -----------
    - The static predicate k21e3pdq defines an undirected connectivity graph
      between z1mgqxkc nodes (we ignore the label argument for connectivity).
    - Each wuqjn5r5 fact encodes the current node of a kukxix7x object.
    - Moving the kuk along one graph edge typically requires one action;
      additional preparatory actions might be needed if the destination node
      lacks ajcpzwcf or if other auxiliary tokens are required. We approximate
      these additional costs with a small per-node penalty.
    - The heuristic need not be admissible; it aims to be informative and
      cheaply computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract from the task:
    - goal_map: mapping from kuk objects (first arg of wuqjn5r5 goals) to
      desired goal node (second arg).
    - adjacency: undirected adjacency list of z1mgqxkc nodes obtained from
      static facts of the form (k21e3pdq NODE1 NODE2 LABEL). The LABEL is
      ignored for connectivity but preserved in static if needed in future.
    The constructor stores these structures so __call__ can perform quick BFS
    distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's facts already satisfy all task goals, return 0.
    2. Parse all wuqjn5r5 facts in the state to determine the current node for
       every kukxix7x object of interest (those appearing in the goal_map).
       If a goal kuk does not appear with a wuqjn5r5 fact in the state, we
       treat it as "unknown" and assign a conservative large cost.
    3. For each goal kuk:
       a. Compute a shortest path (in number of graph edges) from the kuk's
          current node to its goal node using BFS on the static adjacency.
       b. The base cost contribution for that kuk is the path length (each
          edge typically corresponds to one operator application that moves
          the kuk's association along the graph).
       c. Add a small penalty for nodes along the path (excluding the start)
          that currently do NOT have ajcpzwcf in the state. The rationale is
          that destination nodes generally need ajcpzwcf for direct moves, so
          missing ajcpzwcf often requires extra preparatory operations; we
          model this with +1 per such node. This keeps the estimate finite
          and more informative without simulating full action interactions.
       d. If no path exists in the static adjacency (disconnected graph),
          return a large finite cost for that goal (we use LARGE_COST).
    4. Sum the costs for all goal kuks and return the integer total as the
       heuristic value.

    Notes
    -----
    - This heuristic is intentionally approximate (non-admissible) but
      computationally cheap (BFS on the static graph + small checks).
    - It returns 0 only when the entire goal set is satisfied by the state.
    """

    LARGE_COST = 1000  # Returned when something appears unreachable (keeps value finite).

    def __init__(self, task):
        # Goals are kept as the raw goal fact strings (set of strings).
        self.goals = task.goals
        # Static facts for building adjacency.
        static_facts = task.static

        # Build undirected adjacency between z1mgqxkc nodes from k21e3pdq static facts.
        # k21e3pdq has form: (k21e3pdq NODE1 NODE2 LABEL)
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] != "k21e3pdq":
                continue
            _, n1, n2, _label = parts
            if n1 not in self.adjacency:
                self.adjacency[n1] = set()
            if n2 not in self.adjacency:
                self.adjacency[n2] = set()
            self.adjacency[n1].add(n2)
            self.adjacency[n2].add(n1)

        # Extract goal mappings for each kuk that appears in a wuqjn5r5 goal.
        # goal format: "(wuqjn5r5 <kuk> <node>)"
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                kuk, node = parts[1], parts[2]
                self.goal_map[kuk] = node

    def _shortest_path_and_nodes(self, start, target):
        """
        BFS to find shortest path (as list of nodes) between start and target
        using the static adjacency. Returns a list of nodes along the path
        (including start and target). If unreachable, returns None.
        """
        if start == target:
            return [start]
        if start not in self.adjacency or target not in self.adjacency:
            return None
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            curr, path = queue.popleft()
            for nbr in self.adjacency.get(curr, ()):
                if nbr in visited:
                    continue
                visited.add(nbr)
                new_path = path + [nbr]
                if nbr == target:
                    return new_path
                queue.append((nbr, new_path))
        return None

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals are satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Collect which nodes currently have ajcpzwcf and map current positions of kuks.
        ajcp_set = {get_parts(f)[1] for f in state if get_parts(f)[0] == "ajcpzwcf"}
        # Map kuk -> current node from wuqjn5r5 facts in the state.
        current_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                kuk, nodepos = parts[1], parts[2]
                current_positions[kuk] = nodepos

        total_cost = 0

        # For every kuk that must be placed (per goal_map), estimate cost.
        for kuk, goal_node in self.goal_map.items():
            # If already satisfied individually (kuk at goal), skip.
            curr_node = current_positions.get(kuk)
            if curr_node == goal_node:
                continue

            if curr_node is None:
                # Kuk location unknown in this state; assign conservative large cost.
                total_cost += self.LARGE_COST
                continue

            # Compute shortest path in static adjacency.
            path = self._shortest_path_and_nodes(curr_node, goal_node)
            if path is None:
                # If there's no static path, treat as very difficult but finite.
                total_cost += self.LARGE_COST
                continue

            # Base cost: number of edge steps required (path length - 1).
            steps = max(0, len(path) - 1)
            cost = steps

            # Penalty: for each step's destination node (i.e., for each node in path
            # excluding the start) that currently lacks ajcpzwcf, add +1 to model
            # preparatory actions (destination nodes often require ajcpzwcf).
            missing_ajcp = sum(1 for node_name in path[1:] if node_name not in ajcp_set)
            cost += missing_ajcp

            total_cost += cost

        # Ensure heuristic is finite and integer.
        if total_cost >= self.LARGE_COST:
            # Cap to LARGE_COST to avoid unbounded growth.
            return self.LARGE_COST
        return int(total_cost)