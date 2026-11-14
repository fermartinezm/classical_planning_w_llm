from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    Example:
    "(k21e3pdq beg400qv dj19p99h oxpphcgv)" -> ["k21e3pdq", "beg400qv", "dj19p99h", "oxpphcgv"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(wuqjn5r5 ip8xvrv6 z4076g0l)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates how many actions are roughly needed to reach the goal state from a given state.
    The goal in this domain involves establishing a (wuqjn5r5 ?kukxix7x ?z1mgqxkc) fact corresponding to the goal.
    The domain actions revolve mainly around manipulation of the (odqd6scj), (ajcpzwcf), and (wuqjn5r5) predicates,
    and transitions of these predicates on objects of type z1mgqxkc and kukxix7x.
    The heuristic estimates the effort needed by counting:
      - how many target (wuqjn5r5) goal facts are missing,
      - how many intermediate (odqd6scj) and (ajcpzwcf) facts need to be "cycled" or "moved",
      - and leveraging the static relation (k21e3pdq) which constrains connectivity/relations between z1mgqxkc objects
        parameterized by fixed yrvdqfln constants.
    The heuristic effectively counts the minimal number of intermediate steps needed to move the (odqd6scj) "marker" 
    through a chain of z1mgqxkc objects towards the goal, and then to establish the required (wuqjn5r5) relation.

    Assumptions
    -----------
    - The predicates and static relations are the only significant source of information for progress.
    - (k21e3pdq ?from ?to ?const) represents allowed "edges" between z1mgqxkc nodes to move the (odqd6scj) or (ajcpzwcf) markers.
    - The goal is represented solely by a disjunctive set of (wuqjn5r5) facts that must hold.
    - Actions gww2bqr6 and yaz3jy7e respectively move the (odqd6scj) marker and modify (wuqjn5r5) relations along the chain
      where the (ajcpzwcf) predicate marks intermediate targets.
    - The static facts (k21e3pdq) form a directed graph between z1mgqxkc objects through which movement happens.
    - The heuristic only needs to guide greedy best-first search and thus does not have to be admissible, but should avoid large overestimations.

    Heuristic Initialization
    ------------------------
    - Extract all goal (wuqjn5r5) facts that must be true.
    - Extract all static (k21e3pdq ?from ?to ?const) facts and build a graph structure:
        key: ?from z1mgqxkc
        values: list of (?to, ?const) edges.
    - Store all constants of type yrvdqfln appearing in the static facts.
    - Precompute sets of all objects of each type extracted from the static facts for efficiency.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Quickly check if current state is a goal state (all goal (wuqjn5r5) facts present), if yes, return 0.

    2. Parse the state to extract sets of:
       - All current (wuqjn5r5) facts: (kukxix7x, z1mgqxkc) pairs that currently hold.
       - All current (odqd6scj) facts: the current position(s) of the "odqd6scj" marker(s).
       - All current (ajcpzwcf) facts: the z1mgqxkc nodes currently marked as ajcpzwcf.

    3. For each goal (wuqjn5r5 ?kukxix7x ?target_z1mgqxkc), if it does not hold:
       - We try to estimate the minimal number of steps to propagate the (odqd6scj) marker 
         via the action gww2bqr6 and the complex yaz3jy7e to eventually establish the required (wuqjn5r5) fact.

    4. The structure of the problem suggests:
       - The (odqd6scj) marker is moved along the graph edges (k21e3pdq).
       - The (ajcpzwcf) markers indicate intermediate nodes that must be "activated" or "visited".
       - The action yaz3jy7e moves the (wuqjn5r5) fact from one z1mgqxkc to another, swapping (wuqjn5r5) relations on kukxix7x objects,
         linked by an edge in k21e3pdq and with proper preconditions.

    5. We estimate cost for:
       - Moving the (odqd6scj) marker from its current position(s) to a necessary intermediate node(s) by shortest path along graph edges.
       - Count how many ajcpzwcf nodes still need to be visited or changed.
       - Distance to move (wuqjn5r5) from current position(s) to goal required position.

    6. The heuristic sums these components weighted by relevant minimal action counts:
       - one action to move (odqd6scj) between nodes per edge,
       - one action to apply yaz3jy7e changing (wuqjn5r5) per step,
       - plus some constants for intermediate preparations.

    7. To keep the heuristic efficiently computable:
       - Use shortest-path computations on the static graph precomputed.
       - Do not simulate entire action sequences, just approximate costs by distances and counts of missing facts.

    8. Return the summed cost as heuristic value.

    This heuristic guides greedy best-first search toward states that reduce the distance along the graph 
    from current (odqd6scj)/ (wuqjn5r5) positions to goal (wuqjn5r5) positions, thus breaking symmetries and minimizing expansions.
    """

    def __init__(self, task):
        # Extract goal facts (wuqjn5r5 Kukxix7x Z1mgqxkc).
        self.goal_wuqjn5r5 = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "wuqjn5r5" and len(parts) == 3:
                self.goal_wuqjn5r5.add((parts[1], parts[2]))

        # Build adjacency graph from static (k21e3pdq ?from ?to ?const) facts for moving odqd6scj and wuqjn5r5
        self.adjacency = {}  # key: ?from z1mgqxkc, value: list of (?to z1mgqxkc, ?const yrvdqfln)
        self.yrvdqfln_constants = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq" and len(parts) == 4:
                _from, _to, y_const = parts[1], parts[2], parts[3]
                self.adjacency.setdefault(_from, []).append((_to, y_const))
                self.yrvdqfln_constants.add(y_const)

        # Collect all z1mgqxkc objects from static adjacency keys and values for completeness.
        self.z1mgqxkc_nodes = set(self.adjacency.keys())
        for neighbors in self.adjacency.values():
            for (to_node, _) in neighbors:
                self.z1mgqxkc_nodes.add(to_node)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        Parameters:
        - node: the search node containing the current state.

        Returns:
        - Non-negative integer heuristic value estimating distance to goal.
        """
        state = node.state

        # Quick goal check: if all goal (wuqjn5r5) facts hold, heuristic = 0.
        current_wuqjn5r5 = set()
        for fact in state:
            if fact.startswith("(wuqjn5r5 "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    current_wuqjn5r5.add((parts[1], parts[2]))
        if self.goal_wuqjn5r5 <= current_wuqjn5r5:
            return 0

        # Extract current odqd6scj positions (set of nodes with that predicate).
        odqd6scj_nodes = set()
        for fact in state:
            if fact.startswith("(odqd6scj "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    odqd6scj_nodes.add(parts[1])
        if not odqd6scj_nodes:
            # If no odqd6scj marker present, estimate some minimal cost to set or move it:
            # We arbitrarily pick minimal cost of 1 (must bring odqd6scj somewhere).
            odqd6scj_nodes = set()

        # Extract current ajcpzwcf marked nodes.
        ajcpzwcf_nodes = set()
        for fact in state:
            if fact.startswith("(ajcpzwcf "):
                parts = get_parts(fact)
                if len(parts) == 2:
                    ajcpzwcf_nodes.add(parts[1])

        # Extract current wuqjn5r5 predicates, also needed to estimate how far the needed wuqjn5r5
        # facts are from current ones (to estimate moves of wuqjn5r5 relations).
        # We have current_wuqjn5r5 above.

        # For heuristic estimate, precompute shortest path distances on the graph from any node to any node:
        # We only need shortest distances to goal nodes and ajcpzwcf nodes.
        # For efficiency, we compute BFS from each odqd6scj node and from wuqjn5r5 target nodes only as needed.

        # For large graphs this can be costly, so we do a caching of BFS results per call.

        # Define a helper BFS function for shortest path distance ignoring edge labels.
        def bfs_dist(start, targets):
            """
            Breadth-first search from start node to minimal distance to any target node.
            Returns minimal distance to any node in targets or None if unreachable.
            """
            if start in targets:
                return 0
            from collections import deque

            visited = set([start])
            queue = deque([(start, 0)])

            while queue:
                current, dist = queue.popleft()
                for neighbor, _ in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        if neighbor in targets:
                            return dist + 1
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return None  # unreachable

        # Estimate cost contribution from odqd6scj marker movement:
        # Compute minimal distance from any odqd6scj node to any ajcpzwcf node 
        # (since yaz3jy7e requires ajcpzwcf for its precondition).
        odqd6scj_to_ajcpzwcf = None
        if ajcpzwcf_nodes and odqd6scj_nodes:
            distances = []
            for start in odqd6scj_nodes:
                d = bfs_dist(start, ajcpzwcf_nodes)
                if d is not None:
                    distances.append(d)
            if distances:
                odqd6scj_to_ajcpzwcf = min(distances)

        # If no ajcpzwcf nodes but still needed (goal wuqjn5r5 missing):
        # estimate cost as minimal distance to goal wuqjn5r5 nodes.
        odqd6scj_to_goal_wuqjn5r5 = None
        goal_wuqjn5r5_to_move = self.goal_wuqjn5r5 - current_wuqjn5r5
        goal_targets = set(t for _, t in goal_wuqjn5r5_to_move)
        if odqd6scj_nodes and goal_targets:
            distances = []
            for start in odqd6scj_nodes:
                d = bfs_dist(start, goal_targets)
                if d is not None:
                    distances.append(d)
            if distances:
                odqd6scj_to_goal_wuqjn5r5 = min(distances)

        # Estimate cost: compose heuristic from components:
        # - Number of missing goal (wuqjn5r5) facts: roughly corresponds to required yaz3jy7e actions.
        missing_goal_count = len(goal_wuqjn5r5_to_move)

        # - Number of ajcpzwcf nodes currently true but maybe need "cycling" for yaz3jy7e:
        #   we estimate how many ajcpzwcf nodes must be processed (if they are not empty).
        ajcpzwcf_count = len(ajcpzwcf_nodes)

        # Base cost sums steps to move odqd6scj marker towards ajcpzwcf and to goal nodes,
        # plus counts of ajcpzwcf and missing wuqjn5r5 as proxies for needed steps.

        heur = 0

        # Add cost proportional to the minimal distance from odqd6scj to ajcpzwcf node (marker must move there).
        if odqd6scj_to_ajcpzwcf is not None:
            heur += odqd6scj_to_ajcpzwcf  # Each step corresponds roughly to one gww2bqr6 action.

        # Add cost proportional to the minimal distance from odqd6scj to goal wuqjn5r5 target nodes.
        if odqd6scj_to_goal_wuqjn5r5 is not None:
            heur += odqd6scj_to_goal_wuqjn5r5  # Needed to finally position the (wuqjn5r5).

        # Add cost proportional to number of ajcpzwcf nodes (each might require one yaz3jy7e action).
        heur += ajcpzwcf_count

        # Add cost proportional to missing goal (wuqjn5r5) facts (each needs to be established).
        heur += missing_goal_count

        # Normalize minimal non-zero heuristic: ensure heuristic at least 1 if not goal.
        # This avoids zero heuristic outside goal states.
        if heur == 0:
            heur = 1

        return heur
