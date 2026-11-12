from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    E.g. '(k21e3pdq a b c)' -> ['k21e3pdq', 'a', 'b', 'c']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(wuqjn5r5 obj1 loc1)".
    - `args`: The expected pattern (wildcards '*' allowed).
    Return True if all parts match, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    Estimates the minimum number of actions needed to reach a goal state by analyzing 
    the current distribution of facts related to the predicates `wuqjn5r5`, `odqd6scj`, 
    `ajcpzwcf`, and `k21e3pdq` and their relations as encoded in static facts. The key 
    idea is to estimate how many transitions (via the two domain actions) are needed 
    to achieve the goal(s) `wuqjn5r5 <kukxix7x> <z1mgqxkc>`. The heuristic models 
    the propagation of `odqd6scj` states over a graph of `z1mgqxkc` nodes connected 
    via `k21e3pdq` edges, with constraints from `ajcpzwcf` and `wuqjn5r5`.
    
    Assumptions
    -----------
    - The goal facts are of the form `(wuqjn5r5 ?k ?z)` establishing a relation between  
      specific `kukxix7x` and `z1mgqxkc` objects.
    - The `odqd6scj` predicate marks the "current active" or "marker" nodes (`z1mgqxkc`),
      and actions move this marker from one node to another.
    - The `ajcpzwcf` predicate partitions `z1mgqxkc` objects in two "states" relevant 
      for action applicability.
    - The static predicate `k21e3pdq ?from ?to ?obj` defines edges between `z1mgqxkc` 
      nodes gated by the `yrvdqfln` object, which determines valid transitions.
    - The heuristic operates over these edges to count minimal steps needed for all goal 
      `wuqjn5r5` facts to hold, considering the rules enforced by the two operators.
    - Since heuristic admissibility is not required, an approximate "shortest path" 
      estimate on the underlying connectivity graph is computed for each goal.

    Heuristic Initialization
    ------------------------
    - Extract all goal `(wuqjn5r5 ?k ?z)` facts and store pairs `(k, z)` to satisfy.
    - Extract static facts `(k21e3pdq ?from ?to ?obj)` describing directed edges in a 
      graph keyed by `(from, to, obj)`.
    - For each `z1mgqxkc` object, collect adjacency info to allow searching neighbors.
    - No assumptions about `yrvdqfln` constants beyond usage in edges.
    - No full state-space search is done; only efficient lookups in static and current facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state satisfies all goal facts; if so, return 0.
    2. Extract from the current state:
       - The set of `odqd6scj`-marked nodes (marker positions).
       - The set of `ajcpzwcf` nodes (state of some objects).
       - The set of `wuqjn5r5` facts holding (current achieved goals).
    3. For each goal `(k, z)`:
       - If `(wuqjn5r5 k z)` already holds, no cost needed for that goal.
       - Otherwise, estimate the minimal "marker move" actions to reach a configuration 
         where `(wuqjn5r5 k z)` can hold by:
         * Checking if the marker is already at the target node `z` or nearby nodes.
         * Using edges `(k21e3pdq from to obj)` and predicates `odqd6scj`, `ajcpzwcf` 
           to compute shortest approximate distance from some `odqd6scj` node to `z`, 
           following valid transitions allowed by actions.
    4. Use a breadth-first search (BFS) or Dijkstra-like traversal over the graph induced 
       by edges to estimate the minimal steps needed to "propagate" `odqd6scj` to `z` 
       while respecting domain constraints.
    5. The heuristic is the sum over all goals of their respective estimated costs.
    6. Return the total as the heuristic estimate.

    This heuristic is efficiently computable using the static graph and current facts, 
    providing guidance to the search by estimating how many "marker moves" are needed 
    to satisfy the goal relations.
    """

    def __init__(self, task):
        # Store goal pairs (k, z) from wuqjn5r5 goals.
        self.goal_pairs = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "wuqjn5r5" and len(parts) == 3:
                self.goal_pairs.add((parts[1], parts[2]))

        # Build adjacency map keyed by (k, z) -> list of neighbors reachable with some yrvdqfln.
        # Since edges have: (k21e3pdq from to yrvdqfln)
        # We'll build a graph per k (z1mgqxkc) where edges go from `from` to `to`.
        # Note: yrvdqfln constants play a gating role but not used specifically here.
        self.edges_from = {}  # map: z1mgqxkc node -> list of (neighbor_node, yrvdqfln_const)
        self.edges_to = {}    # map: z1mgqxkc node -> list of (from_node, yrvdqfln_const)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq" and len(parts) == 4:
                src, dst, obj = parts[1], parts[2], parts[3]
                self.edges_from.setdefault(src, []).append((dst, obj))
                self.edges_to.setdefault(dst, []).append((src, obj))

    def __call__(self, node):
        state = node.state

        # Quickly return 0 if goal reached.
        if self.goal_pairs <= {(get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(wuqjn5r5")}:
            return 0

        # Extract sets of current relevant facts.
        odqd6scj_nodes = {get_parts(f)[1] for f in state if f.startswith("(odqd6scj")}
        ajcpzwcf_nodes = {get_parts(f)[1] for f in state if f.startswith("(ajcpzwcf")}
        wuqjn5r5_pairs = {(get_parts(f)[1], get_parts(f)[2]) for f in state if f.startswith("(wuqjn5r5")}

        # Heuristic cost accumulator
        total_cost = 0

        # Since actions move `odqd6scj` marker between nodes if certain conditions hold,
        # and the goal requires (wuqjn5r5 ?k ?z), the main challenge is to get odqd6scj marker
        # near z and have ajcpzwcf predicate switched suitably to enable wuqjn5r5.

        # Our approach:
        # For each goal pair (k,z), if already satisfied, cost=0.
        # Else, estimate minimal moves for odqd6scj marker to get "close" to z with required ajcpzwcf condition.

        # Precompute for BFS:
        # We model BFS over `z1mgqxkc` nodes with transitions allowed by edges (k21e3pdq),
        # but constrained by ajcpzwcf presence and odqd6scj marker position in current state.

        # For approximate cost, we do a BFS from all odqd6scj nodes simultaneously to goal z,
        # considering only nodes in ajcpzwcf (since the actions require ajcpzwcf on some nodes).
        # The cost counted is number of marker moves needed.

        for (k, goal_z) in self.goal_pairs:
            # If goal fact already holds, no cost.
            if (k, goal_z) in wuqjn5r5_pairs:
                continue

            # BFS initialization: start from odqd6scj nodes.
            # We find shortest distance from any odqd6scj node to goal_z.
            # Moves are transitions along edges adjacently connected,
            # Only nodes in ajcpzwcf can be intermediate steps (due to preconditions).

            if goal_z not in ajcpzwcf_nodes:
                # If goal_z is not in ajcpzwcf in current state, must first move odqd6scj or change ajcpzwcf (requires at least one step).
                # For simplicity, add 1 penalty for that condition.
                ajcp_switch_cost = 1
            else:
                ajcp_switch_cost = 0

            from collections import deque
            visited = set()
            queue = deque()

            # Start BFS from all odqd6scj nodes, cost 0
            for start_node in odqd6scj_nodes:
                visited.add(start_node)
                queue.append((start_node, 0))

            distance = None
            while queue:
                current_node, dist = queue.popleft()
                if current_node == goal_z:
                    distance = dist
                    break
                # From current_node, expand neighbors that are ajcpzwcf nodes (since action preconditions require ajcpzwcf),
                # or the goal node itself (already checked above).
                for (neighbor, _) in self.edges_from.get(current_node, []):
                    # Only consider neighbor nodes that have ajcpzwcf predicate or equal to goal_z (which might lack ajcpzwcf now).
                    if neighbor not in visited and (neighbor in ajcpzwcf_nodes or neighbor == goal_z):
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            # If cannot reach goal_z from any marker position, assign large cost but finite (e.g., max number of nodes).
            if distance is None:
                # Fail safe: heuristic large value for unreachable
                distance = len(self.edges_from) + 10

            # Actions needed roughly:
            # Each step in BFS corresponds to one marker move (gww2bqr6 action).
            # Also consider that to enable wuqjn5r5 goal from distance=0,
            # likely need one yaz3jy7e action to switch predicates.

            # Thus heuristic per goal approximated as:
            goal_cost = distance + 1 + ajcp_switch_cost  # 1 for possible predicate change

            total_cost += goal_cost

        return total_cost
