from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates how many actions are needed to reach a goal state by focusing on the core "odqd6scj"
    predicate that is transferred through states, and the "wuqjn5r5" predicate that must hold in the goal.
    The heuristic tracks the position of the object involved in the goal predicate (the z1mgqxkc object),
    measures the steps needed to "move" the odqd6scj fact starting from its current location towards the goal object,
    and accounts for required conversions of ajcpzwcf and wuqjn5r5 predicates accordingly.

    Assumptions:
    - The goal requires achieving (wuqjn5r5 ip8xvrv6 <goal-z1-object>).
    - The "odqd6scj" predicate represents a kind of "position marker" among z1mgqxkc objects.
    - The "k21e3pdq" facts define a static directed graph between z1mgqxkc objects, parameterized by a yrvdqfln constant.
    - Actions "gww2bqr6" and "yaz3jy7e" enable moving or swapping the "odqd6scj" fact along edges and changing "wuqjn5r5" relations.
    - The heuristic estimates the minimal number of edge traversals needed to move the "odqd6scj" fact from its
      current location to the relevant goal location in the static graph.
    - ajcpzwcf facts likely indicate special states of z1mgqxkc objects required or affected during the moves.
    - Each graph edge traversal corresponds approximately to one or two actions in the domain.

    Heuristic Initialization:
    - Extract the two key constants of the problem: the "ip8xvrv6" kukxix7x object appearing in the goal wuqjn5r5 predicate,
      and the target z1mgqxkc object appearing in the goal wuqjn5r5.
    - Extract all static edges from k21e3pdq facts as a directed multigraph keyed by (from_node, yrvdqfln value) to to_node.
      We ignore the yrvdqfln for general graph construction, but remember it to refine heuristic approximation.
    - Build a directed graph from the static facts ignoring yrvdqfln type since transitions in the domain depend on it,
      but the heuristic assumes that an edge exists if any yrvdqfln label connects the nodes.
    - Extract the goal wuqjn5r5 fact, to identify the target z1mgqxkc node.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the goal is already reached (the goal wuqjn5r5 fact is in the current state), return heuristic 0.
    2. Identify the current location of the "odqd6scj" predicate in the state. There should be exactly one such fact.
    3. Consider the static graph of z1mgqxkc nodes representing possible "moves" between locations.
    4. Find the shortest path length in the static graph from the current "odqd6scj" location node to the goal node from the goal wuqjn5r5 fact.
       If no path exists, heuristic returns a large finite number.
    5. Approximate the cost of each edge traversal:
       - Each move along an edge requires at least one action ("gww2bqr6").
       - In some cases, a "yaz3jy7e" action might be needed involving ajcpzwcf predicates and wuqjn5r5 swapping.
       - Since the heuristic is not required to be admissible, approximate each edge with a cost of 2 actions for safety.
    6. Add an estimated cost if ajcpzwcf facts need to be toggled for the nodes involved, approximated by the count of ajcpzwcf predicates for nodes on the path.
       Since determining exact ajcpzwcf changes per path is costly, the heuristic only adds a small fixed overhead per edge.
    7. Return the sum as an estimate of required moves to reach the goal.

    This heuristic exploits the static k21e3pdq relations to estimate the minimal number of moves
    of the "odqd6scj" fact needed to arrive at the goal node, which should closely correlate with the number of required actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:

        - The static graph formed by k21e3pdq relations between z1mgqxkc objects.
        - The goal wuqjn5r5 predicate for identifying the target node.
        - The fixed kukxix7x object involved in wuqjn5r5 from the goal.
        """
        self.static_facts = task.static
        self.goals = task.goals

        # Extract the goal wuqjn5r5 predicate: (wuqjn5r5 ?kukxix7x ?z1mgqxkc)
        self.goal_wuq = None
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "wuqjn5r5":
                self.goal_wuq = goal
                _, kuk_obj, z1_obj = parts
                self.goal_kukxix7x = kuk_obj
                self.goal_z1mgqxkc = z1_obj
                break
        else:
            # If no wuqjn5r5 goal predicate - fallback, no heuristic info
            self.goal_wuq = None
            self.goal_kukxix7x = None
            self.goal_z1mgqxkc = None

        # Build a directed graph from k21e3pdq facts: edges from ?cvpg2 to ?ey5ma
        # Graph structure: { node: set(neighbor nodes) }
        self.graph = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq":
                # format: (k21e3pdq from-node to-node label)
                _, from_node, to_node, _label = parts
                if from_node not in self.graph:
                    self.graph[from_node] = set()
                self.graph[from_node].add(to_node)

        # Precompute reachable nodes from each node with shortest distances for quick heuristic
        # We'll lazily compute shortest paths as needed to save preprocessing time.

    def shortest_path_length(self, start, goal):
        """
        Compute the shortest path length from start to goal in the static graph using BFS.

        Return:
        - The number of edges in the shortest path if exists.
        - None if no path exists.
        """
        if start == goal:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            neighbors = self.graph.get(current, [])
            for n in neighbors:
                if n not in visited:
                    queue.append((n, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for a given state node.

        Steps:
        - Return 0 if goal wuqjn5r5 fact is present.
        - Find current location of odqd6scj predicate.
        - Estimate shortest path length from odqd6scj location to goal location.
        - Multiply path length by approximate cost per edge (2 actions).
        - Add a small constant overhead for adjustment.
        - Return heuristic value.
        """
        state = node.state

        # Goal reached check: heuristic is 0 only if goal wuqjn5r5 fact holds
        if self.goal_wuq in state:
            return 0

        # Find current odqd6scj location: fact (odqd6scj ?z1)
        odqd6scj_nodes = [get_parts(fact)[1] for fact in state if fact.startswith("(odqd6scj ")]
        if not odqd6scj_nodes:
            # Unreachable or invalid state; return large cost but finite
            return 1000
        odqd6scj_node = odqd6scj_nodes[0]

        # If no goal known or missing goal node, fallback large number
        if self.goal_z1mgqxkc is None:
            return 1000

        # Compute shortest path length in static graph from odqd6scj_node to goal_z1mgqxkc
        path_len = self.shortest_path_length(odqd6scj_node, self.goal_z1mgqxkc)

        if path_len is None:
            # No path found, very costly (but finite)
            return 1000

        # Each edge corresponds approximately to 2 actions in transition (gww2bqr6 or yaz3jy7e)
        cost_per_edge = 2

        # Additional overhead: number of ajcpzwcf facts currently true approximates extra toggling steps
        num_ajcpzwcf = sum(1 for fact in state if fact.startswith("(ajcpzwcf "))

        # Compute heuristic estimate
        # Use path length * cost_per_edge plus a small fraction of ajcpzwcf counts to refine estimate
        heuristic_value = path_len * cost_per_edge + (num_ajcpzwcf * 0.2)

        return heuristic_value
