from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(k21e3pdq a b c)" -> ["k21e3pdq", "a", "b", "c"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., "(k21e3pdq a b c)".
    - args: The expected pattern (wildcards * allowed).
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' domain.

    Summary:
    This heuristic estimates the number of actions needed to reach the goal state from the current
    state in the secret_domain. The main challenge in the domain is to move the predicate "odqd6scj" 
    (which can be interpreted as a "location" or "state flag") along a chain of z1mgqxkc objects, 
    using the actions provided. The goal requires that a specific "wuqjn5r5" predicate between a 
    kukxix7x object and a z1mgqxkc object holds true.
    
    The heuristic estimates cost based on the distance (in terms of hops via static "k21e3pdq" relations)
    needed to move the unique "odqd6scj" fact to an object where "wuqjn5r5" holds for the given kukxix7x
    and z1mgqxkc pair in the goal, combined with some additional costs related to required "ajcpzwcf"
    and "wuqjn5r5" facts needed along the way.
    
    Assumptions:
    - Exactly one fact with predicate "odqd6scj" is true at any time (represents a "current location").
    - The static relation "k21e3pdq" defines edges between z1mgqxkc objects, parameterized by a yrvdqfln label.
    - The goal is defined by a single "wuqjn5r5" fact connecting a kukxix7x object with a z1mgqxkc object.
    - Costs are roughly approximated by minimal path lengths for transferring "odqd6scj" to the goal z1mgqxkc,
      plus costs to prepare and swap the required "ajcpzwcf" and "wuqjn5r5" facts.

    Heuristic Initialization:
    - Extract the target goal "wuqjn5r5" fact to identify the destination z1mgqxkc object and the kukxix7x object.
    - Parse static facts with predicate "k21e3pdq" to build a bidirectional graph of z1mgqxkc nodes.
    - Store all static "k21e3pdq" edges keyed by their start node for quick neighbor lookup.
    - No other static information is needed for the heuristic.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the goal z1mgqxkc object (destination for "odqd6scj") and kukxix7x object (for "wuqjn5r5").
    2. Identify the current z1mgqxkc object holding "odqd6scj" in the current state.
    3. If the state is already a goal state, heuristic = 0.
    4. Compute shortest number of hops along the undirected graph formed by "k21e3pdq" edges 
       between z1mgqxkc nodes to move the "odqd6scj" from current location to goal location. 
       Each "gww2bqr6" action moves the "odqd6scj" from a node to a neighbor.
       This corresponds directly to the minimum number of "gww2bqr6" actions needed.
    5. Additionally, consider the preparation and swapping caused by the "yaz3jy7e" action:
       - This action swaps "ajcpzwcf" predicates between two nodes and moves "wuqjn5r5" of the kukxix7x object.
       - The heuristic adds extra cost for each "ajcpzwcf" fact that must be altered on the path, approximated by 
         the number of ajcpzwcf nodes currently holding the predicate that are not already at the goal node.
    6. Estimate total cost as the shortest path length for moving "odqd6scj" plus the number of ajcpzwcf nodes not at goal.
    7. Return this sum as the heuristic value.
    """

    def __init__(self, task):
        """
        Extract goal information and build adjacency graph from static facts.

        - self.goal_kukxix7x: target kukxix7x object from goal wuqjn5r5 predicate.
        - self.goal_z1mgqxkc: target z1mgqxkc object from goal wuqjn5r5 predicate.
        - self.adjacency: dict mapping z1mgqxkc nodes to sets of neighbors (undirected).
        """
        self.goals = task.goals
        self.static = task.static

        # Initialize goal objects from the unique wuqjn5r5 goal predicate.
        self.goal_kukxix7x = None
        self.goal_z1mgqxkc = None
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "wuqjn5r5":
                _, kukxix7x_obj, z1mgqxkc_obj = parts
                self.goal_kukxix7x = kukxix7x_obj
                self.goal_z1mgqxkc = z1mgqxkc_obj
                break

        # Build adjacency graph from static k21e3pdq relations. The graph is undirected.
        self.adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq" and len(parts) == 4:
                start, end = parts[1], parts[2]
                self.adjacency.setdefault(start, set()).add(end)
                self.adjacency.setdefault(end, set()).add(start)

    def __call__(self, node):
        """
        Compute heuristic estimate for the number of actions to reach the goal.

        Steps:
        - If goal reached, return 0.
        - Identify current location of odqd6scj predicate.
        - Compute shortest path length from current odqd6scj location to goal_z1mgqxkc location via adjacency graph.
        - Count ajcpzwcf facts in the state whose objects differ from the goal location, contributing to extra cost.
        - Sum above values and return as heuristic.
        """
        state = node.state

        # 1) Return 0 if goal state reached
        if self.goals <= state:
            return 0

        # 2) Extract current odqd6scj location (there should be exactly one)
        current_odqd6scj_nodes = [get_parts(fact)[1] for fact in state if fact.startswith("(odqd6scj ")]
        if not current_odqd6scj_nodes:
            # No odqd6scj fact means no valid current location, heuristic high but finite
            return 1000
        current_node = current_odqd6scj_nodes[0]

        # 3) If current node equals goal, minimal cost due to other factors
        if current_node == self.goal_z1mgqxkc:
            # We might still need to fix wuqjn5r5 and ajcpzwcf predicates,
            # but since wuqjn5r5 must hold only at goal node between goal kukxix7x and goal z1mgqxkc,
            # we check if it already does and ajcpzwcf setup is adequate.

            # Check if "wuqjn5r5 {goal_kukxix7x} {goal_z1mgqxkc}" holds
            wuqjn5r5_goal_fact = f"(wuqjn5r5 {self.goal_kukxix7x} {self.goal_z1mgqxkc})"
            if wuqjn5r5_goal_fact in state:
                # Also check ajcpzwcf facts matching goal node, assume correct if many are there
                # Count ajcpzwcf facts mismatching goal node as cost
                ajcpzwcf_facts = [fact for fact in state if fact.startswith("(ajcpzwcf ")]
                # We cannot match ajcpzwcf to nodes trivially, but they represent some "free" flags on z1mgqxkc objects
                # The heuristic here assumes no extra ajcpzwcf fixes needed if goal wuqjn5r5 holds.
                return 0
            else:
                # Need to fix wuqjn5r5 predicate with an estimated cost 1
                return 1

        # 4) Compute shortest path length from current_node to goal_z1mgqxkc in adjacency graph
        path_length = self._shortest_path_length(current_node, self.goal_z1mgqxkc)

        # 5) Count ajcpzwcf predicates that are "active" (true) in nodes that differ from goal_z1mgqxkc
        # ajcpzwcf predicates indicate additional preparation costs related to swapping action yaz3jy7e
        ajcpzwcf_locations = [get_parts(fact)[1] for fact in state if fact.startswith("(ajcpzwcf ")]
        ajcpzwcf_mismatched = sum(1 for loc in ajcpzwcf_locations if loc != self.goal_z1mgqxkc)

        # 6) Approximate additional costs for fixing wuqjn5r5 predicates if not yet at goal
        # Count how many "wuqjn5r5" predicates of the goal kukxix7x exist on other nodes (usually 0 or 1)
        wuqjn5r5_facts = [fact for fact in state if fact.startswith("(wuqjn5r5 ") and f"(wuqjn5r5 {self.goal_kukxix7x} "* in fact]
        # We'll charge 1 cost if desired wuqjn5r5 not established on the goal node yet
        wuqjn5r5_cost = 0 if f"(wuqjn5r5 {self.goal_kukxix7x} {self.goal_z1mgqxkc})" in state else 1

        # 7) Sum costs: path moves + ajcpzwcf fixes + wuqjn5r5 fixes
        heuristic_value = path_length + ajcpzwcf_mismatched + wuqjn5r5_cost

        # Ensure heuristic is finite for solvable states, else large but finite number
        if heuristic_value < 0:
            heuristic_value = 1000

        return heuristic_value

    def _shortest_path_length(self, start, goal):
        """
        Compute shortest path length in undirected graph self.adjacency from start to goal nodes.
        If no path exists, return a large number.
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
            for neighbor in self.adjacency.get(current, ()):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        # No path found
        return 1000
