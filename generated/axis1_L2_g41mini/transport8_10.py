from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(jft9ynda pewaka9c cxv9w3bk)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (mtfkspuf domain).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal facts of the form (jft9ynda <g83opnlj> <hmhptaxe>) in the secret_domain.
    The domain's primary predicates and actions revolve around relations between objects of types g83opnlj, hodp29q0, r86varlt, and hmhptaxe, connected via predicates like jft9ynda, yarbdfgc, jf4qno7u, uypgj7u3, etc.

    Key observation:
    The goals are expressed as (jft9ynda ?obj1 ?obj2), where:
      - jft9ynda relates an object of type hodp29q0 (like ?pk9o1) to an object of type hmhptaxe (like ?oqsrn).
    The main effect of actions sw2fg9ql, k12p0czo, and vyn92k55 is to change these relationships and other linked predicates involving those objects.

    Assumptions:
    - The main progress towards goals requires achieving the correct (jft9ynda ?obj ?loc) predicates matching the goals.
    - Changing these predicates requires changing connections through the sw2fg9ql, k12p0czo, and vyn92k55 actions.
    - The problem can be simplified by focusing on discrepancies between the current state's (jft9ynda ...) facts and the goal's (jft9ynda ...) facts.
    - We leverage static connectivity information from (z6o4oitz ...) predicates forming a graph over hmhptaxe objects (like locations).
    - The heuristic counts, for each goal (jft9ynda g83opnlj_obj hmhptaxe_obj), how many steps are needed to transform the current position of g83opnlj_obj to the goal position by:
      * tracking the current hmhptaxe location linked via (jft9ynda g83opnlj_obj ?loc)
      * estimating shortest distance in the location graph to the goal location.
    - Each difference or distance roughly corresponds to applying one or more actions to move the object or adjust predicate relationships.

    Heuristic Initialization:
    - Extract from static facts the adjacency graph of hmhptaxe objects via (z6o4oitz loc1 loc2) which is symmetric (in examples appears bidirectional).
    - Extract the set of goal (jft9ynda ...) facts.
    - Precompute adjacency for efficient shortest-path calculations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse static facts:
       - Build an undirected graph G=(V,E) over hmhptaxe objects, where edges exist according to (z6o4oitz loc1 loc2).
    2. For the current state:
       - For every g83opnlj object O that appears in the goal as (jft9ynda O goal_loc):
         - Determine its current location cur_loc by checking (jft9ynda O cur_loc) in the state.
         - If not present, estimate cost as large fixed penalty (e.g., 5).
         - Otherwise, compute shortest distance on the location graph from cur_loc to goal_loc.
    3. Sum over all such objects the shortest distances.
       - Since one move action can move/modify location predicates, distance corresponds to minimal steps needed.
    4. Additionally, check if any goal (jft9ynda O loc) is currently false in the state (the predicate is missing), count at least 1 action to generate it.
    5. The heuristic value is the sum of these estimated minimal required action counts to achieve all goal facts, providing an estimate for a layered sequence of moves.

    Note:
    - Distances are computed with a simple BFS on static adjacency because location connectivity is static.
    - This heuristic is not necessarily admissible but aims to provide a good estimate guiding greedy best-first search.
    - The heuristic is 0 exactly if all goal (jft9ynda ...) predicates hold in the current state.

    """

    def __init__(self, task):
        # Extract goal (jft9ynda ...) facts
        self.goal_jft9ynda = set(
            fact for fact in task.goals if fact.startswith("(jft9ynda ")
        )

        # Build adjacency graph from static facts (z6o4oitz ?loc1 ?loc2)
        self.location_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "z6o4oitz":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        state = node.state

        # Fast check: if state contains all goal jft9ynda facts, heuristic = 0
        if self.goal_jft9ynda <= state:
            return 0

        # Map: g83opnlj_obj -> current hmhptaxe location (from (jft9ynda g83opnlj hmhptaxe))
        current_positions = {}

        # Extract all (jft9ynda ?g83opnlj ?hmhptaxe) in the current state
        for fact in state:
            if fact.startswith("(jft9ynda "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    g83opnlj_obj, hmhptaxe_loc = parts[1], parts[2]
                    current_positions[g83opnlj_obj] = hmhptaxe_loc

        # For each goal (jft9ynda O goal_loc), compute estimated cost:
        # cost = shortest path length in location_graph between current_positions[O] and goal_loc
        # + 1 if (jft9ynda O goal_loc) not currently true (must be established)

        def shortest_path_distance(graph, start, goal):
            """Compute shortest distance between start and goal in undirected graph using BFS."""
            if start == goal:
                return 0
            if start not in graph or goal not in graph:
                return float("inf")
            from collections import deque

            visited = set()
            queue = deque([(start, 0)])
            while queue:
                loc, dist = queue.popleft()
                if loc == goal:
                    return dist
                visited.add(loc)
                for neighbor in graph.get(loc, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            return float("inf")

        total_estimated_cost = 0

        for goal_fact in self.goal_jft9ynda:
            _, g83obj, goal_loc = get_parts(goal_fact)
            # Check if goal is true in current state: if yes, no cost
            if goal_fact in state:
                continue

            # Get current location of g83obj
            cur_loc = current_positions.get(g83obj)
            if cur_loc is None:
                # No known position, assign penalty
                # Because the object must first be associated with some location by an action.
                distance = 5
            else:
                distance = shortest_path_distance(self.location_graph, cur_loc, goal_loc)
                if distance == float("inf"):
                    # No path => large penalty
                    distance = 5

            # We add 1 action for establishing the goal predicate itself if not present.
            cost = distance + 1
            total_estimated_cost += cost

        # Return finite heuristic value
        return total_estimated_cost
