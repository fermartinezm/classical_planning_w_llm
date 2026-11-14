from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state characterized
    by specific (wuqjn5r5 ?ndr6y ?el0fk) facts. Essentially, the goal requires certain
    kukxix7x elements (?ndr6y) to be related via wuqjn5r5 to a particular z1mgqxkc object (?el0fk).

    The domain involves:
    - A connectivity relation given by static facts (k21e3pdq ?el0fk1 ?el0fk2 ?yrvdqfln).
    - Two unary predicates odqd6scj and ajcpzwcf over z1mgqxkc objects where odqd6scj "moves" along
      a path defined by k21e3pdq, effectively shifting truth of odqd6scj from one z1mgqxkc object
      to another via action gww2bqr6.
    - The action yaz3jy7e modifies wuqjn5r5 facts by shifting them along similarly constrained
      z1mgqxkc objects.

    The goal predicates require establishing the wuqjn5r5 predicate for certain fixed kukxix7x
    objects and goal z1mgqxkc objects. Achieving wuqjn5r5 facts appears to depend on moving odqd6scj
    facts along paths and manipulating ajcpzwcf facts to enable wuqjn5r5 transitions.

    Assumptions
    -----------
    - Goal facts are only of form (wuqjn5r5 ?ndr6y ?el0fk), with ?ndr6y a fixed kukxix7x constant.
    - Paths between z1mgqxkc objects are defined by static k21e3pdq relations.
    - Movement of odqd6scj markers along these paths is sequential, and each move corresponds to one gww2bqr6 action.
    - Transition of wuqjn5r5 markers uses more complex actions and require ajcpzwcf conditions but can be abstracted as one step after odqd6scj is correctly positioned.
    - The heuristic counts approximate minimal movements of odqd6scj facts plus the number of wuqjn5r5 goals not yet satisfied.
    - ajcpzwcf predicates are assumed to be abundant and manipulable at low cost, so they do not overly impact cost estimation.
    - We estimate shortest path distances based on static connectivity for minimal odqd6scj moves.

    Heuristic Initialization
    ------------------------
    - Extract all goal (wuqjn5r5 ?ndr6y ?el0fk) facts.
    - Extract the static adjacency graph from (k21e3pdq ?from ?to ?const) facts (ignore the third arg for adjacency).
    - Compute an undirected graph of z1mgqxkc nodes with edges between ?from and ?to from k21e3pdq facts.
    - Goals are stored as desired (ndrx, el0fk) pairs.
    - Store the static graph for shortest path computations.
    - No assumptions on initial locations of odqd6scj or ajcpzwcf besides what's in the given state.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. If the node state satisfies all goals (all wuqjn5r5 goals included), return 0.
    2. Identify the current locations (z1mgqxkc objects) where odqd6scj holds.
       Multiple odqd6scj facts possible, track all positions.
    3. For each goal wuqjn5r5 fact (ndrx, el0fk):
       - If already present in state, cost for this goal is zero.
       - Else estimate minimal cost to "move" an odqd6scj marker along the graph from its closest current position
         to el0fk. This corresponds to minimal number of gww2bqr6 actions to move odqd6scj markers.
       - Since yaz3jy7e actions convert odqd6scj states to wuqjn5r5 shifts, count an additional fixed cost per goal
         for required yaz3jy7e action (heuristically 1) if goal not yet reached.
    4. Sum minimal odqd6scj movement costs plus one yoz3jy7e cost per unsatisfied wuqjn5r5 goal.
    5. Return total as heuristic estimate.

    The heuristic is greedy and non-admissible but intended to guide search efficiently,
    favoring states with odqd6scj markers near goal locations and reducing the number of expanded nodes.

    """

    def __init__(self, task):
        self.goals = set()
        # Extract goal (wuqjn5r5 ?ndr6y ?el0fk) facts only
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "wuqjn5r5" and len(parts) == 3:
                # Record as (kukxix7x_obj, z1mgqxkc_obj)
                self.goals.add((parts[1], parts[2]))

        # Build undirected graph from static adjacency k21e3pdq facts: (k21e3pdq ?from ?to ?_const)
        self.graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq" and len(parts) == 4:
                n1, n2 = parts[1], parts[2]
                self.graph.setdefault(n1, set()).add(n2)
                self.graph.setdefault(n2, set()).add(n1)

    def __call__(self, node):
        state = node.state

        # Goal check: if all wuqjn5r5 goals are satisfied return 0
        if all(f"(wuqjn5r5 {ndrx} {el0fk})" in state for (ndrx, el0fk) in self.goals):
            return 0

        # Find all current odqd6scj locations
        current_odqd6scj = {get_parts(fact)[1] for fact in state if match(fact, "odqd6scj", "*")}

        # If no odqd6scj marker present in state, heuristic is at least the number of goals (must create moves)
        if not current_odqd6scj:
            # Conservative cost: at least one move + one wuqjn5r5 action per goal not yet achieved
            return len(self.goals) * 2  # heuristic minimum cost

        # Precompute shortest path distances in graph using BFS for each current_odqd6scj node
        # We do this lazily per needed goal node below.

        def shortest_path_length(start, goal):
            if start == goal:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                for neighbor in self.graph.get(node, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # If no path, return large number (infinite cost)
            return 1000

        total_cost = 0
        for (ndrx, goal_node) in self.goals:
            goal_fact = f"(wuqjn5r5 {ndrx} {goal_node})"
            if goal_fact in state:
                # Goal predicate satisfied, zero cost
                continue

            # Estimate minimal odqd6scj movements needed:
            # Find minimal odqd6scj marker distance to goal_node
            min_move_cost = min(shortest_path_length(pos, goal_node) for pos in current_odqd6scj)

            # Each odqd6scj move corresponds to one action (gww2bqr6)
            # Plus one yaz3jy7e action needed to shift wuqjn5r5 once at location

            total_cost += min_move_cost + 1  # +1 for yaz3jy7e action to establish wuqjn5r5

        return total_cost
