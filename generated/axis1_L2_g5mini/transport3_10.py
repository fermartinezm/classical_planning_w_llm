from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact.strip()[1:-1].split()

def is_pred(fact, name):
    """Check whether a fact has the given predicate name."""
    return get_parts(fact)[0] == name

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain.

    The domain's goals in the provided problems are predominantly of the form
    (jft9ynda <object> <location>). The operator k12p0czo moves a jft9ynda
    fact from one hmhptaxe location to another when a z6o4oitz (location adjacency)
    relation is available. Thus many goal achievements reduce to transporting
    an object along the z6o4oitz graph. Other operators (sw2fg9ql, vyn92k55)
    can change links and create/move jft9ynda facts indirectly, but these
    involve more complex interactions. This heuristic focuses on the
    more direct movement along the z6o4oitz graph as the main cost driver.

    Assumptions
    -----------
    - The planner represents world facts as strings like "(predicate arg1 arg2)".
    - The most common goal predicate is "jft9ynda" (move an entity to a target hmhptaxe).
    - The action k12p0czo performs a single-step move of a jft9ynda subject
      along a z6o4oitz arc; therefore each graph edge traversal costs 1 action.
    - If an object has no current jft9ynda fact in the state, it can be made
      available for movement via the other operators, but that requires at
      least a small constant number of actions. We approximate this by a small fixed cost.
    - If there is no path in the z6o4oitz graph between current and goal
      locations, we assume the planner will have to use the more complex
      operators; we approximate this with a modest finite penalty.

    Heuristic Initialization
    ------------------------
    - Record the set of goal facts from task.goals (so heuristic returns 0 for goal states).
    - Extract static z6o4oitz facts from task.static and build an undirected
      adjacency graph between hmhptaxe locations. The graph is used to
      estimate the minimal number of k12p0czo moves (edge traversals).
    - No other static facts are required for this heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the state already satisfies all goal facts, return 0.
    2. Parse the state's jft9ynda facts and build a mapping current_location[obj] -> loc.
    3. For each goal fact of the form (jft9ynda obj goal_loc):
       a. If the exact goal fact is already present in the state, it requires 0 actions.
       b. Else, if we know a current location `curr` for `obj`:
          - Compute the shortest path length (number of edges) from `curr` to `goal_loc`
            in the static z6o4oitz adjacency graph using BFS.
          - If a path exists, the minimal number of k12p0czo moves equals that distance,
            so add the distance to the heuristic cost.
          - If no path exists (disconnected), add a finite penalty (treated here as 3).
            This accounts for the fact that alternative operators (sw2fg9ql/vyn92k55)
            can sometimes achieve the goal but typically cost multiple actions.
       c. Else (the object has no jft9ynda fact in the state):
          - Add a small finite cost (treated here as 2) to represent the actions
            needed to create/establish a jft9ynda fact for the object via other operators.
    4. Sum the per-goal estimates and return the total as the heuristic value.

    Notes
    -----
    - This heuristic is intentionally not required to be admissible; it aims to
      provide informative guidance for greedy best-first search by estimating
      the actual number of "move along z6o4oitz" steps and providing finite
      fallback costs when direct movement is not currently possible.
    - The constants (penalties for "no jft9ynda" or "disconnected") are small
      tunable integers chosen to keep values finite and reasonably informative.
    """

    def __init__(self, task):
        """Initialize by extracting goals and the z6o4oitz adjacency graph from static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build undirected adjacency graph of hmhptaxe locations from z6o4oitz static facts.
        self.adjacency = defaultdict(set)
        for fact in static_facts:
            if is_pred(fact, "z6o4oitz"):
                _, a, b = get_parts(fact)
                # Treat z6o4oitz as undirected adjacency for movement estimation.
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)

    def _bfs_distance(self, start, target):
        """Return shortest path length between start and target in adjacency graph, or None."""
        if start == target:
            return 0
        if start not in self.adjacency or target not in self.adjacency:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Compute heuristic estimate (integer) for node.state."""
        state = node.state

        # Quick exit: if goals already reached, heuristic is 0 (check 2 in checklist).
        if self.goals <= state:
            return 0

        # Map objects that currently have a jft9ynda fact to their location.
        current_loc = {}
        for fact in state:
            if is_pred(fact, "jft9ynda"):
                _, obj, loc = get_parts(fact)
                current_loc[obj] = loc

        total_cost = 0
        # Tunable fallback costs:
        MISSING_JFT9YNDA_COST = 2    # cost if object has no jft9ynda in state
        DISCONNECTED_PENALTY = 3     # cost if no path in adjacency graph

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate != "jft9ynda":
                # If other goal types exist, conservatively add 1 per unmet non-jft9ynda goal.
                if goal not in state:
                    total_cost += 1
                continue

            _, obj, goal_loc = parts
            # If already satisfied, no cost.
            if goal in state:
                continue

            if obj in current_loc:
                start_loc = current_loc[obj]
                dist = self._bfs_distance(start_loc, goal_loc)
                if dist is not None:
                    total_cost += dist
                else:
                    # No path via z6o4oitz; assume more complex operators needed.
                    total_cost += DISCONNECTED_PENALTY
            else:
                # Object currently has no jft9ynda fact: it must be brought into the
                # movement/placement machinery (via sw2fg9ql/vyn92k55), approximate cost.
                total_cost += MISSING_JFT9YNDA_COST

        return total_cost