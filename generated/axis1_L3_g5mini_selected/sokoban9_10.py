from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'miconic' (secret_domain) elevator-style domain.
    The domain models an elevator (lift) that can change "origin" pointers (origin <passenger-2> <passenger>)
    by using operators that move boarding status and the lift between passenger-nodes connected by
    static `served` relations. Goals require certain origin mappings to point to specific passenger nodes.
    This heuristic estimates how many operator applications are required to change each origin mapping
    from its current passenger node to the target passenger node specified by the goal.

    Assumptions
    -----------
    - The static facts of the task contain `served` facts of the form
      "(served passengerA passengerB floorX)". These define adjacency between passenger nodes.
      We treat these as undirected graph edges between passenger nodes.
    - Each `up` (or related) operator can effectively move the origin mapping along a single edge
      in that graph. Thus the number of edge traversals (shortest-path distance) between current
      and goal passenger nodes is a reasonable base estimate for the number of operator applications
      needed to re-point an origin.
    - There is no explicit "board" operator in the domain; `boarded` facts are moved around by
      other operators. At least one boarded passenger is required to perform `up` transitions in the real domain
      (per operator preconditions). If the current state contains no `boarded` facts, we add a small
      overhead because the planner must first create a boarded passenger by using the available operators.
    - We ignore detailed ordering/interaction effects between multiple origin goals and
      conservatively sum per-origin shortest-path distances. This is not admissible but gives a
      domain-aware, efficiently-computable estimate that usually guides greedy Best-First Search well.

    Heuristic Initialization
    ------------------------
    - Extract the goals that are `origin` predicates and construct a mapping:
        goal_origins: passenger-2 -> desired passenger
    - From static facts, build an undirected adjacency graph of passenger nodes using the `served` facts.
      Each `served a b floor` fact contributes an undirected edge between 'a' and 'b'.
    - Save the set of all passenger nodes seen in the `served` facts (graph nodes).
    - Keep the full set of goal fact strings (self.goals) so that heuristic is 0 when all goal facts are satisfied.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goal facts are already present in the state's facts, return 0.
    2. Extract the current mapping of origins found in the state: facts of the form "(origin passenger-2 passengerX)".
       For each goal origin (passenger-2 -> desired passenger), get the current passengerX.
       - If the origin fact is missing in the state, treat it as "unknown"; we conservatively estimate
         the cost as the maximum possible graph distance (a finite upper bound).
    3. For each goal origin where current != desired:
       - Compute the shortest-path distance between current passenger node and goal passenger node in the
         undirected served-graph using BFS. This distance is taken as the base number of "moves" needed.
       - If either node is not part of the static graph, treat the distance as a conservative finite upper bound.
    4. Sum the distances across all origin goals to produce the base heuristic.
    5. If the state contains no `(boarded ...)` facts, add a small penalty (1) because applying `up`-style transitions
       typically requires at least one boarded passenger to be present; this models the extra work to create one.
    6. Return the sum as the heuristic estimate (finite integer). The heuristic is intentionally non-admissible
       but tuned to reflect topological distances implied by the static `served` graph, which often reduces
       expansions in greedy search.

    Notes
    -----
    - The heuristic intentionally keeps computation cheap (BFS on a small graph per origin goal).
    - If the static `served` graph is disconnected and a goal passenger is unreachable from the current
      passenger, the heuristic returns a finite large bound (based on the number of known nodes).
    """

    def __init__(self, task):
        # The set of goal fact strings. If all these facts are present in a state, heuristic should return 0.
        self.goals = set(task.goals)

        # Static facts from the task (facts that do not change).
        static_facts = task.static

        # Build undirected graph of passenger nodes using static 'served' facts.
        # Each served fact: (served passengerA passengerB floorX)
        self.graph = {}  # node -> set(neighbors)
        self.nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                self.nodes.add(a)
                self.nodes.add(b)
                self.graph.setdefault(a, set()).add(b)
                self.graph.setdefault(b, set()).add(a)

        # Parse goal origin mappings into a dict: passenger-2 -> goal passenger
        self.goal_origins = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]  # passenger-2 identifier
                target = parts[2]  # passenger goal node
                self.goal_origins[p2] = target

        # Upper bound used when nodes are disconnected or unknown: use number of known nodes + 1
        self.unreachable_cost = max(1, len(self.nodes)) + 1

    def bfs_distance(self, start, goal):
        """Return shortest-path distance between start and goal in self.graph (undirected).
        If either node unknown or unreachable, return a finite large bound self.unreachable_cost."""
        if start == goal:
            return 0
        if start not in self.nodes or goal not in self.nodes:
            return self.unreachable_cost
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, dist = dq.popleft()
            for nb in self.graph.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    dq.append((nb, dist + 1))
        return self.unreachable_cost

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # 1) If all goal facts already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Extract current origins from state: map passenger-2 -> passenger
        current_origins = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                cur = parts[2]
                current_origins[p2] = cur

        # 3) Determine whether there is at least one boarded passenger in the state.
        boarded_exists = any(get_parts(f)[0] == "boarded" for f in state)

        total = 0
        # For each origin goal, compute cost to change current mapping to target mapping.
        for p2, target in self.goal_origins.items():
            cur = current_origins.get(p2)
            if cur is None:
                # Missing origin in current state; conservative finite cost.
                total += self.unreachable_cost
            else:
                total += self.bfs_distance(cur, target)

        # 4) If no boarded passenger exists, add a small penalty because an 'up'-like move
        # typically requires a boarded passenger to be present (domain-dependent nuance).
        if not boarded_exists:
            total += 1

        return int(total)