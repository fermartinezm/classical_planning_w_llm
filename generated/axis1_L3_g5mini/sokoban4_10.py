from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing the surrounding
    parentheses and splitting on whitespace.

    Example:
      "(served passenger1 passenger2 floor3)" -> ["served", "passenger1", "passenger2", "floor3"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `miconic` (secret_domain) elevator-like domain.
    The heuristic estimates the number of actions required to reach the goal(s) that
    only involve changing the mapping expressed by `origin <passenger-2> <passenger>`.
    It uses the static `served` relations to build a graph over passenger nodes and
    computes shortest-path distances between the current `origin` mapping and the
    goal passenger(s). Each graph hop is mapped to an approximate number of actions
    required in the real domain (a small constant factor) to produce an informative
    guide for greedy best-first search.

    Assumptions
    -----------
    - Goals reference only `origin` facts of the form (origin <passenger-2> <passenger>).
    - The static facts contain `served` relations that connect passenger nodes; these
      are treated as (undirected) adjacency edges for estimating how many steps are
      required to move an `origin` mapping from one passenger to another.
    - The planner's state always contains an `origin` fact for each passenger-2 object
      referenced in the goals (if missing, a large but finite penalty is used).
    - The heuristic is not required to be admissible; it aims to be informative and
      cheaply computable.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: the goal facts (task.goals).
    - self.served_graph: an adjacency dictionary mapping passenger node -> set(neighbors)
      constructed from all static `served` facts of the form:
        (served <passengerA> <passengerB> <floor>)
      Both <passengerA> and <passengerB> are added as graph nodes and an undirected
      edge is created between them.
    - self.all_nodes_count: number of distinct passenger nodes seen in the served graph,
      used to compute a finite large penalty for unreachable goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's facts already satisfy all goals, return 0.
    2. Build a mapping current_origin_map from each passenger-2 object present in the state's
       `origin` facts to its currently mapped passenger node.
    3. For each goal of the form (origin P2_goal target_passenger):
       a. Find the current passenger node cur = current_origin_map.get(P2_goal).
          - If cur is equal to target_passenger, the cost contribution is 0.
          - If cur is missing, treat as unknown and assign a finite large penalty.
       b. Compute the shortest path distance `d` (in number of hops) between cur and
          target_passenger on the served_graph using BFS.
          - If a path exists, d is the number of edges (hops).
          - If no path exists, use a finite large penalty proportional to number of nodes.
       c. Convert the hop-distance into an action estimate. Empirically, moving the
          origin one hop typically requires a small fixed number of domain actions
          (e.g., repositioning the lift, swapping who is boarded, and performing the
          up operation). We therefore map each hop to `HOP_COST = 2` actions to obtain
          a simple, informative numeric heuristic. (This multiplier is intentionally
          optimistic but not required to be admissible.)
    4. Sum the contributions of all goal origin mappings to obtain the final heuristic
       value. The returned value is always a finite integer.

    Notes
    -----
    - The heuristic is deliberately simple and fast: BFS on a static small graph and
      basic lookups in the current state. This keeps evaluation cheap while being
      significantly more informative than a trivial heuristic.
    - If desired, the HOP_COST can be tuned (or made state-dependent) to reflect
      more accurately the real action cost per hop; for guiding greedy best-first
      search a fixed small multiplier often suffices.
    """

    HOP_COST = 2  # approximate number of actions per graph hop (tunable)

    def __init__(self, task):
        """
        Extract static `served` relations and goal origin facts.

        Parameters
        ----------
        task : Task
            The planning task object provided by the planner infrastructure.
        """
        # Save goal set
        self.goals = task.goals

        # Build undirected adjacency graph from static 'served' facts.
        self.served_graph = {}  # node -> set of neighbors
        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "served" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                # add nodes and undirected edge
                self.served_graph.setdefault(a, set()).add(b)
                self.served_graph.setdefault(b, set()).add(a)

        # number of nodes for fallback penalty
        self.all_nodes_count = len(self.served_graph)

    def _bfs_distance(self, start, target):
        """
        Compute shortest-path distance (number of edges) between start and target in the
        served_graph using BFS. If either node is missing or unreachable, return None.
        """
        if start == target:
            return 0
        if start not in self.served_graph or target not in self.served_graph:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self.served_graph.get(node, ()):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Return
        ------
        int
            Estimated number of actions to reach the goal from node.state.
        """
        state = node.state

        # If goal already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Build current origin mapping from passenger-2 -> passenger from the current state
        current_origin = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                p = parts[2]
                current_origin[p2] = p

        total_cost = 0
        # finite large penalty for unreachable / missing origin facts
        fallback_penalty = max(1, self.all_nodes_count) * self.HOP_COST + 5

        # For each origin goal, estimate cost
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue
            if parts[0] != "origin" or len(parts) < 3:
                # we only handle origin goals in this heuristic; others are treated as 0-cost
                continue
            p2_goal = parts[1]
            target_passenger = parts[2]

            cur_passenger = current_origin.get(p2_goal)
            if cur_passenger is None:
                # missing origin in state: apply fallback finite penalty
                total_cost += fallback_penalty
                continue

            if cur_passenger == target_passenger:
                # already satisfied for this passenger-2
                continue

            # Compute graph distance (number of hops)
            d = self._bfs_distance(cur_passenger, target_passenger)
            if d is None:
                # unreachable in static graph: use a finite large penalty
                total_cost += fallback_penalty
            else:
                # map hops to action estimate
                total_cost += d * self.HOP_COST

        # Ensure non-negative integer
        if total_cost < 0:
            total_cost = 0
        return int(total_cost)