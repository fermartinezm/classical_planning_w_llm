from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string: remove surrounding parentheses and split."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain.
    This heuristic focuses on achieving goal facts of the form
    (jft9ynda <object> <location>) which are the primary goals in provided
    instances. The heuristic estimates the number of actions required by
    approximating how many single-edge moves (k12p0czo) are necessary to
    transport each object from its current location to its goal location,
    using the static z6o4oitz predicate as the adjacency graph of locations.
    When an object has no explicit current jft9ynda location in the state,
    the heuristic provides a small finite optimistic cost to account for
    the (typically required) creation or placement of a jft9ynda fact.

    Assumptions
    -----------
    - The goals that matter are jft9ynda facts (this matches the example
      instances where the goal conjuncts are jft9ynda predicates).
    - The operator k12p0czo moves a jft9ynda fact along a single z6o4oitz
      adjacency (one edge per application). Therefore shortest path length
      in the z6o4oitz graph is a good proxy for the number of k12p0czo
      actions required.
    - Other operators (sw2fg9ql, vyn92k55) can affect jft9ynda as well,
      sometimes creating or relocating jft9ynda facts in one action, but
      modeling all interactions precisely is costly. The heuristic therefore
      uses the graph distance as the main signal and falls back to a small
      optimistic constant cost when the object's current jft9ynda fact is
      not present in the state.
    - The z6o4oitz static facts form the connectivity graph of the
      "location" (hmhptaxe) objects. Edges are treated as undirected for
      pathfinding (the static facts in examples include both directions,
      but treating them as undirected is robust).

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Extract all z6o4oitz static facts and build an adjacency dictionary
      mapping each location node to its neighbors. Edges are inserted
      as undirected.
    - No heavy precomputation beyond the adjacency graph is performed.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's facts already satisfy all goals, return 0.
    2. Build a mapping of current jft9ynda facts in the state:
         object -> current_location
       by scanning all facts of the form (jft9ynda X Y).
    3. For each goal fact that is not yet satisfied (i.e., not present
       in the state), parse it as (jft9ynda obj goal_loc) and estimate the
       cost to achieve it:
       a. If obj has a current jft9ynda location cur_loc, compute the
          shortest-path length between cur_loc and goal_loc in the
          z6o4oitz adjacency graph using BFS. That distance (number of
          edges) is used as the estimated number of k12p0czo moves.
       b. If either cur_loc or goal_loc is missing from the adjacency
          graph (no static z6o4oitz info), or BFS finds no path, the
          heuristic uses a small finite optimistic penalty (UNSEEN_COST)
          to remain finite and optimistic.
       c. If the object has no current jft9ynda fact in the state, an
          optimistic small creation cost (MISSING_OBJ_COST) is added
          to account for the fact that some operator (sw2fg9ql or
          vyn92k55) will likely create/assign a jft9ynda fact. This
          cost is intentionally small to avoid overestimating while
          keeping the heuristic finite.
    4. The heuristic value returned is the sum of the per-goal estimates.
       This is not guaranteed to be admissible (it is optimistic and
       domain-specific), but is cheap to compute and provides informative
       guidance toward placing objects at the requested locations.

    Notes on constants:
    - UNSEEN_COST: cost when the adjacency graph lacks the relevant nodes
      or no path exists. Set to a moderate small finite value.
    - MISSING_OBJ_COST: cost when an object currently lacks any jft9ynda
      fact. Set to a small optimistic constant.
    """

    MISSING_OBJ_COST = 2   # optimistic cost to create/assign a jft9ynda for an object missing it
    UNSEEN_COST = 6        # optimistic cost when no graph path is known (keeps heuristic finite)

    def __init__(self, task):
        """Extract goals and static adjacency information from the task."""
        self.goals = task.goals  # set of goal fact strings
        self.static = task.static

        # Build adjacency graph from static z6o4oitz facts.
        # Treat edges as undirected for pathfinding robustness.
        self.adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "z6o4oitz" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)

    def _bfs_distance(self, start, goal):
        """
        Compute shortest path distance (in number of edges) between start and goal
        in the adjacency graph. Return None if either node is missing or goal
        is unreachable.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Return the estimated number of actions to reach the goal from node.state."""
        state = node.state

        # 1) If goal already satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # 2) Build current mapping of jft9ynda facts: object -> location
        current_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                current_loc[obj] = loc

        total_cost = 0

        # 3) For each jft9ynda goal not satisfied, estimate cost
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts or parts[0] != "jft9ynda" or len(parts) < 3:
                # If goals include other predicates, conservatively add 1 (rare in instances).
                if goal_fact not in state:
                    total_cost += 1
                continue

            obj, goal_loc = parts[1], parts[2]
            if goal_fact in state:
                # Already achieved
                continue

            if obj in current_loc:
                start_loc = current_loc[obj]
                # Try BFS distance on the z6o4oitz graph
                dist = self._bfs_distance(start_loc, goal_loc)
                if dist is None:
                    # No known path or missing nodes -> optimistic fallback
                    total_cost += self.UNSEEN_COST
                else:
                    total_cost += dist
            else:
                # Object lacks a jft9ynda fact in the current state.
                # Optimistically assume it can be created and placed with small cost.
                total_cost += self.MISSING_OBJ_COST

        # Ensure we always return a finite integer
        return int(total_cost)