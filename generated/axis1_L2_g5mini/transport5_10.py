from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a fact string '(pred a b ...)' as a list of tokens."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (mtfkspuf) domain.

    This heuristic focuses on goal facts of the form (jft9ynda <obj> <loc>) which
    appear in the provided example problems and in the domain's goal conditions.
    Intuitively, jft9ynda relates an object (val3ypio / g83opnlj) to a location
    (hmhptaxe). The operator k12p0czo moves a jft9ynda fact from one hmhptaxe
    to another when a static connectivity fact (z6o4oitz) exists between those
    locations. Therefore, a natural, informative estimate of how many actions
    are required to achieve a missing jft9ynda goal is the distance (in the
    z6o4oitz graph) from the object's current location to the goal location.

    Assumptions
    -----------
    - Goals that matter for guidance are of the form (jft9ynda <obj> <loc>).
    - Movement along hmhptaxe nodes is captured by static facts (z6o4oitz X Y).
      Each application of operator k12p0czo corresponds to one step along such
      an edge (directed as given by static facts). The static graph is used to
      compute shortest-path distances.
    - If an object does not currently have a jft9ynda fact in the state, or
      if no path exists between current and goal locations in the z6o4oitz
      graph, we return a finite penalty for that goal (so the heuristic
      remains finite).
    - The heuristic is intentionally optimistic w.r.t. pure movement cost
      (counts number of k12p0czo-like moves). It ignores more complex
      multi-object interactions that could make costs lower or higher; this
      keeps the heuristic fast and informative.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract all goal jft9ynda facts and map each object to its desired goal location.
    - Build a directed adjacency list (graph) from static facts of the form
      (z6o4oitz X Y). The graph may be asymmetric; we treat edges as directed
      according to the static facts. (Many problem files include both directions
      explicitly; treating them as directed preserves correctness.)
    - Optionally prepare a small distance cache used across calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node (with a state = frozenset of fact strings):

    1. If the state's facts already satisfy all goals, return 0.

    2. For each goal of the form (jft9ynda obj goal_loc):
       a) Find the object's current location by scanning state's facts for a
          fact whose predicate is jft9ynda and whose first argument equals obj.
          - If found, current_loc is the second argument of that fact.
          - If not found, we treat the object as 'unlocated' w.r.t. jft9ynda
            and assign a finite penalty (see below).
       b) If current_loc == goal_loc then this goal contributes 0.
       c) Otherwise compute the shortest-path distance (number of directed
          z6o4oitz edges) from current_loc to goal_loc using BFS on the static
          adjacency built at initialization.
          - If a path is found, the distance (an integer >= 1) is added to the
            heuristic sum (each step corresponds to a k12p0czo-like action).
          - If no path exists (disconnected), add a fixed finite penalty to
            the heuristic for this goal. The penalty is intended to be larger
            than a single edge step so disconnected goals are prioritized but
            the heuristic stays finite.

    3. The final heuristic value is the sum of the per-goal contributions.
       This yields 0 exactly when all goals are already present in the state,
       and a finite positive integer otherwise.

    Notes on Design Choices
    -----------------------
    - The heuristic purposely focuses on movement in the z6o4oitz graph because
      the k12p0czo operator provides an explicit, cheap way to move jft9ynda
      facts along z6o4oitz edges; these moves are often the dominant component
      needed to satisfy jft9ynda goals.
    - More complex operators (sw2fg9ql, vyn92k55) manipulate additional
      predicates (yarbdfgc, uypgj7u3, jf4qno7u) and can sometimes deliver a
      goal in fewer or more steps depending on the state. Modeling all such
      interactions would be expensive; the chosen approximation balances
      informativeness and efficiency.
    """

    def __init__(self, task):
        # Goals: map objects -> target location for jft9ynda predicates.
        self.goals = task.goals
        self.goal_jft9ynda = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.goal_jft9ynda[obj] = loc

        # Build adjacency list from static facts (z6o4oitz X Y).
        self.adj = defaultdict(list)  # directed adjacency: node -> [neighbors]
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "z6o4oitz" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj[src].append(dst)

        # Small cache for computed shortest distances between pairs of nodes:
        # (src, dst) -> distance (int) or None if unreachable.
        self._dist_cache = {}

        # Penalty for cases where no jft9ynda current location is found or
        # when goal is unreachable in the static graph.
        self._unreachable_penalty = 5

    def _bfs_distance(self, src, dst):
        """Return the directed shortest-path length from src to dst in the z6o4oitz graph.
        If unreachable, return None. Results cached."""
        if (src, dst) in self._dist_cache:
            return self._dist_cache[(src, dst)]

        if src == dst:
            self._dist_cache[(src, dst)] = 0
            return 0

        # Standard BFS on directed graph.
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            for neigh in self.adj.get(node, ()):
                if neigh == dst:
                    self._dist_cache[(src, dst)] = d + 1
                    return d + 1
                if neigh not in visited:
                    visited.add(neigh)
                    q.append((neigh, d + 1))

        self._dist_cache[(src, dst)] = None
        return None

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If goals are already satisfied, return 0.
        if self.goals <= state:
            return 0

        # Build mapping from objects to their current jft9ynda location (if any).
        current_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                current_loc[obj] = loc

        total = 0
        # For each jft9ynda goal, add estimated cost.
        for obj, goal_loc in self.goal_jft9ynda.items():
            # If goal already satisfied in state, skip.
            if f"(jft9ynda {obj} {goal_loc})" in state:
                continue

            # Determine current location (if available).
            if obj in current_loc:
                src_loc = current_loc[obj]
                # If same location, no cost.
                if src_loc == goal_loc:
                    continue
                # Compute graph distance.
                dist = self._bfs_distance(src_loc, goal_loc)
                if dist is None:
                    # No directed path: add penalty
                    total += self._unreachable_penalty
                else:
                    total += dist
            else:
                # Object has no jft9ynda fact in the state: assign penalty.
                total += self._unreachable_penalty

        # Ensure heuristic is at least 1 for non-goal states (helps greedy selection).
        return max(1, total) if not (self.goals <= state) else 0