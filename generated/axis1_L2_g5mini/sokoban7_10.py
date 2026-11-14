from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact as a list (without surrounding parentheses)."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5). The domain models
    moving objects of type kukxix7x between locations of type z1mgqxkc using
    a small set of actions. The static predicate `k21e3pdq` encodes directed
    edges between locations; goals are typically `wuqjn5r5 <kuk> <location>`
    facts that require bringing a specific kukxix7x object to a target location.
    This heuristic estimates the number of elementary moves required to place
    each goal object at its goal location by approximating the required number
    of single-step moves along the static k21e3pdq graph.

    Assumptions
    -----------
    - The important structural information is encoded in static k21e3pdq facts:
      (k21e3pdq src dst label). We treat these as directed edges src -> dst.
    - Each application of the domain action that moves a kukxix7x object
      advances it along one directed edge (this is the main contributor to cost).
    - Secondary constraints (presence and movement of the odqd6scj token,
      ajcpzwcf flags, and matching labels) can add extra overhead in reality,
      but modeling those precisely is expensive. We therefore approximate the
      cost by the graph distance (number of directed edges) between current and
      goal locations. If a directed path is not available, we fall back to an
      undirected path (still gives a finite cost estimate).
    - If the heuristic cannot find any node for a required object, a large but
      finite penalty is returned to guide search away from such states.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Store the set of goal facts (task.goals).
    - Parse static facts and build:
      * a directed adjacency map self.adj where self.adj[src] is a list of dst
        locations such that (k21e3pdq src dst label) exists.
      * an undirected adjacency map self.undir_adj used as a fallback if a
        directed path does not exist.
    - Extract the subset of goals that are `wuqjn5r5` predicates (these are
      the locations we try to reach for specific kukxix7x objects).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state) the heuristic works as follows:

    1. If the state's facts already satisfy all task goals, return 0.

    2. Build a mapping of the current locations of kukxix7x objects by scanning
       the state's facts for `wuqjn5r5` predicates: `(wuqjn5r5 <kuk> <loc>)`.

    3. For each `wuqjn5r5` goal (i.e., each goal that requests a particular
       kukxix7x object at a specific location):
       a. Determine the object's current location using the mapping from step 2.
          - If the mapping does not contain the object (no wuqjn5r5 fact in the
            state for that object), we consider that as an unknown location and
            add a finite large penalty to the heuristic (so the value stays finite).
       b. Compute the shortest directed-path distance from the object's current
          location to the goal location using BFS on the directed adjacency map.
       c. If no directed path exists, compute a shortest path on the undirected
          adjacency map (fallback). If still no path exists (disconnected graph),
          add a large finite penalty.
       d. The cost contribution for this goal is the number of edges in the
          found path (each edge corresponds to at least one domain action to
          advance the object by one step).

    4. Sum the individual costs for all unsatisfied `wuqjn5r5` goals and return
       that sum as the heuristic estimate.

    Notes on quality and admissibility:
    - The heuristic is not admissible: it underestimates in many situations by
      ignoring the token and flag management and possible interactions between
      multiple objects. However it is cheap to compute and often informative,
      because the main cost in this domain is usually moving objects along the
      static graph, which the heuristic captures.
    """

    def __init__(self, task):
        # Store goals for quick membership checks.
        self.goals = task.goals
        static_facts = task.static

        # Build directed adjacency from static k21e3pdq facts.
        # k21e3pdq facts have form: (k21e3pdq src dst label)
        self.adj = {}
        self.undir_adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] != "k21e3pdq":
                continue
            _, src, dst, _label = parts
            self.adj.setdefault(src, []).append(dst)
            # undirected adjacency
            self.undir_adj.setdefault(src, []).append(dst)
            self.undir_adj.setdefault(dst, []).append(src)

        # Extract wuqjn5r5 goals (object -> goal_location)
        self.wuq_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                self.wuq_goals[obj] = loc

        # A finite large penalty for unreachable/unknown situations.
        # Keep it moderate so the heuristic remains useful but finite.
        self.unreachable_penalty = max(50, len(self.adj) + 10)

    def _bfs_distance(self, start, target, directed=True):
        """
        BFS returning the minimal number of edges from start to target.
        If directed is True, uses directed adjacency self.adj; otherwise uses undirected.
        Returns None if no path exists.
        """
        if start == target:
            return 0
        adj = self.adj if directed else self.undir_adj
        if start not in adj and start not in (target,):
            # If start has no outgoing edges (in directed mode), quick reject.
            if directed:
                return None
        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            neighbors = adj.get(node, [])
            for nb in neighbors:
                if nb in visited:
                    continue
                if nb == target:
                    return dist + 1
                visited.add(nb)
                queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Returns an integer estimate of the number of actions to reach a goal state.
        """
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map current locations of kukxix7x objects from wuqjn5r5 facts in the state.
        current_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                current_loc[obj] = loc

        total_cost = 0

        # For each goal that requires a wuqjn5r5 fact, estimate cost to bring it about.
        for obj, goal_loc in self.wuq_goals.items():
            # If the specific goal fact is already true in the state, skip it.
            # (We already checked goal subset overall, but handle partial completion.)
            goal_fact = f"(wuqjn5r5 {obj} {goal_loc})"
            if goal_fact in state:
                continue

            # Find object's current location; if unknown, add penalty.
            if obj not in current_loc:
                total_cost += self.unreachable_penalty
                continue
            start_loc = current_loc[obj]

            # Try directed shortest path first (most faithful to domain).
            dist = self._bfs_distance(start_loc, goal_loc, directed=True)
            if dist is None:
                # Fallback to undirected connectivity (may represent reversing moves).
                dist = self._bfs_distance(start_loc, goal_loc, directed=False)

            if dist is None:
                # As a last resort, add the unreachable penalty.
                total_cost += self.unreachable_penalty
            else:
                # Each edge traversal typically requires at least one domain action
                # that advances the object by one step; we use distance as cost.
                total_cost += dist

        return int(total_cost)