from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into components."""
    return fact.strip()[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'miconic' (secret_domain) domain.
    The goal in this domain is expressed as origin facts of the form
    (origin <passenger-2> <passenger>). The only actions that change an
    origin mapping are 'up' actions that move the mapped passenger along
    neighbor relations given by the static 'served' facts. This heuristic
    estimates how many such "moves" are required to transform each current
    origin mapping into its goal passenger and sums those distances across
    all origin-goal obligations.

    Assumptions
    -----------
    - The static relation "served A B floorX" induces an (undirected)
      adjacency graph among passenger nodes. We treat served links as edges
      between passengers and ignore the floor label for distance computation.
    - Each 'up' operation can be viewed (abstractly) as moving an origin
      one step along that adjacency graph. The heuristic therefore uses
      shortest-path (graph) distance between the current mapped passenger
      and the goal passenger as the number of required 'up' steps.
    - The planner state maintains origin facts for all passenger-2 objects
      referenced by the goals (if not, the heuristic treats that case as
      unreachable and returns a large finite estimate).
    - We do not attempt to model precisely extra auxiliary actions (like
      depart moves to reposition boarded tokens or lift-at positioning).
      The heuristic intentionally focuses on the number of origin-movement
      steps, which provides a compact, efficiently computable and
      informative estimate for greedy best-first search.

    Heuristic Initialization
    ------------------------
    During construction the heuristic:
    - Extracts all goal origin requirements from task.goals and stores a
      mapping from passenger-2 object to target passenger.
    - Parses the static facts (task.static) and collects all 'served'
      facts into an adjacency dictionary mapping each passenger to its
      neighbor passengers (undirected).
    - Keeps the set of passengers that appear in the adjacency dictionary
      to validate reachability.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node/state:
    1. If all goal origin facts are already satisfied in the state, return 0.
    2. For each goal (origin P2 -> goal_passenger):
       a. Find the current mapping in the state: the fact "(origin P2 current)".
          - If no such current mapping exists in the state, the heuristic
            returns a large finite estimate (treat as effectively unreachable).
       b. Compute the shortest-path distance (number of edges) between
          current and goal in the undirected adjacency graph derived from
          'served' static facts using BFS.
          - If there is no path, return a large finite estimate.
       c. Add that distance to the running heuristic total.
    3. Return the summed distance across all origin requirements.
    Notes:
    - The heuristic returns an integer >= 0.
    - The heuristic focuses on the minimal number of origin-moving steps
      (up-operations) and does not try to exactly count supporting actions
      (depart, repositioning of lift-at, etc.). This makes it fast to
      compute while still informative for guidance in search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with information extracted from the task.

        Parameters:
        - task: a Task object, which provides .goals (set of goal facts as strings)
                and .static (set of static facts as strings).
        """
        # Store goal origin mapping: passenger-2 -> passenger
        self.goal_map = {}  # e.g., {'passenger50': 'passenger11'}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2, passenger = parts[1], parts[2]
                self.goal_map[p2] = passenger

        # Build adjacency graph from static 'served' facts.
        # Served facts: (served p3 p4 floorX) -> connect p3 <-> p4
        self.adj = {}  # passenger -> set(neighbor_passengers)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 4:
                a, b = parts[1], parts[2]
                if a not in self.adj:
                    self.adj[a] = set()
                if b not in self.adj:
                    self.adj[b] = set()
                self.adj[a].add(b)
                self.adj[b].add(a)

        # A large finite cost used to indicate unreachable/very expensive states.
        self.INF_COST = 9999

    def _shortest_distance(self, start, goal):
        """
        Compute BFS shortest-path distance (number of edges) between start and goal
        in the adjacency graph self.adj. If start==goal returns 0. If either
        node is unknown or goal not reachable, returns None.
        """
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Procedure:
        - If all origin goals are present in the node.state return 0.
        - Otherwise, for each required origin mapping (p2 -> goal_passenger),
          find the state's current mapping (origin p2 current). Use BFS on the
          served graph to compute distance from current to goal. Sum distances.
        - If any mapping is missing or unreachable in the graph, return INF_COST.

        Returns:
        - Integer heuristic estimate (0 for goal states; finite positive otherwise).
        """
        state = node.state  # frozenset of PDDL fact strings

        # Quick goal check: if every goal fact is already in the state, heuristic is 0.
        if self.goal_map:
            all_goals_present = True
            for p2, goal_passenger in self.goal_map.items():
                fact = f"(origin {p2} {goal_passenger})"
                if fact not in state:
                    all_goals_present = False
                    break
            if all_goals_present:
                return 0

        total_distance = 0

        # For each required origin mapping, find current mapping and compute distance.
        for p2, goal_passenger in self.goal_map.items():
            # Find current mapping fact "(origin p2 current)" in the state.
            current = None
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "origin" and len(parts) >= 3 and parts[1] == p2:
                    current = parts[2]
                    break

            if current is None:
                # No mapping present in state for this passenger-2 -> treat as unreachable.
                return self.INF_COST

            if current == goal_passenger:
                # Already at desired passenger; zero cost for this mapping.
                continue

            dist = self._shortest_distance(current, goal_passenger)
            if dist is None:
                # Unreachable in the served graph (or missing nodes) -> very costly.
                return self.INF_COST

            # Add the number of origin-moving steps required (one 'up'-like move per edge).
            total_distance += dist

        return total_distance