from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'miconic' (secret_domain) problem.
    The domain models movement of an "origin" pointer (predicate: (origin P2 P))
    along a passenger graph described by the static predicate (served P1 P2 floor).
    The goal typically requires a particular passenger (P) to be the current
    target of a passenger-2 object (P2): (origin P2 goalPassenger).

    This heuristic estimates the number of actions required to move each origin
    pointer from its current passenger to the goal passenger by computing the
    shortest path length in the undirected graph induced by the served relation.
    Each edge on that path corresponds to one (up) step needed to advance the
    origin pointer along the served chain; thus the path length is used as the
    estimated number of actions for that origin mapping. The heuristic sums the
    distances for all origin goals.

    Assumptions
    -----------
    - The static served facts describe connectivity between passenger nodes:
      a fact "(served A B floorX)" implies an adjacency between passenger A
      and passenger B. The heuristic treats served as an undirected edge
      between A and B (because in many instances both directions are present
      in static facts or can be traversed via domain operators).
    - Moving the origin pointer one hop along the served graph can be achieved
      with one up-like action in the ideal case. We therefore assign unit cost
      per edge in the served graph. (This heuristic is not required to be
      admissible; it aims for informed guidance.)
    - If no path exists between current and goal passenger, the heuristic
      returns a large but finite number to avoid returning infinity.

    Heuristic Initialization
    ------------------------
    - Extract all goal origin facts from task.goals and store them as pairs
      (passenger-2 -> desired passenger).
    - Build an adjacency dictionary of passengers from static facts that match
      the pattern "(served passengerA passengerB floorX)". For every such
      fact we add an undirected edge passengerA <-> passengerB.
    - Keep a conservative finite "infeasible" distance (large_cost) to use
      when the graph does not connect a current passenger to its goal.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse current origin mappings from the state's "(origin P2 Pcur)" facts.
    3. For every origin goal (origin P2 Pgoal):
       a. Find the current passenger Pcur that P2 points to in the state.
          - If no current mapping exists in the state (unexpected but possible),
            use a conservative large cost.
       b. If Pcur == Pgoal, cost contribution is 0.
       c. Else, compute the shortest path length (number of edges) between
          Pcur and Pgoal in the served adjacency graph using BFS.
          - If a path exists, its length is added to the heuristic sum.
          - If not, add large_cost (finite).
    4. Return the sum of costs for all origin goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic.

        Parameters
        ----------
        task : Task
            The planning task. We extract:
            - self.goals: the set of goal facts.
            - self.origin_goals: mapping from passenger-2 objects to their goal passenger.
            - self.adjacency: undirected graph from 'served' static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Map passenger-2 objects to their desired passenger according to the goals.
        self.origin_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                passenger = parts[2]
                self.origin_goals[p2] = passenger

        # Build undirected adjacency graph between passenger nodes from served facts.
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if a not in self.adjacency:
                    self.adjacency[a] = set()
                if b not in self.adjacency:
                    self.adjacency[b] = set()
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)

        # A conservative finite cost for unreachable cases.
        # Use number of adjacency nodes + 10 as a safe upper bound.
        self.large_cost = max(1000, len(self.adjacency) + 10)

    def _bfs_distance(self, start, target):
        """
        Compute shortest path length between start and target in self.adjacency.
        Returns integer length (number of edges). If unreachable, returns None.
        """
        if start == target:
            return 0
        if start not in self.adjacency or target not in self.adjacency:
            return None
        visited = set([start])
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
        """
        Compute heuristic estimate for a given search node.

        Parameters
        ----------
        node : search node (has attribute 'state' which is a frozenset of facts)

        Returns
        -------
        int
            Estimated number of actions to reach a goal state (finite).
        """
        state = node.state

        # If goal already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract current origin mappings from the state.
        current_origins = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                passenger = parts[2]
                current_origins[p2] = passenger

        total_cost = 0

        # For each origin goal, compute distance from current passenger to goal passenger.
        for p2, goal_passenger in self.origin_goals.items():
            current_passenger = current_origins.get(p2)
            if current_passenger is None:
                # No current mapping found: add conservative large cost.
                total_cost += self.large_cost
                continue

            if current_passenger == goal_passenger:
                # Already correct for this origin.
                continue

            dist = self._bfs_distance(current_passenger, goal_passenger)
            if dist is None:
                # Unreachable in served graph: add conservative large cost.
                total_cost += self.large_cost
            else:
                # Use distance as estimated number of actions (one up-like step per edge).
                total_cost += dist

        return int(total_cost)