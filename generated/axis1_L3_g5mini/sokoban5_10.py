from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "miconic" (secret_domain) elevator-like domain.
    The heuristic estimates the number of "up"-like transitions required to transform
    the current origin mappings into the goal origin mappings. Each unit of cost
    corresponds to one origin-shift (one application of the domain operator that
    changes an `origin` fact), approximated by the shortest-path distance in a
    graph induced by the static `served` relations.

    Assumptions
    -----------
    - The only predicates relevant to computing how many origin updates are needed
      are `origin` (the facts the goals require) and the static `served` facts
      (which define adjacency between passenger positions).
    - The static `served` relations implicitly define connectivity between passenger
      nodes. A sequence of adjacent `served` facts (treated as undirected edges)
      corresponds to a sequence of origin-shifts that an `up`-like operator can
      produce (we ignore boarding/lift precondition constraints for heuristic
      simplicity).
    - Each single step of the shortest path corresponds roughly to one operator
      application that moves an `origin` mapping forward by one passenger.
    - If a passenger-2's `origin` is already the desired target in the current
      state, its contribution is 0. If no path exists in the static graph,
      the heuristic returns a finite large value (so the heuristic is always finite).

    Heuristic Initialization
    ------------------------
    - self.goals: set of goal facts (from task.goals).
    - self.goal_origin_map: mapping passenger-2 -> desired passenger (extracted from goals)
      for all goal origin facts of the form "(origin <p2> <p>)".
    - self.adjacency: undirected adjacency dictionary built from all static facts
      of the form "(served <p1> <p2> <floor>)". For each such fact we add an undirected
      edge between <p1> and <p2>. This adjacency graph is used to compute shortest
      path distances between passengers.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the node's state already satisfies all goals (self.goals subset of state),
       return 0.
    2. Parse the current state's `origin` facts to obtain current_origin_map:
       a mapping from passenger-2 objects to the passenger they currently point to.
    3. For every goal origin (passenger-2 -> target passenger):
       a. If the current mapping equals the target, cost contribution is 0.
       b. Otherwise, compute the shortest-path distance (number of edges) between
          the current passenger node and the desired passenger node in the undirected
          adjacency graph built from `served` facts using BFS. The distance is an
          estimate of how many origin-shift actions are required.
       c. If BFS finds no path, use a finite fallback cost (graph_size + 1).
    4. Sum the individual contributions for all goal origin facts and return the sum.
       This sum is used as the heuristic estimate for greedy best-first search.
    """

    def __init__(self, task):
        # Goals (set of goal fact strings)
        self.goals = task.goals

        # Build mapping from goal origin facts: passenger-2 -> target passenger
        self.goal_origin_map = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2, target = parts[1], parts[2]
                self.goal_origin_map[p2] = target

        # Build an undirected adjacency graph from static "served" facts.
        # For each "(served p1 p2 floor)" we connect p1 <-> p2.
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                if a not in self.adjacency:
                    self.adjacency[a] = set()
                if b not in self.adjacency:
                    self.adjacency[b] = set()
                # treat served as undirected adjacency for heuristic reachability
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)

        # Precompute a finite fallback cost (used when no path exists between nodes).
        # Use number of known nodes + 1 to remain finite but large relative to graph size.
        self.fallback_cost = (len(self.adjacency) + 1)

    def _bfs_distance(self, start, goal):
        """Return shortest-path distance between start and goal in adjacency graph.
        If either node is unknown or unreachable, return None.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If all goals satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Build current mapping of origin facts from the state:
        # passenger-2 -> passenger
        current_origin_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2, cur = parts[1], parts[2]
                current_origin_map[p2] = cur

        total_cost = 0
        # For each goal origin requirement, add estimated cost to change current mapping to goal mapping.
        for p2, desired in self.goal_origin_map.items():
            current = current_origin_map.get(p2)
            if current == desired:
                continue  # no cost for this goal
            if current is None:
                # If the current state doesn't contain an origin for this passenger-2,
                # use fallback cost (we still return a finite heuristic).
                total_cost += self.fallback_cost
                continue
            dist = self._bfs_distance(current, desired)
            if dist is None:
                # unreachable in static graph -> fallback finite cost
                total_cost += self.fallback_cost
            else:
                total_cost += dist

        # Return aggregate estimated number of origin-shift actions required.
        return total_cost