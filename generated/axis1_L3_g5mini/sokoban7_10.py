from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `miconic` (secret_domain) elevator-like domain.
    The task in this domain is to move `origin` associations (facts of the form
    "(origin <passenger-2> <passenger>)") so that they match the goal origin
    passengers. The only action that directly changes an `origin` fact is `up`,
    which moves an origin from one passenger to a "next" passenger following the
    relation expressed by `served` facts. Thus this heuristic treats each
    origin token as an agent that must be routed along the directed graph
    induced by `served` facts from its current passenger to its goal passenger.

    Assumptions
    -----------
    - The static relation (served A B floorX) defines a directed adjacency A -> B
      that can be traversed by a single `up` action (provided other action
      preconditions are met in the true dynamics).
    - The number of `up` actions required to move an origin from passenger P to
      passenger Q is approximated by the shortest-path distance from P to Q in
      the graph induced by `served` facts (treated as directed edges).
    - Other domain constraints (e.g., needing a boarded passenger or a lift-at
      fact at intermediate nodes) are not modeled exactly; they are approximated
      by a small, cheap penalty when the state lacks boarded passengers.
    - Summing distances for all origin facts gives a reasonable estimate of the
      number of actions still required (this heuristic is inadmissible but
      informative for greedy best-first search).

    Heuristic Initialization
    ------------------------
    - Extract the set of goal origin facts (self.goals).
    - Build a directed adjacency graph of passengers from static `served`
      facts: for each `(served A B floorX)` add a directed edge A -> B.
    - Record the set of passenger nodes observed in the static `served` facts
      so that we can reason about unreachable nodes.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node / state:
    1. If the state's facts already satisfy all goal facts, return 0.
    2. For each goal origin fact (origin <p2> <goal_passenger>):
       a. Find the current origin fact for that passenger-2 in the state:
          "(origin <p2> <current_passenger>)". If such a fact is missing,
          assume the origin is not yet created at any passenger and treat it as
          needing a routing from an unknown location: penalize with a finite,
          conservative large value.
       b. Compute the shortest-path distance in the adjacency graph from
          current_passenger to goal_passenger using BFS. This distance
          approximates the number of `up` actions required.
       c. If no path exists in the graph, count a conservative finite penalty
          (graph size + 5) for that origin to keep the heuristic finite.
    3. Sum distances for all origin tokens.
    4. Add a small extra penalty of 1 if the state has no `(boarded ...)`
       facts but the total distance > 0. This approximates the fact that `up`
       requires a boarded passenger to operate, so at least one boarding-like
       action may be necessary before the first `up`.
    5. Return the computed integer cost.
    """

    def __init__(self, task):
        """
        Extracts:
        - self.goals: the goal facts (task.goals)
        - self.adj: dict mapping passenger -> set(of next passengers) from static `served` facts
        - self.nodes_count: number of distinct passenger nodes encountered (used for finite penalties)
        """
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graph from static served facts: served A B floor -> edge A -> B
        adj = {}
        nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                src = parts[1]
                dst = parts[2]
                nodes.add(src)
                nodes.add(dst)
                if src not in adj:
                    adj[src] = set()
                adj[src].add(dst)
        self.adj = adj
        self.nodes_count = max(1, len(nodes))

    def bfs_distance(self, src, dst):
        """Return shortest-path distance from src to dst in self.adj (directed).
        If src == dst return 0. If either node is unknown or dst unreachable,
        return None.
        """
        if src == dst:
            return 0
        if src not in self.adj and src != dst:
            # src has no outgoing edges
            return None
        visited = {src}
        queue = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == dst:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # 0 if goals already satisfied
        if self.goals <= state:
            return 0

        # Map passenger-2 -> current passenger from state's origin facts
        current_origins = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                p = parts[2]
                current_origins[p2] = p

        total_cost = 0
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                goal_p = parts[2]
                cur_p = current_origins.get(p2)
                if cur_p is None:
                    # origin not present in this state for this passenger-2;
                    # assign a conservative finite penalty so heuristic remains finite.
                    total_cost += self.nodes_count + 5
                    continue
                dist = self.bfs_distance(cur_p, goal_p)
                if dist is None:
                    # unreachable in static adjacency: assign finite conservative penalty
                    total_cost += self.nodes_count + 5
                else:
                    total_cost += dist

        # Small heuristic adjustment: if there are no boarded passengers but we need moves,
        # at least one boarding-like action will likely be required before the first `up`.
        if total_cost > 0:
            has_boarded = any(get_parts(f)[0] == "boarded" for f in state)
            if not has_boarded:
                total_cost += 1

        return int(total_cost)