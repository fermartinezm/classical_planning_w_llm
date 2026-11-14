from heuristics.heuristic_base import Heuristic
from collections import defaultdict
import heapq

def get_parts(fact):
    """Return the tokens of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5) domain.

    This heuristic provides an informed estimate of the number of actions
    needed to achieve goal wuqjn5r5 predicates (i.e., to move each kukxix7x
    object to its goal z1mgqxkc node). The estimate is based on the static
    k21e3pdq relation which defines adjacency between z1mgqxkc nodes. The
    heuristic computes a (weighted) shortest-path distance in the graph of
    z1mgqxkc nodes and sums these distances for each wuqjn5r5 goal that is
    not yet satisfied.

    Assumptions
    -----------
    - Goals of interest are wuqjn5r5 <kukxix7x> <z1mgqxkc> facts: moving the
      kukxix7x object along nodes is the main planning effort.
    - k21e3pdq facts in the static information describe adjacency among
      z1mgqxkc nodes. The heuristic treats these as edges in a graph.
    - Moving a kukxix7x object by one graph hop typically requires at least
      one domain action (yaz3jy7e), and that action has extra preconditions
      (e.g. the target node often needs to be ajcpzwcf). Thus moving into a
      node that is not currently ajcpzwcf is more costly in practice; the
      heuristic penalizes such moves slightly.
    - The heuristic need not be admissible; it should be informative and
      efficiently computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static adjacency information from the
    task.static facts. Each static k21e3pdq fact of the form:
        (k21e3pdq NODE_A NODE_B LABEL)
    is interpreted as an adjacency relation between NODE_A and NODE_B.
    For heuristic purposes this relation is treated as an undirected edge
    between NODE_A and NODE_B (both directions are possible in the search).
    The constructor builds an adjacency mapping (node -> neighboring nodes)
    and a set of all known z1mgqxkc nodes.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (containing a state), the heuristic performs the
    following steps to compute an estimate:

    1. If all goal facts are already true in the state, return 0.

    2. Parse the state and collect:
       - Current positions of all kukxix7x objects from wuqjn5r5 facts.
       - The set of nodes that currently satisfy (ajcpzwcf NODE).

    3. For each goal fact of interest (wuqjn5r5 AGENT TARGET):
       - If the exact goal fact is present in the state, contribute 0 cost.
       - Otherwise, locate the current node of AGENT by inspecting the
         state's wuqjn5r5 facts. If the agent has no known position in the
         state, assign a conservative large estimate.
       - Compute a weighted shortest-path distance from the agent's current
         node to the TARGET in the static adjacency graph. The edge cost
         used for traversing from node U to neighbor V is:
             base_cost = 1
             if V is not currently ajcpzwcf in the state: base_cost += 1
         This reflects that moving into nodes that are not ``available''
         (ajcpzwcf) typically requires additional preparatory actions.
       - If TARGET is unreachable in the static graph, use a conservative
         fallback cost proportional to the graph size.

    4. Sum the costs for all wuqjn5r5 goals that are not yet achieved and
       return the resulting integer as the heuristic value.

    Notes
    -----
    - The heuristic is intentionally optimistic about the possibility of
      moving along static edges (treats edges as undirected) while adding a
      small preparatory penalty for non-ajcpzwcf targets. This gives a
      balance between efficiency and informativeness without simulating
      the full precondition interactions of the domain.
    - The heuristic always returns 0 for goal states, and returns finite
      numeric estimates for states where there is at least some path in
      the static graph (and a large finite fallback when not).
    """

    def __init__(self, task):
        # Store goal facts (set of strings)
        self.goals = task.goals

        # Build adjacency graph from static k21e3pdq facts.
        # We treat k21e3pdq A B L as an undirected adjacency A <-> B.
        self.adjacency = defaultdict(set)
        self.nodes = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)
                self.nodes.add(a)
                self.nodes.add(b)

        # A conservative large cost to use when no path exists; relative to domain size.
        self.unreachable_cost = max(20, len(self.nodes) + 5)

    def __call__(self, node):
        state = node.state

        # If the state already meets all goals, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current wuqjn5r5 positions and ajcpzwcf availability from state.
        agent_pos = {}   # agent -> node
        ajcp_available = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "wuqjn5r5" and len(parts) >= 3:
                agent = parts[1]
                node_name = parts[2]
                agent_pos[agent] = node_name
            elif pred == "ajcpzwcf" and len(parts) >= 2:
                ajcp_available.add(parts[1])

        total_cost = 0

        # Helper: weighted shortest path (Dijkstra) from src to dst on adjacency graph.
        def shortest_weighted_distance(src, dst):
            if src == dst:
                return 0
            if src not in self.nodes or dst not in self.nodes:
                return self.unreachable_cost
            # Min-heap: (cost, node)
            heap = [(0, src)]
            seen = {}
            while heap:
                cost, u = heapq.heappop(heap)
                if u in seen:
                    continue
                seen[u] = cost
                if u == dst:
                    return cost
                for v in self.adjacency.get(u, ()):
                    # base traversal cost is 1; if target node v is not currently ajcpzwcf,
                    # add small penalty because preparatory actions are usually required.
                    step_cost = 1 + (0 if v in ajcp_available else 1)
                    new_cost = cost + step_cost
                    if v not in seen:
                        heapq.heappush(heap, (new_cost, v))
            return self.unreachable_cost

        # For each goal of the form (wuqjn5r5 AGENT TARGET), estimate cost if not satisfied.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue
            if parts[0] != "wuqjn5r5" or len(parts) < 3:
                # This heuristic currently focuses on wuqjn5r5 goals only.
                continue
            agent = parts[1]
            target = parts[2]
            # If goal fact is already true in current state, cost contribution is 0.
            if goal_fact in state:
                continue

            # Find current position of agent.
            if agent in agent_pos:
                src = agent_pos[agent]
            else:
                # Agent position unknown in the state; assign fallback cost.
                total_cost += self.unreachable_cost
                continue

            # Compute weighted shortest distance from src to target.
            dist = shortest_weighted_distance(src, target)
            total_cost += dist

        # Ensure heuristic returns an integer.
        return int(total_cost)