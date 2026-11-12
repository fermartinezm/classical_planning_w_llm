from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses from a PDDL fact string and split it into parts.
    Example: "(served passenger1 passenger2 floor1)" -> ["served", "passenger1", "passenger2", "floor1"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the `miconic` (secret_domain) domain.
    The domain's relevant structure for this heuristic:
    - Facts `served X Y FLOOR` define directed adjacency relations between
      passenger-nodes (X -> Y). The `up` action advances an `origin` along
      such a served edge (origin moves from Y to the successor), so moving
      the `origin` from a current node to a goal node is fundamentally a
      graph traversal problem.
    - Goals are expressed as `origin P2 P` facts (for one or more passenger-2
      objects P2). Each such goal requires moving the corresponding origin
      to the requested passenger node P.

    This heuristic estimates the number of actions required by computing
    shortest-path distances (in number of edge-traversals) in the directed
    graph induced by `served` facts from the current origin node(s) to the
    desired goal node(s). The BFS distance corresponds to the minimal number
    of `up`-like moves needed under an abstraction that ignores operational
    prerequisites such as the exact placement of a boarded passenger or the
    lift-at facts. The heuristic is intentionally not admissible (it may
    ignore some prerequisites) but is computationally cheap and usually
    informative for guiding greedy best-first search in this domain.

    Assumptions
    -----------
    - Each `served` fact encodes a directed edge between passenger nodes:
      served A B F  =>  edge A -> B (floor F is ignored for cost).
    - The `origin` predicate is unique per passenger-2 object (there is at
      most one `origin P2 ?` fact at a time). The heuristic sums the costs
      to satisfy each origin-goal pair independently (they concern different
      passenger-2 objects).
    - The main cost driver is the number of `up` (edge) traversals needed to
      move an origin from its current passenger node to the target node.
    - The heuristic ignores secondary preparatory actions (boarding/depart)
      required in the real domain to enable a particular `up`. This makes
      the heuristic optimistic in some cases and thus not strictly admissible,
      but it remains an effective guide.

    Heuristic Initialization
    ------------------------
    - Extract all `served` facts from the task's static information and build
      a directed adjacency map: for each served A B F, add B as a successor
      of A.
    - Extract the goal origin facts (those of the form `origin P2 P`) and
      record them as target mapping: passenger-2 -> goal passenger.
    - Keep the set of all passenger nodes observed in the `served` facts so
      that we can detect unreachable targets quickly.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node (state):
    1. If all goal facts already hold in the state (self.goals subset of
       state), immediately return 0.
    2. Parse the state's `origin` facts to determine the current passenger
       node for each passenger-2 object that appears in the goals.
       - If a goal passenger-2 does not have a corresponding origin fact in
         the current state, we treat that goal as currently unreachable under
         the abstraction and assign a large finite penalty.
    3. For each goal origin pair (passenger-2 -> target passenger):
       - Run a directed BFS in the adjacency graph (constructed from static
         served facts) from the current origin node to the target node.
       - If a path exists, the heuristic contribution for this goal is the
         length of the shortest path (number of served-edge traversals).
       - If no path exists in the static graph, add a finite large penalty
         equal to (number_of_nodes + 5). (This keeps the heuristic finite
         while indicating a high cost / likely unsolvable under the static
         connectivity.)
    4. Sum the contributions for all goal origin pairs and return the sum.
       This sum estimates the total number of edge-like moves required to
       satisfy all origin goals.

    Notes
    -----
    - The heuristic is fast: BFS on a small graph built from static facts.
    - It satisfies: heuristic(state) == 0 iff the goal facts hold in state.
    - For solvable instances (where static served connectivity provides a
      path), the heuristic returns a finite integer (the BFS path length).
    """

    def __init__(self, task):
        """
        Initialize the heuristic.

        Parameters
        ----------
        task : Task
            The planning task. We expect task.static to contain `served` facts
            and task.goals to contain `origin` goals.
        """
        # Store the raw goal set for the quick goal-check in __call__.
        self.goals = task.goals

        # Build directed adjacency from static `served` facts.
        # adjacency[A] is a set of successors B such that (served A B floor) exists.
        self.adjacency = {}
        self.nodes = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                self.nodes.add(a)
                self.nodes.add(b)
                self.adjacency.setdefault(a, set()).add(b)

        # Extract origin goals: map passenger-2 -> target passenger.
        # We only take goals that are 'origin' predicates.
        self.origin_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                target = parts[2]
                self.origin_goals[p2] = target

        # A finite large penalty for unreachable targets under the static graph.
        self.unreachable_penalty = max(1, len(self.nodes)) + 5

    def _shortest_distance(self, start, target):
        """
        Compute shortest directed path length from start to target in the
        adjacency graph using BFS. Returns an integer distance if reachable,
        or None if target is not reachable.
        """
        if start == target:
            return 0
        if start not in self.adjacency:
            return None
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for succ in self.adjacency.get(node, ()):
                if succ in visited:
                    continue
                if succ == target:
                    return dist + 1
                visited.add(succ)
                q.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic value for the given search node.

        Parameters
        ----------
        node : search node
            The node contains attribute `state` which is a frozenset of PDDL fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach the goal (non-negative integer).
        """
        state = node.state

        # 1) If goals already satisfied, return zero.
        if self.goals <= state:
            return 0

        # 2) Extract current origin facts from the state: passenger-2 -> passenger
        current_origins = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                pos = parts[2]
                current_origins[p2] = pos

        total_cost = 0
        # 3) For each origin goal, compute distance from current origin to target.
        for p2, target in self.origin_goals.items():
            current = current_origins.get(p2, None)
            if current is None:
                # No current origin fact for this passenger-2 in this state:
                # treat as high-cost (finite) so search can continue.
                total_cost += self.unreachable_penalty
                continue

            # Use BFS on the static adjacency graph to estimate number of edge traversals.
            dist = self._shortest_distance(current, target)
            if dist is None:
                # No path under static connectivity: assign a large finite penalty.
                total_cost += self.unreachable_penalty
            else:
                total_cost += dist

        return total_cost