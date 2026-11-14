from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact string by stripping the surrounding
    parentheses and splitting on whitespace.
    Example: "(on item1 item2)" -> ["on", "item1", "item2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain ("blocksworld"-like) domain.
    The domain's goals are expressed as arm-empty predicates of the form
    (arm-empty <arm> <object>). This heuristic estimates how many "swap-like"
    moves are required to make each required arm-empty predicate true by
    reasoning about the graph of "on" relationships. Each individual step
    along an "on" chain (i.e., an (on X Y) fact) corresponds to a single
    pickup/transfer operation that can move the arm-empty association from
    X to Y (according to the pickup operator in the domain).

    Assumptions
    -----------
    - The relevant operator that changes arm-empty relations is `pickup`, whose
      effect effectively moves the "arm-empty" association from one second
      argument to another when an (on ...) relationship exists.
    - The set of static and current-state (on ...) facts defines a directed
      graph where an edge X -> Y means (on X Y) is true; following an edge
      corresponds to one application of a pickup-like operator that changes
      (arm-empty <arm> X) into (arm-empty <arm> Y).
    - Goals consist only of arm-empty predicates (as in provided instances).
    - We ignore more complex interactions (stack/unstack side-effects,
      holding requirements, clear, on-table peculiarities). The heuristic is
      intentionally optimistic about being able to perform a chain of pickups
      along available on-edges.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (task.goals).
    - Extract all static "on" edges from task.static for quick graph construction.
      The constructor builds an adjacency dictionary mapping a node -> set(neighbors)
      for directed edges defined by (on X Y).
    - No other static structures are required for this heuristic, but state
      "on" facts will be combined with static ones when estimating distances.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node with a state (frozenset of fact strings):
    1. If all goal facts are already in the state, return 0.
    2. Build the directed "on" graph for the current state by combining:
       - static "on" facts gathered at initialization, and
       - current-state "on" facts (these may change during planning).
       The graph has an edge X -> Y for every (on X Y).
    3. For every goal of the form (arm-empty ARM TARGET):
       a. If the exact goal fact is already present in the state, it contributes 0.
       b. Otherwise, locate the current object(s) CUR such that (arm-empty ARM CUR)
          holds in the current state. If none is found, use a small default cost
          (fallback) because we cannot see where ARM is currently associated.
       c. If one or more CUR are found, compute the shortest directed path length
          (number of edges) from any CUR to TARGET in the combined "on" graph
          using BFS. Each edge corresponds to one pickup-like operation that moves
          the arm association one step closer to TARGET, so the path length is the
          estimated required number of actions.
       d. If no path exists between any CUR and TARGET, use a modest fallback cost
          (finite) to keep the heuristic finite for solvable states.
    4. Sum the estimated costs for all missing arm-empty goals and return this sum.
       (This sum gives a stronger guidance signal than a max-based estimate,
       preferring states that already satisfy many of the arm-empty goals.)

    Notes
    -----
    - The heuristic is not admissible (it may be optimistic or pessimistic),
      but it is designed to be informative and efficient to compute.
    - The heuristic always returns a finite integer (fallback costs are finite),
      and returns 0 iff all goal arm-empty facts are present in the state.
    """

    def __init__(self, task):
        # Store goals (set of strings)
        self.goals = task.goals

        # Build adjacency dict from static "on" facts: node -> set(neighbors)
        self.static_on_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.static_on_graph.setdefault(src, set()).add(dst)

    def _build_state_on_graph(self, state):
        """
        Combine static on-graph with current-state on facts to produce a
        merged adjacency mapping for BFS.
        """
        graph = {k: set(v) for k, v in self.static_on_graph.items()}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                graph.setdefault(src, set()).add(dst)
        return graph

    def _bfs_shortest_distance(self, graph, sources, target):
        """
        BFS to find shortest distance (number of directed edges) from any of the
        source nodes to the target node in the directed graph.
        - graph: dict node -> set(neighbors)
        - sources: iterable of start nodes
        - target: target node name
        Returns the integer distance if a path exists, otherwise None.
        """
        # Fast path: if target is in sources
        if target in sources:
            return 0

        visited = set()
        q = deque()
        for s in sources:
            q.append((s, 0))
            visited.add(s)

        while q:
            node, dist = q.popleft()
            for neigh in graph.get(node, ()):
                if neigh == target:
                    return dist + 1
                if neigh not in visited:
                    visited.add(neigh)
                    q.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Returns an integer estimate of the number of actions required to
        achieve all arm-empty goals from the node's state.
        """
        state = node.state  # frozenset of fact strings

        # Quick goal check: if all goal facts are in the state -> heuristic 0
        if self.goals <= state:
            return 0

        # Build combined on-graph (static + current-state on facts)
        graph = self._build_state_on_graph(state)

        # Map from ARM -> set(current_partner_objects) for facts like (arm-empty ARM OBJ)
        current_arm_empty = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, obj = parts[1], parts[2]
                current_arm_empty.setdefault(arm, set()).add(obj)

        total_cost = 0
        # Heuristic fallback costs when we cannot find a precise chain
        FALLBACK_NO_ARM = 2   # ARM not found at all in current state
        FALLBACK_NO_PATH = 3  # ARM present but no directed path to target

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, target = parts[1], parts[2]
                # If already satisfied, cost 0
                if goal in state:
                    continue

                # If we know where this arm is currently associated (one or more objects),
                # try to find shortest chain from any such object to the target.
                if arm in current_arm_empty and current_arm_empty[arm]:
                    sources = current_arm_empty[arm]
                    dist = self._bfs_shortest_distance(graph, sources, target)
                    if dist is not None:
                        total_cost += dist
                    else:
                        # No directed path found; use a modest fallback cost
                        total_cost += FALLBACK_NO_PATH
                else:
                    # No information about this arm in current state; assume a small cost
                    total_cost += FALLBACK_NO_ARM

        # Ensure the heuristic is at least 1 for non-goal states (helps some search setups)
        return int(total_cost)