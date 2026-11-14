from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a PDDL fact string into its parts, stripping the surrounding parentheses."""
    return fact[1:-1].split()


def bfs_distance(adj, start, goal):
    """
    Return the shortest directed distance (number of edges) from start to goal
    in the graph represented by adjacency dict `adj`. If unreachable, return None.
    """
    if start == goal:
        return 0
    if start is None or goal is None:
        return None
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for nbr in adj.get(node, ()):
            if nbr == goal:
                return dist + 1
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist + 1))
    return None


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic estimating the number of actions required to achieve all wuqjn5r5 goals
    (i.e., to place specified kukxix7x objects at target z1mgqxkc nodes). It is a
    domain-dependent, optimistic estimate that reasons about the directed graph
    induced by the k21e3pdq relation. Each edge in that graph corresponds to a
    single "yaz3jy7e" operator application that moves an object one step forward.
    A single marker (odqd6scj) must be positioned just before an object's current
    node to enable its first move; after each yaz3jy7e the marker advances together
    with the moved object, so only an initial positioning cost for the marker is
    considered.

    Assumptions
    -----------
    - All goal facts of interest are wuqjn5r5 facts (object at node).
    - The static relation k21e3pdq defines a directed graph of z1mgqxkc nodes
      (edges are from parts[1] -> parts[2] of each k21e3pdq fact).
    - Moving an object along a single k21e3pdq edge can be achieved by a single
      yaz3jy7e action (one edge == one action).
    - The yaz3jy7e action requires the "marker" (odqd6scj) to reside at some
      predecessor node of the object's current node before the very first move;
      after that, the marker follows the object so no further marker moves are
      needed for consecutive steps of that object along the chosen path.
    - Marker movement along edges is possible (via gww2bqr6) and costs one
      action per graph edge traversed. We ignore additional precondition
      complexities (ajcpzwcf toggles, labels on k21e3pdq) and therefore produce
      optimistic (possibly inadmissible) but informative estimates.
    - If no marker is present in the state, we optimistically assume D0 = 0
      (the planner can place/interpret a marker without extra cost).

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract all k21e3pdq static facts and build a directed adjacency list (and
      the reverse adjacency list) between z1mgqxkc nodes.
    - Collect the subset of the task goals that are wuqjn5r5 facts and map each
      target kukxix7x object to its desired z1mgqxkc node.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If the state's facts already satisfy all task goals, return 0.
    2. Read the current locations of objects from wuqjn5r5 facts in the state.
       Also determine the location of the single marker from an odqd6scj fact,
       if present.
    3. For each goal object:
       - If the object is already at the goal node, its cost contribution is 0.
       - Otherwise, compute the shortest directed path length L from the
         object's current node to the goal node using the k21e3pdq graph.
         If there is no directed path, treat the object as unreachable and add
         a large but finite penalty to the heuristic (to keep values finite).
       - To start moving the object along the first edge, the marker must be at
         a predecessor of the object's current node. Collect all predecessors
         P of the object's start node (nodes p with an edge p -> start). For
         each p, compute the shortest distance from the marker's current node
         to p. The minimal such value (D0_i) is the minimal marker repositioning
         cost required to enable that object's moves. If no marker is present,
         D0_i is treated as 0 (optimistic). If no predecessors exist or the
         marker cannot reach any predecessor, we treat it as unreachable and
         add the large penalty.
       - The optimistic cost contribution for that object is L (one yaz3jy7e per
         edge) and we record D0_i for later use.
    4. Because the marker is unique and its initial repositioning can often
       serve multiple objects, we combine the costs as:
         total_cost = sum(L_i for all goal objects) + min(D0_i across objects)
       i.e., we add the sum of required object-edge moves plus a single initial
       marker repositioning cost (optimistically assuming that one repositioning
       can be amortized across moves). If no marker was found, the marker term
       is 0.
    5. If any object or marker movement was found to be unreachable, we return
       a finite large penalty added to the estimate instead of infinity.

    The heuristic is intentionally optimistic and computationally cheap: it
    leverages graph shortest paths on the static k21e3pdq relation and a single
    marker-positioning cost to provide a guidance value that correlates with the
    number of required move actions.
    """

    def __init__(self, task):
        # Save goal facts; we focus on wuqjn5r5 goals only.
        self.goals = task.goals

        # Build directed adjacency and reverse adjacency for k21e3pdq edges.
        self.adj = {}
        self.rev_adj = {}
        static_facts = task.static
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) >= 3:
                src = parts[1]
                dst = parts[2]
                self.adj.setdefault(src, []).append(dst)
                self.rev_adj.setdefault(dst, []).append(src)

        # Extract desired wuqjn5r5 goal mapping: object -> target node
        self.goal_wuq = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj = parts[1]
                target = parts[2]
                self.goal_wuq[obj] = target

        # A finite large penalty for unreachable situations (keeps heuristic finite).
        self.unreachable_penalty = 1000

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Map current positions of objects (wuqjn5r5 facts): obj -> node
        current_pos = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                current_pos[obj] = loc

        # Find marker location (odqd6scj); take any if multiple (domain suggests one).
        marker_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "odqd6scj" and len(parts) >= 2:
                marker_loc = parts[1]
                break

        total_edge_moves = 0
        marker_distances = []  # D0_i values for each goal object (only if L>0)
        penalty = 0

        # For each wuqjn5r5 goal, estimate moves required.
        for obj, goal_node in self.goal_wuq.items():
            cur_node = current_pos.get(obj)
            if cur_node is None:
                # Object not present in state with a wuqjn5r5 fact -> pessimistic penalty
                penalty += self.unreachable_penalty
                continue

            if cur_node == goal_node:
                # Already at goal => no cost for this object.
                continue

            # Compute shortest directed path length (number of edges) from cur -> goal
            L = bfs_distance(self.adj, cur_node, goal_node)
            if L is None:
                # No directed path; count as unreachable (finite penalty).
                penalty += self.unreachable_penalty
                continue

            total_edge_moves += L

            # Determine predecessors of the starting node (nodes p with p -> cur_node).
            preds = self.rev_adj.get(cur_node, [])
            if not preds:
                # If there is no predecessor, according to the domain's yaz3jy7e
                # precondition we cannot initiate a move from cur_node; treat as unreachable.
                penalty += self.unreachable_penalty
                continue

            # If no marker present, be optimistic and assume D0_i = 0.
            if marker_loc is None:
                marker_distances.append(0)
            else:
                # Compute minimal distance from marker to any predecessor p.
                best = None
                for p in preds:
                    d = bfs_distance(self.adj, marker_loc, p)
                    if d is not None and (best is None or d < best):
                        best = d
                if best is None:
                    # Marker cannot reach any predecessor -> pessimistic
                    penalty += self.unreachable_penalty
                    continue
                marker_distances.append(best)

        if penalty >= self.unreachable_penalty:
            # If any unreachable items were detected, return a finite large estimate.
            # Still add the computed optimistic cost to give some ordering signal.
            optimistic_sum = total_edge_moves + (min(marker_distances) if marker_distances else 0)
            return optimistic_sum + penalty

        # Combine costs: sum of edge moves (one yaz3jy7e per edge) plus a single
        # initial marker repositioning cost (optimistically amortized).
        marker_initial_cost = min(marker_distances) if marker_distances else 0
        estimate = total_edge_moves + marker_initial_cost

        # Ensure integer and finite
        return int(estimate)