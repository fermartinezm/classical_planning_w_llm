from collections import deque
from itertools import permutations
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the sokoban domain.

    Assumptions
    -----------
    - The domain's "adjacent" facts describe directed adjacency edges (they typically
      appear in both directions where movement is possible both ways).
    - A push moves a single box one directed adjacency step forward; the push
      precondition requires that the robot stands directly behind the box in the
      same direction used to move the box.
    - For estimating the number of pushes required, we compute shortest directed-path
      distances from each box to each goal (this ignores dynamic blocking by other
      boxes and hence is optimistic for pushes).
    - For estimating robot travel to start pushing, we compute a shortest-path for the
      robot on the static adjacency graph treating current box locations as blocked
      (robot cannot move into locations currently occupied by boxes).
    - Number of boxes is assumed small in practice; when small we compute an optimal
      assignment (permutation) of boxes to goals. For larger numbers a greedy strategy
      would be used (but the code uses permutations up to reasonable sizes).

    Heuristic Initialization
    ------------------------
    - Extract the directed adjacency graph from static facts "adjacent loc1 loc2 dir".
      The graph is stored as:
        * edges: dict mapping loc -> list of (neighbor_loc, dir)
        * dir_of_edge: dict mapping (loc, neighbor) -> dir
        * incoming: dict mapping (loc, dir) -> set of predecessor locations (i.e., r such that adjacent r loc dir)
    - Extract the goal locations for each box from the task.goals (facts of form "(at box loc)").

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a current state:
    1. If the goals are already satisfied, return 0.
    2. Parse the state to obtain:
       - robot location
       - current box locations (set)
    3. For each box and each goal, compute the minimal number of directed pushes needed:
       - Use BFS on the directed adjacency graph (ignoring other boxes) to obtain the shortest
         directed path length from the box location to the goal location. This gives the number
         of pushes required for that box to reach that goal.
       - If a box cannot reach a goal by directed adjacency, treat the cost as very large (unreachable).
    4. Compute a minimum-cost assignment of boxes to goals:
       - For up to a modest number of boxes we try every permutation to find the assignment that
         minimizes the total number of pushes (exact minimum). For larger numbers the same
         method will still work but may be slow; in typical Sokoban benchmarks the box count is small.
    5. Add a robot-travel estimate:
       - For the selected box->goal matches we compute, for each matched box, the first directed step
         along the shortest path from the box to its goal (call it b0 -> b1). For that first step
         there must exist at least one location r with an adjacency (r -> b0) using the same direction
         (this r is a candidate robot push-position).
       - We compute the shortest robot path (BFS on directed adjacency) from the robot's current location
         to any candidate r, treating all current box-locations as blocked (robot cannot move into them).
       - The minimum such robot-move distance across all matched boxes is taken as an estimate of the
         robot repositioning cost to start pushing. (This is an optimistic but informative estimate:
         it does not account for interleaving pushes for multiple boxes.)
    6. The heuristic is then:
         heuristic = total_number_of_pushes (from matching) + robot_min_travel_distance
       where robot distances and push counts are measured in numbers of primitive actions (moves or pushes).
    7. If some box/goal is unreachable (no directed path), return a large finite value to indicate
       the state is unlikely to lead to a goal without rearrangement.

    Notes
    -----
    - The heuristic is not required to be admissible. The method above aims to provide a
      fairly accurate and cheaply computable estimate that guides greedy best-first search
      efficiently by combining push distances with robot repositioning cost.
    """

    def __init__(self, task):
        # Store goal facts and static facts.
        self.goals = task.goals
        static_facts = task.static

        # Directed adjacency graph: edges[src] = [(dst, dir), ...]
        self.edges = {}
        # Map (src,dst) -> dir for quick direction lookup.
        self.dir_of_edge = {}
        # incoming[(loc, dir)] = set of predecessors r such that adjacent r loc dir
        self.incoming = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                src, dst, dir = parts[1], parts[2], parts[3]
                self.edges.setdefault(src, []).append((dst, dir))
                self.dir_of_edge[(src, dst)] = dir
                self.incoming.setdefault((dst, dir), set()).add(src)

        # Extract goal locations for boxes: goals are of form "(at box loc)"
        self.goal_locations = []
        self.goal_by_box = {}  # if goal specifies particular box
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "at":
                box, loc = parts[1], parts[2]
                self.goal_locations.append(loc)
                self.goal_by_box[box] = loc

    def _directed_bfs(self, start):
        """
        BFS over the directed adjacency graph (self.edges).
        Returns a dict distance: node -> distance and parent mapping for path reconstruction:
        parent[node] = predecessor_node so that predecessor -> node is an edge.
        """
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for (v, _) in self.edges.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        return dist, parent

    def _directed_shortest_path(self, src, dst):
        """
        Return a shortest directed path (list of nodes) from src to dst using self.edges.
        If no path exists, return None.
        """
        dist, parent = self._directed_bfs(src)
        if dst not in dist:
            return None
        # Reconstruct path
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def _robot_bfs_distance(self, start, targets, blocked):
        """
        BFS for the robot movement: directed moves along adjacency facts (move action).
        The robot may move from u to v only if adjacency(u,v,dir) exists AND v not in blocked.
        Returns the minimal distance (number of moves) from start to any node in targets.
        If no target reachable, returns None.
        """
        if start in targets:
            return 0
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for (v, _) in self.edges.get(u, ()):
                if v in blocked:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if v in targets:
                        return dist[v]
                    q.append(v)
        return None

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals hold, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state: find robot location and current box positions.
        robot_loc = None
        box_locations = {}  # box -> loc
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at":
                # (at box loc)
                obj, loc = parts[1], parts[2]
                box_locations[obj] = loc

        # Collect list of boxes and goals to match.
        boxes = list(box_locations.keys())
        goals = list(self.goal_locations)

        # Basic sanity: if number of boxes != number of goals, proceed with min(len(boxes), len(goals))
        n_boxes = len(boxes)
        n_goals = len(goals)
        if n_boxes == 0 or n_goals == 0:
            # No boxes or no goals relevant: return 0 or small cost depending on goal check earlier.
            return 0

        # Precompute directed shortest distances (push counts) and paths for each box->goal.
        # We ignore other boxes for push distance computation (optimistic about clearing).
        push_dist = {}   # (box, goal) -> distance (number of pushes)
        push_path = {}   # (box, goal) -> path list if reachable else None
        LARGE_COST = 100000  # large finite cost for unreachable situations

        for b in boxes:
            b_loc = box_locations[b]
            # BFS from b_loc over directed edges
            dist_map, _ = self._directed_bfs(b_loc)
            for g in goals:
                if g in dist_map:
                    push_dist[(b, g)] = dist_map[g]
                    push_path[(b, g)] = self._directed_shortest_path(b_loc, g)
                else:
                    push_dist[(b, g)] = None
                    push_path[(b, g)] = None

        # Build cost matrix and compute best assignment (exact via permutations for modest sizes).
        # If there are more goals than boxes or vice versa, we only assign min(n_boxes, n_goals).
        m = min(n_boxes, n_goals)
        # For matching we will use permutations of assignments of boxes to m goals.
        # To keep it simple when there are equal counts we match each box to a distinct goal.
        best_total_pushes = None
        best_assignment = None  # list of (box, goal) pairs

        # If there are more goals than boxes, consider only subsets of goals of size n_boxes?
        # For simplicity, when n_goals > n_boxes, only consider the first n_boxes goals (common in sokoban).
        # (Typically n_boxes == n_goals in Sokoban.)
        if n_goals >= n_boxes:
            candidate_goals_list = goals
            # try permutations of goals of length n_boxes
            # For small n this is fine.
            for perm in permutations(candidate_goals_list, n_boxes):
                total = 0
                feasible = True
                for b, g in zip(boxes, perm):
                    d = push_dist.get((b, g))
                    if d is None:
                        feasible = False
                        break
                    total += d
                if not feasible:
                    continue
                if best_total_pushes is None or total < best_total_pushes:
                    best_total_pushes = total
                    best_assignment = list(zip(boxes, perm))
        else:
            # fewer goals than boxes: assign subset of boxes to goals (rare). We assign first n_goals boxes.
            candidate_boxes_list = boxes
            for perm in permutations(candidate_boxes_list, n_goals):
                total = 0
                feasible = True
                for b, g in zip(perm, goals):
                    d = push_dist.get((b, g))
                    if d is None:
                        feasible = False
                        break
                    total += d
                if not feasible:
                    continue
                if best_total_pushes is None or total < best_total_pushes:
                    best_total_pushes = total
                    best_assignment = list(zip(perm, goals))

        if best_total_pushes is None:
            # No feasible matching found (some box cannot reach any goal by directed adjacency).
            # Return a large finite value (not infinity) so planner can still proceed.
            return LARGE_COST

        # Now estimate robot travel cost to get to a push-start position for the matched boxes.
        # For each matched (box,goal) we compute the first step along the path box->goal:
        # if path is [b0, b1, ...], we need a robot position r such that adjacent r b0 dir and adjacent b0 b1 dir.
        # All candidates r are incoming[(b0, dir)].
        blocked_by_boxes = set(box_locations.values())  # robot cannot walk into these
        robot_min_dist = None

        for (b, g) in best_assignment:
            path = push_path.get((b, g))
            if path is None or len(path) < 2:
                # No move needed or unreachable; if box already at goal we need no robot travel for it.
                if path is None:
                    continue
                else:
                    continue
            b0 = path[0]
            b1 = path[1]
            # direction of edge b0 -> b1
            dir = self.dir_of_edge.get((b0, b1))
            if dir is None:
                # Should not happen because path was found via directed BFS, but be robust.
                continue
            # candidate robot positions that can push b0->b1 are predecessors r with same dir: adjacent r b0 dir
            candidates = self.incoming.get((b0, dir), set())
            if not candidates:
                # cannot stand behind box to push in that direction; skip
                continue
            # Compute robot distance to any candidate
            dist_to_candidates = self._robot_bfs_distance(robot_loc, candidates, blocked_by_boxes)
            if dist_to_candidates is None:
                # candidate not reachable because boxes block robot; skip
                continue
            if robot_min_dist is None or dist_to_candidates < robot_min_dist:
                robot_min_dist = dist_to_candidates

        # If robot_min_dist is None, it means robot cannot reach any initial push position (maybe boxed-in).
        # We still return pushes cost but add a penalty to reflect blocked robot mobility.
        if robot_min_dist is None:
            # Penalize by 1 extra move (optimistic) or a moderate penalty. Choose moderate to avoid infinite.
            robot_min_dist = 2

        heuristic_value = int(best_total_pushes + robot_min_dist)

        # Ensure non-negative
        if heuristic_value < 0:
            heuristic_value = 0

        return heuristic_value