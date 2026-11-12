from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    The domain contains:
    - zdufuwne LOC : the robot is at LOC
    - ldlu1rwo BOX LOC : BOX (a rullurva object) is at LOC
    - qsw2k0xh LOC : LOC is free / traversable
    - h80z8tvp A B LABEL : a directed static edge A->B labelled LABEL (labels encode direction)
      A push from A (robot) -> B (box) -> C (destination for the box) is possible iff
      there are two consecutive edges with the same LABEL: A->B LABEL and B->C LABEL.
      The push action moves the box from B to C while the robot moves into B.

    Assumptions
    -----------
    - We count pushes (yi0kp1zg) as the dominating costly actions to achieve box goals.
    - Robot move actions (x042b0fi) also cost 1; we estimate only an initial robot travel
      cost: the number of robot-steps to reach the position needed to start the first push.
      After performing the first push, subsequent pushes are assumed to be contiguous (robot
      moves into the box's previous position) and therefore we do not add extra travel
      costs for them.
    - We ignore dynamic blocking by other boxes when estimating push reachability. Instead
      we use the static push-graph induced by h80z8tvp to compute a lower-bound on the
      number of pushes required for each box to reach a goal location. This is efficient
      and informative for guiding greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract goal facts of the form (ldlu1rwo BOX LOC) and store desired locations by box.
    - Extract all static directed edges (h80z8tvp A B LABEL) and build:
        * incoming_map[B]  = list of (A, LABEL)
        * outgoing_map[A]  = list of (B, LABEL)
      From those we build the push graph:
        push_graph[B] contains entries (C, A) meaning a push can move a box from B->C
        if there exists a predecessor A such that A->B LABEL and B->C LABEL with same LABEL.
      The push_graph is thus independent of dynamic occupancy and depends only on static edges.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals return 0.
    2. Parse the state to find:
       - robot location (the single fact '(zdufuwne X)'),
       - box locations (facts '(ldlu1rwo BOX LOC)'),
       - set of currently free/traversable locations (facts '(qsw2k0xh LOC)').
    3. For each box not already at its goal, run a BFS over the push_graph to find the
       minimal number of pushes required to reach any of its goal locations. While doing this
       BFS we also record the "first push predecessor" (a robot position A behind the box
       for the very first push in the computed shortest push path). The BFS runs only over
       static push edges and is therefore efficient.
    4. Sum the minimal number of pushes for all boxes. If a box has no path in the push_graph
       to any goal, we treat its push-cost as a large finite penalty (so the heuristic remains
       finite).
    5. Compute an estimate of the robot travel cost to get into position to perform the first
       push for some box. We run a BFS on the (dynamic) robot-move graph: the robot may move
       from U to V iff there is a static h80z8tvp U->V LABEL and V is currently qsw2k0xh (free).
       We take the minimal robot-distance among the recorded first-push predecessors of all
       boxes that require pushes. If the robot is already at such a predecessor, the distance
       is 0. If none of the boxes requires pushing, this term is 0.
    6. Heuristic value = total_pushes + minimal_initial_robot_travel.
       - Each push is counted as 1 action (yi0kp1zg).
       - Each robot-step is counted as 1 action (x042b0fi).

    Notes on admissibility and usefulness
    -------------------------------------
    - The heuristic is not admissible (it may underestimate or overestimate depending on
      dynamic blockings), but it is tuned to estimate the dominant cost (number of pushes).
    - It is cheaply computable (BFS on a graph of locations) and informative for steering
      search towards reducing the number of pushes remaining.
    """

    def __init__(self, task):
        # store goal facts (set of strings) for quick goal-check
        self.goals = task.goals
        # static facts: directed edges h80z8tvp A B LABEL
        static_facts = task.static

        # Build incoming and outgoing maps from static h80z8tvp facts
        self.incoming_map = {}  # node -> list of (pred, label)
        self.outgoing_map = {}  # node -> list of (succ, label)
        self.locations = set()  # all wlo7ezwc nodes discovered

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) == 4:
                a, b, label = parts[1], parts[2], parts[3]
                self.locations.add(a)
                self.locations.add(b)
                self.outgoing_map.setdefault(a, []).append((b, label))
                self.incoming_map.setdefault(b, []).append((a, label))

        # Construct push graph:
        # A push moves a box from B -> C if exists predecessor A s.t. A->B LABEL and B->C LABEL.
        # We record for each B a list of (C, A) meaning "box at B can be pushed to C by a robot at A".
        self.push_graph = {loc: [] for loc in self.locations}
        for b in self.locations:
            preds = self.incoming_map.get(b, [])
            succs = self.outgoing_map.get(b, [])
            if not preds or not succs:
                continue
            # For every combination of pred and succ, add push if labels match
            for (a, lab1) in preds:
                for (c, lab2) in succs:
                    if lab1 == lab2:
                        # push from b to c is possible with robot at a
                        self.push_graph.setdefault(b, []).append((c, a))

    def _bfs_robot_distance(self, start, allowed_targets):
        """
        BFS for robot movement in the current state:
        - start: the robot's current location (string)
        - allowed_targets: set/list of target locations for which we need distances (we compute
          distances to all nodes but stop early if all targets are reached).
        Robot can move U->V iff there is a static edge h80z8tvp U V LABEL and V is currently free.
        Returns a dict: location -> distance (in robot steps). If a target is unreachable, it won't
        be present in the dict.
        """
        # This method is only called from __call__ where we know qsw2k0xh set; so we do not store
        # any static filtering here. The adjacency to use for robot moves is self.outgoing_map,
        # but we will filter neighbors by runtime free_set handled by caller.
        # The caller will pass allowed_targets and we will return distances to the discovered nodes.
        raise NotImplementedError("Should not be called directly without free-set; see __call__.")

    def __call__(self, node):
        state = node.state

        # If the goals are already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "zdufuwne":
                if len(parts) >= 2:
                    robot_loc = parts[1]
                    break

        # Parse box locations: ldlu1rwo BOX LOC
        box_locations = {}  # box -> loc
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        # Parse current free/traversable locations qsw2k0xh LOC
        free_set = {get_parts(f)[1] for f in state if get_parts(f)[0] == "qsw2k0xh" and len(get_parts(f)) == 2}

        # Precompute robot adjacency that respects current free_set:
        # robot can move from u -> v if static outgoing edge u->v exists and v is currently free.
        robot_adj = {}
        for u, out_list in self.outgoing_map.items():
            # neighbors v such that v in free_set
            neighbors = [v for (v, _) in out_list if v in free_set]
            if neighbors:
                robot_adj[u] = neighbors

        # Helper: BFS on push_graph to compute minimal pushes from box_pos to any of goal_locs.
        # Also we record the first predecessor position (robot position) required for the first push of that path.
        def pushes_bfs(box_pos, goal_locs):
            """
            Returns (pushes_distance, first_push_pred):
            - pushes_distance: minimal number of pushes to reach any location in goal_locs (int)
              or None if unreachable in push_graph.
            - first_push_pred: the robot predecessor location for the first push on one shortest
              path (string), or None if pushes_distance is 0 (already at goal) or unreachable.
            """
            if box_pos in goal_locs:
                return 0, None

            visited = set([box_pos])
            # queue entries: (current_box_loc, pushes_so_far, first_pred)
            # for neighbors directly reachable from box_pos, first_pred is the associated pred (A)
            q = deque()
            for (succ, pred) in self.push_graph.get(box_pos, []):
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, 1, pred))  # one push, first predecessor = pred

            while q:
                cur, dist, first_pred = q.popleft()
                if cur in goal_locs:
                    return dist, first_pred
                for (succ, pred) in self.push_graph.get(cur, []):
                    if succ not in visited:
                        visited.add(succ)
                        # preserve the first_pred from the initial expansion
                        q.append((succ, dist + 1, first_pred))
            # unreachable
            return None, None

        # Build set of goal locations for each box (task.goals contains ldlu1rwo facts)
        # It's possible multiple goals are present: map boxes to set of allowed goal locations.
        box_to_goal_locs = {}
        for gfact in self.goals:
            parts = get_parts(gfact)
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                box_to_goal_locs.setdefault(box, set()).add(loc)

        # Sum minimal pushes for all boxes (if a box has no goal in the goals set, ignore it)
        total_pushes = 0
        first_push_positions = []  # list of robot positions (preds) needed for the first push per box
        LARGE_PENALTY = max(1000, len(self.locations) * 10)

        for box, box_pos in box_locations.items():
            goal_locs = box_to_goal_locs.get(box, None)
            if goal_locs is None:
                # No goal for this box, ignore it (it might not be a goal object)
                continue
            pushes_needed, first_pred = pushes_bfs(box_pos, goal_locs)
            if pushes_needed is None:
                # treat unreachable as large finite cost
                total_pushes += LARGE_PENALTY
                # cannot determine first_pred reliably; skip adding to first_push_positions
            else:
                total_pushes += pushes_needed
                if pushes_needed > 0 and first_pred is not None:
                    first_push_positions.append(first_pred)

        # If no pushes are required (all boxes at goals), heuristic is 0 (we already returned earlier on goals subset
        # but in case goals include other stuff, still ensure heuristic 0)
        if total_pushes == 0:
            return 0

        # Compute minimal robot travel distance to any of the required first-push positions.
        # If robot location is unknown, we conservatively assume a small travel cost of 1.
        if not first_push_positions:
            # No recorded first push positions (e.g., because unreachable or boxes already at goal).
            # Return total_pushes as heuristic (we can't estimate robot travel).
            return total_pushes

        if robot_loc is None:
            min_robot_distance = 1
        else:
            # BFS from robot_loc over robot_adj graph
            visited = set([robot_loc])
            q = deque([(robot_loc, 0)])
            found_dists = {}
            target_set = set(first_push_positions)
            # If robot is already on a required position, distance is 0
            if robot_loc in target_set:
                min_robot_distance = 0
            else:
                min_robot_distance = None
                while q:
                    cur, dist = q.popleft()
                    neighs = robot_adj.get(cur, [])
                    for nb in neighs:
                        if nb not in visited:
                            visited.add(nb)
                            nd = dist + 1
                            if nb in target_set:
                                if min_robot_distance is None or nd < min_robot_distance:
                                    min_robot_distance = nd
                                    # we can stop early if we found distance 0 or 1; but continue to find any smaller
                                    if min_robot_distance == 0:
                                        break
                            q.append((nb, nd))
                    if min_robot_distance == 0:
                        break
                if min_robot_distance is None:
                    # target unreachable by simple robot moves (due to current free-set). Use a penalty.
                    min_robot_distance = LARGE_PENALTY // 10

        return total_pushes + min_robot_distance