from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (kyjn6f1j).
    The domain models a Sokoban-like puzzle:
      - zdufuwne LOC   : the robot is at location LOC
      - ldlu1rwo BOX LOC : BOX is at location LOC
      - qsw2k0xh LOC   : location LOC is free/clear
      - h80z8tvp U V LABEL : a directed labeled edge from U to V with label LABEL
    There are two action types:
      - x042b0fi: robot moves from one location to an adjacent free location
      - yi0kp1zg: robot pushes a box from P to Q when robot stands at R,
                 if edges (R -> P LABEL) and (P -> Q LABEL) exist with the same LABEL,
                 and Q is free. The push moves the box P->Q and the robot R->P.

    The heuristic estimates required actions to achieve all box placement goals
    (ldlu1rwo facts). It focuses on (1) the minimum number of pushes for each box
    (the dominant cost in Sokoban-like instances) and (2) a lightweight estimate
    of the robot movement required to get into position for performing the first push.

    Assumptions
    -----------
    - Each goal is of the form (ldlu1rwo BOX GOAL_LOC). Each BOX appears in the initial
      state with one (ldlu1rwo BOX CUR_LOC).
    - A push moves a box exactly one step along the push-direction (one yi0kp1zg).
    - Robot move (x042b0fi) moves along the directed h80z8tvp edges but only into
      currently free locations (qsw2k0xh facts).
    - We ignore complex dynamic interactions between multiple boxes for tractability:
      the heuristic uses current occupancies to restrict robot movement when possible,
      and treats box-to-goal reachability via a derived "box-move graph" that captures
      allowed single-step pushes (independent of the actual robot position).
    - The heuristic is not required to be admissible; it aims for good guidance while
      remaining efficient to compute.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract goal mappings (box -> goal location) from task.goals.
    - Parse static h80z8tvp facts (edges) from task.static and build:
        * out_edges[u] = list of (v,label)
        * in_edges[v]  = list of (u,label)
      These are used to build the box-move graph and for robot path searches.
    - Build the derived "box-move graph" where an edge p -> q exists if there is a label
      L such that some node r has h80z8tvp r p L and h80z8tvp p q L. This encodes that a
      push from p to q is possible (robot standing at some r), disregarding whether r is
      currently reachable by the robot.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state (node.state) the heuristic does:
    1. If all goal facts are satisfied, return 0.
    2. Parse the state to obtain:
       - robot_loc : the unique location where (zdufuwne LOC) holds.
       - free_locs : set of locations with (qsw2k0xh LOC) in the state.
       - cur_box_locs: mapping box -> current location from (ldlu1rwo BOX LOC) facts.
    3. For each goal box:
       a) If the box is already at its goal location, contribute 0 to the estimate.
       b) Otherwise compute the minimum number of pushes needed to move the box from
          its current location to its goal location using BFS over the precomputed
          box-move graph. Each BFS step corresponds to one push (yi0kp1zg).
          If the box is unreachable in this graph, a large penalty is added (this
          typically indicates the planning task is unsolvable under push semantics).
    4. Robot positioning cost for initiating pushes:
       - For boxes that need at least one push, identify candidate robot positions R
         from which the first push is possible: an R such that (R -> box_loc LABEL)
         exists and (box_loc -> Q LABEL) exists for some Q that is currently free (Q in free_locs).
       - Try to find the minimum number of robot moves from the current robot location
         to any such candidate R using a BFS over the robot movement graph that respects
         current free cells (x042b0fi requires destination to be qsw2k0xh).
       - If no candidate is reachable under current occupancy, fall back to an
         unblocked robot-distance (ignoring free-cell restrictions) to provide a
         weaker but finite estimate.
       - Add the minimal such robot distance once (we assume the robot can position itself
         to start moving one of the boxes first; subsequent robot motion during pushes
         is accounted for by push counts because each push also moves the robot).
    5. The final heuristic is:
         heuristic = sum(min_pushes_for_each_box) + min_robot_distance_to_first_push
       with penalties for unreachable boxes to produce a high value.
    """

    def __init__(self, task):
        # Store goals (set of goal facts)
        self.goals = set(task.goals)
        # Static facts containing h80z8tvp facts (edges)
        static_facts = set(task.static)

        # Parse goals: map each box -> goal location (from ldlu1rwo facts)
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

        # Build directed labeled edges from static facts.
        # out_edges[u] = list of (v,label)
        # in_edges[v]  = list of (u,label)
        self.out_edges = {}
        self.in_edges = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp" and len(parts) >= 4:
                u, v, label = parts[1], parts[2], parts[3]
                self.out_edges.setdefault(u, []).append((v, label))
                self.in_edges.setdefault(v, []).append((u, label))

        # Build the "box-move graph": possible single-step box moves (pushes).
        # There is an edge p -> q if there exists a label L such that
        # some r has (r -> p L) and p has (p -> q L).
        self.box_adj = {}
        # For each location p, find labels present on incoming edges and outgoing edges.
        for p in set(list(self.out_edges.keys()) + list(self.in_edges.keys())):
            # map label -> list of predecessors r
            incoming_by_label = {}
            for (r, lab) in self.in_edges.get(p, []):
                incoming_by_label.setdefault(lab, []).append(r)
            # map label -> list of successors q
            outgoing_by_label = {}
            for (q, lab) in self.out_edges.get(p, []):
                outgoing_by_label.setdefault(lab, []).append(q)
            # for any label present on both sides, add edges p -> q for all those q
            succs = set()
            for lab, qs in outgoing_by_label.items():
                if lab in incoming_by_label:
                    succs.update(qs)
            if succs:
                self.box_adj[p] = list(succs)
            else:
                self.box_adj.setdefault(p, [])  # ensure key exists

    def _bfs_shortest_box_pushes(self, start, goal):
        """
        BFS on the box-move graph to compute minimum number of pushes from start -> goal.
        Returns integer distance or None if unreachable.
        """
        if start == goal:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nb in self.box_adj.get(node, []):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return None

    def _bfs_robot_distance(self, start, targets, free_locs, respect_free=True):
        """
        BFS for the robot on the location graph defined by static out_edges.
        If respect_free is True, only edges u->v where v in free_locs are allowed (x042b0fi).
        If respect_free is False, all edges from out_edges are allowed (unblocked fallback).
        `targets` is an iterable of desired target nodes; returns minimum distance (int)
        from start to any node in targets, or None if none reachable.
        """
        targets_set = set(targets)
        if start in targets_set:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for (v, lab) in self.out_edges.get(node, []):
                # If respecting free cells, only traverse to v if v currently free.
                if respect_free and v not in free_locs:
                    continue
                if v in targets_set:
                    return d + 1
                if v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
        return None

    def __call__(self, node):
        state = set(node.state)

        # 1) If goal already reached, return 0
        # Note: Task.goal_reached is not available here, use self.goals subset check.
        if self.goals <= state:
            return 0

        # 2) Parse current robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "zdufuwne":
                robot_loc = parts[1]
                break

        # 3) Parse free locations (qsw2k0xh) and current box locations (ldlu1rwo)
        free_locs = {get_parts(f)[1] for f in state if get_parts(f)[0] == "qsw2k0xh"}
        cur_box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                cur_box_locs[box] = loc

        # Some basic safety checks
        # If robot position unknown, assume worst-case small penalty (robot must move into play)
        if robot_loc is None:
            robot_loc = next(iter(free_locs), None)

        # 4) For each box goal, compute minimal pushes to goal
        total_pushes = 0
        unreachable_penalty = 100000  # large penalty for unreachable push paths
        boxes_needing_push = []  # list of (box, current_loc) that require at least one push
        for box, goal_loc in self.box_goals.items():
            cur_loc = cur_box_locs.get(box)
            if cur_loc is None:
                # If box position unknown, penalize heavily
                total_pushes += unreachable_penalty
                continue
            if cur_loc == goal_loc:
                continue
            pushes = self._bfs_shortest_box_pushes(cur_loc, goal_loc)
            if pushes is None:
                total_pushes += unreachable_penalty
            else:
                total_pushes += pushes
                # only consider boxes that actually need pushing for robot positioning
                if pushes > 0:
                    boxes_needing_push.append((box, cur_loc))

        # 5) Estimate robot movement required to reach the first push position.
        min_robot_distance = None
        if boxes_needing_push and robot_loc is not None:
            # Collect candidate robot positions R for first push of each box:
            # R must satisfy: h80z8tvp R -> box_loc LABEL and exists box_loc -> Q LABEL with Q free.
            candidate_positions = set()
            for _, box_loc in boxes_needing_push:
                # outgoing from box_loc: (box_loc -> q, label)
                for (q, lab_out) in self.out_edges.get(box_loc, []):
                    # q must be currently free, because push requires destination free
                    if q not in free_locs:
                        continue
                    # find any r with (r -> box_loc lab_out)
                    for (r, lab_in) in self.in_edges.get(box_loc, []):
                        if lab_in == lab_out:
                            candidate_positions.add(r)
            # If no immediate candidates found (due to occupancy), degrade to more permissive candidates:
            if not candidate_positions:
                # Try to produce candidates ignoring current free status of q (maybe q will become free later)
                for _, box_loc in boxes_needing_push:
                    for (q, lab_out) in self.out_edges.get(box_loc, []):
                        for (r, lab_in) in self.in_edges.get(box_loc, []):
                            if lab_in == lab_out:
                                candidate_positions.add(r)

            # If still empty (no push predecessor at all), then box cannot be pushed at all (already accounted by unreachable)
            if candidate_positions:
                # First try BFS respecting current free cells
                dist_respecting = self._bfs_robot_distance(robot_loc, candidate_positions, free_locs, respect_free=True)
                if dist_respecting is not None:
                    min_robot_distance = dist_respecting
                else:
                    # Fallback: BFS ignoring free-cells (unblocked)
                    dist_unblocked = self._bfs_robot_distance(robot_loc, candidate_positions, free_locs, respect_free=False)
                    min_robot_distance = dist_unblocked

        # If no boxes need push, robot distance is 0
        if not boxes_needing_push:
            min_robot_distance = 0

        # If still None (e.g., no robot known or no candidates), set to 0 to avoid infinite heurics
        if min_robot_distance is None:
            min_robot_distance = 0

        # Final heuristic: pushes dominate, add robot approach once
        heuristic_value = total_pushes + min_robot_distance

        # Ensure non-negative integer
        try:
            return int(heuristic_value)
        except Exception:
            return 0