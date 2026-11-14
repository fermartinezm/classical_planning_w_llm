from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, trimming the surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic tailored for the (secret) sokoban-like domain provided.
    The heuristic estimates the number of actions needed to achieve all goals of the
    form (adjacent-2 boxX). These goals are produced by the `push` action which
    requires that:
      - the box has predicate (at boxX) true, and
      - the robot (at-robot-2 LOC) is at a specific location LOC that matches an
        (at-robot boxX LOC) fact (static or present in the state).
    To obtain (at boxX) usually the `move` action is used; it requires an (adjacent
    boxX LOC_ADJ) fact and that the robot is at LOC_ADJ (at-robot-2 LOC_ADJ).
    The robot moves between locations using `move-2` / `push-2` which consume/require
    `clear` facts. These clear facts form a directed graph of robot-2 movement.

    Assumptions
    -----------
    - Goals are conjunctions of facts like (adjacent-2 boxN).
    - The graph of robot locations is given by static `clear` facts (directed edges).
    - A box may already have (at boxX) in the current state; if not,
      some (adjacent boxX LOC_ADJ) facts must be present in the state to allow
      a `move` to produce (at boxX).
    - The required robot location for pushing each box is available either in
      static facts (task.static) as (at-robot boxX LOC_PUSH) or dynamically in
      the current state as (at-robot boxX LOC_PUSH). The dynamic (state) version
      overrides/augments static declarations.
    - The heuristic need not be admissible; it is designed to be informative and
      efficiently computable (BFS on the static clear graph for distances).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - self.goals: the set of goal fact strings (from task.goals).
      - self.clear_graph: directed adjacency list of locations from static facts
        of the form (clear A B) meaning a robot-2 can move from A to B in one step.
      - self.static_at_robot: mapping box -> location from static facts of the
        form (at-robot BOX LOC). (Dynamic at-robot facts in a state are read
        at evaluation time and may override these.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal box G = (adjacent-2 BOX):
      1. If (adjacent-2 BOX) is already true in the current state, the box
         contributes 0 to the heuristic.
      2. Otherwise, one `push` action is necessary to make (adjacent-2 BOX).
         The preconditions of `push` require:
           - (at BOX) to hold, and
           - the robot to be at the required pushing location LOC_PUSH
             (given by at-robot BOX LOC_PUSH).
      3. If (at BOX) is not yet true:
           - We must perform one `move` action to produce (at BOX). A `move`
             is possible only if the state contains some (adjacent BOX LOC_ADJ)
             and the robot stands at LOC_ADJ.
           - Hence, to get (at BOX) we often need to move the robot to some
             LOC_ADJ (cost = distance in clear graph) and then perform `move`
             (cost = 1).
      4. After (at BOX) holds, the robot must be at LOC_PUSH to do `push`.
         So we may need to move from the location after the `move` (LOC_ADJ)
         to LOC_PUSH (cost = distance in clear graph). If (at BOX) already
         held initially, we compute the cost just to move from the current
         robot position to LOC_PUSH and then `push`.
      5. Distances in the location graph are computed by BFS on the directed
         graph defined by static `clear` facts. Each step in this graph
         corresponds to a single `move-2` / `push-2` action (cost=1).
      6. For a given box we therefore take the minimum over available adjacency
         locations LOC_ADJ in the state of:
             dist(robot_pos, LOC_ADJ) + 1 (move) + dist(LOC_ADJ, LOC_PUSH) + 1 (push)
         If (at BOX) already true:
             dist(robot_pos, LOC_PUSH) + 1 (push)
      7. If distances are unreachable (no path), the heuristic returns a large
         finite penalty for that box. We sum the per-box costs. This produces
         an admissible/optimistic-ish estimation for reachable problems, but it
         can also be larger — admissibility is not required.

    The heuristic is zero exactly in goal states (all (adjacent-2 BOX) are present).
    """

    INF = 10**6

    def __init__(self, task):
        # Goals (set of goal fact strings)
        self.goals = task.goals

        # Parse static facts for clear edges and at-robot mappings.
        static = task.static
        self.clear_graph = {}  # directed: loc -> list of neighbor locs
        self.static_at_robot = {}  # box -> location (from static facts)

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "clear" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.clear_graph.setdefault(src, []).append(dst)
                # Ensure node exists even if it has no outgoing edges
                self.clear_graph.setdefault(dst, self.clear_graph.get(dst, []))
            elif pred == "at-robot" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.static_at_robot[box] = loc

        # Ensure there is at least an empty graph entry if no clear facts exist.
        if not self.clear_graph:
            self.clear_graph = {}

    def _bfs_distance(self, start, goal):
        """
        Compute shortest directed distance (number of edges) from start to goal
        in the clear_graph. Return INF if unreachable or if start/goal is None.
        """
        if start is None or goal is None:
            return self.INF
        if start == goal:
            return 0
        if start not in self.clear_graph:
            return self.INF

        dq = deque([start])
        dist = {start: 0}
        while dq:
            u = dq.popleft()
            for v in self.clear_graph.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if v == goal:
                        return dist[v]
                    dq.append(v)
        return self.INF

    def __call__(self, node):
        """
        Estimate the number of actions to reach a goal from the node.state.

        Algorithm:
          - If all goals already hold in the state, return 0.
          - Determine current robot location from (at-robot-2 LOC) in the state.
          - Gather dynamic facts:
                * which boxes currently have (at BOX)
                * adjacent facts (adjacent BOX LOC_ADJ)
                * any dynamic (at-robot BOX LOC_PUSH) facts in the state
          - For each goal box:
                * If (adjacent-2 BOX) already in state: cost 0
                * Else compute minimal cost as described in the class docstring.
          - Sum per-box costs and return that sum.
        """
        state = node.state

        # 1) Zero for goal states
        if self.goals <= state:
            return 0

        # 2) Robot current location (at-robot-2 LOC) -- there should typically be exactly one.
        robot_pos = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_pos = parts[1]
                break
        # If robot position not present in state, optimistically pick an arbitrary location
        # from the clear graph (so distances can be computed). If graph is empty, leave None.
        if robot_pos is None and self.clear_graph:
            # pick some node to make distances finite (heuristic remains informative).
            robot_pos = next(iter(self.clear_graph.keys()))

        # 3) Dynamic information from state
        dynamic_at_boxes = set()
        adj_map = {}  # box -> list of adjacency locations (from state facts)
        dynamic_at_robot = {}  # box -> loc (if present in state)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 2:
                dynamic_at_boxes.add(parts[1])
            elif pred == "adjacent" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                adj_map.setdefault(box, []).append(loc)
            elif pred == "at-robot" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                dynamic_at_robot[box] = loc
            # we ignore other predicates here

        total_cost = 0

        # For every goal of form (adjacent-2 BOX) compute minimal cost to produce it.
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts or parts[0] != "adjacent-2" or len(parts) < 2:
                # skip non-adjacent-2 goals; domain's goals are expected to be adjacent-2
                continue
            box = parts[1]
            # If already satisfied in state, no cost.
            if f"(adjacent-2 {box})" in state:
                continue

            # Determine required push location for this box:
            l_push = dynamic_at_robot.get(box, self.static_at_robot.get(box, None))

            # If box already has (at BOX), we only need to move robot to l_push and push
            if box in dynamic_at_boxes:
                move_steps = 0 if (robot_pos == l_push) else self._bfs_distance(robot_pos, l_push)
                if move_steps >= self.INF:
                    # unreachable -> large penalty but finite
                    box_cost = 1000
                else:
                    box_cost = move_steps + 1  # +1 for the push action
                total_cost += box_cost
                continue

            # Otherwise we must create (at BOX) first using a `move`:
            candidate_adjs = adj_map.get(box, [])
            if not candidate_adjs:
                # No adjacent facts to enable `move` -> penalize heavily (likely unsolvable)
                total_cost += 1000
                continue

            best_cost = self.INF
            for loc_adj in candidate_adjs:
                # cost to get robot to loc_adj, then perform move, then move to l_push, then push
                d1 = 0 if robot_pos == loc_adj else self._bfs_distance(robot_pos, loc_adj)
                d2 = 0 if loc_adj == l_push else self._bfs_distance(loc_adj, l_push)
                if d1 >= self.INF or d2 >= self.INF:
                    continue
                # +1 for the `move` action (to generate (at BOX)), +1 for the `push`
                candidate_cost = d1 + 1 + d2 + 1
                if candidate_cost < best_cost:
                    best_cost = candidate_cost

            if best_cost >= self.INF:
                # unreachable all candidates -> large penalty
                total_cost += 1000
            else:
                total_cost += best_cost

        # Return an integer heuristic estimate (finite for most solvable states).
        return int(total_cost)