from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the space-separated parts of a PDDL fact string, without surrounding parentheses.
    Example: "(ldlu1rwo box1 loc_3_5)" -> ["ldlu1rwo", "box1", "loc_3_5"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A heuristic tailored to the secret_domain (Sokoban-like) domain. The domain
    contains:
      - zdufuwne L         : the agent is at location L
      - ldlu1rwo BOX L     : a box BOX is at location L
      - qsw2k0xh L         : the location L is "free" (no box), i.e. traversable
      - h80z8tvp A B D     : a directed edge A->B labeled D (labels encode push/move directions)
    There are two action types:
      - a simple move of the agent along an h80z8tvp edge if the destination is free
      - a push: the agent pushes a box B->C if there is an A->B edge with label L
        and a B->C edge with the same label L (agent must be on A to push B to C).

    The heuristic estimates the number of actions to reach the goal by combining:
      1) a push-distance estimate: minimal number of pushes required to move each
         box to its goal using the static push graph derived from h80z8tvp labels
         (this ignores dynamic blocking by other boxes — optimistic but informative),
      2) a short agent walking cost to reach the first required pushing position
         for the first box we plan to move (we compute agent shortest path on the
         current-state free graph, treating current box locations as obstacles).

    The heuristic is not required to be admissible; it is designed to be
    informative and efficiently computable to reduce node expansions.

    Assumptions
    -----------
    - Goals are of the form (ldlu1rwo BOX LOC) and refer to specific boxes.
    - The static connectivity (h80z8tvp facts) encodes the map and push
      directions via labels; these are invariant across states.
    - We approximate multi-box interactions by summing per-box optimistic
      push-distances (ignores blocking by other boxes). This keeps the
      heuristic efficient while remaining domain-informed.
    - Agent walking cost is computed on the current state's qsw2k0xh (free)
      facts. Boxes currently in the state are treated as obstacles for agent walking.

    Heuristic Initialization
    ------------------------
    - Parse static facts (task.static) to build:
      * adjacency graph for agent moves: edges_from[u] -> list of (v, label)
      * push graph for boxes: push_edges[box_loc] -> list of possible destination locations
        derived from pairs of directed edges (A->B label L) and (B->C label L).
    - Extract the goals mapping box -> goal_location from task.goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (node.state):

    1) If the state already satisfies all goals, return 0.

    2) Extract from the state:
       - current agent location (the single zdufuwne fact),
       - current box locations (from ldlu1rwo facts),
       - current free locations (qsw2k0xh facts).

    3) For each box that has a goal:
       - Compute the minimal number of pushes required to move the box from
         its current location to its goal location using a BFS on the static
         push graph (this BFS treats the map as static, ignoring other boxes).
       - If the goal is unreachable in the abstract push graph, give a large
         finite penalty (we return a large integer to avoid infinity).

    4) Sum the push distances of all boxes as the main component of the heuristic.

    5) To capture the agent movement cost to start pushing:
       - Choose a target box to start with (we pick the box with smallest
         remaining push-distance > 0).
       - For that box, identify the set of predecessor agent positions A that
         enable the first push (i.e., A where A->B and B->C have the same label).
       - Run a BFS for the agent on the current walk graph (edges from h80z8tvp
         but only stepping into locations that are currently free: qsw2k0xh facts)
         and where all current box locations are considered blocked.
       - Add the minimal agent walking steps required to reach one of those A
         positions to the heuristic (0 if already in place).

    6) Return the sum: total_pushes + initial_agent_walk_cost.

    This combination gives a compact and informed estimate: the number of
    pushes (dominant cost in Sokoban-like problems) plus the initial walking
    overhead for the agent to reach a pushing position.
    """

    def __init__(self, task):
        # Goals: map box -> desired location (from goal facts).
        self.goals = {}
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ldlu1rwo" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.goals[box] = loc

        # Build adjacency and push graphs from static h80z8tvp facts.
        # edges_from[u] = list of (v, label)
        self.edges_from = {}  # directed edges for agent movement and push analysis
        # Also record existence of nodes
        self.locations = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp" and len(parts) == 4:
                a, b, lab = parts[1], parts[2], parts[3]
                self.locations.add(a)
                self.locations.add(b)
                self.edges_from.setdefault(a, []).append((b, lab))

        # Build push graph: a push from B -> C is possible if there exists A such that
        # A->B label L and B->C label L. We do not check dynamic occupancy here.
        self.push_edges = {loc: [] for loc in self.locations}
        # For quick lookup of edges by (source,label)
        # map (src,label) -> set of targets
        edges_by_src_label = {}
        for src, out_list in self.edges_from.items():
            for tgt, lab in out_list:
                edges_by_src_label.setdefault((src, lab), set()).add(tgt)

        # For each possible (A->B lab) and (B->C lab), add push edge B->C
        for a, out_list in self.edges_from.items():
            for b, lab in out_list:
                # edges from b with same label
                targets = {t for (src_lab, label), tgts in edges_by_src_label.items() for t in ()}
                # faster: check edges_from[b]
                for c, lab2 in self.edges_from.get(b, []):
                    if lab2 == lab:
                        # push from b to c (agent must be at a to push)
                        self.push_edges.setdefault(b, []).append(c)

        # For BFS convenience convert lists to tuples (optional)
        for k in list(self.push_edges.keys()):
            self.push_edges[k] = list(dict.fromkeys(self.push_edges[k]))  # unique neighbors

    def _bfs_push_distance(self, start, goal):
        """Compute minimal number of pushes from start to goal using the static push graph.
        If unreachable, return None.
        """
        if start == goal:
            return 0
        if start not in self.push_edges:
            return None
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            loc, d = dq.popleft()
            for nxt in self.push_edges.get(loc, ()):
                if nxt == goal:
                    return d + 1
                if nxt not in visited:
                    visited.add(nxt)
                    dq.append((nxt, d + 1))
        return None

    def _agent_walk_distance(self, agent_loc, free_set, blocked_set, targets):
        """BFS for agent walking distance from agent_loc to any node in `targets`.
        - free_set: set of nodes that are (currently) free and therefore can be entered.
        - blocked_set: set of nodes that are occupied by boxes and must be treated as blocked.
        - targets: set of nodes (pre-push positions) we want to reach.
        Returns minimal steps (number of moves) or None if unreachable.
        """
        if agent_loc in targets:
            return 0
        # BFS on directed edges edges_from, stepping into neighbors v only if
        # v in free_set and v not in blocked_set.
        visited = {agent_loc}
        dq = deque([(agent_loc, 0)])
        while dq:
            loc, d = dq.popleft()
            for neigh, _lab in self.edges_from.get(loc, []):
                if neigh in visited:
                    continue
                # cannot walk into a location occupied by a box (blocked_set)
                if neigh in blocked_set:
                    continue
                # can only move into locations that are currently free (qsw2k0xh)
                if neigh not in free_set:
                    continue
                if neigh in targets:
                    return d + 1
                visited.add(neigh)
                dq.append((neigh, d + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If the state already satisfies goals, heuristic is 0.
        # Note: Task goal check is available through self.goals but using set inclusion:
        # We'll construct set of goal facts and compare to state.
        goal_facts = {"({} {} {})".format("ldlu1rwo", box, loc) for box, loc in self.goals.items()}
        if goal_facts <= state:
            return 0

        # Extract agent location (zdufuwne ...)
        agent_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "zdufuwne" and len(parts) == 2:
                agent_loc = parts[1]
                break

        # Extract current box locations and free cells
        box_locs = {}  # box -> loc
        free_cells = set()  # locations with qsw2k0xh in current state
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "ldlu1rwo" and len(parts) == 3:
                box_locs[parts[1]] = parts[2]
            elif parts[0] == "qsw2k0xh" and len(parts) == 2:
                free_cells.add(parts[1])

        # Compute push distances for boxes that have goals.
        total_pushes = 0
        unreachable_penalty = len(self.locations) + 10  # finite large penalty
        # collect boxes still needing movement
        boxes_to_move = []
        for box, goal_loc in self.goals.items():
            cur_loc = box_locs.get(box)
            if cur_loc is None:
                # box not present in state? Treat as large penalty (shouldn't happen).
                total_pushes += unreachable_penalty
                continue
            if cur_loc == goal_loc:
                # already in place
                continue
            dist = self._bfs_push_distance(cur_loc, goal_loc)
            if dist is None:
                total_pushes += unreachable_penalty
                # still add to list so we may try to estimate walk cost as well
                boxes_to_move.append((box, cur_loc, goal_loc, None))
            else:
                total_pushes += dist
                boxes_to_move.append((box, cur_loc, goal_loc, dist))

        # If there are no boxes to move, only perhaps agent needs to move (but goals are only boxes).
        if not boxes_to_move:
            return 0

        # Choose a box to start with: prefer the one with smallest push distance > 0,
        # breaking ties arbitrarily. If some had unreachable distance (None), they are deprioritized.
        candidate = None
        best_push = None
        for tup in boxes_to_move:
            _, cur_loc, _, dist = tup
            if dist is None:
                continue
            if dist > 0 and (best_push is None or dist < best_push):
                candidate = tup
                best_push = dist
        if candidate is None:
            # all had None or already zero; pick the first box with dist != 0
            candidate = boxes_to_move[0]

        _, box_start_loc, _, _ = candidate

        # Determine the set of agent predecessor positions A for the first push of this box:
        # A is any node such that there exists an edge A->B with label L and B->C with same L
        # where B==box_start_loc (so possible pushes from B->C). So predecessors are A's
        # that allow pushing box_start_loc to some neighbor C.
        pre_push_positions = set()
        # find all A such that A->B label L exists and B->C label L exists
        for a_src, out_list in self.edges_from.items():
            for b_tgt, lab in out_list:
                if b_tgt != box_start_loc:
                    continue
                # check for B->C with same label
                for c_tgt, lab2 in self.edges_from.get(box_start_loc, []):
                    if lab2 == lab:
                        pre_push_positions.add(a_src)

        # If there are no valid pre-push positions (isolated box in push-graph), agent walk cost is large.
        agent_walk_cost = 0
        if pre_push_positions:
            # Agent walking BFS must treat all current boxes as blocked (cannot walk through boxes)
            blocked = set(box_locs.values())
            # Allow agent to start even if its cell is not included in free_cells (but typically it is).
            # The agent can only move into neighboring cells that are in free_cells and not blocked.
            if agent_loc is None:
                # No agent in state? Highly irregular; assign a penalty
                agent_walk_cost = 3
            else:
                # If the predecessor position is the same as agent current loc, cost 0
                if agent_loc in pre_push_positions:
                    agent_walk_cost = 0
                else:
                    # BFS
                    walk_dist = self._agent_walk_distance(agent_loc, free_cells, blocked, pre_push_positions)
                    if walk_dist is None:
                        # unreachable: give a moderate penalty
                        agent_walk_cost = 5
                    else:
                        agent_walk_cost = walk_dist

        # Total heuristic: pushes (dominant) + initial agent movement overhead
        heuristic_value = total_pushes + agent_walk_cost

        # Ensure finite integer
        if heuristic_value is None:
            return unreachable_penalty
        return int(heuristic_value)