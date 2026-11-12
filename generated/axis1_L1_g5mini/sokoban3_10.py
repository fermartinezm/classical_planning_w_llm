from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a fact string by stripping surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (kyjn6f1j).
    The domain models a robot that can move between locations and can push/move boxes
    along location-to-location edges. Movement of the robot requires the destination
    location to have the predicate `qsw2k0xh` true. Pushing a box from location B to
    C requires the robot to be positioned in some location A and two matching edges
    (A->B and B->C) that share the same label. Pushing toggles `qsw2k0xh` facts for
    the involved locations and moves the robot to the box's former position.

    The heuristic estimates the number of actions required to reach the goal by:
    - Summing an estimate of the number of pushes required to move each box from its
      current location to its goal location (estimated by shortest-path length in the
      static location graph derived from h80z8tvp facts, ignoring labels).
    - Adding an estimate for the robot travel needed to reach the first box to be
      acted upon (computed on the directed movement graph using the current state's
      qsw2k0xh facts to restrict reachable destinations).
    This yields a simple, fast-to-compute estimate that balances box-distance (pushes)
    and robot-travel distance (moves). The heuristic is not required to be admissible
    and is intended to be informative for greedy best-first search.

    Assumptions
    -----------
    - Each push operation (yi0kp1zg) moves a box exactly one graph edge; therefore
      the number of pushes required for a box is approximated by the number of edges
      along a shortest path between current and goal location in the static graph.
    - The static relation h80z8tvp defines directed edges between locations; labels
      on these edges are ignored for the push-distance estimate (this simplifies and
      speeds up computation).
    - Robot movement (x042b0fi) follows static h80z8tvp directed edges but the
      destination must currently have qsw2k0xh true. Robot travel cost is counted
      as the number of such moves (each move is one action).
    - If a location is unreachable for the robot in the current state (due to
      qsw2k0xh restrictions), the heuristic treats the robot travel cost as a
      bounded penalty rather than infinity to keep values finite and guide search.
    - If a box is already at its goal, it contributes 0 to the heuristic.

    Heuristic Initialization
    ------------------------
    - self.goals: stored goal facts (task.goals).
    - self.static: static facts from the task (task.static).
    - Build:
      - `self.undirected_neighbors`: undirected adjacency mapping (location -> set(locations))
        derived from all h80z8tvp facts (labels ignored). Used to estimate number of pushes
        (shortest path lengths).
      - `self.directed_neighbors`: directed adjacency mapping (location -> set(locations))
        derived from h80z8tvp facts preserving direction. Used to compute robot movement
        reachability (movement edges are constrained at runtime by qsw2k0xh facts).
    - Extract the set of box-goal mappings from the goals (predicate ldlu1rwo).

    Step-By-Step Thinking for Computing Heuristic (on a given node/state)
    --------------------------------------------------------------------
    1. If the goal is already reached (all goal facts are present in the state),
       return 0.

    2. Parse the current state to obtain:
       - Current robot location (the fact zdufuwne <loc>).
       - Set of locations that currently have qsw2k0xh true (these are valid
         robot destinations for x042b0fi).
       - Current locations of boxes (facts ldlu1rwo <box> <loc>).

    3. For each box that has a goal location defined (from task goals):
       - If the box is already at the goal, it contributes 0.
       - Otherwise, compute the shortest-path length between its current location
         and its goal location using breadth-first search on the undirected graph
         `self.undirected_neighbors`. This path length is taken as the estimated
         number of pushes required for that box (one push per edge traversed).
       - If there is no path in the static graph, treat that box's push cost as
         a large finite penalty (so the heuristic is still finite).

    4. Robot-travel estimate:
       - Compute distances from the current robot location to all other locations
         using BFS on the directed graph `self.directed_neighbors`, BUT only
         follow edges whose destination is currently declared in the state with
         qsw2k0xh (movement precondition). Each edge traversed is counted as one
         robot move (x042b0fi).
       - Approximate the robot cost as the minimum number of moves to reach any
         box-location that still needs to be moved. If no such box-location is
         reachable under current qsw2k0xh constraints, use a small bounded penalty
         instead of infinity (this keeps heuristic finite while still penalizing
         unreachable configurations).

    5. Combine costs:
       - Heuristic value = sum(push_costs for all boxes) + robot_travel_estimate.
       - Return that integer estimate.

    Notes
    -----
    - This heuristic is intentionally approximate: it ignores detailed label
      constraints on pushes and the exact interplays of qsw2k0xh toggling across
      pushes. Those interactions are expensive to reason about exactly and would
      slow down heuristic computation. The provided approximation is efficient
      and captures the dominant contributors to effort (distance to move boxes
      plus robot travel).
    """

    def __init__(self, task):
        # store goals and static facts
        self.goals = task.goals
        static_facts = task.static

        # adjacency structures
        # undirected neighbors (for box-push distance estimation)
        self.undirected_neighbors = {}
        # directed neighbors (for robot travel estimation)
        self.directed_neighbors = {}

        # collect all locations that appear in h80z8tvp facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                # undirected
                self.undirected_neighbors.setdefault(src, set()).add(dst)
                self.undirected_neighbors.setdefault(dst, set()).add(src)
                # directed
                self.directed_neighbors.setdefault(src, set()).add(dst)
                # ensure nodes exist even if they have no outgoing edges
                self.directed_neighbors.setdefault(dst, set())

        # parse goals to find box -> goal_location mapping
        # goal predicate is ldlu1rwo (box, location)
        self.box_goal = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.box_goal[box] = loc

    def _bfs_distance_undirected(self, start, target):
        """BFS on undirected_neighbors, return shortest path length or None if unreachable."""
        if start == target:
            return 0
        if start not in self.undirected_neighbors or target not in self.undirected_neighbors:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in self.undirected_neighbors.get(node, ()):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return None

    def _bfs_robot_distances(self, robot_loc, qsw2k0xh_set):
        """
        BFS on directed_neighbors but only allow traversing an edge u->v if v is in
        qsw2k0xh_set (movement precondition). Returns a dict location -> distance.
        If robot_loc is None, returns empty dict.
        """
        distances = {}
        if robot_loc is None:
            return distances
        if robot_loc not in self.directed_neighbors:
            distances[robot_loc] = 0
            return distances

        visited = {robot_loc}
        q = deque([(robot_loc, 0)])
        distances[robot_loc] = 0
        while q:
            node, d = q.popleft()
            for nb in self.directed_neighbors.get(node, ()):
                # movement to nb is allowed only if nb currently has qsw2k0xh true
                if nb not in qsw2k0xh_set:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    distances[nb] = d + 1
                    q.append((nb, d + 1))
        return distances

    def __call__(self, node):
        # quick check: goal reached -> heuristic 0
        state = node.state
        if self.goals <= state:
            return 0

        # parse current state
        # robot location: zdufuwne <loc>
        robot_loc = None
        qsw2k0xh_set = set()
        box_locations = {}  # box -> loc
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "zdufuwne" and len(parts) >= 2:
                robot_loc = parts[1]
            elif pred == "qsw2k0xh" and len(parts) >= 2:
                qsw2k0xh_set.add(parts[1])
            elif pred == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                box_locations[box] = loc

        total_pushes_est = 0
        large_penalty = max(50, len(self.undirected_neighbors) + 10)

        # For each box that has a goal, estimate pushes required
        for box, goal_loc in self.box_goal.items():
            curr_loc = box_locations.get(box)
            if curr_loc is None:
                # box location unknown in state (should not happen), penalize
                total_pushes_est += large_penalty
                continue
            if curr_loc == goal_loc:
                continue
            dist = self._bfs_distance_undirected(curr_loc, goal_loc)
            if dist is None:
                total_pushes_est += large_penalty
            else:
                total_pushes_est += dist

        # Robot distance estimate to nearest box that still needs moving
        robot_dist_est = 0
        # If there is no robot location information, we cannot compute travel — add small penalty
        if robot_loc is None:
            # If robot unknown, assume some travel is needed unless no pushes are needed
            robot_dist_est = 1 if total_pushes_est > 0 else 0
        else:
            robot_distances = self._bfs_robot_distances(robot_loc, qsw2k0xh_set)
            # find boxes that still need moving and are reachable; take minimal distance to their location
            min_dist = None
            for box, goal_loc in self.box_goal.items():
                curr_loc = box_locations.get(box)
                if curr_loc is None:
                    continue
                if curr_loc == goal_loc:
                    continue
                # robot ideally needs to get close to the box to start pushing.
                # We approximate by requiring robot to reach the box location itself (admissible simplification).
                d = robot_distances.get(curr_loc)
                if d is not None:
                    if min_dist is None or d < min_dist:
                        min_dist = d
            if min_dist is None:
                # no needed box is reachable under current qsw2k0xh restrictions;
                # use a bounded penalty to keep heuristic finite while encouraging states
                # that make boxes reachable.
                robot_dist_est = 3 if total_pushes_est > 0 else 0
            else:
                robot_dist_est = min_dist

        heuristic_value = total_pushes_est + robot_dist_est
        # ensure integer
        return int(heuristic_value)