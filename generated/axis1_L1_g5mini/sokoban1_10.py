from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return components of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    This heuristic estimates the number of actions required to reach the goal
    by combining:
      - an estimate of the minimal number of pushes required to move each
        box from its current location to its goal location (each push moves
        a box along one edge), and
      - a small estimate of the robot repositioning effort to start pushing
        (distance from the robot to a pushable neighbor of a box).

    The heuristic is intentionally not admissible; it trades admissibility
    for better guidance (fewer expanded nodes) by summing per-box push
    distances and a single robot positioning cost.

    Assumptions
    -----------
    - Facts of interest:
        * (ldlu1rwo BOX LOC)  -- a box is at a location
        * (zdufuwne LOC)      -- the robot is at a location
        * (h80z8tvp A B LABEL) -- a (directed) adjacency between locations A and B
    - A single operator (yi0kp1zg) performs a push of a box from B to C while
      the robot moves from A to B (push moves the box along one h80z8tvp edge).
      We therefore treat a single push as covering one graph edge for the box.
    - Robot movement between locations uses x042b0fi and costs one action per
      step along an adjacency.
    - For efficiency we ignore the push-label constraint (that two edges must
      share the same third-parameter label for a single push) and compute
      distances on the undirected adjacency graph induced by h80z8tvp.
      This produces an informative (but optimistic/approximate) estimate of
      pushes required.

    Heuristic Initialization
    ------------------------
    In the constructor we:
      - extract the goal mapping box -> goal_location from task.goals,
      - build an undirected adjacency graph of locations from static facts
        of the form (h80z8tvp A B LABEL),
      - precompute shortest-path distances (in number of edges) between all
        locations in the adjacency graph using BFS for each node. These
        distances are used at runtime to quickly estimate pushes and robot
        travel distances.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
      1. If the state's facts already satisfy all goals, return 0.
      2. Read the robot location from any fact (zdufuwne LOC) in the state.
      3. For each box that has a goal (from initialization):
         a. Read its current location from (ldlu1rwo BOX LOC) in the state.
         b. If the box is already at its goal, its contribution is 0.
         c. Otherwise, look up the precomputed shortest-path distance (in
            graph edges) from its current location to its goal location.
            Each edge corresponds to one push of the box → pushes_needed.
         d. If no path exists in the static adjacency graph (unreachable),
            return a large but finite heuristic (we treat the state as hard to
            solve but keep finiteness).
      4. Sum pushes_needed across all boxes (this estimates the number of
         yi0kp1zg push actions required).
      5. Robot positioning: find the minimal robot distance (in robot moves)
         to reach a suitable neighbour-of-box position to perform the first
         push. Concretely, for each box not at goal, consider its neighbors
         in the adjacency graph and take the shortest (robot -> neighbor)
         distance using precomputed distances. Take the minimum across all
         such neighbors/boxes. If no robot fact is present, treat robot
         positioning cost as 0.
      6. Heuristic value = sum(pushes_needed) + minimal_robot_positioning_cost.
         If any required distance was unavailable (unreachable), return a large
         finite number.

    Note: The heuristic ignores many Sokoban constraints (push direction label
    consistency and box-blocking interactions) for efficiency. This yields a
    fast, informative estimate that usually reduces node expansions even
    though it may not always be tight or admissible.
    """

    def __init__(self, task):
        """
        Extract static information and goals, and precompute location distances.

        - task.goals: a set of goal facts (strings).
        - task.static: static facts (frozenset of strings), from which we
                       extract h80z8tvp adjacency facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Map boxes to their goal locations from goal facts.
        self.goal_locations = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "ldlu1rwo":
                # format: (ldlu1rwo box location)
                _, box, loc = parts
                self.goal_locations[box] = loc

        # Build undirected adjacency graph from static h80z8tvp facts.
        # We ignore the third parameter (label) for the graph.
        self.adj = {}  # node -> set(neighbor nodes)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp" and len(parts) >= 3:
                _, a, b, *rest = parts
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Collect the set of all known locations (graph nodes).
        self.locations = set(self.adj.keys())

        # Precompute all-pairs shortest path distances (in edges) between locations.
        # We'll use BFS from each node. If the graph is disconnected, unreachable
        # pairs will simply not be present in the dict for that start node.
        self.distances = {}  # start -> {target: distance}
        for start in self.locations:
            dist = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for nb in self.adj.get(cur, ()):
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            self.distances[start] = dist

        # A large finite value to return for (practically) unreachable/goalless cases.
        self.INF = 10**6

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node/state."""
        state = node.state

        # 1) Goal check: if all goals satisfied, heuristic = 0.
        if self.goals <= state:
            return 0

        # 2) Extract robot location (there should be exactly one zdufuwne fact).
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "zdufuwne":
                # format: (zdufuwne loc)
                if len(parts) >= 2:
                    robot_loc = parts[1]
                break  # only one robot location expected

        # 3) Extract current box locations for boxes that appear in goals.
        box_current = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                _, box, loc = parts
                if box in self.goal_locations:
                    box_current[box] = loc

        # Ensure every goal-box has a recorded current location. If missing,
        # treat as unreachable (very costly).
        for box in self.goal_locations:
            if box not in box_current:
                return self.INF

        total_pushes = 0
        boxes_needing_push = []  # boxes not yet at goal

        # 4) For each box, compute minimal number of pushes (graph-edge distance) to goal.
        for box, goal_loc in self.goal_locations.items():
            cur_loc = box_current[box]
            if cur_loc == goal_loc:
                continue
            # If either location is not in the static graph, unreachable.
            if cur_loc not in self.distances or goal_loc not in self.locations:
                return self.INF
            dist_map = self.distances[cur_loc]
            if goal_loc not in dist_map:
                return self.INF
            pushes_needed = dist_map[goal_loc]
            total_pushes += pushes_needed
            boxes_needing_push.append(cur_loc)

        # 5) Robot positioning estimate:
        # Find minimal distance (in robot moves) from robot to any neighbor-of-box
        # that would allow performing the first push. For box at position B,
        # a robot must be at some node A neighbor to B to push B along edge B->C;
        # we conservatively consider any neighbor A of B as a candidate.
        robot_pos_cost = 0
        if boxes_needing_push and robot_loc is not None:
            min_cost = None
            # Precompute robot distance map if robot_loc in graph
            if robot_loc in self.distances:
                robot_dist_map = self.distances[robot_loc]
            else:
                robot_dist_map = {}
            for bpos in boxes_needing_push:
                # consider neighbors of bpos; if none, box is stuck -> INF
                neighbors = self.adj.get(bpos, ())
                if not neighbors:
                    return self.INF
                # For each neighbor, look up distance robot->neighbor
                for nb in neighbors:
                    if nb == robot_loc:
                        # robot already in right spot
                        candidate = 0
                    else:
                        candidate = robot_dist_map.get(nb, None)
                        if candidate is None:
                            # If robot cannot reach this neighbor, skip it
                            continue
                    if min_cost is None or candidate < min_cost:
                        min_cost = candidate
            if min_cost is None:
                # Robot cannot reach any neighbor of any box -> treat as unreachable
                return self.INF
            robot_pos_cost = min_cost
        else:
            # No boxes need pushing or no robot info -> no robot positioning cost.
            robot_pos_cost = 0

        # 6) Final heuristic is pushes + robot positioning cost.
        hval = total_pushes + robot_pos_cost

        # Safety: ensure finite and integer
        if hval >= self.INF:
            return self.INF
        return int(hval)