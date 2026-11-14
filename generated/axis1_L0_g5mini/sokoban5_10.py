from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

INF = 10 ** 6


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for Sokoban. The heuristic estimates the number
    of actions required to reach a goal state by combining:
      - the number of pushes needed to move each box from its current location
        to its goal location (computed on the static adjacency graph),
      - an estimate of the robot travel cost to reach a pushing position for the
        nearest box,
      - a small overhead per non-goal box to account for maneuvering.
    The heuristic is intentionally not admissible but attempts to be informative
    while remaining fast to compute.

    Assumptions
    -----------
    - Static facts include directed adjacency facts of the form
      "(adjacent locA locB dir)" for all passable neighbor pairs. Both directions
      are typically present (with complementary directions).
    - Goals are conjunctions of box-location facts "(at boxX locY)". We use
      these to determine the desired final location for each box.
    - A push moves a box one graph edge toward its goal; thus the number of
      pushes required equals the shortest-path distance (in edges) between the
      box's current location and its goal location (ignoring other boxes).
    - Deadlocks that are trivially detectable (box resting in a static corner
      or dead-end that is not a goal) are treated as unsolvable and mapped to a
      large heuristic value to strongly discourage searching that branch.

    Heuristic Initialization
    ------------------------
    The constructor extracts static adjacency information and:
      - builds an adjacency map for all locations,
      - records the direction label for each directed edge (used for simple
        corner/dead-end detection),
      - precomputes shortest-path distances between all locations using BFS,
      - extracts the mapping of boxes to their goal locations from the task goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If the state's goals are already satisfied, return 0.
    2. Parse the state to find the robot location and the current location of
       every box.
    3. For every box that is not already at its goal:
       a) If the box is placed in a trivial static deadlock (a dead-end or a
          corner determined from adjacency directions) and that location is not
          a goal, return a very large heuristic (treat as unsolvable).
       b) Compute the number of pushes required as the precomputed shortest-path
          distance (number of edges) from the box's current location to its
          goal location. If no path exists in the static adjacency graph, treat
          it as unsolvable (large heuristic).
    4. Sum the pushes required for all boxes (this lower-level estimate assumes
       no dynamic blocking by other boxes).
    5. Estimate robot travel cost: compute the minimal static shortest-path
       distance from the robot's current location to any neighbour cell of any
       non-goal box (the robot must reach a cell adjacent to a box to start
       pushing it). If that is unreachable, return large heuristic.
    6. Combine estimates:
       heuristic = sum_of_pushes + robot_travel_to_nearest_push_pos
                   + overhead_per_box (small integer per box not at goal).
    7. Return the resulting integer heuristic value.

    Notes
    -----
    - This heuristic purposely keeps computation cheap by using static shortest
      paths and only simple deadlock checks (degree 1 dead-ends and degree 2
      perpendicular corners). It does not attempt to detect complex deadlocks
      caused by interactions among boxes.
    - The heuristic value is finite for solvable states because distances on the
      static graph are finite when a path exists; unsolvable or obviously
      deadlocked configurations are mapped to a large finite value.
    """

    def __init__(self, task):
        """
        Initialize static structures:
          - adjacency: dict location -> set(neighbor locations)
          - adj_dir: dict location -> dict(neighbor -> direction)
          - all_locs: set of all locations in the static graph
          - distances: dict loc -> dict(loc2 -> distance in edges)
          - goal_for_box: dict box -> goal_location
          - goal_locations: set of locations that are box goals
        """
        self.goals = task.goals
        self.static = task.static

        # Build adjacency map and store direction labels for edges.
        self.adjacency = defaultdict(set)
        self.adj_dir = defaultdict(dict)
        all_locs = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                # fact: (adjacent locA locB dir)
                _, loc_a, loc_b, direction = parts
                self.adjacency[loc_a].add(loc_b)
                self.adj_dir[loc_a][loc_b] = direction
                all_locs.add(loc_a)
                all_locs.add(loc_b)

        self.all_locs = all_locs

        # Precompute BFS distances between all locations.
        # distances[loc][other] = minimal number of edges (pushes/moves) from loc to other
        self.distances = {}
        for src in all_locs:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adjacency.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.distances[src] = dist

        # Extract box goals from task.goals. Goals are strings like "(at box1 loc_2_4)".
        self.goal_for_box = {}
        self.goal_locations = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "at":
                _, box, loc = parts
                self.goal_for_box[box] = loc
                self.goal_locations.add(loc)

    def __call__(self, node):
        """
        Compute the heuristic estimate for the node's state.

        Returns an integer estimate of remaining actions to reach the goal.
        """
        state = node.state

        # Quick goal check: if all goals are satisfied return 0
        if frozenset(self.goals) <= state:
            return 0

        # Parse robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
                break
        if robot_loc is None:
            # No robot location in state -> treat as very bad
            return INF

        # Parse box locations (facts of form "(at boxX locY)")
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # keep entries for things that look like boxes (we expect box names to match those in goals)
                if obj in self.goal_for_box or obj.startswith("box"):
                    box_locs[obj] = loc

        # For boxes mentioned in goals but not present in state (unlikely), assume large cost:
        for box in self.goal_for_box:
            if box not in box_locs:
                box_locs[box] = None

        total_pushes = 0
        boxes_not_at_goal = 0

        # Precompute occupied locations by boxes (for simple deadlock tests)
        occupied_by_box = {loc for loc in box_locs.values() if loc is not None}

        # Simple deadlock detection: if a box is on a static dead-end or in a corner (degree 1 or perpendicular degree 2)
        # and that location is not a goal, treat as unsolvable -> large heuristic.
        def is_deadlock_location(loc):
            # If location not in static graph treat as deadlocked
            if loc not in self.all_locs:
                return True
            neighbors = list(self.adjacency.get(loc, ()))
            deg = len(neighbors)
            if deg == 0:
                return True
            if deg == 1 and loc not in self.goal_locations:
                return True
            if deg == 2 and loc not in self.goal_locations:
                # Check whether the two directions are opposite (straight corridor) or perpendicular (corner).
                n1, n2 = neighbors
                dir1 = self.adj_dir.get(loc, {}).get(n1)
                dir2 = self.adj_dir.get(loc, {}).get(n2)
                if dir1 is None or dir2 is None:
                    # Missing direction info means be conservative and consider deadlock
                    return True
                # Opposite pairs: left-right or up-down
                opposite = {("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")}
                if (dir1, dir2) not in opposite:
                    # perpendicular -> corner -> deadlock (if not a goal)
                    return True
            return False

        for box, loc in box_locs.items():
            if loc is None:
                # missing box info -> penalize heavily
                return INF
            goal_loc = self.goal_for_box.get(box)
            if goal_loc is None:
                # No explicit goal for this box (shouldn't happen), add small overhead and continue
                boxes_not_at_goal += 1
                continue
            if loc != goal_loc:
                boxes_not_at_goal += 1

                # Deadlock detection
                if is_deadlock_location(loc):
                    # Box is stuck in a corner/dead-end not a goal -> unsolvable branch
                    return INF

                # Number of pushes needed is the shortest path distance between box location and goal location.
                dist_map = self.distances.get(loc, {})
                pushes_needed = dist_map.get(goal_loc)
                if pushes_needed is None:
                    # No static path -> unsolvable
                    return INF
                total_pushes += pushes_needed

        # If no boxes need moving, cost is simply robot travel to finish (0)
        if boxes_not_at_goal == 0:
            return 0

        # Estimate robot travel cost to reach a pushing position:
        # For each non-goal box, consider all neighbor cells of the box as possible push positions.
        # Robot needs to reach at least one such neighbor to start pushing.
        min_robot_travel = INF
        for box, loc in box_locs.items():
            goal_loc = self.goal_for_box.get(box)
            if goal_loc is None or loc == goal_loc:
                continue
            # neighbors of box where robot could stand to push
            neighbors = self.adjacency.get(loc, ())
            # For each neighbor, compute robot distance to that neighbor (from precomputed distances)
            for nbr in neighbors:
                # The robot must be able to reach this neighbor ignoring box occupancy (static graph)
                dist_from_robot = self.distances.get(robot_loc, {}).get(nbr)
                if dist_from_robot is None:
                    continue
                if dist_from_robot < min_robot_travel:
                    min_robot_travel = dist_from_robot

        if min_robot_travel == INF:
            # Robot cannot reach any pushing position -> unsolvable in static graph
            return INF

        # Combine costs. Each push is one push action. Robot travel is counted as move actions.
        # Add a small overhead per non-goal box to account for maneuvering around boxes,
        # pick the value 1 per box.
        overhead_per_box = 1
        heuristic_value = total_pushes + min_robot_travel + overhead_per_box * boxes_not_at_goal

        # Cap heuristic to INF - 1 to avoid returning the exact INF sentinel
        if heuristic_value >= INF:
            return INF - 1
        return int(heuristic_value)