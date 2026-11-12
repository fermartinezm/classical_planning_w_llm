from collections import deque, defaultdict
from itertools import permutations
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing the surrounding
    parentheses and splitting on whitespace.

    Example:
      "(at box1 loc_3_5)" -> ["at", "box1", "loc_3_5"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain.

    This heuristic estimates the number of actions (moves + pushes) required to
    reach a goal state from the current state. It is intended for greedy
    best-first search and therefore does not need to be admissible. The
    heuristic blends a simple estimate of the number of pushes still required
    (boxes -> goal assignments) with an estimate of the robot travel cost to
    reach a box to start pushing.

    Assumptions
    -----------
    - Goals are of the form (at <box> <location>); every goal refers to a box.
    - Static facts include (adjacent <loc1> <loc2> <direction>) facts that
      define the grid connectivity. Both directions may be present as separate
      facts; we treat adjacency as a bidirectional relationship if present.
    - PDDL facts in states follow the domain predicates:
        (at-robot <loc>), (at <box> <loc>), (clear <loc>), (adjacent ...)
    - We assume push and move actions each count as 1 unit of cost.
    - The heuristic is allowed to be optimistic or optimistic on some parts;
      priority is computational efficiency and good guidance for greedy search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goal_boxes: mapping from goal box name -> goal location (from task.goals)
    - self.adj: adjacency dictionary mapping each location to a set of neighboring locations
      (built from static facts). We interpret adjacent facts as edges in a graph.
    - self.locations: set of all known locations (keys of adjacency map)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state node:
    1. If the goal is already reached (task.goals subset of state), return 0.
    2. Parse the state's facts to get:
       - robot_loc: where the robot currently stands (from "(at-robot ...)")
       - boxes: dictionary mapping box name -> current location (from "(at <box> <loc>)")
       - occupied: set of locations currently occupied by boxes (used to restrict robot movement)
    3. Estimate remaining pushes:
       - Build a cost matrix C where C[i][j] is the shortest graph distance (number of steps)
         from box_i's location to goal_j's location using the adjacency graph.
         This distance is a simple proxy for the number of pushes needed to move the box
         to that goal assuming an unobstructed push-path.
       - Assign boxes to goals by a greedy minimum-cost matching: repeatedly pick the
         smallest remaining pair and remove its row and column. (This is fast and
         effective for the small number of boxes typically found in Sokoban problems.)
       - Sum the chosen distances -> pushes_sum. Each unit corresponds to one push action.
    4. Estimate robot travel to start pushing:
       - For each box, consider its neighbors (cells from which a push could be initiated).
         A valid neighbor must be present in the adjacency graph and must be currently clear
         (not occupied by any box).
       - Compute the shortest distance (in robot move actions) from robot_loc to any valid
         neighbor, using BFS on the adjacency graph while treating occupied cells as blocked.
       - Take the minimal such robot distance among all boxes -> robot_dist_to_box.
       - If robot is already adjacent to a chosen box, this distance can be zero.
    5. Combine estimates:
       - heuristic_value = pushes_sum + robot_dist_to_box
         (pushes account for push actions; robot_dist accounts for initial movement to the first box).
       - If either pushes_sum or the robot distance is found to be unreachable in the graph,
         a large finite penalty is returned to keep the heuristic finite but discouraging.
    6. Return the heuristic value (integer).

    Implementation notes:
    - Distances for box->goal are computed ignoring other boxes (fast, optimistic).
    - Robot path distances are computed considering boxes as obstacles (more realistic).
    - The matching is greedy (O(n^2)), which is usually sufficient and much cheaper than
      exact assignment for the small numbers of boxes encountered.
    """

    def __init__(self, task):
        # Goals: map box -> goal location(s). We store a list of goal locations (usually one per box).
        self.goals = task.goals
        self.goal_locations = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "at" and len(parts) >= 3:
                _, box, loc = parts[:3]
                self.goal_locations.append((box, loc))

        # Build adjacency graph from static facts.
        self.adj = defaultdict(set)
        for fact in task.static:
            parts = get_parts(fact)
            # expected structure: ["adjacent", loc1, loc2, dir]
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                _, l1, l2 = parts[:3]
                # Treat adjacency as directed as given, but also store the reverse if present in static facts.
                # To be robust, we add the directed edge; many instances include both directions.
                self.adj[l1].add(l2)
                # We do not force l2->l1 here; if the static facts contain it, it will be added too.

        # Collect known locations
        self.locations = set(self.adj.keys())
        for nbrs in self.adj.values():
            self.locations.update(nbrs)

        # A large finite penalty for unreachable situations
        self.LARGE = 10 ** 6

    def bfs_distance(self, start, goals, avoid=None, max_dist=None):
        """
        BFS from start to the nearest node in goals.
        - start: starting location (string)
        - goals: a set of target locations (strings) or a single location (string)
        - avoid: a set of nodes to treat as blocked (e.g., box-occupied locations)
        - max_dist: optional cutoff for early termination

        Returns minimal distance (int) or None if unreachable.
        """
        if start is None or goals is None:
            return None
        if isinstance(goals, str):
            goals = {goals}
        avoid = avoid or set()
        if start in goals:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            if max_dist is not None and d >= max_dist:
                continue
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in avoid:
                    continue
                if nbr in goals:
                    return d + 1
                visited.add(nbr)
                q.append((nbr, d + 1))
        return None

    def get_box_goal_cost_matrix(self, boxes, goals):
        """
        Return a matrix (list of lists) of distances between each box location and each goal location.
        distances are computed by BFS on the adjacency graph, ignoring other boxes (optimistic).
        If no path exists, a large penalty is inserted instead of None.
        """
        n = len(boxes)
        m = len(goals)
        matrix = [[self.LARGE for _ in range(m)] for _ in range(n)]
        for i, (_, bloc) in enumerate(boxes):
            for j, (_, gloc) in enumerate(goals):
                if bloc == gloc:
                    matrix[i][j] = 0
                else:
                    d = self.bfs_distance(bloc, gloc, avoid=set())
                    matrix[i][j] = d if d is not None else self.LARGE
        return matrix

    def greedy_minimum_matching_cost(self, cost_matrix):
        """
        Greedy matching: repeatedly pick the smallest remaining cost entry and remove
        its row and column. Returns the sum of chosen costs.
        This is O(n^2) and effective for small n.
        """
        total = 0
        n = len(cost_matrix)
        if n == 0:
            return 0
        rows = set(range(n))
        cols = set(range(len(cost_matrix[0])))
        # Create a list of (cost, row, col) and sort it
        entries = []
        for i in rows:
            for j in cols:
                entries.append((cost_matrix[i][j], i, j))
        entries.sort(key=lambda x: x[0])
        matched_rows = set()
        matched_cols = set()
        for cost, i, j in entries:
            if i in matched_rows or j in matched_cols:
                continue
            matched_rows.add(i)
            matched_cols.add(j)
            total += cost
            if len(matched_rows) == n or len(matched_cols) == n:
                break
        return total

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Extract robot location, box locations and occupied locations
        robot_loc = None
        boxes = []  # list of tuples (box_name, location)
        occupied = set()  # locations with boxes

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
            elif pred == "at" and len(parts) >= 3:
                obj = parts[1]
                loc = parts[2]
                # In Sokoban domain, these "at" facts refer to boxes.
                boxes.append((obj, loc))
                occupied.add(loc)

        # If there are no boxes or no goals specified, fallback to distance from robot to goal area
        if not boxes or not self.goal_locations:
            # fallback: distance from robot to any goal location (if available)
            goal_locs = {loc for (_, loc) in self.goal_locations}
            if robot_loc is None:
                return 0
            d = self.bfs_distance(robot_loc, goal_locs, avoid=occupied)
            return d if d is not None else self.LARGE

        # 3) Compute box->goal cost matrix (push counts). Ignore other boxes for this computation.
        # Build lists of boxes and goals (we only need goal locations; boxes may not be same names as goal boxes).
        goal_list = list(self.goal_locations)
        box_list = boxes
        # If number of boxes and number of goals differ, match as many as possible by using min(len(boxes), len(goals))
        # For matrix build, we align rows with boxes and columns with goals.
        cost_matrix = self.get_box_goal_cost_matrix(box_list, goal_list)

        # 4) Find a (greedy) minimal assignment and sum pushes
        pushes_sum = self.greedy_minimum_matching_cost(cost_matrix)

        # 5) Compute robot distance to nearest push-adjacent cell (consider boxes as obstacles)
        robot_dist_to_box = self.LARGE
        if robot_loc is None:
            robot_dist_to_box = self.LARGE
        else:
            for _, bloc in box_list:
                # for each neighbor of the box location, the robot can stand there to push
                neighbors = self.adj.get(bloc, set())
                valid_neighbors = [n for n in neighbors if n not in occupied]
                if not valid_neighbors:
                    # no valid pushing position for this box currently
                    continue
                d = self.bfs_distance(robot_loc, set(valid_neighbors), avoid=occupied)
                if d is not None:
                    robot_dist_to_box = min(robot_dist_to_box, d)

        if robot_dist_to_box == self.LARGE and pushes_sum == 0:
            # No pushes required but robot unreachable -> weird state, return 0 if goals satisfied, else small penalty.
            return 1

        # 6) Combine estimates: pushes are push actions; robot distance counts move actions to reach first push.
        # This is intentionally simple: we count the robot travel to the first box plus the sum of pushes.
        heuristic_value = pushes_sum + (robot_dist_to_box if robot_dist_to_box != self.LARGE else 0)

        # Cap to LARGE if something was unreachable
        if pushes_sum >= self.LARGE:
            return self.LARGE
        return int(heuristic_value)