from itertools import permutations
from collections import deque
from heuristics.heuristic_base import Heuristic
import math


def get_parts(fact):
    """Return the parts of a PDDL fact string without surrounding parentheses."""
    # fact is like "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain. The heuristic estimates
    the number of actions (robot moves + pushes) required to bring all boxes to
    their goal locations. It is not guaranteed to be admissible, but it aims to
    be informative and cheap to compute to guide greedy best-first search.

    Assumptions
    -----------
    - Every "at" goal refers to a box being at a specific location.
    - The static adjacency relation describes the (grid-like) connectivity of
      locations. Adjacency facts are given as (adjacent l1 l2 dir).
    - We ignore dynamic blocking by other boxes when computing shortest-path
      distances (i.e., distances are computed on the static graph).
    - The number of boxes in typical instances is small enough that trying
      all permutations for an optimal assignment is feasible; if too large,
      a greedy assignment is used as fallback.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract the set of goal facts (expected to be of the form "(at boxX locY)").
      We collect goal locations for each goal (goal_locations list).
    - Parse static facts to build an undirected graph of locations using the
      "adjacent" static facts.
    - Precompute shortest-path distances (in number of edges) between all
      locations using BFS from each location. These distances represent the
      minimal number of robot moves (ignoring boxes) between two locations,
      and the minimal number of single-tile pushes needed to move a box from
      one location to another (one push moves the box by one edge).
    - Store the list of all recognized locations (nodes of the graph).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. If the state's facts already satisfy all goal facts, return 0.

    2. Extract current robot location and current box locations from the state
       (facts "(at-robot ...)" and "(at box ...)" respectively).

    3. For each pair (box_i, goal_j) compute the minimal number of pushes to
       move box_i from its current location to goal_j. This is simply the
       static shortest-path distance between the two locations in the adjacency
       graph (if no path exists, a large penalty is used).

    4. Solve a minimum-weight assignment problem that pairs each box with a
       unique goal to minimize the sum of pushes. For small numbers of boxes
       (<= 8) an optimal assignment is computed by trying all permutations.
       For larger numbers a greedy assignment is used (assign each box to the
       nearest available goal). The sum of assigned shortest-path distances is
       named total_pushes (estimated number of pushes required).

    5. Estimate the robot travel cost to begin pushing. The robot must at least
       reach a square adjacent to some box to start the first push. We
       approximate this as the minimal static distance from the robot to any
       location that is adjacent to any box. (If the robot is already adjacent
       to some box that can be pushed, this cost will be 0.)

    6. Account for additional robot movement during the sequence of pushes.
       A simple but effective approximation is to assume that, in addition to
       the push actions themselves, the robot needs roughly one move per push
       to reposition itself between pushes (this models the robot moving
       around the box to get into pushing position). Therefore we add an
       extra cost proportional to total_pushes.

    7. The resulting heuristic value is:
         heuristic = total_pushes (push actions)
                     + robot_to_box_cost (initial robot travel)
                     + total_pushes (approximate repositioning moves)
       which simplifies to:
         heuristic = 2 * total_pushes + robot_to_box_cost

    Notes on design choices
    -----------------------
    - Distances ignore box blocking. This typically underestimates the true
      number of robot moves but yields an optimistic and fast-to-compute
      heuristic that still captures structural difficulty (how far boxes are
      from goals).
    - Using the optimal assignment of boxes to goals reduces overcounting for
      multiple-box problems.
    - The factor "2 * total_pushes" (pushes + repositioning moves) is a tuned
      compromise between under- and over-estimation that generally leads to
      fewer node expansions in greedy search compared to using only pushes.
    """

    def __init__(self, task):
        # Goals (set of strings) and static facts
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations: list of locations that boxes must reach.
        # We keep the goals as a list of location strings; goal identities (which
        # box to which goal) are decided by the assignment algorithm.
        self.goal_locations = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "at" and len(parts) == 3:
                _, box, loc = parts
                # store target location (we do not assume which box)
                self.goal_locations.append(loc)

        # Build adjacency graph from static facts: treat as undirected graph.
        self.graph = {}  # loc -> set(neighbor_locs)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                # (adjacent l1 l2 dir)
                _, l1, l2 = parts[:3]
                self.graph.setdefault(l1, set()).add(l2)
                # Note: static may already contain reverse adjacency; ensure undirected
                self.graph.setdefault(l2, set()).add(l1)

        # Ensure we have a canonical list of locations
        self.locations = sorted(self.graph.keys())

        # Precompute BFS distances between all pairs of locations (number of edges)
        # distances[loc][other_loc] = integer distance or math.inf if unreachable
        self.distances = {}
        for loc in self.locations:
            dist = {loc: 0}
            q = deque([loc])
            while q:
                u = q.popleft()
                for v in self.graph.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Fill missing nodes with infinity
            full_dist = {other: dist.get(other, math.inf) for other in self.locations}
            self.distances[loc] = full_dist

    def _min_cost_assignment(self, cost_matrix):
        """
        Compute a (near-)optimal assignment for the square matrix cost_matrix
        where cost_matrix[i][j] is the cost of assigning row i to column j.

        For small sizes (<=8) we try all permutations for optimality. For larger
        sizes we fall back to a greedy assignment for performance.
        """
        n = len(cost_matrix)
        if n == 0:
            return 0

        # If small, try all permutations (optimal)
        if n <= 8:
            min_cost = math.inf
            for perm in permutations(range(n)):
                cost = 0
                for i, j in enumerate(perm):
                    c = cost_matrix[i][j]
                    cost += c
                    if cost >= min_cost:
                        break
                if cost < min_cost:
                    min_cost = cost
            return min_cost

        # Greedy fallback: assign each row to its nearest free column
        min_cost = 0
        free_cols = set(range(n))
        # rows sorted by their minimum available cost to try to reduce mistakes
        row_order = sorted(range(n), key=lambda r: min(cost_matrix[r][c] for c in free_cols))
        for r in row_order:
            best_c = None
            best_cost = math.inf
            for c in free_cols:
                if cost_matrix[r][c] < best_cost:
                    best_cost = cost_matrix[r][c]
                    best_c = c
            if best_c is None:
                # no available column (should not happen)
                return math.inf
            min_cost += best_cost
            free_cols.remove(best_c)
        return min_cost

    def __call__(self, node):
        state = node.state

        # 1) If all goal facts already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Extract robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                break

        # If robot location not found in state (should not happen), assume far away.
        if robot_loc is None:
            robot_loc = None

        # 3) Extract current box locations
        boxes = []  # list of (box_name, loc)
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                # (at boxX locY)
                obj, loc = parts[1], parts[2]
                # Heuristic cares only about boxes that appear in goal set (or any box)
                boxes.append((obj, loc))

        # Filter boxes to only those that correspond to box objects involved in goals
        # Create list of current box locations for the number of goals (we only need to
        # move boxes that have target locations). If there are more boxes than goals,
        # we consider only the boxes that currently are not already at any goal.
        goal_locs = list(self.goal_locations)  # copy

        if not goal_locs:
            # No explicit box goals => nothing to do
            return 0

        # Build mapping box -> current location for boxes appearing in state
        # Keep only boxes that are not already at a goal location (since those don't require pushes)
        box_current_locs = []
        # set of goal locations for quick check
        goal_loc_set = set(goal_locs)
        for obj, loc in boxes:
            if loc in goal_loc_set:
                # If this box is already at some goal location, it may still be needed
                # to satisfy matching (if multiple boxes target same location), but we
                # simply prefer to not include boxes already on any goal.
                continue
            box_current_locs.append((obj, loc))

        # If all boxes already on goals (or no boxes relevant), return small cost:
        if not box_current_locs:
            # Might still need robot moves to satisfy goals if e.g. wrong box is on a goal;
            # but we return 0 as goals <= state would have caught exact match. Use 0.
            return 0

        # Number of boxes to consider and goals
        n_boxes = len(box_current_locs)
        n_goals = len(goal_locs)

        # For assignment we want square matrix of size max(n_boxes, n_goals)
        n = max(n_boxes, n_goals)
        INF = 10 ** 6

        # Build cost matrix: cost of pushing box i to goal j = shortest-path distance
        # between current location and goal location (in number of edges).
        # If location not present in precomputed distances, treat as INF.
        cost_matrix = [[INF] * n for _ in range(n)]
        for i in range(n_boxes):
            _, box_loc = box_current_locs[i]
            for j in range(n_goals):
                goal_loc = goal_locs[j]
                if box_loc in self.distances and goal_loc in self.distances[box_loc]:
                    d = self.distances[box_loc][goal_loc]
                    if d == math.inf:
                        cost = INF
                    else:
                        cost = int(d)  # number of pushes needed
                else:
                    cost = INF
                cost_matrix[i][j] = cost

        # Fill dummy rows/cols with INF to make square
        # (already initialized to INF)

        # Compute minimal total pushes via assignment
        total_pushes = self._min_cost_assignment(cost_matrix)
        if total_pushes >= INF:
            # Unreachable assignment (no path); return a large finite heuristic
            return INF // 10

        # Estimate robot travel to start pushing:
        # For each box location compute min distance from robot to any neighbor of the box.
        # That is the minimal number of moves for the robot to reach a pushing position.
        robot_to_box_cost = math.inf
        if robot_loc is not None and robot_loc in self.locations:
            for _, box_loc in box_current_locs:
                # neighbors of box_loc (positions from which the robot can push the box)
                neighbors = self.graph.get(box_loc, ())
                for neigh in neighbors:
                    # distance from robot to neigh
                    if robot_loc in self.distances and neigh in self.distances[robot_loc]:
                        d = self.distances[robot_loc][neigh]
                        if d < robot_to_box_cost:
                            robot_to_box_cost = d
        else:
            # If robot location unknown, assume a conservative estimate
            robot_to_box_cost = 5

        if robot_to_box_cost == math.inf:
            # unreachable neighbor (shouldn't normally happen), use a penalty
            robot_to_box_cost = 10

        # Combine costs:
        # - total_pushes actions for pushing
        # - robot_to_box_cost to reach the first pushing position
        # - approximately one extra robot move per push for repositioning between pushes
        heuristic_value = int(2 * total_pushes + robot_to_box_cost)

        # Ensure heuristic is finite and at least 1 for non-goal states
        if heuristic_value < 1:
            heuristic_value = 1

        return heuristic_value