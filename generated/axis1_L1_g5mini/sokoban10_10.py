from fnmatch import fnmatch
from collections import deque
from itertools import permutations
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.

    The heuristic estimates the number of push-actions (and thus roughly the
    number of primitive actions) required to achieve the box-placement goals.
    It computes, for each box, the shortest number of single-box pushes needed
    to reach its goal position on the static connectivity graph, and then
    combines these estimates into a single cost:
      - If goals bind boxes to specific goal-locations (e.g. (ldlu1rwo box1 locX))
        the heuristic sums the shortest push-distances from each box to its
        designated goal.
      - If goals only state that some boxes must occupy some locations
        (multiple goal locations without box-to-location mapping), the heuristic
        computes the minimum sum of push-distances over all assignments
        between current box positions and goal locations (optimal matching by
        permutation enumeration, suitable when number of boxes is small).

    Assumptions
    -----------
    - Static predicate `h80z8tvp A B L` encodes adjacency between locations A and B.
      The heuristic treats adjacency as undirected: a single push moves a box
      from a location to an adjacent location (ignoring label constraints and
      dynamic occupancy).
    - The heuristic ignores the robot's precise position and the need for the
      robot to walk around to perform pushes. (This keeps it fast to compute.)
    - The heuristic ignores temporary blocking by other boxes (i.e., it uses
      a relaxed model where boxes do not block each other).
    - If a box cannot reach its goal on the static graph, a large finite value
      is returned to indicate (practically) that the state is bad/unsolvable.

    Heuristic Initialization
    ------------------------
    - Extract goal facts from task.goals and build:
        - A mapping box -> specific goal location for goals that bind each box.
        - If goals do not bind boxes explicitly, collect the set of goal locations.
    - Extract static adjacency from task.static by parsing `h80z8tvp` facts and
      building an undirected adjacency dictionary for all location nodes.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state already satisfies all goals, return 0.
    2. Parse the current state to obtain the list of boxes and their current locations
       (facts of the form `(ldlu1rwo box loc)`).
    3. For each relevant box-goal pair (either as explicitly mapped in the goals or
       for combinations when goals are not bound to boxes), compute the shortest
       number of pushes needed to move the box from its current location to the
       goal location using BFS on the undirected adjacency graph built from
       `h80z8tvp` static facts. Each BFS step corresponds to one push.
    4. Combine per-box push distances:
       - If goals specify a unique goal location per box, sum those distances.
       - Otherwise, choose an assignment of boxes to goal locations that minimizes
         the total sum (all permutations are considered; this is practical since
         the number of boxes in typical instances is small).
    5. If any required distance is unreachable (no path on the static graph),
       return a large finite constant to reflect that reaching the goal requires
       breaking assumptions (practically unsolvable state).
    6. Return the resulting sum as the heuristic estimate (an integer).
    """

    def __init__(self, task):
        """
        Initialize the heuristic using static facts and the problem goals.

        Parameters:
        - task: an instance of the Task class (provides .goals and .static).
        """
        self.goals = task.goals
        static_facts = task.static

        # Build undirected adjacency graph of locations from static h80z8tvp facts.
        # Each fact has the form: (h80z8tvp locA locB LABEL)
        self.adj = {}  # node -> set(neighbor_nodes)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "h80z8tvp" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Parse goals:
        # If goals bind specific boxes to target locations, record mapping box->loc.
        # Otherwise collect goal locations to be matched to boxes later.
        self.goal_for_box = {}   # box_name -> goal_location
        self.goal_locations = []  # list of goal locations (if not box-bound)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                # Record mapping (there may be multiple goals; last wins if repeated)
                self.goal_for_box[box] = loc
                self.goal_locations.append(loc)

        # If goal locations were provided but not uniquely bound to box names,
        # keep them unique (useful when goals don't mention box names explicitly).
        self.goal_locations = list(dict.fromkeys(self.goal_locations))

        # A large finite penalty for unreachable goals (keeps heuristic finite).
        self.large_cost = 100000

    def bfs_distance(self, start, target):
        """
        Compute shortest-path length (number of edges) between start and target on
        the undirected adjacency graph. Returns None if no path exists.
        """
        if start == target:
            return 0
        if start not in self.adj or target not in self.adj:
            return None

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self.adj.get(node, ()):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """Return estimated number of push-actions to reach the goal from node.state."""
        state = node.state

        # If the goals are already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract box positions from state: facts like (ldlu1rwo box loc)
        boxes = []  # list of (box_name, location)
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                boxes.append((box, loc))

        if not boxes:
            # No boxes present => nothing to do; if goals not satisfied we return large cost.
            return 0 if self.goals <= state else self.large_cost

        # Case 1: goals explicitly map each box to a target location.
        if all(box in self.goal_for_box for box, _ in boxes):
            total = 0
            for box, loc in boxes:
                target = self.goal_for_box[box]
                dist = self.bfs_distance(loc, target)
                if dist is None:
                    return self.large_cost
                total += dist
            return total

        # Case 2: goals provide a set of goal locations but not per-box binding.
        # We compute minimal sum of distances over all assignments (permutations).
        goal_locs = list(self.goal_for_box.values()) if self.goal_for_box else list(self.goal_locations)

        # If there are more boxes than goal locations, it's unusual but handle gracefully:
        if len(goal_locs) == 0:
            # No goal locations known -> cannot estimate; return large finite cost.
            return self.large_cost

        # Build distance matrix between boxes and goal locations.
        box_locs = [loc for _, loc in boxes]
        n_boxes = len(box_locs)
        n_goals = len(goal_locs)

        # If there are fewer goals than boxes, some boxes cannot be assigned -> penalize.
        if n_goals < n_boxes:
            return self.large_cost

        # Precompute distances
        dist_matrix = []
        for b_loc in box_locs:
            row = []
            for g_loc in goal_locs:
                d = self.bfs_distance(b_loc, g_loc)
                # If unreachable, treat as large to avoid selecting that pairing unless unavoidable.
                row.append(self.large_cost if d is None else d)
            dist_matrix.append(row)

        # Try all permutations of goal assignment for the boxes (choose n_boxes out of n_goals).
        # For efficiency, if n_goals == n_boxes enumerate permutations of goal indices,
        # otherwise enumerate combinations of goal indices and their permutations.
        best = None
        # If equal counts, simpler:
        if n_goals == n_boxes:
            for perm in permutations(range(n_goals), n_boxes):
                s = 0
                for i, g_idx in enumerate(perm):
                    s += dist_matrix[i][g_idx]
                    # early pruning
                    if best is not None and s >= best:
                        break
                else:
                    if best is None or s < best:
                        best = s
        else:
            # n_goals > n_boxes: choose permutations of goals taken n_boxes at a time
            # by iterating permutations over goal indices of length n_boxes.
            for perm in permutations(range(n_goals), n_boxes):
                s = 0
                for i, g_idx in enumerate(perm):
                    s += dist_matrix[i][g_idx]
                    if best is not None and s >= best:
                        break
                else:
                    if best is None or s < best:
                        best = s

        # If best remained None (shouldn't happen), return large cost.
        if best is None:
            return self.large_cost
        return int(best)