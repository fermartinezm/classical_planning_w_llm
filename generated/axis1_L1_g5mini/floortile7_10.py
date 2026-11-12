from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact components match the given pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (w133ormt). The domain's
    goals require painting specific tiles with given colors. Robots can move
    across tiles using several directed adjacency relations and can paint an
    adjacent tile if they hold the correct color. Changing the robot's held
    color is a separate action. This heuristic estimates the number of actions
    required to achieve all painting goals from a given state.

    Assumptions
    -----------
    - The goal only contains predicates of the form (yh6q5j8r tile color).
    - Each painting action requires the robot to be on a neighboring tile Y
      such that a static adjacency fact (pred tile Y) exists for one of the
      adjacency predicates seen in the domain.
    - Changing the held color costs 1 action (operator z7pcmkeu).
    - Painting a tile costs 1 action (operators c1qh1p5g / cadwr4wd).
    - Moving the robot between adjacent tiles costs 1 action per step.
    - If a tile is already painted with a different color than required by
      the goal, the problem is assumed unsolvable (the domain lacks a repaint
      operator), and the heuristic returns a large finite value.
    - For efficiency, the heuristic treats robots independently and sums a
      per-goal optimistic cost (i.e., it does not solve the multi-goal route
      optimization problem exactly). This is a deliberate relaxation to keep
      the heuristic fast while being informative.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - The painting goals: a mapping goal_tile -> required_color.
    - The set of available colors (from static facts (cdb9op3o ...)).
    - A directed adjacency graph between tiles based on static adjacency
      predicates. Static facts such as (xr2t1c1d A B) provide a directed edge
      B -> A (because the movement operators in the domain move the robot
      from the second argument to the first).
    - For each tile X, the set of neighbor tiles Y from which X can be painted
      (i.e., static facts of the form (pred X Y)).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. If any tile required by the goals is already painted with a different
       color in the state, return a large finite value (unsolvable under the
       domain's operators).
    3. Determine current robot positions and each robot's currently held color
       from facts (gk9u3vg3 robot tile) and (i4u94rl9 robot color).
    4. Precompute BFS shortest-path distances (in number of move actions) on
       the static directed adjacency graph from every robot's tile to all
       other tiles. (Distances are computed on the static graph and ignore the
       dynamic d7ekk7vb precondition to keep the heuristic fast and
       informative. This may make the heuristic inadmissible but practical.)
    5. For each goal tile that is not yet correctly painted:
       a. Identify all tiles Y such that a static fact (pred goal_tile Y)
          exists. These are the robot positions from which the tile can be
          painted with one painting action.
       b. For each robot and each such neighbor Y compute:
          - move_cost = distance(robot_tile -> Y) (INF if unreachable)
          - color_change_cost = 0 if robot already holds required color else 1
          - paint_cost = 1
          Total candidate cost = move_cost + color_change_cost + paint_cost
       c. Take the minimal candidate cost across all robots and neighbor Y.
       d. If minimal cost is INF (unreachable by any robot), return a large
          finite value.
    6. Sum these minimal costs for all unpainted goal tiles and return the sum.
       This summed cost is the heuristic estimate. It is efficient to compute
       (BFS from each robot position on a static graph and simple lookups).
    """

    def __init__(self, task):
        """Extract goals and static adjacency information for efficient heuristics."""
        self.goals = set(task.goals)  # goal fact strings

        # Parse goal facts of the form (yh6q5j8r tile color) into mapping tile->color.
        self.goal_tiles = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "yh6q5j8r" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Static facts (not affected by actions).
        static_facts = task.static

        # Available colors
        self.colors = {get_parts(f)[1] for f in static_facts if match(f, "cdb9op3o", "*")}

        # Adjacency predicates that enable movement/painting (identified from domain).
        self.adj_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Build directed adjacency graph for robot moves:
        # For any static fact (pred A B) where pred in adj_preds, add edge B -> A.
        self.adj_graph = defaultdict(set)  # node -> set of neighbors reachable in one move
        # Also build for each tile the set of neighbor positions from which it can be painted:
        # If static contains (pred tile Y), then robot at Y can paint tile.
        self.paint_neighbors = defaultdict(set)  # tile -> set of neighbor tiles Y

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in self.adj_preds and len(parts) == 3:
                a, b = parts[1], parts[2]
                # Movement edge: robot can move from b to a
                self.adj_graph[b].add(a)
                # For painting the tile 'a' the robot must stand on 'b' when pred is of form (pred a b)
                # But note: some painting actions also used the same pattern; we uniformly treat (pred a b) as
                # robot standing on b to affect a.
                self.paint_neighbors[a].add(b)

    def _bfs_distances(self, start):
        """Return a dict with shortest distances (in moves) from start to all reachable tiles."""
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adj_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """Compute the heuristic estimate for node.state."""
        state = node.state

        # If already in a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Quick check: if any goal tile is already painted with a different color,
        # problem is unsolvable under current domain (no repaint action). Return INF.
        for tile, required_color in self.goal_tiles.items():
            # Is there any yh6q5j8r fact for this tile in the current state?
            for fact in state:
                if match(fact, "yh6q5j8r", tile, "*"):
                    # Extract the current color
                    curr_color = get_parts(fact)[2]
                    if curr_color != required_color:
                        return INF

        # Gather robots' positions and held colors.
        robots = {}  # robot -> tile
        robot_colors = {}  # robot -> color (if present)
        for fact in state:
            if match(fact, "gk9u3vg3", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                robots[robot] = tile
            elif match(fact, "i4u94rl9", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        if not robots:
            # No robots available to paint -> unsolvable
            return INF

        # Precompute BFS distances from each robot's position.
        robot_dist_maps = {}
        for robot, pos in robots.items():
            robot_dist_maps[robot] = self._bfs_distances(pos)

        total_cost = 0

        # For each goal tile not yet satisfied, compute minimal cost to paint it.
        for tile, required_color in self.goal_tiles.items():
            goal_fact = f"(yh6q5j8r {tile} {required_color})"
            if goal_fact in state:
                continue  # already satisfied

            # Candidate neighbor positions from which the tile can be painted
            neighbors = self.paint_neighbors.get(tile, set())
            if not neighbors:
                # No static neighbor from which this tile can possibly be painted -> unreachable
                return INF

            best_cost_for_tile = INF

            for robot, dist_map in robot_dist_maps.items():
                # robot may not have a recorded color; assume color change required in that case
                r_color = robot_colors.get(robot, None)
                color_change_cost = 0 if r_color == required_color else 1

                for neigh in neighbors:
                    move_cost = dist_map.get(neigh, INF)
                    if move_cost >= INF:
                        continue
                    # 1 action to paint
                    candidate = move_cost + color_change_cost + 1
                    if candidate < best_cost_for_tile:
                        best_cost_for_tile = candidate

            if best_cost_for_tile >= INF:
                # This tile cannot be reached/painted by any robot -> unsolvable
                return INF

            total_cost += best_cost_for_tile

        # Return the summed estimate.
        return int(total_cost)