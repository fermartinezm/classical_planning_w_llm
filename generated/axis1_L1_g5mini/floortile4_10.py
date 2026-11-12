from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (grid painting) domain.

    The domain contains robots that occupy tiles and hold a paint color.
    Robots can:
      - move along directed adjacency edges between tiles,
      - switch their held color in one action,
      - paint an adjacent tile with their current color (one paint action).
    Goals require certain tiles to have a specific color (predicate yh6q5j8r).

    This heuristic estimates the remaining number of actions as the sum of,
    for each goal tile that is not yet satisfied, the minimal estimated cost
    for some robot to (a) move to a tile from which it can paint the goal tile,
    (b) switch color if necessary, and (c) perform the paint action.

    Assumptions
    -----------
    - Painting a goal tile always requires exactly one paint action.
    - Switching the robot's held color to any other color takes one action.
    - Movement between adjacent tiles costs one action per directed edge.
    - The static adjacency relations (xr2t1c1d, oh0smogc, w224no0l, g7ya35is)
      define the directed movement/painting graph. A static fact
      "(pred B A)" indicates an action moving/acting from A to B.
    - We ignore occupancy/blocking complexities (d7ekk7vb) and compute
      shortest path distances on the static adjacency graph. This makes the
      heuristic non-admissible but efficiently computable and informative.
    - If multiple robots exist we pick the robot that yields the minimal
      per-goal estimate; we sum those minimal per-goal estimates. This
      approximation may double-count shared movements but is fast.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - the set of goal (tile, color) pairs from task.goals,
      - the directed adjacency graph from static facts (for movement/painting),
      - predecessor lists (tiles from which a specific tile can be painted),
      - the list of valid colors (from cdb9op3o), for completeness (not strictly needed).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (state):
     1. If all goal facts are satisfied in the state return 0.
     2. Parse the state to find:
        - robot positions (gk9u3vg3 robot tile),
        - robot held colors (i4u94rl9 robot color),
        - currently painted tiles (yh6q5j8r tile color).
     3. For each robot run a BFS on the adjacency graph to compute shortest
        move-distances from its current tile to every tile.
     4. For every unsatisfied goal (tile T, color C):
        - Determine the set of painting positions P: tiles A such that there
          is a static adjacency indicating A -> T (i.e., a robot standing on A
          can paint T).
        - For each robot r compute:
            distance = min_{p in P} dist_r[p]  (infinite if no reachable p)
            color_change_cost = 0 if robot holds C else 1
            paint_cost = 1
            total_r = distance + color_change_cost + paint_cost
        - The goal's estimate is the minimal total_r over all robots.
     5. The heuristic value is the sum of the estimates for all unsatisfied goals.
        If some goal has no reachable painting position from any robot we
        return a large finite estimate (a high integer) so the value remains finite.
    """

    LARGE_COST = 100000  # finite large number for unreachable goals

    def __init__(self, task):
        # Store goal facts (strings) for reference and also parse them into (tile, color).
        self.goals = set(task.goals)

        # Build adjacency graph from static facts.
        # Movement/painting static predicates that imply a directed edge from A -> B
        # are: xr2t1c1d, oh0smogc, w224no0l, g7ya35is
        self._edge_predicates = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        self.adjacency = defaultdict(list)   # adjacency[from_tile] -> list of to_tiles
        self.predecessors = defaultdict(list)  # predecessors[to_tile] -> list of from_tiles
        self.colors = set()
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in self._edge_predicates and len(parts) >= 3:
                # static fact example: (xr2t1c1d tile_1_1 tile_0_1)
                # This indicates an edge from tile_0_1 -> tile_1_1
                to_tile = parts[1]
                from_tile = parts[2]
                self.adjacency[from_tile].append(to_tile)
                self.predecessors[to_tile].append(from_tile)
            elif pred == "cdb9op3o" and len(parts) >= 2:
                self.colors.add(parts[1])

        # Parse goal facts into (tile, color) pairs for quick iteration.
        self.goal_tile_color = []
        for goal in self.goals:
            gparts = get_parts(goal)
            if gparts and gparts[0] == "yh6q5j8r" and len(gparts) >= 3:
                tile = gparts[1]
                color = gparts[2]
                self.goal_tile_color.append((tile, color))

    def __call__(self, node):
        state = node.state

        # If goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state for robots positions, robot colors, and currently painted tiles.
        robot_positions = {}  # robot -> tile
        robot_colors = {}     # robot -> color
        painted = set()       # set of (tile, color) currently painted

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color
            elif pred == "yh6q5j8r" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                painted.add((tile, color))

        robots = list(robot_positions.keys())
        if not robots:
            # No robots to perform actions -> unsolvable; return large finite cost.
            return self.LARGE_COST

        # Precompute BFS distances from each robot position to all tiles.
        distances_by_robot = {}
        for robot in robots:
            start = robot_positions[robot]
            # BFS
            dist = {start: 0}
            dq = deque([start])
            while dq:
                cur = dq.popleft()
                for nb in self.adjacency.get(cur, []):
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        dq.append(nb)
            distances_by_robot[robot] = dist

        total_estimate = 0

        for (goal_tile, goal_color) in self.goal_tile_color:
            # Skip if already satisfied exactly.
            if (goal_tile, goal_color) in painted:
                continue

            # Candidate painting positions are tiles from which a robot can paint goal_tile.
            painting_positions = self.predecessors.get(goal_tile, [])
            if not painting_positions:
                # No static predecessor, cannot paint => very large estimate but finite.
                return self.LARGE_COST

            # For each robot compute minimal cost to achieve this goal tile.
            best_cost_for_goal = None
            for robot in robots:
                dist_map = distances_by_robot.get(robot, {})
                # Find minimal distance to any painting position
                min_dist = None
                for pos in painting_positions:
                    if pos == robot_positions[robot]:
                        # already at painting position
                        d = 0
                    else:
                        d = dist_map.get(pos)
                    if d is None:
                        continue
                    if (min_dist is None) or (d < min_dist):
                        min_dist = d
                if min_dist is None:
                    # this robot cannot reach any painting position
                    continue

                # color change cost: 0 if robot already holds the color, else 1.
                held_color = robot_colors.get(robot)
                color_change_cost = 0 if held_color == goal_color else 1

                # painting action cost = 1
                paint_cost = 1

                total_r = min_dist + color_change_cost + paint_cost

                if (best_cost_for_goal is None) or (total_r < best_cost_for_goal):
                    best_cost_for_goal = total_r

            if best_cost_for_goal is None:
                # no robot can reach painting positions -> unsolvable from here (approx).
                return self.LARGE_COST

            total_estimate += best_cost_for_goal

        # Ensure heuristic is integer and finite.
        return int(total_estimate)