from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a PDDL fact string, stripping surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact matches a pattern with wildcards '*'.
    Example: match("(painted tile_1_1 white)", "painted", "*", "white") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'floortile' domain. The heuristic estimates
    the number of actions required to achieve the goal painted facts from the
    current state. It is intended to guide greedy best-first search (not required
    to be admissible) and therefore trades admissibility for informedness and
    fast computation.

    Assumptions
    -----------
    - Each paint action paints one tile with one color and requires the tile to
      be clear and a robot to be on an adjacent tile.
    - A robot holds exactly one color at a time (predicate robot-has). Changing
      the robot's color to any available color costs one action (change_color).
    - Movement between adjacent tiles costs one action per step.
    - Painting costs one action.
    - If a tile is already painted with a different color (and hence not clear),
      there is no action in the domain to repaint it; such a goal is treated as
      unreachable (we return a large heuristic value).
    - The heuristic ignores dynamic clear constraints when computing distances
      (i.e., it treats the grid connectivity statically). It adds a small fixed
      penalty if the tile to be painted is currently not clear (typically
      because a robot sits on it), accounting approximately for the effort to
      make it clear.
    - Multiple robots and multiple goals are handled by independently estimating
      the cost to paint each goal tile (taking the best robot for that tile).
      This is a heuristic approximation that is fast to compute and often
      informative in practice.

    Heuristic Initialization
    ------------------------
    - Extract goal painted facts.
    - Build an undirected adjacency graph of tiles from static facts (up/down/left/right).
    - Extract set of available colors from static facts. (Not essential for cost
      computation, but kept for clarity.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse the state to find:
       - Robot positions (from 'robot-at' facts).
       - The color each robot currently holds (from 'robot-has' facts).
       - Which tiles are clear (from 'clear' facts).
       - Which tiles are already painted and with what colors (from 'painted' facts).
    3. For each goal of the form (painted TILE COLOR):
       a. If the exact painted fact is already present in the state, cost 0 for that goal.
       b. If TILE is painted with some other color (i.e., there exists a different
          painted fact (painted TILE X) and X != COLOR), we consider the goal
          unreachable in this domain and return a large heuristic value.
       c. Otherwise (TILE not painted yet), estimate the minimum cost over all
          robots to paint TILE:
           - Distance cost: shortest path length (on the static adjacency graph)
             from the robot's current tile to any tile that is adjacent to TILE.
             (A robot must stand on an adjacent tile to paint TILE.)
           - Color-change cost: 0 if robot already has COLOR, otherwise 1.
           - Paint cost: 1 (the paint action).
           - Clear penalty: if TILE is not currently clear (and not already painted),
             add +1 to approximately account for making the tile clear (e.g., moving
             the occupying robot away). If TILE is clear, penalty is 0.
       d. Take the minimum of the above costs across robots as the estimated cost
          to achieve that painted goal.
    4. Sum the per-goal estimates to obtain the final heuristic value.
       (This sums independent optimistic estimates and is fast to compute.)
    """

    # Large value used when a goal is impossible or adjacency is missing.
    LARGE = 10000

    def __init__(self, task):
        """Extract static adjacency and goals from the planning task."""
        self.goals = task.goals  # set of goal fact strings
        static = task.static

        # Build undirected adjacency graph between tiles from up/down/left/right facts.
        # Each such static fact has the form "(up tileA tileB)" meaning tileA is up of tileB.
        self.adj = {}  # tile -> set(neighbor_tiles)
        for fact in static:
            parts = get_parts(fact)
            if parts[0] in ("up", "down", "left", "right"):
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Collect available colors if needed.
        self.available_colors = {get_parts(f)[1] for f in static if match(f, "available-color", "*")}

    def _bfs_distance(self, start, targets):
        """
        BFS on the static adjacency graph to compute shortest distance from start
        tile to any tile in targets. Returns LARGE if no path exists.
        """
        if start in targets:
            return 0
        if start not in self.adj:
            return self.LARGE if hasattr(self, "LARGE") else NewDomainDependentHeuristic.LARGE

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                queue.append((nbr, dist + 1))
        return NewDomainDependentHeuristic.LARGE

    def __call__(self, node):
        """Compute the heuristic estimate for the node's state."""
        state = node.state

        # 1) If already goal, return 0.
        if self.goals <= state:
            return 0

        # 2) Parse current facts: robots positions, robot colors, clear tiles, painted tiles.
        robot_pos = {}  # robot -> tile
        robot_color = {}  # robot -> color
        clear_tiles = set()
        painted_map = {}  # tile -> set(colors)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_pos[robot] = tile
            elif pred == "robot-has":
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif pred == "clear":
                clear_tiles.add(parts[1])
            elif pred == "painted":
                tile, color = parts[1], parts[2]
                painted_map.setdefault(tile, set()).add(color)

        # Quick unsolvability check:
        # If a goal requires (painted T C) but T is already painted with some other color
        # and C is not present, then the goal cannot be achieved (painting is irreversible).
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                if tile in painted_map and color not in painted_map[tile]:
                    # Tile already painted with different color -> unreachable
                    return NewDomainDependentHeuristic.LARGE

        # Gather set of tiles (use adjacency keys and state info as fallback)
        all_tiles = set(self.adj.keys())
        # include any tiles that appear in state facts that may not be in static adjacency
        for fact in state:
            for tok in get_parts(fact)[1:]:
                if tok.startswith("tile_"):
                    all_tiles.add(tok)

        total_cost = 0

        # For every painted goal that is not yet satisfied, estimate cost.
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "painted":
                # floortile goals are expected to be painted facts; skip others conservatively
                continue
            tile_goal, color_goal = parts[1], parts[2]
            goal_fact = f"(painted {tile_goal} {color_goal})"
            if goal_fact in state:
                continue  # already achieved

            # If tile has been painted with other colors we returned earlier as unreachable.

            # If tile has no neighbors in static adjacency, it's impossible to stand adjacently -> unreachable.
            neighbors = self.adj.get(tile_goal, set())
            if not neighbors:
                return NewDomainDependentHeuristic.LARGE

            # The tile must be clear to paint. If not clear and not painted, it's likely
            # occupied by a robot; account for that by a small penalty.
            clear_penalty = 0
            if tile_goal not in clear_tiles:
                # If tile is already painted with the desired color we'd have returned earlier,
                # so here non-clear implies occupancy (or painted-other which is handled above).
                clear_penalty = 1

            # Compute minimal estimated cost across all robots to paint this tile.
            best_cost_for_tile = NewDomainDependentHeuristic.LARGE

            if not robot_pos:
                # no robots present -> impossible
                return NewDomainDependentHeuristic.LARGE

            for robot, pos in robot_pos.items():
                # distance to any neighboring tile (robot needs to stand on a neighbor)
                dist = self._bfs_distance(pos, neighbors)
                if dist >= NewDomainDependentHeuristic.LARGE:
                    continue  # this robot cannot reach any neighbor

                # color change cost: 0 if already has desired color, else 1
                r_color = robot_color.get(robot)
                color_change_cost = 0 if r_color == color_goal else 1

                # paint action cost = 1
                paint_cost = 1

                # Total cost for this robot to paint this tile (approx.)
                cost = dist + color_change_cost + paint_cost + clear_penalty

                if cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            # If no robot can reach the tile's neighbors, unreachable
            if best_cost_for_tile >= NewDomainDependentHeuristic.LARGE:
                return NewDomainDependentHeuristic.LARGE

            total_cost += best_cost_for_tile

        # Return a finite integer heuristic estimate.
        return int(total_cost)