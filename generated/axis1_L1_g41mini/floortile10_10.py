from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern elements (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (w133ormt).

    Summary:
    ----------
    This heuristic estimates the approximate number of actions needed
    to reach the goal state from the current state. The domain involves
    robots located on tiles, objects/colors associated to those tiles,
    and movements or coloring actions affecting tiles and the robot.

    Assumptions:
    ------------
    - Each goal is to have certain tiles colored with a particular color.
    - The robot can move between tiles exploiting a directed graph defined
      by static predicates (xr2t1c1d and oh0smogc) representing tile adjacency.
    - Only one color can be "held" by a robot at a time.
    - The robot can change its held color with the z7pcmkeu action using white/black color objects.
    - To color a tile, the robot must be on a tile related to the target tile and the target tile must be "d7ekk7vb".
    - The robot's moves and color changes all have unit cost.
    - The heuristic is not necessarily admissible but tries to minimize node expansions.

    Heuristic Initialization:
    --------------------------
    - Extract all goal facts of the form (yh6q5j8r tile color) that define the target
      color assignments for specific tiles.
    - Extract static connectivity relations between tiles from the static facts:
      - Directed graph edges from predicates (xr2t1c1d), (oh0smogc), and (w224no0l), (g7ya35is)
        describe possible moves or adjacency.
    - Extract available color objects (cdb9op3o facts).
    - Build connectivity graphs to estimate shortest distances between tiles efficiently.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify which tiles still need to be painted with their respective goal colors
       by comparing the goal color predicates with the state's current y h6q5j8r predicates.
    2. Identify the current position of the robot (tile) and the color it holds.
    3. For each tile that still requires coloring:
       - Estimate minimal number of moves for the robot to get to a tile allowing
         it to color the target tile: a tile from which (gk9u3vg3 robot tile) predicate
         can hold true (using connections) and the tile is "d7ekk7vb".
       - The robot must hold the correct color object to paint the tile; a color change
         may be required if the robot currently holds a different color.
    4. Model the problem approximately as solving these subgoals:
       - Traveling between tiles to paint tiles in order (choose a cheap ordering),
       - Switching colors as needed (color changes have cost 1 per switch),
       - Coloring a tile costs 1 action.
    5. Use shortest path distance on the tile graph to estimate move costs.
    6. Sum minimal color switches needed plus estimated traveling and coloring costs.
    7. The heuristic returns 0 if all goals are satisfied (goal state).

    This heuristic thus aggressively estimates travel and painting costs, attempting to
    guide greedy best-first search efficiently toward solved painted tile configurations.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal tile coloring: (yh6q5j8r tile color)
        self.goal_tile_colors = {}  # tile -> color
        for g in self.goals:
            p, *args = get_parts(g)
            if p == "yh6q5j8r" and len(args) == 2:
                tile, color = args
                self.goal_tile_colors[tile] = color

        # Extract all tile objects (from static and facts)
        # We get tiles from static facts mentioning tiles via predicates:
        # (xr2t1c1d ?tile1 ?tile2), (oh0smogc ?tile1 ?tile2), (w224no0l ?tile1 ?tile2), (g7ya35is ?tile1 ?tile2)
        self.tiles = set()
        for fact in self.static:
            p, *args = get_parts(fact)
            if p in {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"} and len(args) == 2:
                self.tiles.add(args[0])
                self.tiles.add(args[1])

        # Build adjacency (directed) graph from static connectivity predicates
        # We consider all directed edges from (xr2t1c1d), (oh0smogc), (w224no0l)
        self.adjacency = {tile: set() for tile in self.tiles}
        for fact in self.static:
            p, *args = get_parts(fact)
            if p in {"xr2t1c1d", "oh0smogc", "w224no0l"} and len(args) == 2:
                # Add edge from args[0] -> args[1]
                self.adjacency[args[0]].add(args[1])

        # Also add edges from (g7ya35is) (likely horizontal adjacency)
        # It appears undirected, so add both directions
        for fact in self.static:
            p, *args = get_parts(fact)
            if p == "g7ya35is" and len(args) == 2:
                self.adjacency.setdefault(args[0], set()).add(args[1])
                self.adjacency.setdefault(args[1], set()).add(args[0])

        # Extract colors that can be held, from (cdb9op3o color)
        self.colors = set()
        for fact in self.static:
            if match(fact, "cdb9op3o", "*"):
                color = get_parts(fact)[1]
                self.colors.add(color)

        # Store tiles that can be colored (which appear with d7ekk7vb predicate)
        # This will be checked later dynamically.

        # Precompute shortest paths between tiles using BFS to enable quick distance lookup
        # We'll cache distances on demand in __call__
        self._dist_cache = {}

    def _compute_shortest_path(self, start, goal):
        """
        Compute shortest path length between two tiles in the adjacency graph.

        Returns:
            int: length of shortest path or large number if unreachable.
        """
        if (start, goal) in self._dist_cache:
            return self._dist_cache[(start, goal)]
        if start == goal:
            self._dist_cache[(start, goal)] = 0
            return 0

        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == goal:
                self._dist_cache[(start, goal)] = dist
                return dist
            if current not in visited:
                visited.add(current)
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        # If unreachable, return a large number
        self._dist_cache[(start, goal)] = 1000
        return 1000

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal.

        Details:
        - If already in goal state, return 0.
        - For each tile that must be colored a color different from current,
          estimate costs:
          - Movement of robot to a tile enabling painting (tile with gk9u3vg3 robot tile and d7ekk7vb tile predicates).
          - Switching color held by robot if needed.
          - Painting action for the tile.
        - Sum estimates for all tiles.
        """
        state = node.state

        # Quick check: if goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Extract robot(s) and their position(s) and held color(s).
        # The domain may have multiple robots - handle them individually.
        # Predicates:
        # (gk9u3vg3 robot tile) - robot position
        # (i4u94rl9 robot color) - robot holds color

        robots = set()
        robot_positions = {}
        robot_colors = {}

        for fact in state:
            p, *args = get_parts(fact)
            if p == "gk9u3vg3" and len(args) == 2:
                robot, tile = args
                robots.add(robot)
                robot_positions[robot] = tile
            elif p == "i4u94rl9" and len(args) == 2:
                robot, color = args
                robots.add(robot)
                robot_colors[robot] = color

        # Extract which tiles are currently colored by which color: predicates (yh6q5j8r tile color)
        current_tile_colors = {}
        for fact in state:
            if match(fact, "yh6q5j8r", "*", "*"):
                _, tile, color = get_parts(fact)
                current_tile_colors[tile] = color

        # Extract tiles that are "d7ekk7vb" - those available to be painted currently
        d7_tiles = set()
        for fact in state:
            if match(fact, "d7ekk7vb", "*"):
                tile = get_parts(fact)[1]
                d7_tiles.add(tile)

        # Function to find tiles reachable by the robot to paint a specific tile.
        # Only tiles with gk9u3vg3 robot tile and d7ekk7vb tile predicates,
        # Also the robot must be on a tile related to the target tile by gk9u3vg3 or xr2t1c1d etc.
        # However, painting actions require the robot to be on a tile satisfying gk9u3vg3 robot tile condition (robot position)
        # and tile to be painted is d7ekk7vb, and connected through gk9u3vg3 predicate.

        # In domain, painting actions require:
        # preconditions:
        # - robot is on a tile for which (gk9u3vg3 robot tile) holds,
        # - tile to paint is d7ekk7vb,
        # - target tile related by xr2t1c1d or oh0smogc (depends on action).

        # Approximate movement cost: robot moves between tiles along adjacency graph;
        # painting a tile costs 1; switching color costs 1 if mismatch.

        # We simplify by assuming robot must move to the tile it stays on (robot's current tile),
        # perform a chain of moves to a tile adjacent (via adjacency relations) to the tile needing painting,
        # then paint. I.e. for tile t to paint, robot must be on some tile r s.t. gk9u3vg3 robot r and
        # r is the tile described in the (gk9u3vg3 robot r) predicate and which is linked to tile t via
        # xr2t1c1d or oh0smogc predicates (directed edges indicating adjacency).

        # To estimate the minimal distance to paint a tile, find minimal path from current robot tile to
        # any tile r such that r is linked to tile t, and that tile r is in adjacency graph (reachable).

        # For this, build helper maps from tile to adjacent painting origin tiles:
        # Tiles from which tile t can be painted:
        # those tiles r where (xr2t1c1d tile t) or (oh0smogc tile t) or (w224no0l tile t) or (g7ya35is tile t) predicates exist.

        # We'll preprocess from static facts a reversed adjacency for painting:
        self.painting_origins_for = getattr(self, "painting_origins_for", None)
        if self.painting_origins_for is None:
            painting_origins_for = {tile: set() for tile in self.tiles}
            for fact in self.static:
                p, a1, a2 = get_parts(fact)
                if p in {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}:
                    painting_origins_for.setdefault(a2, set()).add(a1)
            self.painting_origins_for = painting_origins_for

        painting_origins_for = self.painting_origins_for

        total_heuristic = 0

        # If multiple robots, we will heuristically assign tiles roughly by closest robot.
        # But for simplicity, assume a single robot (use first found if multiple).
        if not robots:
            # No robot? Hard to proceed, but return a large heuristic to avoid dead ends.
            return 1000
        robot = next(iter(robots))
        robot_tile = robot_positions.get(robot, None)
        robot_color = robot_colors.get(robot, None)

        if robot_tile is None:
            # Robot position missing -> large cost
            return 1000

        # Goals not satisfied yet: we iterate over all tiles that still need coloring
        # Those are tiles in goal_tile_colors where current color differs or tile unpainted
        tiles_to_paint = []
        for tile, goal_color in self.goal_tile_colors.items():
            current_color = current_tile_colors.get(tile, None)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))

        # We assume robot must paint all tiles in tiles_to_paint.

        # To roughly estimate the cost:
        # For each tile:
        # - If robot_color != needed color: +1 for color switch
        # - Move cost: minimal distance from robot_tile to any painting origin tile for tile
        # - Paint cost: 1
        # Accumulate costs and update robot_tile and robot_color as if sequentially painting tiles greedily.
        # Since ordering is not given, select closest tile next.

        # We'll greedily find next tile from current robot_tile minimizing:
        # color switch cost + movement cost + 1 paint cost

        unpainted = tiles_to_paint[:]
        current_pos = robot_tile
        current_color_held = robot_color if robot_color in self.colors else None

        while unpainted:
            best_cost = None
            best_index = None
            best_move_cost = None
            best_painting_origin = None
            best_tile = None
            best_needed_color = None

            for i, (tile, needed_color) in enumerate(unpainted):
                # Determine color switch cost:
                color_switch_cost = 0 if current_color_held == needed_color else 1
                # Find painting origins for tile that are d7ekk7vb and reachable:
                candidate_origins = self.painting_origins_for.get(tile, set()) & d7_tiles
                if not candidate_origins:
                    # No tile from which to paint (no valid d7ekk7vb precondition), large penalty
                    continue
                # Compute minimal move cost from current_pos to any painting origin
                move_cost = min(self._compute_shortest_path(current_pos, origin) for origin in candidate_origins)
                if move_cost >= 1000:
                    # unreachable
                    continue
                # Paint cost fixed 1 action
                paint_cost = 1
                cost = color_switch_cost + move_cost + paint_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_index = i
                    best_move_cost = move_cost
                    best_tile = tile
                    best_needed_color = needed_color

            if best_cost is None:
                # No reachable tile to paint found -> penalize with large number
                return 1000

            # Accumulate best_cost
            total_heuristic += best_cost

            # Update robot position and color after painting this tile
            # After painting, robot stays on the painting origin tile chosen:
            # For accuracy: choose the painting origin with minimal distance again:
            candidate_origins = self.painting_origins_for.get(best_tile, set()) & d7_tiles
            best_painting_origin = min(candidate_origins, key=lambda o: self._compute_shortest_path(current_pos, o))
            current_pos = best_painting_origin
            current_color_held = best_needed_color

            # Remove painted tile from list
            unpainted.pop(best_index)

        return total_heuristic
