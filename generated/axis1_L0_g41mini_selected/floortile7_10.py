from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    Summary
    -------
    This heuristic estimates the number of actions remaining to reach the goal state in the floortile domain.
    It considers the number and locations of tiles that still need to be painted, the current color the robot carries,
    and the distances (in moves) between the robot and those tiles. The heuristic calculates a lower-bound-like estimate
    but does not need to be admissible. It guides a greedy best-first search efficiently by focusing on tiles still
    needing painting, minimizing color changes, and estimating robot travel moves realistically on the tile grid.

    Assumptions
    -----------
    - There is one or more robots operating on a grid of tiles with adjacency relations defined (up/down/left/right).
    - Each goal fact requires a certain tile to be painted a certain color.
    - The robot can paint only adjacent tiles.
    - Painting consumes a paint action after robot moves and possible color changes.
    - Robots can carry exactly one color at a time; changing color costs one action.
    - Tiles must be clear (unpainted) before painting.
    - Moving from one tile to an adjacent tile costs one action.
    - The heuristic ignores concurrent robot moves but assumes the minimal number of moves by the closest robot to each tile.

    Heuristic Initialization
    ------------------------
    - Extract all goal conditions of the form (painted tile color).
    - Extract all static adjacency relations (up/down/left/right) to build a bidirectional connectivity graph between tiles.
    - Extract the set of available colors.
    - Extract the robot identifiers.
    - Precompute an adjacency map to quickly compute shortest distances between tiles using BFS per query.
      (Distances are computed lazily on demand, cached for performance.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which tiles still need painting to reach the goal:
       - Compare current painted facts with goal painted facts to gather tiles pending painting and their required colors.

    2. Extract the current positions and carried colors of all robots.

    3. For each tile that still needs painting:
       - Compute the shortest distance (in tile moves) from each robot to an adjacent tile of the target tile
         (since painting applies to adjacent tiles).
       - Choose the minimal distance among all robots.

    4. Group tiles by required color, because the robot must carry matching paint color to paint a tile.
       If a robot needs to paint tiles of different colors, it must do color-change actions.

    5. Estimate color-change costs minimally, assuming optimal color order of painting.

    6. Compute and sum:
       - The minimal travel moves for painting all tiles (the sum of closest distances to paint each tile).
         This models the travel cost to get in position for painting each tile.
       - The color-change costs (minimum number of color changes needed).
       - The painting action costs (one per tile).

    7. Return the sum as the heuristic value.

    This heuristic balances painting costs, travel costs, and color-change costs,
    focusing on reachable tiles and carrying paint colors efficiently.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal tiles and required colors: {tile: color}
        self.goal_painted = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Build a graph of tile adjacencies (bidirectional edges)
        # The domain has predicates: up, down, left, right
        # All are pairs tile -> tile representing adjacency.
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ("up", "down", "left", "right"):
                src, dst = parts[1], parts[2]
                if src not in self.adj:
                    self.adj[src] = set()
                if dst not in self.adj:
                    self.adj[dst] = set()
                # Add bidirectional adjacency
                self.adj[src].add(dst)
                self.adj[dst].add(src)

        # Cache for shortest distances between tiles, computed lazily
        self._dist_cache = {}

        # Extract available colors to handle color switching context if needed
        self.available_colors = {get_parts(fact)[1] for fact in self.static if get_parts(fact)[0] == "available-color"}

    def shortest_distance(self, start, goals):
        """
        Compute shortest distance from start tile to any of the goal tiles (set of tiles).
        Uses BFS on adjacency graph.
        Caches results for performance.
        """
        if not goals:
            return 0

        key = (start, frozenset(goals))
        if key in self._dist_cache:
            return self._dist_cache[key]

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            if current in goals:
                self._dist_cache[key] = dist
                return dist
            for neighbor in self.adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Return a large distance if unreachable
        self._dist_cache[key] = float('inf')
        return float('inf')

    def __call__(self, node):
        state = node.state

        # If state is a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract painted tiles/colors in current state: {tile: color}
        painted = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 3:
                painted[parts[1]] = parts[2]

        # Identify tiles requiring painting:
        # tiles in goal but either not painted or wrong color
        tiles_to_paint = {
            tile: color
            for tile, color in self.goal_painted.items()
            if painted.get(tile) != color
        }

        if not tiles_to_paint:
            # No tiles to paint, so heuristic=0
            return 0

        # Extract robot locations and their currently held colors: {robot: (tile, color)}
        robot_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_positions.setdefault(robot, {})["pos"] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_positions.setdefault(robot, {})["color"] = color

        # Default: if a robot does not have a color fact, it has no color (treated as None)
        for r in robot_positions:
            if "color" not in robot_positions[r]:
                robot_positions[r]["color"] = None

        # Function to get all adjacency tiles neighboring a given tile
        def neighbors(tile):
            return self.adj.get(tile, set())

        # For each tile to paint, find minimal distance cost from any robot to an adjacent tile of that tile
        # Because painting requires robot-at on a tile adjacent to the tile to paint.
        # Distances are tile moves (each move = 1 action).

        # Group tiles by color for minimal color changes path estimation
        color_groups = {}
        for tile, color in tiles_to_paint.items():
            color_groups.setdefault(color, set()).add(tile)

        # Compute per-color minimal painting cost estimate:
        # We'll estimate as:
        #   Sum over tiles in color group of minimal distance to an adjacent tile from any robot
        # Color changes cost = number_of_color_groups - 1 (assuming optimal color change ordering)
        # Painting cost = one paint action per tile
        # Moving from one tile adjacency to next tile adjacency is not fully accounted for in this heuristic,
        # but the repeated sum of distances from robots to tiles is a proxy for travel cost.

        # For each tile, precompute adjacency tiles (robot must be adjacent to paint)
        tiles_adjacents = {tile: neighbors(tile) for tile in tiles_to_paint}

        # Compute robot starting colors to estimate color change costs more accurately
        robots_by_color = {}
        for r, info in robot_positions.items():
            c = info["color"]
            robots_by_color.setdefault(c, set()).add(r)

        total_movement = 0

        for color, tiles in color_groups.items():
            for tile in tiles:
                adj_tiles = tiles_adjacents[tile]
                if not adj_tiles:
                    # No adjacent tiles means tile isolated, consider distance from robots to tile itself (rare)
                    adj_tiles = {tile}

                # Find minimal distance from any robot to any adjacency tile
                min_dist = float('inf')
                for r, info in robot_positions.items():
                    robot_tile = info["pos"]
                    dist = self.shortest_distance(robot_tile, adj_tiles)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist == float('inf'):
                    # unreachable tile -> heuristic large but finite
                    min_dist = 1000
                total_movement += min_dist

        # Number of color changes minimally needed:
        # At least (number of distinct colors to paint) - 1 changes.
        # But if a robot currently carries one of these colors, it may save one color change.
        needed_colors = set(color_groups.keys())

        # Count how many robots currently carry a needed color
        carried_needed_colors = set(info["color"] for info in robot_positions.values()) & needed_colors

        # Minimally, color changes = number of needed colors - max(1, number_of_robots_carrying_needed_color)
        # Because if at least one robot already carries a needed color, we save one color change.
        color_changes = max(0, len(needed_colors) - max(1, len(carried_needed_colors)))

        # Painting cost: one paint action per tile to paint
        painting_cost = len(tiles_to_paint)

        # Total heuristic estimate sum:
        heuristic_value = total_movement + color_changes + painting_cost

        # Ensure heuristic is at least 1 for non-goal (except when no tiles to paint)
        return max(1, int(heuristic_value))
