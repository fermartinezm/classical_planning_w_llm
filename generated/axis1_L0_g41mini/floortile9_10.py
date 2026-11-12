from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract individual components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches the given pattern (supports '*' wildcards).
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    Summary
    -------
    This heuristic estimates the number of actions required to paint all tiles as specified by the goal. It
    does so by approximating costs based on:
      - The distance the robot must travel to reach unpainted tiles adjacent to its current position.
      - The number of tiles still needing painting.
      - The cost of changing the robot's painting color when needed.

    Assumptions
    -----------
    - The robot can paint only from adjacent tiles (up, down, left, right).
    - Only one robot is handled by this heuristic.
    - The robot can hold only one color at a time.
    - The robot must move onto tiles adjacent to a target tile to paint it.
    - The heuristic is not required to be admissible; approximate distances are sufficient.
    - Tiles that are already painted with the goal color require no action.
    - Changing painting colors always costs at least one action.
    - Moving between tiles costs one action per step.
    - Painting a tile costs exactly one action.

    Heuristic Initialization
    ------------------------
    During initialization:
      - Extract the goal tiles and their required colors from the task's goal conditions.
      - Precompute adjacency relations (up, down, left, right) from static facts to represent the floorgrid as a graph.
      - Identify all tiles and robots present in the problem.
      - Store available colors for reference.
    This information is used to efficiently estimate distances and color switches during heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which goal-painted tiles are still not painted with the required color.
    2. Extract the robot's current location and its current paint color.
    3. Construct a graph of tiles using static adjacency facts to easily compute shortest paths.
    4. For each unpainted goal tile:
       - Compute the minimum distance (in tile moves) from the robot's current position to a tile adjacent to the goal tile,
         because painting requires adjacency.
    5. Select the closest such tile (for painting next).
    6. Heuristic cost components:
       - Robot movement cost: estimated by the shortest path distance to the chosen adjacent tile.
       - Painting cost: 1 action per tile to paint.
       - Color change cost: 1 action each time the robot must change its current painting color to the required color of a tile.
    7. To approximate minimal color changes:
       - Count the number of color clusters (groups of tiles requiring the same color), since switching color between clusters is needed.
       - If the robot currently holds a color, possibly reduce the number of color switches by one if it matches some cluster.
    8. Sum these costs for the heuristic estimate.
    9. Return 0 if the goal is already reached.

    This heuristic balances the travel needed to reach tiles with the number of painting and color-changing actions,
    providing an informed estimate that guides the search efficiently.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal painting requirements: tile -> required color
        self.goal_tiles_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles_colors[tile] = color

        # Extract adjacency relations - create an undirected graph (tile -> set of neighboring tiles)
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("up", "down", "left", "right") and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                # Add both directions, since painting can be done from adjacent tiles and moves are bidirectional
                self.adj.setdefault(tile1, set()).add(tile2)
                self.adj.setdefault(tile2, set()).add(tile1)

        # Identify robot(s) - we expect only one robot for the heuristic
        self.robots = set()
        for fact in task.initial_state | task.static:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and len(parts) == 3:
                self.robots.add(parts[1])

        # Identify available colors
        self.available_colors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "available-color" and len(parts) == 2:
                self.available_colors.add(parts[1])

    def __call__(self, node):
        state = node.state

        # --------- Helper functions ---------

        def robot_position_and_color():
            """Return the robot's current tile and color it holds."""
            robot_tile = None
            robot_color = None
            # There should be exactly one robot for this heuristic
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot_tile = parts[2]
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot_color = parts[2]
            return robot_tile, robot_color

        def painted_tiles_by_color():
            """Return dictionary tile -> color for currently painted tiles."""
            painted = {}
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "painted" and len(parts) == 3:
                    painted[parts[1]] = parts[2]
            return painted

        def neighbors(tile):
            """Return neighboring tiles of a given tile."""
            return self.adj.get(tile, set())

        def adjacent_tiles(tile):
            """Return tiles adjacent to the given tile (same as neighbors)."""
            return neighbors(tile)

        def bfs(start, goals_set):
            """
            BFS over tiles graph from start to any tile in goals_set.
            Returns minimal distance or None if no path found.
            """
            if start in goals_set:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                current, dist = queue.pop(0)
                for nbr in neighbors(current):
                    if nbr in visited:
                        continue
                    if nbr in goals_set:
                        return dist + 1
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
            return None

        # --------- Start heuristic calculation ---------

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        robot_tile, robot_color = robot_position_and_color()
        if robot_tile is None:
            # Defensive fallback: if robot position unknown, cannot estimate - return large cost
            return 1000

        painted = painted_tiles_by_color()

        # Identify tiles still needing painting:
        # Tiles that are in goal_tiles_colors but either not painted or painted in wrong color
        unpainted_goal_tiles = {
            tile for tile, col in self.goal_tiles_colors.items()
            if painted.get(tile) != col
        }
        if not unpainted_goal_tiles:
            return 0  # All goal tiles correctly painted

        # Compute for each unpainted goal tile the set of tiles adjacent to it (these are candidate painting positions)
        # Note: A tile must be clear to paint from, but we ignore that for heuristic efficiency

        # Aggregate all candidate painting positions (adjacent tiles of unpainted goal tiles)
        candidate_positions = set()
        tile_to_color = {}
        for tile in unpainted_goal_tiles:
            tile_to_color[tile] = self.goal_tiles_colors[tile]
            adj_tiles = adjacent_tiles(tile)
            candidate_positions.update(adj_tiles)

        # If any candidate position is not reachable or no candidate positions, we fallback
        if not candidate_positions:
            # No tile to stand on to paint (isolated goal tile?), estimate with large cost
            return len(unpainted_goal_tiles)*3

        # Find minimal distance from robot_tile to any candidate painting position
        dist_to_closest = bfs(robot_tile, candidate_positions)
        if dist_to_closest is None:
            # Unreachable, assign large cost
            dist_to_closest = 50

        # Calculate color changes needed

        # Group unpainted goal tiles by required color
        color_groups = {}
        for tile in unpainted_goal_tiles:
            c = self.goal_tiles_colors[tile]
            color_groups.setdefault(c, set()).add(tile)

        num_color_groups = len(color_groups)

        # If robot holds a color and that color is among the groups, one less color change needed
        color_changes = max(num_color_groups - 1, 0)
        if robot_color is None or robot_color not in color_groups:
            color_changes = num_color_groups  # robot must switch color for every color group

        # Painting cost: one paint action per unpainted tile
        paint_actions = len(unpainted_goal_tiles)

        # Summation:
        # movement cost to get adjacent to first tile + painting cost + color switching cost
        # We consider movement only once for heuristic, since the robot can plan an efficient route afterwards
        heuristic_value = dist_to_closest + paint_actions + color_changes

        return heuristic_value
