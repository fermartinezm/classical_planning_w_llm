from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    E.g., "(available-color color1 tile4)" -> ["available-color", "color1", "tile4"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (color) domain.

    Summary
    -------
    This heuristic estimates the number of required actions to reach the goal state
    in the color domain where robots manipulate colors available on tiles and paint
    certain tiles by moving "up" through connected tiles, changing color availability,
    moving down, and painting tiles.

    Assumptions
    -----------
    - The main goal is to paint all goal tiles (tile-2 objects) flagged in the goal.
    - To paint a tile, the robot must:
      * have a free color,
      * be robot-at some color,
      * the tile must be "down",
      * the color must be available on the tile over which painting occurs.
    - The robot must navigate colors availability through tiles moving "up" and "down".
    - An action `change_color` moves the robot's available color from one tile to an "up" tile.
    - An action `move_down` moves the robot-at predicate from one color to another color at the same tile.
    - An action `paint_up` paints a down tile; it consumes the free-color of a robot color and removes the down predicate of the tile.
    - The "up" relation between tiles is static.

    Heuristic Initialization
    ------------------------
    - Extract from static facts the "up" edges between tiles into a graph-like structure.
    - Extract the set of goal tiles that must be painted.
    - Extract from the initial state which tiles are initially "down".
    - Preprocessing done to enable quick lookups of adjacency and goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal tile that must be painted but is not yet painted in the current state:
    1) The tile must currently have the predicate (down tileX).
    2) To paint it, one robot color must be robot-at some color C that is free-color,
       and the color C must be available on the tile immediately "below" (the up edge goes from the lower tile to the higher one, so painting requires the tile below).
    3) Estimating the number of steps for painting a single tile roughly involves:
       - Moving the available color upward along the "up" edges until it reaches the tile below the tile to paint,
         counting one `change_color` action per tile step upwards.
       - Moving down among colors (robot-at changes) until the robot has a free color to paint.
         We approximate this cost by counting how many robot-at colors currently free-color; if none free-color, assume 1 move_down needed.
       - Performing the paint_up action itself (cost=1).
    4) Summing these costs for all goal tiles not yet painted gives an estimate of the remaining cost.
    5) Because the robot can have multiple free colors and manipulations may overlap, we take minimum costs per tile and sum them to obtain a heuristic.
    6) In cases where no free-color exists for the robot-at color, we add some penalty for obtaining a free-color by moving down.

    Note: The heuristic is non-admissible but tries to balance accuracy and efficiency.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.operators = task.operators

        # Extract the "up" relation edges between tiles for quick graph traversal.
        self.up_graph = {}  # key: tile, value: set of tiles that are "up" of it
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up":
                below, above = parts[1], parts[2]
                if below not in self.up_graph:
                    self.up_graph[below] = set()
                self.up_graph[below].add(above)

        # Keep inverse graph from above to below, useful for stepping down in the tile graph.
        self.down_graph = {}
        for below, uppers in self.up_graph.items():
            for upper in uppers:
                if upper not in self.down_graph:
                    self.down_graph[upper] = set()
                self.down_graph[upper].add(below)

        # Extract the tile-2 objects that must be painted from goals
        self.goal_tiles = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "painted":
                tile = parts[1]
                self.goal_tiles.add(tile)

        # Precompute all tiles connected via up relation for path calculations
        # We'll use BFS for path length estimates later.

    def _bfs_distance(self, start_tile, target_tile):
        """
        Compute shortest distance from start_tile to target_tile by moving via "up" edges.
        Distance is number of 'change_color' actions needed.
        Returns infinity if no path exists.
        """
        if start_tile == target_tile:
            return 0
        from collections import deque

        visited = set()
        queue = deque([(start_tile, 0)])

        while queue:
            tile, dist = queue.popleft()
            if tile == target_tile:
                return dist
            if tile in visited:
                continue
            visited.add(tile)
            for neighbor in self.up_graph.get(tile, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract painted tiles in current state
        painted_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                painted_tiles.add(parts[1])

        # Extract which tiles are currently down
        down_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "down":
                down_tiles.add(parts[1])

        # Extract robot colors (color-2) that are currently robot-at and their colors (color)
        # store as robot_at_facts: key = robot-color-2, value = color
        robot_at_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                r_color2, color = parts[1], parts[2]
                robot_at_colors[r_color2] = color

        # Extract all free colors (color) currently that may be available for painting
        free_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "free-color":
                free_colors.add(parts[1])

        # Extract the availability: which robot-color-2 has which tiles available with color:
        # Create a dict: key = robot-color-2, value = set of tiles it has available-color on
        available_color = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                obj1, obj2 = parts[1], parts[2]
                # By PDDL domain predicates available-color ?v1 - robot ?v2 - tile
                # But types are mixed in domain file, so we keep as is.
                if obj1 not in available_color:
                    available_color[obj1] = set()
                available_color[obj1].add(obj2)

        total_estimated_cost = 0

        for goal_tile in self.goal_tiles:
            # Skip if already painted
            if goal_tile in painted_tiles:
                continue

            # The tile to paint must currently be down.
            if goal_tile not in down_tiles:
                # Since painting needs down tile, if not down probably need extra actions.
                # To keep heuristic finite, add penalty 3 (paint_up + make down + others)
                total_estimated_cost += 3
                continue

            # The paint_up action requires the available-color of a robot-color-2 on a tile "below" the tile to paint.
            # Find all tiles t such that (up t goal_tile) - tile immediately below goal_tile
            # If multiple tiles are below, consider minimal cost among them.

            tiles_below = self.down_graph.get(goal_tile, set())
            if not tiles_below:
                # If no tile below goal_tile (no up edge), we assume that available color must be on goal_tile itself.
                tiles_below = {goal_tile}

            # We try to estimate minimal cost for getting available-color on a tile below goal_tile for some robot color.
            # We'll estimate costs per possible robot color.

            min_cost_for_tile = float('inf')

            for robot_col2, current_color in robot_at_colors.items():
                # 1) Find shortest distance from any available-color tile for this robot_col2 to some tile below goal_tile by moving with change_color actions up.
                # The robot can move available-color upward by change_color action (precondition: available-color on tile X, up X Y).

                # Find closest tile in available-color[robot_col2] that can reach one tile below goal_tile by going up via change_color

                if robot_col2 not in available_color:
                    # If robot_col2 has no available-color tiles, assign big penalty
                    distance_available_to_goal = float('inf')
                else:
                    distances = []
                    for tile_with_avail_color in available_color[robot_col2]:
                        for tile_below in tiles_below:
                            dist = self._bfs_distance(tile_with_avail_color, tile_below)
                            if dist != float('inf'):
                                distances.append(dist)
                    distance_available_to_goal = min(distances) if distances else float('inf')

                if distance_available_to_goal == float('inf'):
                    # Cannot reach needed tile below goal_tile; consider penalty cost of 5 (moving available color plus change)
                    distance_available_to_goal = 5

                # 2) Check if current_color is free-color
                has_free_color = current_color in free_colors

                # If robot-at color is free-color, painting can occur immediately after moving available-color.
                # Else, robot must do move_down action(s) to get free-color robot-at (estimate 1 move_down action).

                move_down_cost = 0 if has_free_color else 1

                # 3) Finally, painting costs exactly 1 action (paint_up).

                paint_cost = 1

                cost_for_robot_color = distance_available_to_goal + move_down_cost + paint_cost

                if cost_for_robot_color < min_cost_for_tile:
                    min_cost_for_tile = cost_for_robot_color

            # If no robot-at colors available at all (unlikely but possible), assign penalty for painting this tile
            if min_cost_for_tile == float('inf'):
                min_cost_for_tile = 7

            # Add min cost for painting this tile
            total_estimated_cost += min_cost_for_tile

        # Return finite estimate (0 for goal, positive otherwise)
        return int(total_estimated_cost)
