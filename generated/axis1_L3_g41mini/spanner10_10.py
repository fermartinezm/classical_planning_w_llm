from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (color).

    Summary
    -------
    This heuristic estimates the (not necessarily minimal) number of actions needed 
    to reach the goal state where all goal tiles are painted. It accounts for the 
    necessary robot movements along tile chains, color changes, and painting actions.

    Assumptions
    -----------
    - The goal is to paint certain tiles of type tile-2.
    - Painting requires the robot to be at a color matching the "down" tile 
      of the tile-2 to paint.
    - The robot can move "down" (move_down action) and change colors along "up" relations (change_color action).
    - Painting consumes the free-color resource and the robot must have that color.
    - Up/down relations form chains of tiles; painting requires navigating these chains.
    - The heuristic can underestimate or overestimate (not required to be admissible) but must be informative to reduce expansions.
    - The domain's static facts include the "up" relations between tiles which determine adjacency for movement and color changes.

    Heuristic Initialization
    -----------------------
    - Extract the "up" relations from the static facts and build bidirectional graph structures (up and down edges).
    - Extract the list of goal tiles to paint.
    - Precompute for each tile its chain of reachable tiles upward/downward for quick distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic performs:

    1. Identify which goal tiles remain unpainted.
       - If none remain, heuristic cost is 0.

    2. Identify robot "colors" currently present by facts of the form (robot-at ?color-2 ?color).

    3. Identify which tiles are currently "available-color" for each robot color or tile.
       - This restricts where the robot/color can move.

    4. For each unpainted goal tile (tile-2), 
       - Determine the minimum estimated cost for the robot to reach the tile and paint it:
          a) The robot must have access to the color corresponding to the "down" tile linked with the tile-2 (since paint_up requires free-color ?v3 and down ?v6 tiles).
          b) Estimate the minimal number of change_color actions needed to move the robot color along "up" chains from its current available tile to the needed tile.
          c) Estimate the number of move_down actions to move robot colors down if applicable (paint_up includes usage of move_down semantics).
          d) Count 1 paint_up action per tile to be painted.

    5. Sum the estimated costs for all unpainted goal tiles, or take the max to avoid overestimating paths that can be combined efficiently.

    This heuristic captures the essence of navigating tile-color adjacency, switching colors by moving along "up" relations, and painting tiles requiring coordination of robot colors and tiles. It remains computationally efficient by using preprocessed adjacency and ignoring some complex interactions but strongly guides search towards painting all target tiles.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts
        self.static = task.static

        # Extract up relations from static facts to build graph edges for tiles
        self.up_edges = {}
        self.down_edges = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up":
                # up tileX tileY
                tile_from, tile_to = parts[1], parts[2]
                self.up_edges.setdefault(tile_from, set()).add(tile_to)
                self.down_edges.setdefault(tile_to, set()).add(tile_from)

        # For quick distance computations, precompute all reachable tiles upwards and downwards via BFS
        # We'll build adjacency in both directions
        self.all_up_reachable = {}   # tile -> set of tiles reachable by repeatedly following up edges
        self.all_down_reachable = {} # tile -> set of tiles reachable by repeatedly following down edges

        # Compute all upward reachable tiles
        for tile in set(self.up_edges.keys()).union(self.down_edges.keys()):
            # Up reachable
            visited = set()
            queue = [tile]
            while queue:
                cur = queue.pop()
                for nxt in self.up_edges.get(cur, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            self.all_up_reachable[tile] = visited

        # Compute all downward reachable tiles
        for tile in set(self.up_edges.keys()).union(self.down_edges.keys()):
            visited = set()
            queue = [tile]
            while queue:
                cur = queue.pop()
                for nxt in self.down_edges.get(cur, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            self.all_down_reachable[tile] = visited

        # Extract goal tiles to paint (tile-2 types) from goals by looking for (painted tile)
        self.goal_painted_tiles = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "painted":
                self.goal_painted_tiles.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick check if goals already achieved: all goal tiles painted
        if all(f"(painted {tile})" in state for tile in self.goal_painted_tiles):
            return 0

        # Identify which goal tiles remain unpainted
        unpainted_tiles = [tile for tile in self.goal_painted_tiles if f"(painted {tile})" not in state]
        if not unpainted_tiles:
            return 0  # all goals painted

        # Extract current robot color locations, from (robot-at color-2 color)
        # Map robot-color (color-2) -> set of colors currently at
        robot_color_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_color = parts[1]      # color-2 object
                color = parts[2]            # color object
                robot_color_positions.setdefault(robot_color, set()).add(color)

        # Extract available-color facts: available-color ?v1 ?v2
        # This shows which robot color or tile can be "available-color"
        available_color_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                key, val = parts[1], parts[2]
                available_color_map.setdefault(key, set()).add(val)

        # Extract free-color tiles from state for quick lookup
        free_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "free-color":
                free_colors.add(parts[1])

        # Helper to find minimal steps to move from one tile to another along up edges (change_color)
        # That corresponds to the number of change_color moves needed (up moves along up edges)
        def min_up_steps(start_tile, end_tile):
            if start_tile == end_tile:
                return 0
            # BFS from start_tile upwards
            visited = {start_tile}
            queue = [(start_tile, 0)]
            while queue:
                cur, dist = queue.pop(0)
                for nxt in self.up_edges.get(cur, []):
                    if nxt == end_tile:
                        return dist + 1
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))
            # Not reachable by going up directly
            return None

        # Helper to find minimal steps to move from one tile to another along down edges (move_down)
        def min_down_steps(start_tile, end_tile):
            if start_tile == end_tile:
                return 0
            visited = {start_tile}
            queue = [(start_tile, 0)]
            while queue:
                cur, dist = queue.pop(0)
                for nxt in self.down_edges.get(cur, []):
                    if nxt == end_tile:
                        return dist + 1
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))
            return None

        # Find the corresponding tile and color needed for painting each unpainted tile
        # paint_up action precondition includes: 
        #   (available-color ?v1 ?v2)        -- robot color availability at tile ?v2
        #   (available-color ?v6 ?v2)        -- tile-2 availability at tile ?v2
        #   (robot-at ?v1 ?v3)               -- robot color at color ?v3
        #   (free-color ?v3)                 -- color must be free
        #   (down ?v6)                      -- the tile-2 has down relation (i.e. tile-2 mapped to tile)

        # We need to find for each tile-2 (the goal tile) the corresponding tile (?v6), its down tile (?v2),
        # and the color ?v3 so that robot can paint.

        # First, for each tile-2 to paint, find all candidate "down" tiles (the tile the robot must be available on)
        # Using static facts, each tile-2 that has a (down tile-2) fact associates with a tile where painting happens
        tile2_down_tile = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "down":
                tile2obj = parts[1]
                tile2_down_tile[tile2obj] = True  # Mark existence

        # The set tile2_down_tile keys must include all tile-2 (goal tiles) per domain design.
        # But we don't know the actual down tile object here; however, from the examples,
        # it seems the down tile is the tile-2 itself or connected implicitly. Since in the domain,
        # down ?v6 is a property of tile-2, probably the painting is done relative to the tile-2.

        # We assume that for the paint_up action, the robot must be available on the tile linked with the tile-2.
        # From the domain definition, paint_up preconditions:
        # (available-color ?v1 ?v2) and (available-color ?v6 ?v2) and (robot-at ?v1 ?v3) and (free-color ?v3) and (down ?v6)
        # =>
        # => tile-2 ?v6 has down relation and ?v2 tile is the "location" for availability.

        # We interpret that the tile ?v2 is the location on which robot and tile-2 availability must coincide,
        # so for simplicity, assume the down tile is the tile-2 itself or closest related tile.

        # To approximate, use the "available-color" facts linking tile-2 (?v6) and some tiles (?v2).
        # Build reverse map: tile-2 -> set(tiles) where available-color (tile-2 tile) holds
        tile2_available_tiles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                obj1, obj2 = parts[1], parts[2]
                if obj1 in self.goal_painted_tiles:
                    tile2_available_tiles.setdefault(obj1, set()).add(obj2)

        # The robot colors and their current tiles (available-color robot_color tile)
        robot_color_current_tiles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                obj1, obj2 = parts[1], parts[2]
                # robot colors are color-2 types; assuming robot colors are keys that appear in robot_color_positions
                if obj1 in robot_color_positions:
                    robot_color_current_tiles.setdefault(obj1, set()).add(obj2)

        # free colors that are robot colors (free-color ?v3)
        free_colors_in_state = {fact.split()[1][:-1] for fact in state if fact.startswith("(free-color ")}

        # Estimate cost for painting each unpainted tile:
        # painting cost = cost to move robot color (change_color) along up edges to needed tile
        #               + cost to move robot down (move_down if any)
        #               + one paint_up action

        cost_per_goal = []

        for tile2 in unpainted_tiles:
            # Candidate tiles where painting can happen (where tile2 is available-color)
            candidate_tiles = tile2_available_tiles.get(tile2, set())
            if not candidate_tiles:
                # Fallback: if no available info, guess 1 paint action plus at least 1 move
                cost_per_goal.append(2)
                continue

            min_cost_for_this_tile = None

            # For each robot color in the robot_color_positions, find cost to move that robot color to correct tile to paint
            robot_colors = list(robot_color_positions.keys())

            for robot_color in robot_colors:
                # Check if robot color is free to paint (free-color)
                # According to paint_up, robot color must be at free-color
                is_free = any(f"(free-color {c})" in state for c in robot_color_positions.get(robot_color, []))
                if not is_free:
                    # Robot color is not free to paint; painting with this color not possible now.
                    continue

                # Robot_color must be at some color ?v3 where free-color ?v3 holds, this is checked above.
                # Find where robot_color is currently "available-color" (set of tiles)
                robot_tiles = robot_color_current_tiles.get(robot_color, set())
                if not robot_tiles:
                    # If no available-color facts for this robot_color, fallback
                    continue

                for paint_tile in candidate_tiles:
                    # For each robot tile location and each paint tile, estimate steps needed.

                    # Steps to move robot color from robot_tiles to paint_tile along up relations (change_color)
                    # We estimate minimal number of up steps between one of robot_tiles and paint_tile
                    # We pick the minimal among all robot tiles.

                    min_steps_up = None
                    for rt in robot_tiles:
                        steps = min_up_steps(rt, paint_tile)
                        if steps is not None:
                            if min_steps_up is None or steps < min_steps_up:
                                min_steps_up = steps

                    if min_steps_up is None:
                        # No upward path found; possibly downward path needed (but domain allows only change_color is up movement)
                        # So possibly not reachable this way; skip this combo
                        continue

                    # Steps down (move_down action) might be needed if paint_tile is "down" from the tile-2.
                    # From domain: paint_up requires (down ?v6) i.e. tile-2 has down relation.
                    # But not clear if move_down action is needed for the robot color to be on specific color.
                    # move_down action precondition needs available-color for color-2 and color on same tile, and changes robot-at
                    # Approximate by 0 or 1 move_down cost since this is complex and unknown in heuristic.

                    move_down_cost = 0

                    # Add 1 paint action
                    paint_cost = 1

                    total_cost = min_steps_up + move_down_cost + paint_cost

                    if min_cost_for_this_tile is None or total_cost < min_cost_for_this_tile:
                        min_cost_for_this_tile = total_cost

            # If no suitable robot color path found, assign default cost to encourage search:
            if min_cost_for_this_tile is None:
                min_cost_for_this_tile = 5  # arbitrary penalty for unknown cost

            cost_per_goal.append(min_cost_for_this_tile)

        # Combine the costs for all unpainted tiles.
        # We take the sum, as all unpainted tiles require painting.
        heuristic_value = sum(cost_per_goal)

        return heuristic_value
