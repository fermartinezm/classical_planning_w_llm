import math # Required for float('inf')

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        This heuristic estimates the total number of actions required to satisfy
        all unsatisfied goal conditions, which are typically (painted TILE COLOR)
        facts. For each goal tile that is not yet painted with the correct color,
        it estimates the minimum cost to get a robot into a position to paint
        that tile with the required color, plus the cost of the paint action itself.
        The cost for a robot to paint a tile includes the cost to change color
        (if needed) and the estimated movement cost (Manhattan distance) to a
        clear tile adjacent (specifically, above or below) to the target tile
        where the paint action can be performed. If a goal tile or all potential
        paint positions for it are blocked (e.g., not clear), the heuristic adds
        infinity, guiding the search away from such states.

    Assumptions:
        - Tile names follow the format 'tile_R_C' where R and C are integers
          representing row and column, allowing Manhattan distance calculation.
        - The grid structure defined by 'up', 'down', 'left', 'right' predicates
          corresponds to this row/column structure.
        - The 'paint_up' and 'paint_down' actions require the robot to be
          at a tile directly above or below the tile being painted, as indicated
          by the 'up' and 'down' static predicates.
        - Movement actions require the destination tile to be 'clear'. The
          Manhattan distance estimate ignores potential obstacles on the path
          but verifies that the final adjacent tile is clear.
        - Solvable problems do not require repainting tiles that are already
          painted with the wrong color, nor do they require complex unblocking
          of goal tiles or paint positions if they are not clear.
        - All colors mentioned in goal facts are available (available-color is true).

    Heuristic Initialization:
        The constructor processes the task definition to extract static information:
        - `goal_facts`: The set of facts that must be true in the goal state.
        - `static_facts`: The set of facts that are true in all states (e.g., adjacency).
        - `all_tiles`: A set of all tile object names in the problem.
        - `paint_positions`: A dictionary mapping each tile to a set of tiles
          where a robot must be located to paint that tile (based on 'up' and 'down'
          static predicates).
        - `robots`: A set of all robot object names.
        - `available_colors`: A set of all available color object names.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Extract the current positions and held colors for all robots from the state.
        3. Identify all tiles that are currently 'clear' in the state.
        4. Iterate through each goal fact. Goal facts are expected to be of the form `(painted TILE COLOR)`.
        5. If the current goal fact `(painted TILE COLOR)` is already present in the state, this goal is satisfied; continue to the next goal fact.
        6. If the goal fact is not satisfied, identify the `TargetTile` and `TargetColor`.
        7. Check if the `TargetTile` itself is 'clear' in the current state. The 'paint' action requires the target tile to be clear. If it is not clear (e.g., painted with the wrong color, or occupied by a robot), this goal tile cannot be painted directly. Add `infinity` to `h` and move to the next goal fact, as this state is likely a dead end or requires unmodeled actions.
        8. Find the set of potential tiles where a robot must stand to paint the `TargetTile`. These are the tiles `P` such that `(up TargetTile P)` or `(down TargetTile P)` is a static fact.
        9. Filter this set of potential paint positions to keep only those that are currently 'clear' in the state. Let this be `ClearPaintPositions`. A robot can only move onto or occupy a clear tile.
        10. If `ClearPaintPositions` is empty, it means there is no clear spot adjacent (above/below) to the `TargetTile` where a robot can stand to paint it. This state is likely blocked. Add `infinity` to `h` and move to the next goal fact.
        11. If `ClearPaintPositions` is not empty, find the minimum cost to get *any* robot to *any* tile in `ClearPaintPositions` while holding the `TargetColor`.
            - For each robot:
                - Determine the cost to acquire the `TargetColor`: 0 if the robot already has it, 1 otherwise (assuming all goal colors are available).
                - Calculate the minimum Manhattan distance from the robot's current position to each tile in `ClearPaintPositions`. The minimum of these distances is the estimated movement cost for this robot.
                - The total cost for this robot for this goal tile is the color cost plus the minimum movement cost.
            - The minimum cost across all robots for this goal tile is the minimum of these sums over all robots.
        12. Add this minimum robot cost (movement + color) plus 1 (for the paint action itself) to the total heuristic value `h`.
        13. After iterating through all goal facts, return the total accumulated heuristic value `h`.
    """
    def __init__(self, task):
        self.goal_facts = task.goals
        self.static_facts = task.static

        # Extract all tiles from initial state, goals, and static facts
        self.all_tiles = set()
        # Collect all objects mentioned in initial state, goals, and static facts
        all_objects = set()
        for fact in task.initial_state | task.goals | self.static_facts:
             parts = self._parse_fact(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Filter for objects that look like tiles
        self.all_tiles = {obj for obj in all_objects if obj.startswith('tile_')}


        # Build paint positions map: {tile_to_paint: set_of_robot_positions}
        # A robot at tile X can paint tile Y if (up Y X) or (down Y X) is true.
        self.paint_positions = {tile: set() for tile in self.all_tiles}
        for fact in self.static_facts:
            parts = self._parse_fact(fact)
            if parts[0] == 'up' and len(parts) == 3: # (up ?y ?x) -> robot at ?x can paint ?y
                tile_y, tile_x = parts[1], parts[2]
                if tile_y in self.paint_positions: # Ensure tile_y is a known tile
                    self.paint_positions[tile_y].add(tile_x)
            elif parts[0] == 'down' and len(parts) == 3: # (down ?y ?x) -> robot at ?x can paint ?y
                 tile_y, tile_x = parts[1], parts[2]
                 if tile_y in self.paint_positions: # Ensure tile_y is a known tile
                    self.paint_positions[tile_y].add(tile_x)

        # Identify robots (assuming they are present in initial state with robot-at fact)
        self.robots = set()
        for fact in task.initial_state:
            if fact.startswith('(robot-at ') and len(self._parse_fact(fact)) >= 2:
                self.robots.add(self._parse_fact(fact)[1])

        # Identify available colors (assuming they are static)
        self.available_colors = set()
        for fact in self.static_facts:
             if fact.startswith('(available-color ') and len(self._parse_fact(fact)) >= 2:
                 self.available_colors.add(self._parse_fact(fact)[1])


    def _parse_fact(self, fact_str):
        """Parses a fact string '(predicate arg1 arg2)' into ['predicate', 'arg1', 'arg2']."""
        # Remove parentheses and split by space
        # Handle potential empty string after split if there are multiple spaces
        return [part for part in fact_str[1:-1].split(' ') if part]

    def _parse_tile_name(self, tile_str):
        """Parses tile name 'tile_R_C' into (R, C) integers."""
        parts = tile_str.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                # This should not happen with valid domain inputs
                raise ValueError(f"Unexpected tile name format: {tile_str}")
        # This should not happen with valid domain inputs
        raise ValueError(f"Unexpected tile name format: {tile_str}")


    def manhattan_distance(self, tile1_str, tile2_str):
        """Calculates Manhattan distance between two tiles based on their names."""
        r1, c1 = self._parse_tile_name(tile1_str)
        r2, c2 = self._parse_tile_name(tile2_str)
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        h = 0
        state_set = set(state) # Convert frozenset to set for potentially faster lookups

        # Get current robot positions and colors
        robot_info = {} # {robot_name: {'pos': tile_name, 'color': color_name}}
        for robot_name in self.robots:
             robot_info[robot_name] = {'pos': None, 'color': None} # Initialize info

        for fact in state_set:
            if fact.startswith('(robot-at '):
                parts = self._parse_fact(fact)
                if len(parts) == 3 and parts[1] in self.robots:
                    robot, pos = parts[1], parts[2]
                    robot_info[robot]['pos'] = pos
            elif fact.startswith('(robot-has '):
                parts = self._parse_fact(fact)
                if len(parts) == 3 and parts[1] in self.robots:
                    robot, color = parts[1], parts[2]
                    robot_info[robot]['color'] = color

        # Get current clear tiles
        clear_tiles = {self._parse_fact(fact)[1] for fact in state_set if fact.startswith('(clear ') and len(self._parse_fact(fact)) == 2}

        # Iterate through goal facts
        for goal_fact in self.goal_facts:
            # Goal facts are assumed to be (painted TILE COLOR)
            if goal_fact in state_set:
                continue # Goal already satisfied

            parts = self._parse_fact(goal_fact)
            if parts[0] != 'painted' or len(parts) != 3:
                 # Should not happen in this domain based on examples
                 continue

            target_tile = parts[1]
            target_color = parts[2]

            # Check if the target tile itself is clear. Paint action requires (clear ?y).
            # If it's not clear, it's blocked (painted wrong color, or occupied).
            if target_tile not in clear_tiles:
                 # If it's not clear, it must be painted or occupied.
                 # If painted with the wrong color, it's a dead end.
                 # If occupied, the robot needs to move off, which isn't directly modeled.
                 # Assume blocked states are unsolvable for this goal tile.
                 h += float('inf')
                 continue # Move to next goal fact

            # Find potential robot paint positions for target_tile (tiles above/below)
            potential_paint_positions = self.paint_positions.get(target_tile, set())

            # Filter paint positions to keep only those that are currently clear
            clear_paint_positions = {pos for pos in potential_paint_positions if pos in clear_tiles}

            # If no clear tile exists where a robot can stand to paint the target tile
            if not clear_paint_positions:
                # This state is likely unsolvable or requires complex unblocking
                h += float('inf')
                continue # Move to next goal fact

            # Find the minimum cost to get any robot to a clear paint position with the target color
            min_cost_for_this_tile = float('inf')

            for robot_name, info in robot_info.items():
                robot_pos = info.get('pos')
                robot_color = info.get('color')

                if robot_pos is None or robot_color is None:
                    # Robot info is incomplete, skip this robot for this calculation
                    continue

                # Cost to get the correct color
                # Assumes target_color is available, which is static.
                color_cost = 0 if robot_color == target_color else 1

                # Minimum movement cost from robot's current position to any clear paint position
                min_move_cost = float('inf')
                for clear_paint_pos in clear_paint_positions:
                    try:
                        move_cost = self.manhattan_distance(robot_pos, clear_paint_pos)
                        min_move_cost = min(min_move_cost, move_cost)
                    except ValueError:
                         # Handle cases where tile name parsing fails for some reason
                         min_move_cost = float('inf') # Treat as unreachable

                # Total cost for this robot to paint this tile
                cost_for_robot = color_cost + min_move_cost

                # Update minimum cost across all robots for this tile
                min_cost_for_this_tile = min(min_cost_for_this_tile, cost_for_robot)

            # Add the minimum cost (movement + color) plus the paint action cost (1)
            if min_cost_for_this_tile == float('inf'):
                 # This should only happen if there are robots but none can reach a clear paint position
                 # (e.g., tile name parsing failed for all reachable clear paint positions, or no robots exist)
                 # Treat as blocked.
                 h += float('inf')
            else:
                 h += min_cost_for_this_tile + 1 # Add cost to get robot ready + paint action

        return h
