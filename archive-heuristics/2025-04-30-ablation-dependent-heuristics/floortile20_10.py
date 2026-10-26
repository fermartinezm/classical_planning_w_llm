import math

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        estimated costs for each tile that needs to be painted according to the
        goal but is not yet painted correctly in the current state. The cost
        for each such tile includes the paint action itself, the cost to clear
        the tile if occupied, the minimum movement cost for any robot to reach
        an adjacent tile, and a global cost component for acquiring colors
        that are needed but not currently held by any robot.

    Assumptions:
        - The problem instances represent a grid structure where tiles are named
          in the format 'tile_R_C' where R and C are integers representing row
          and column.
        - Adjacency relations (up, down, left, right) correspond to movements
          between adjacent cells in this grid structure.
        - Tiles that need to be painted according to the goal have at least one
          adjacent tile in the grid structure.
        - If a tile is painted with a color different from the goal color, the
          state is considered unsolvable (heuristic returns infinity), as there
          is no action to unpaint or change the color of an already painted tile.
        - Available colors are static and listed in the initial state.
        - The 'free-color' predicate is not used in the domain actions and is ignored.

    Heuristic Initialization:
        The constructor processes the static facts from the task definition.
        It builds an adjacency map representing the grid structure based on
        'up', 'down', 'left', 'right' predicates. It also collects all unique
        tile names mentioned in static, initial, and goal facts and precomputes
        their row/column coordinates by parsing the tile names. Available colors
        are also stored. This precomputation makes the heuristic calculation
        for each state faster.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Parse the current state to extract robot positions, robot held colors,
           and the current painted status of tiles. Also, identify occupied tiles.
        3. Parse the goal state to identify the desired painted status of tiles.
        4. Iterate through the tiles specified in the goal state:
           - If a tile is currently painted with a color different from the goal
             color, the state is considered unsolvable, and the heuristic
             immediately returns `float('inf')`.
           - If a tile is not painted with the goal color (either clear or
             painted with the correct color in the goal but not yet in the state),
             add it to a list of 'unpainted goal tiles' and note the required color.
        5. If there are no unpainted goal tiles, the goal is reached, return 0.
        6. Calculate a color acquisition cost: Identify the set of colors needed
           by the unpainted goal tiles. Identify the set of colors currently held
           by robots. The number of colors that are needed but not held by any
           robot contributes to the heuristic. Add the size of this set to `h`.
           This estimates the minimum number of 'change_color' actions required
           to make all necessary colors available to at least one robot.
        7. Iterate through the list of 'unpainted goal tiles':
           - For each unpainted tile, add 1 to `h` representing the cost of the
             'paint' action itself.
           - Check if the tile is currently occupied by any robot. If it is, add 1
             to `h` representing the estimated cost to move the occupying robot off
             the tile to make it clear for painting.
           - Calculate the minimum Manhattan distance from any robot's current
             position to any tile adjacent to the unpainted goal tile. This
             estimates the minimum movement cost to get a robot into a position
             from which it can paint the tile. Add this minimum distance to `h`.
             If the tile has no adjacent tiles (and is a known tile), or if no
             robot can reach any adjacent tile, the state is considered unsolvable,
             and the heuristic returns `float('inf')`.
        8. Return the final calculated value of `h`.
    """

    def __init__(self, task):
        self.task = task
        self.adjacency_map = {} # tile -> list of adjacent tiles
        self.all_tiles = set() # Collect all unique tile names
        self.coords = {} # tile -> (row, col)
        self.available_colors = set()

        # Process static facts
        for fact in task.static:
            fact_str = str(fact) # Ensure it's a string
            if fact_str.startswith(('(up ', '(down ', '(left ', '(right ')):
                parts = fact_str.split()
                # Example: '(up tile_1_1 tile_0_1)' -> parts = ['(up', 'tile_1_1', 'tile_0_1)']
                # adj_tile is tile_1_1, main_tile is tile_0_1
                if len(parts) == 3:
                    adj_tile = parts[1]
                    main_tile = parts[2][:-1] # Remove ')'
                    self.adjacency_map.setdefault(main_tile, []).append(adj_tile)
                    self.adjacency_map.setdefault(adj_tile, []).append(main_tile) # Adjacency is symmetric
                    self.all_tiles.add(adj_tile)
                    self.all_tiles.add(main_tile)
            elif fact_str.startswith('(available-color '):
                parts = fact_str.split()
                if len(parts) == 2:
                    color = parts[1].rstrip(')')
                    self.available_colors.add(color)

        # Collect all tile names from initial state and goal state
        # This ensures we know about all tiles in the problem, even if they
        # don't appear in static adjacency facts (e.g., small grids or isolated tiles)
        for fact in task.initial_state:
             fact_str = str(fact)
             if fact_str.startswith('(robot-at '):
                 parts = fact_str.split()
                 if len(parts) == 3:
                     tile = parts[2].rstrip(')')
                     self.all_tiles.add(tile)
                     if tile not in self.adjacency_map:
                         self.adjacency_map[tile] = [] # Ensure tile exists in map
             elif fact_str.startswith('(clear '):
                 parts = fact_str.split()
                 if len(parts) == 2:
                     tile = parts[1].rstrip(')')
                     self.all_tiles.add(tile)
                     if tile not in self.adjacency_map:
                         self.adjacency_map[tile] = [] # Ensure tile exists in map
             elif fact_str.startswith('(painted '):
                 parts = fact_str.split()
                 if len(parts) == 3:
                     tile = parts[1]
                     self.all_tiles.add(tile)
                     if tile not in self.adjacency_map:
                         self.adjacency_map[tile] = [] # Ensure tile exists in map


        for fact in task.goals:
             fact_str = str(fact)
             if fact_str.startswith('(painted '):
                 parts = fact_str.split()
                 if len(parts) == 3:
                     tile = parts[1]
                     self.all_tiles.add(tile)
                     if tile not in self.adjacency_map:
                         self.adjacency_map[tile] = [] # Ensure tile exists in map

        # Precompute coordinates for Manhattan distance for all known tiles
        for tile in self.all_tiles:
            try:
                self.coords[tile] = self.get_coords(tile)
            except ValueError:
                 # If a tile name doesn't match the expected format,
                 # it's likely an invalid problem instance for this heuristic.
                 # The heuristic will return infinity later if it tries to use
                 # coordinates for this tile.
                 pass


    def get_coords(self, tile_name):
        """Parses tile name 'tile_R_C' to get (row, col)."""
        # Assumes tile names are always in the format tile_R_C
        parts = tile_name.split('_')
        # Check if parts has enough elements and row/col are integers
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                # Row or column is not an integer
                raise ValueError(f"Invalid tile coordinate format: {tile_name}")
        else:
            # Tile name does not match tile_R_C format
            raise ValueError(f"Invalid tile name format: {tile_name}")


    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles."""
        if tile1_name not in self.coords or tile2_name not in self.coords:
            # This indicates an issue with tile name parsing or collection
            # of all_tiles if valid tile names are missing coordinates.
            # For valid problems, this should not happen.
            return float('inf') # Cannot calculate distance

        r1, c1 = self.coords[tile1_name]
        r2, c2 = self.coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, state):
        """
        Compute the heuristic value for the given state.
        """
        h = 0

        robot_positions = {} # robot -> tile
        robot_colors = {} # robot -> color
        occupied_tiles = set() # tiles occupied by robots
        current_painted = {} # tile -> color

        # Parse state
        for fact in state:
            fact_str = str(fact) # Ensure it's a string
            if fact_str.startswith('(robot-at '):
                parts = fact_str.split()
                if len(parts) == 3:
                    robot = parts[1]
                    tile = parts[2][:-1] # Remove ')'
                    robot_positions[robot] = tile
                    occupied_tiles.add(tile)
            elif fact_str.startswith('(robot-has '):
                parts = fact_str.split()
                if len(parts) == 3:
                    robot = parts[1]
                    color = parts[2][:-1] # Remove ')'
                    robot_colors[robot] = color
            elif fact_str.startswith('(painted '):
                parts = fact_str.split()
                if len(parts) == 3:
                    tile = parts[1]
                    color = parts[2][:-1] # Remove ')'
                    current_painted[tile] = color
            # Ignore other facts like 'clear' as they are derived or not needed directly for this heuristic

        # Parse goal (from self.task.goals)
        goal_painted = {} # tile -> color
        for goal_fact in self.task.goals:
             goal_fact_str = str(goal_fact) # Ensure it's a string
             if goal_fact_str.startswith('(painted '):
                parts = goal_fact_str.split()
                if len(parts) == 3:
                    tile = parts[1]
                    color = parts[2][:-1] # Remove ')'
                    goal_painted[tile] = color

        unpainted_goal_tiles_info = [] # List of (tile, color_needed)
        colors_needed_by_unpainted = set()

        # Check for wrongly painted tiles and collect unpainted goal tiles
        for tile, color_needed in goal_painted.items():
            if tile in current_painted:
                if current_painted[tile] != color_needed:
                    # Tile is painted the wrong color - likely unsolvable
                    # Return infinity or a very large number
                    return float('inf')
            else:
                # Tile is not painted with the correct color, needs painting
                unpainted_goal_tiles_info.append((tile, color_needed))
                colors_needed_by_unpainted.add(color_needed)

        # If no unpainted goal tiles, goal is reached
        if not unpainted_goal_tiles_info:
            return 0

        # Calculate color cost: Number of needed colors not held by any robot
        colors_robots_have = set(robot_colors.values())
        colors_to_acquire = colors_needed_by_unpainted - colors_robots_have
        h += len(colors_to_acquire)

        # Calculate movement and paint cost for each unpainted tile
        for tile, color_needed in unpainted_goal_tiles_info:
            # Cost for paint action
            h += 1

            # Cost to clear tile if occupied
            if tile in occupied_tiles:
                h += 1

            # Cost to get a robot adjacent
            min_dist_to_adjacent = float('inf')
            adjacent_tiles = self.adjacency_map.get(tile, [])

            # Check if the goal tile is paintable (has adjacent tiles)
            # and is a known tile in the grid structure with valid coordinates.
            # If a tile is in adjacency_map but has no adjacent_tiles, it's isolated.
            if tile not in self.all_tiles or tile not in self.coords or (tile in self.adjacency_map and not adjacent_tiles):
                 # A goal tile that is not part of the known grid or has no neighbours
                 # cannot be painted by adjacent paint actions. Unsolvable.
                 return float('inf')

            # Calculate min distance from any robot to any adjacent tile
            if not robot_positions: # No robots to paint
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile of the target tile
            found_reachable_adjacent = False
            for robot, pos_r in robot_positions.items():
                # Ensure robot position is a known tile with coordinates
                if pos_r not in self.coords:
                     # Robot is on an unknown tile? Unsolvable.
                     return float('inf')

                for adj_t in adjacent_tiles:
                     # Ensure adjacent tile is a known tile with coordinates
                     if adj_t not in self.coords:
                          # Adjacent tile is unknown? Unsolvable.
                          return float('inf')

                     dist = self.manhattan_distance(pos_r, adj_t)
                     min_dist_to_adjacent = min(min_dist_to_adjacent, dist)
                     found_reachable_adjacent = True # At least one robot can reach at least one adjacent tile

            # If min_dist_to_adjacent is still inf, no robot can reach any adjacent tile.
            # This implies unsolvability (e.g., disconnected grid).
            if min_dist_to_adjacent == float('inf'):
                 return float('inf')

            h += min_dist_to_adjacent

        return h
