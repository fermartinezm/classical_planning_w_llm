import re

def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and arguments.
    e.g., '(robot-at robot1 tile_0_1)' -> ('robot-at', ['robot1', 'tile_0_1'])
    """
    # Remove leading/trailing parentheses and split by spaces
    # Handle potential inner parentheses if predicates had complex arguments,
    # but based on the domain, simple split is fine after stripping outer parens.
    parts = fact_string.strip("()").split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing the costs for each
        unsatisfied goal tile. The cost for a tile includes:
        1. A paint action.
        2. An extra move action if a robot is currently occupying the goal tile
           (as the tile must be clear to be painted).
        3. The minimum Manhattan distance for any robot to reach a tile
           adjacent to the target tile.
        4. A cost of 1 if the required color for this tile is not currently
           held by any robot (counted once per color needed across all
           unsatisfied tiles).

    Assumptions:
        - The input state is reachable from the initial state and consistent
          with the domain rules (a tile is either clear, painted, or occupied).
        - If a goal tile is painted with the wrong color, the state is considered
          a dead end (heuristic returns infinity).
        - The grid structure is defined by 'up', 'down', 'left', 'right'
          predicates between 'tile_r_c' objects, allowing parsing of
          coordinates (r, c).
        - Adjacency is symmetric (if A is up from B, B is down from A, etc.).
          The static facts explicitly define all adjacencies.
        - Manhattan distance on the grid is a reasonable estimate for movement cost,
          ignoring potential blockages by non-clear tiles (other than the robot's
          current position).

    Heuristic Initialization:
        The constructor pre-processes the task definition:
        - Identifies all robots, tiles, and colors from initial state and goals.
        - Parses tile names ('tile_r_c') to extract grid coordinates (r, c) and
          builds mappings between names and coordinates.
        - Builds a neighbor map for tiles based on 'up', 'down', 'left', 'right'
          static facts.
        - Stores the required color for each goal tile.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine robot positions, robot held colors,
           and the presence of `(clear tile)` and `(painted tile color)` facts.
        2. Initialize heuristic value `h = 0`.
        3. Identify unsatisfied goal tiles: Iterate through the goal tiles stored
           during initialization. For each goal tile requiring color C:
           a. Check if the tile is painted correctly (`(painted tile C)` in state). If yes, continue.
           b. Check if the tile is painted with a wrong color (`(painted tile C')` in state for C' != C). If yes, return infinity.
           c. If the tile is not painted correctly, it needs painting. Determine its current status: 'clear' if `(clear tile)` is in state, 'occupied' if `(robot-at robot tile)` is in state for some robot. If neither, the state is inconsistent, return infinity.
           d. Add the tile, required color, and its status ('clear' or 'occupied') to a list of unsatisfied tiles. Add C to a set of needed colors.
        4. If there are no unsatisfied goal tiles, the state is a goal state,
           return `h = 0`.
        5. Calculate color acquisition cost: For each color in the set of needed
           colors, check if any robot currently holds that color. If not, add 1
           to a temporary color cost. This counts the minimum number of
           `change_color` actions needed to make all required colors available
           to at least one robot. Add this temporary cost to `h`.
        6. Calculate movement and paint cost: For each unsatisfied tile:
           a. Add 1 to `h` for the paint action required for this tile.
           b. If the tile's status is 'occupied', add 1 to `h` for
              the move action required to free the tile so it can become clear.
           c. Calculate the minimum Manhattan distance from the current position
              of *any* robot to *any* tile adjacent to the unsatisfied tile.
           d. Add this minimum distance to `h`.
        7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.robots = set()
        self.tiles = set()
        self.colors = set()
        self.tile_coords = {}  # tile_name -> (row, col)
        self.coords_tile = {}  # (row, col) -> tile_name
        self.tile_neighbors = {} # tile_name -> set of adjacent tile_names
        self.goal_colors = {}  # tile_name -> required_color

        # Collect all object names and types by parsing facts from initial state, goals, and static
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_string in all_facts:
            pred, args = parse_fact(fact_string)
            if pred == 'robot-at':
                self.robots.add(args[0])
                if len(args) > 1: self.tiles.add(args[1])
            elif pred == 'robot-has':
                self.robots.add(args[0])
                if len(args) > 1: self.colors.add(args[1])
            elif pred == 'painted':
                if len(args) > 0: self.tiles.add(args[0])
                if len(args) > 1: self.colors.add(args[1])
            elif pred == 'clear':
                if len(args) > 0: self.tiles.add(args[0])
            elif pred in {'up', 'down', 'left', 'right'}:
                if len(args) > 0: self.tiles.add(args[0])
                if len(args) > 1: self.tiles.add(args[1])
            elif pred == 'available-color':
                 if len(args) > 0: self.colors.add(args[0])

        # Parse tile coordinates and build neighbor map
        for tile_name in self.tiles:
            match = re.match(r'tile_(\d+)_(\d+)', tile_name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                self.tile_coords[tile_name] = (r, c)
                self.coords_tile[(r, c)] = tile_name
            self.tile_neighbors[tile_name] = set() # Initialize neighbor set

        # Build neighbor map from static facts
        for fact_string in task.static:
             pred, args = parse_fact(fact_string)
             if pred in {'up', 'down', 'left', 'right'}:
                 if len(args) == 2:
                     tile1, tile2 = args[0], args[1]
                     # Ensure tiles were parsed and exist in our tile set
                     if tile1 in self.tile_neighbors and tile2 in self.tile_neighbors:
                         self.tile_neighbors[tile1].add(tile2)
                         self.tile_neighbors[tile2].add(tile1) # Assuming symmetric adjacency

        # Parse goal colors
        for goal_fact in task.goals:
            pred, args = parse_fact(goal_fact)
            if pred == 'painted':
                if len(args) == 2:
                    tile, color = args[0], args[1]
                    self.goal_colors[tile] = color
                # Handle potential non-painted goal facts if domain was extended
                # For this domain, goals are only painted facts.

    def __call__(self, state):
        # Convert state frozenset to set for easier checking
        state_set = set(state)

        # Parse robot positions and colors for quick lookup
        robot_positions = {} # robot_name -> tile_name
        robot_on_tile = {} # tile_name -> robot_name
        robot_colors = {} # robot_name -> color_name

        for fact_string in state_set:
            pred, args = parse_fact(fact_string)
            if pred == 'robot-at' and len(args) == 2:
                robot_positions[args[0]] = args[1]
                robot_on_tile[args[1]] = args[0]
            elif pred == 'robot-has' and len(args) == 2:
                robot_colors[args[0]] = args[1]

        # Identify unsatisfied goal tiles and needed colors
        unsatisfied_tiles_info = [] # List of (tile_name, required_color, status) # status is 'clear' or 'occupied'
        needed_colors_now = set()

        for tile, required_color in self.goal_colors.items():
            # Check painted status
            is_painted_correctly = f'(painted {tile} {required_color})' in state_set
            is_painted_wrong = False
            if not is_painted_correctly:
                 # Check if painted with any other color
                 for fact_string in state_set:
                     pred, args = parse_fact(fact_string)
                     if pred == 'painted' and len(args) == 2 and args[0] == tile and args[1] != required_color:
                         is_painted_wrong = True
                         break

            if is_painted_correctly:
                continue # Goal satisfied for this tile
            elif is_painted_wrong:
                return float('inf') # Painted with wrong color - unsolvable
            else: # Tile is not painted correctly (and not painted wrong)
                # It needs painting. Determine if it's clear or occupied.
                is_clear = f'(clear {tile})' in state_set
                is_occupied = tile in robot_on_tile

                if is_clear:
                    status_detail = 'clear'
                elif is_occupied:
                    status_detail = 'occupied'
                else:
                    # Tile is not painted, not clear, and not occupied. This state is inconsistent.
                    # Based on domain effects, a tile is either clear, painted, or occupied.
                    # If it's not painted correctly, and not clear, it must be occupied.
                    # If it's not occupied either, the state is invalid.
                    return float('inf')

                unsatisfied_tiles_info.append((tile, required_color, status_detail))
                needed_colors_now.add(required_color)

        # If no unsatisfied goal tiles, the state is a goal state
        if not unsatisfied_tiles_info:
            return 0

        # Calculate heuristic components
        h_paint = len(unsatisfied_tiles_info) # Each needs one paint action

        h_color_change = 0
        for color in needed_colors_now:
            # Check if any robot currently has this color
            has_color = False
            for robot_name in self.robots:
                if robot_colors.get(robot_name) == color:
                    has_color = True
                    break
            if not has_color:
                h_color_change += 1 # Need to acquire this color

        h_movement = 0
        for tile_X_Y, color_Z, status_detail in unsatisfied_tiles_info:
            # Add cost for moving robot off if it's currently on the tile
            if status_detail == 'occupied':
                 h_movement += 1 # Cost to move robot off the tile

            min_dist_to_adj = float('inf')
            tile_coords = self.tile_coords.get(tile_X_Y)

            if tile_coords is None:
                 # Should not happen for valid goal tiles parsed from task
                 return float('inf')

            adjacent_tiles = self.tile_neighbors.get(tile_X_Y, set())

            if not adjacent_tiles:
                 # Goal tile has no neighbors - cannot be painted. Unsolvable.
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            for robot_name, robot_tile in robot_positions.items():
                robot_coords = self.tile_coords.get(robot_tile)

                if robot_coords is None:
                    # Should not happen if robot is on a valid tile
                    continue # Skip robot if its position tile coordinates are unknown

                for adj_tile in adjacent_tiles:
                     adj_coords = self.tile_coords.get(adj_tile)

                     if adj_coords is None:
                         # Should not happen if adjacent tile is a valid tile
                         continue # Skip adjacent tile if its coordinates are unknown

                     # Manhattan distance between robot's current tile and adjacent tile
                     dist = abs(robot_coords[0] - adj_coords[0]) + abs(robot_coords[1] - adj_coords[1])
                     min_dist_to_adj = min(min_dist_to_adj, dist)

            # If min_dist_to_adj is still infinity, it means no robot can reach any adjacent tile.
            # This implies unsolvability in a connected grid.
            if min_dist_to_adj == float('inf'):
                 return float('inf')

            h_movement += min_dist_to_adj

        # Total heuristic is the sum of components
        h = h_paint + h_color_change + h_movement

        return h
