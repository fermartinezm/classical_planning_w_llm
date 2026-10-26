import math # For float('inf')
from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available via import or in the same scope
# from task import Task, Operator # Not needed for the heuristic itself, but good to know context

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated cost for each
    unsatisfied goal tile. For each tile that needs to be painted with a specific
    color, the heuristic estimates the minimum cost for any robot to paint that
    tile. This minimum cost is calculated as 1 (for the paint action) plus the
    cost for the robot to acquire the correct color (0 or 1) plus the Manhattan
    distance from the robot's current location to the closest tile adjacent to
    the target tile. This heuristic relaxes the constraint that movement is only
    possible through clear tiles, using Manhattan distance on the grid instead
    of shortest path on the dynamic clear-tile graph. It also simplifies the
    multi-robot coordination by summing minimum costs per tile across robots,
    rather than assigning specific robots to specific tiles.

    Assumptions:
    - The grid structure is defined by 'up', 'down', 'left', 'right' static predicates.
    - Tile names are in the format 'tile_row_col' where row and col are integers.
    - Robots always possess a color (as per domain definition and examples).
    - Tiles are either 'clear' or 'painted' with exactly one color.
    - Unpainting or repainting a tile is not possible. If a tile is painted
      with the wrong color according to the goal, the problem is unsolvable.

    Heuristic Initialization:
    - Parses all tile names from the task's facts (initial state, goals, static)
      and stores their grid coordinates in `self.tile_coords`.
    - Builds an adjacency list representation of the grid graph based on static
      'up', 'down', 'left', 'right' predicates in `self.grid_adj`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h` to 0.
    2. Parse the current state to identify:
       - Which tiles are painted and with which color (`painted_tiles`).
       - The location and color of each robot (`robot_info`).
    3. Identify unsatisfied goal tiles: Iterate through the goal facts. For each
       `(painted tile c)` goal:
       - Check if the tile is in `painted_tiles`.
       - If yes, check if the color matches the goal color. If not, the tile is
         painted incorrectly, the problem is unsolvable, return `float('inf')`.
       - If no (the tile is not in `painted_tiles`), it must be clear (by domain
         definition), and it needs to be painted. Add `(tile, c)` to a list of
         `unsatisfied_goals`.
    4. If the list of `unsatisfied_goals` is empty, all goals are satisfied, return 0.
    5. For each `(tile_t, color_c)` in `unsatisfied_goals`:
       a. Initialize `min_cost_for_tile` to `float('inf')`.
       b. Find all tiles adjacent to `tile_t` using the precomputed `self.grid_adj`.
       c. If `tile_t` has no adjacent tiles, it cannot be painted, return `float('inf')`.
       d. For each robot `robot_r` and its information (`info`) in `robot_info`:
          i. Get the robot's current location `r_loc` and color `r_color`.
          ii. Calculate the cost for the robot to acquire the correct color `color_c`:
              `color_cost = 0` if `r_color == color_c`, otherwise `1`.
          iii. Calculate the minimum Manhattan distance from `r_loc` to any tile
               in the list of tiles adjacent to `tile_t`. This is the estimated
               movement cost.
          iv. Calculate the total estimated cost for `robot_r` to paint `tile_t`:
              `cost_this_robot = 1 (paint action) + color_cost + min_manhattan_distance`.
          v. Update `min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)`.
       e. After checking all robots, if `min_cost_for_tile` is still `float('inf')`,
          it means no robot can reach an adjacent tile (this should ideally not
          happen on a connected grid unless there are no robots, which is handled).
          Return `float('inf')` as a safeguard.
       f. Add `min_cost_for_tile` to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Data structures for grid information
        self.tile_coords = {} # tile_name -> (row, col)
        self.grid_adj = {}    # tile_name -> [adjacent_tile_name, ...]

        # 1. Extract all tile names and parse coordinates
        all_tile_names = set()
        # Look for tile names in initial state, goals, and static facts
        # We need to parse facts carefully to extract arguments
        for fact_str in task.initial_state | task.goals | self.static:
             # Simple split by space and clean up parentheses
             # Handle potential issues with malformed facts defensively
             fact_str = fact_str.strip()
             if not fact_str.startswith('(') or not fact_str.endswith(')'):
                 continue # Skip malformed facts

             # Remove outer parentheses and split
             parts = fact_str[1:-1].split()
             if not parts: continue

             # Check arguments for tile names
             for part in parts[1:]: # Skip predicate name
                 # Clean up potential trailing parentheses if part is the last argument
                 part = part.rstrip(')')
                 if part.startswith('tile_'):
                     all_tile_names.add(part)

        for tile_name in all_tile_names:
            coords = self.parse_tile_coords(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
                self.grid_adj[tile_name] = [] # Initialize adjacency list

        # 2. Build grid adjacency list from static facts
        for fact_str in self.static:
            fact_str = fact_str.strip()
            if not fact_str.startswith('(') or not fact_str.endswith(')'):
                 continue # Skip malformed facts
            parts = fact_str[1:-1].split()

            if len(parts) == 3:
                pred, tile1, tile2 = parts
                if pred in ['up', 'down', 'left', 'right']:
                    # Add bidirectional edge if both tiles exist in our parsed list
                    if tile1 in self.grid_adj and tile2 in self.grid_adj:
                         self.grid_adj[tile1].append(tile2)
                         self.grid_adj[tile2].append(tile1) # Assuming connectivity is symmetric

        # Remove duplicates from adjacency lists (shouldn't create duplicates with set conversion)
        for tile in self.grid_adj:
            self.grid_adj[tile] = list(set(self.grid_adj[tile]))


    def parse_tile_coords(self, tile_name):
        """Parses tile name 'tile_row_col' into (row, col) tuple."""
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None
        return None

    def get_adjacent_tiles(self, tile_name):
        """Returns a list of tile names adjacent to the given tile."""
        return self.grid_adj.get(tile_name, [])

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles."""
        coord1 = self.tile_coords.get(tile1_name)
        coord2 = self.tile_coords.get(tile2_name)
        if coord1 is None or coord2 is None:
            # This indicates an issue if tile names from state/goals/static
            # were not found during initialization. Should not happen in valid problems.
            # Return infinity to indicate unreachable/problematic state.
            return float('inf')
        r1, c1 = coord1
        r2, c2 = coord2
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.

        Keyword arguments:
        node -- the current state node (node.state is the frozenset of facts)
        """
        state = node.state

        unsatisfied_goals = []
        painted_tiles = {} # tile -> color

        # Parse current state facts
        robot_info = {} # robot -> {'loc': tile, 'color': color}

        for fact_str in state:
            fact_str = fact_str.strip()
            if not fact_str.startswith('(') or not fact_str.endswith(')'):
                 continue # Skip malformed facts
            parts = fact_str[1:-1].split()
            if not parts: continue

            pred = parts[0]
            if pred == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            elif pred == 'robot-at' and len(parts) == 3:
                robot, loc = parts[1], parts[2]
                if robot not in robot_info: robot_info[robot] = {}
                robot_info[robot]['loc'] = loc
            elif pred == 'robot-has' and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot not in robot_info: robot_info[robot] = {}
                 robot_info[robot]['color'] = color
            # 'clear' facts are implicitly handled: if a tile is not in painted_tiles, it's clear.

        # Check goals and identify unsatisfied ones
        for goal_fact_str in self.goals:
            goal_fact_str = goal_fact_str.strip()
            if not goal_fact_str.startswith('(') or not goal_fact_str.endswith(')'):
                 continue # Skip malformed facts
            parts = goal_fact_str[1:-1].split()

            if not parts or parts[0] != 'painted' or len(parts) != 3:
                 # This should not happen for valid floortile goals, but handle defensively
                 continue

            goal_tile, goal_color = parts[1], parts[2]

            if goal_tile in painted_tiles:
                current_color = painted_tiles[goal_tile]
                if current_color != goal_color:
                    # Tile is painted with the wrong color - unsolvable
                    return float('inf')
                # Else: current_color == goal_color, goal satisfied for this tile
            else:
                # Tile is not in painted_tiles. By domain definition, it must be clear.
                # It needs painting.
                unsatisfied_goals.append((goal_tile, goal_color))

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        # Calculate heuristic sum for unsatisfied goals
        h = 0
        for tile_t, color_c in unsatisfied_goals:
            min_cost_for_tile = float('inf')
            target_adjacent_tiles = self.get_adjacent_tiles(tile_t)

            # If a tile has no adjacent tiles (e.g., 1x1 grid, or isolated tile), it's impossible to paint.
            if not target_adjacent_tiles:
                 return float('inf') # Unsolvable

            # If there are no robots but there are unsatisfied goals, it's unsolvable
            if not robot_info:
                 return float('inf')

            for robot_r, info in robot_info.items():
                r_loc = info.get('loc')
                r_color = info.get('color')

                # Ensure robot location and color are known (should be from initial state)
                if r_loc is None or r_color is None:
                     # This robot's state is incomplete, maybe ignore or handle error
                     continue # Skip this robot

                color_cost = 0 if r_color == color_c else 1

                # Calculate min Manhattan distance from r_loc to any target_adjacent_tile
                min_dist_to_adjacent = float('inf')
                for adj_t in target_adjacent_tiles:
                    dist = self.manhattan_distance(r_loc, adj_t)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                # min_dist_to_adjacent should be finite if the grid is connected and tiles exist
                if min_dist_to_adjacent != float('inf'):
                     # Cost for this robot to paint this tile:
                     # 1 (paint action) + color change cost + movement cost
                     cost_this_robot = 1 + color_cost + min_dist_to_adjacent
                     min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

            # If after checking all robots, no robot can paint this tile (shouldn't happen
            # if grid is connected and tiles exist, and there's at least one robot)
            # If min_cost_for_tile is still inf, it implies an issue or unsolvable part.
            # Given the problem structure, this case should ideally not be reached for solvable problems
            # unless target_adjacent_tiles was empty (handled) or no robots exist (handled).
            # If it happens, it's safer to return inf.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            h += min_cost_for_tile

        return h
