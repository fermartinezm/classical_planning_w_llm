import math

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal state by summing the minimum
    estimated cost for each tile that needs to be painted. The estimated
    cost for a single tile includes the cost for a robot to change color
    (if needed), move to the tile's location (using Manhattan distance as
    a relaxation), and perform the paint action. It assumes each tile
    can be painted independently by the most suitable robot.

    Assumptions:
    - Tile names are in the format 'tile_X_Y' where X and Y are integers
      representing row and column indices. Row index increases downwards,
      column index increases rightwards.
    - The grid structure implied by 'up', 'down', 'left', 'right' predicates
      corresponds to Manhattan distance.
    - Repainting a tile is not possible or required; tiles are either clear,
      painted correctly, or painted incorrectly (unsolvable state).
    - All robots are always located at some tile in any valid state.
    - All colors required by the goal are available.

    Heuristic Initialization:
    The constructor processes the task definition to extract static information:
    - Identifies all goal predicates of the form (painted tile color).
    - Identifies all objects (robots, tiles, colors) present in the initial
      state, static facts, and goal facts by parsing predicate arguments and
      inferring types based on predicate usage.
    - Builds a mapping from tile names ('tile_X_Y') to (row, col) coordinates
      and vice-versa, assuming the 'tile_X_Y' naming convention. Tiles not
      following this convention are ignored for coordinate mapping but still
      considered for painted status if they appear in the goal.
    - Stores the set of all robot names and available color names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine the location and color held by
       each robot, and the painted status or clearness of each tile.
    2. Identify the set of tiles that need painting. A tile needs painting
       if the goal requires it to be painted with a specific color, and
       in the current state, it is either clear or painted with a different color.
    3. If any tile needing painting is currently painted with the wrong color,
       the state is considered unsolvable, and the heuristic returns infinity.
    4. If there are no tiles needing painting, the current state satisfies
       all painting goals, so it is a goal state, and the heuristic returns 0.
    5. Initialize the total heuristic value `h` to 0.
    6. For each tile `t` that needs to be painted with color `c`:
       a. Get the coordinate `loc_t` for tile `t` using the pre-calculated map.
          If the tile name doesn't have coordinates (e.g., not 'tile_X_Y'),
          this tile cannot be painted using actions involving movement on the grid,
          making the state likely unsolvable if it's a goal tile. Return infinity.
       b. Initialize `min_robot_cost_for_tile` to infinity. This will store
          the minimum estimated cost for any single robot to paint tile `t`.
       c. Iterate through all robots identified during initialization:
          i. Get the robot's current location `loc_r` (tile name) from the
             parsed state information. If the robot's location is not found
             in the state (which shouldn't happen in valid states for this domain),
             skip this robot.
          ii. Get the coordinate `loc_r_coord` for the robot's location `loc_r`.
              If the location tile doesn't have coordinates, skip this robot.
          iii. Get the robot's current color `color_r` from the parsed state
               information. A robot might not have a color initially or in some
               intermediate states if the domain allowed dropping colors (not in this domain).
               Assume robot-has is always present if robot has a color.
          iv. Calculate the estimated color change cost: 1 if the robot's
              current color `color_r` is not the required color `c`, and 0 otherwise.
              (Assumes 1 action to change from any color to any available color).
          v. Calculate the estimated movement cost: The Manhattan distance
              between the robot's location coordinate `loc_r_coord` and the
              tile's location coordinate `loc_t`. This is a relaxation that
              ignores obstacles (`clear` tiles) and the requirement to be on
              an adjacent tile for painting.
          vi. Calculate the total estimated cost for this specific robot
              to paint tile `t`: `color_cost + move_cost + 1` (where 1 is the
              cost of the paint action itself).
          vii. Update `min_robot_cost_for_tile` with the minimum cost found
               so far across all robots for painting tile `t`.
       d. If `min_robot_cost_for_tile` is still infinity after checking all robots,
          it means no robot could be located or had valid coordinates to estimate
          the cost for this tile. This state is likely unsolvable. Return infinity.
       e. Add `min_robot_cost_for_tile` to the total heuristic value `h`.
    7. Return the accumulated total heuristic value `h`.
    """
    def __init__(self, task):
        self.goal_tiles = {}  # {tile_name: color_name}
        self.tile_coords = {} # {tile_name: (row, col)}
        self.coords_tile = {} # {(row, col): tile_name}
        self.all_robots = set()
        self.available_colors = set()

        # Collect all objects and identify types based on predicates
        temp_robots = set()
        temp_tiles = set()
        temp_colors = set()

        # Examine facts from initial state, static, and goal to identify objects and types
        for fact in task.initial_state | task.static | task.goals:
            parts = fact[1:-1].split()
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            args = parts[1:]

            # Infer types based on predicate structure from domain file
            if predicate == 'robot-at' and len(args) == 2:
                temp_robots.add(args[0])
                temp_tiles.add(args[1])
            elif predicate in ['up', 'down', 'right', 'left'] and len(args) == 2:
                temp_tiles.update(args)
            elif predicate == 'clear' and len(args) == 1:
                temp_tiles.add(args[0])
            elif predicate == 'painted' and len(args) == 2:
                temp_tiles.add(args[0])
                temp_colors.add(args[1])
            elif predicate == 'robot-has' and len(args) == 2:
                temp_robots.add(args[0])
                temp_colors.add(args[1])
            elif predicate == 'available-color' and len(args) == 1:
                temp_colors.add(args[0])

        self.all_robots = frozenset(temp_robots)
        self.available_colors = frozenset(temp_colors)

        # Parse tile coordinates for identified tiles
        for tile_name in temp_tiles:
             try:
                 # Assuming tile names are like 'tile_X_Y'
                 parts = tile_name.split('_')
                 if len(parts) == 3 and parts[0] == 'tile':
                     row, col = int(parts[1]), int(parts[2])
                     self.tile_coords[tile_name] = (row, col)
                     self.coords_tile[(row, col)] = tile_name
             except (ValueError, IndexError):
                 # Ignore tiles that don't fit the 'tile_X_Y' pattern or cannot be parsed
                 pass

        # Parse goal tiles
        for fact in task.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'painted' and len(parts) == 3:
                tile_name = parts[1]
                color_name = parts[2]
                self.goal_tiles[tile_name] = color_name

    def __call__(self, state):
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        tile_status = {} # {tile_name: 'clear' or color_name}

        # Parse current state
        for fact in state:
            parts = fact[1:-1].split()
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'robot-at' and len(args) == 2:
                robot_name, tile_name = args
                robot_locations[robot_name] = tile_name
            elif predicate == 'robot-has' and len(args) == 2:
                robot_name, color_name = args
                robot_colors[robot_name] = color_name
            elif predicate == 'painted' and len(args) == 2:
                tile_name, color_name = args
                tile_status[tile_name] = color_name
            elif predicate == 'clear' and len(args) == 1:
                tile_name = args[0]
                tile_status[tile_name] = 'clear'

        needs_painting = [] # List of (tile_name, goal_color)

        # Check unsatisfied goals and identify tiles needing painting
        for tile_name, goal_color in self.goal_tiles.items():
            current_status = tile_status.get(tile_name)

            if current_status == goal_color:
                # Goal satisfied for this tile
                pass
            elif current_status is not None and current_status != 'clear':
                # Tile is painted with the wrong color
                # This state is likely unsolvable in this domain as repainting is not possible
                return math.inf
            else: # current_status is 'clear' or None (assume clear if not mentioned as painted/clear)
                # Tile needs painting
                needs_painting.append((tile_name, goal_color))

        # If all goals are satisfied, needs_painting is empty
        if not needs_painting:
             return 0

        # If there are no robots but tiles need painting, it's unsolvable
        if not self.all_robots:
             return math.inf

        h = 0
        # Calculate heuristic for tiles needing painting
        for tile_name, goal_color in needs_painting:
            # Check if the required color is available
            if goal_color not in self.available_colors:
                 # Cannot get this color, unsolvable
                 return math.inf

            tile_coord = self.tile_coords.get(tile_name)
            if tile_coord is None:
                # Tile needing painting doesn't have coordinates (e.g., not tile_X_Y)
                # Cannot estimate movement cost. Treat as unsolvable.
                return math.inf

            min_robot_cost_for_tile = math.inf

            # Find the minimum cost for any robot to paint this tile
            for robot_name in self.all_robots:
                robot_loc_name = robot_locations.get(robot_name)
                if robot_loc_name is None:
                    # Robot location unknown in current state facts, skip this robot
                    continue

                robot_loc_coord = self.tile_coords.get(robot_loc_name)
                if robot_loc_coord is None:
                    # Robot is at a location without coordinates, skip this robot
                    continue

                robot_current_color = robot_colors.get(robot_name) # None if robot-has is not in state

                # Cost to get the right color
                # Assumes 1 action to change from any color to any available color
                color_cost = 1 if robot_current_color != goal_color else 0

                # Cost to move to the tile (using Manhattan distance)
                # This is a relaxation, ignoring clear tiles for path and adjacency requirement
                move_cost = abs(robot_loc_coord[0] - tile_coord[0]) + abs(robot_loc_coord[1] - tile_coord[1])

                # Total estimated cost for this robot for this tile:
                # color change + movement + paint action
                robot_cost_for_tile = color_cost + move_cost + 1

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost_for_tile)

            # If min_robot_cost_for_tile is still inf, it means no robot could be located
            # with valid coordinates to paint this tile.
            if min_robot_cost_for_tile == math.inf:
                 return math.inf

            h += min_robot_cost_for_tile

        return h
