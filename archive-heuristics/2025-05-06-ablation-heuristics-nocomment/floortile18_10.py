from heuristics.heuristic_base import Heuristic
# Although fnmatch is not strictly needed for this specific heuristic's parsing,
# it was present in the example code structure, so we include it for consistency
# if needed for more complex pattern matching in other heuristics.
# from fnmatch import fnmatch

# Helper function to parse PDDL facts
def get_parts(fact):
    """Strips parentheses and splits a PDDL fact string into parts."""
    # Ensure fact is a string and starts/ends with parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return empty list for non-standard facts
    return []

# Although fnmatch is not used in the final heuristic logic,
# the example code included a match helper, so we could include one
# if needed for more complex predicate matching. For this heuristic,
# direct string comparison of the first part is sufficient.
# def match(fact, *args):
#     """Checks if a fact matches a pattern using fnmatch."""
#     parts = get_parts(fact)
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the estimated costs associated with each unpainted goal tile. The
        estimated cost for satisfying the goal for a single tile includes:
        1. A base cost for the paint action itself.
        2. A cost related to acquiring the necessary color if no robot currently has it.
        3. A cost related to moving a robot to a tile adjacent to the target tile.
        4. A cost related to moving a robot off the target tile if one is currently
           occupying it, as paint actions require the target tile to be clear.
        The heuristic returns a large penalty value if a goal tile is found
        to be painted with the wrong color, indicating an unsolvable state.

    Assumptions:
        - Tile names follow the format 'tile_row_col' where row and col are integers.
        - The PDDL predicates 'up', 'down', 'left', 'right' define a grid structure
          where (direction Y X) means X is adjacent to Y in the specified direction.
          Specifically, (up Y X) means X is directly below Y, (down Y X) means X is
          directly above Y, (right Y X) means X is directly left of Y, and (left Y X)
          means X is directly right of Y. This convention is consistent with
          tile names `tile_r_c` where `r` is the row index increasing downwards
          and `c` is the column index increasing rightwards.
        - Tiles painted with a color cannot be unpainted or repainted with a different color.
          If a goal tile is painted with the wrong color, the problem is unsolvable.
        - Robots can only move onto clear tiles.
        - Paint actions require the target tile to be clear and the robot to be
          at an adjacent tile with the correct color.
        - The problem instance is solvable unless a goal tile is painted with the wrong color.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic precomputes and stores static
        information from the task description for efficient lookup during search:
        - `goal_painted_tiles`: A dictionary mapping goal tile names to their required colors,
          extracted from the task's goal facts.
        - `tile_coords`: A dictionary mapping tile names (like 'tile_0_1') to their
          corresponding (row, col) integer coordinates, parsed directly from the tile names.
          This allows for efficient Manhattan distance calculations.
        - `robots`: A list of all robot names identified from the initial state or static facts.
        - `available_colors`: A set of all available colors in the domain.
        - `adj_tiles`: A dictionary mapping each tile name to a list of tile names
          that are adjacent to it, derived from the 'up', 'down', 'left', 'right'
          static facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (represented as a frozenset of fact strings):
        1. Initialize the heuristic value `h` to 0.
        2. Extract the current state information:
           - `current_painted_tiles`: A dictionary mapping tiles that are currently painted
             to their color, from facts like `(painted tile color)`.
           - `robot_locations`: A dictionary mapping each robot to its current tile location,
             from facts like `(robot-at robot tile)`.
           - `robot_colors`: A dictionary mapping each robot to the color it currently holds,
             from facts like `(robot-has robot color)`.
        3. Identify `unpainted_goals`: A list of `(tile, color)` pairs for all goal tiles
           that are not currently painted with the correct color in the state.
        4. Check for unsolvable states: Iterate through the goal tiles. If a goal tile
           is found in `current_painted_tiles` but its color is different from the
           required goal color, return a large penalty value (e.g., 1000000) as this
           state cannot lead to the goal.
        5. If `unpainted_goals` is empty, the state is a goal state, so return 0.
        6. Add the base cost for painting: For each `(goal_tile, goal_color)` in `unpainted_goals`,
           add 1 to `h`. This accounts for the `paint` action required for each tile.
        7. Calculate color acquisition cost:
           - Determine the set of `needed_colors` required by the tiles in `unpainted_goals`.
           - Determine the set of `held_colors` currently held by any robot.
           - The set of `colors_to_acquire` is `needed_colors - held_colors`.
           - Add the number of colors in `colors_to_acquire` to `h`. This estimates the
             minimum number of `change_color` actions needed across all robots to ensure
             all required colors are available.
        8. Calculate movement and clearing costs: For each `(goal_tile, goal_color)` in `unpainted_goals`:
           - Find the minimum Manhattan distance from *any* robot's current location
             to *any* tile adjacent to `goal_tile`. Let this be `min_dist_to_adjacent`.
             Iterate through all robots and all adjacent tiles of the `goal_tile`,
             calculating the Manhattan distance between the robot's location and the
             adjacent tile, and keeping track of the minimum found distance.
           - Add `min_dist_to_adjacent` to `h`. This estimates the movement cost for
             the closest robot to get into a position from which it can paint the tile.
           - Check if any robot is currently located *on* the `goal_tile`. If a robot
             is on the tile, it must move off before the tile can be painted (as paint
             requires the target tile to be clear). Add 1 to `h` if a robot is on the tile.
        9. Return the final calculated value of `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all robots

        # 1. Parse goal painted tiles
        self.goal_painted_tiles = {} # {tile: color}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

        # 2. Identify all tiles and parse their coordinates
        self.tile_coords = {} # {tile_name: (row, col)}
        tile_names = set()
        # Scan static, initial state, and goal facts for tile names
        for fact in static_facts | initial_state:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     tile_names.add(part)
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith("tile_"):
                     tile_names.add(part)

        for tile_name in tile_names:
            try:
                # Assuming format tile_row_col
                _, row_str, col_str = tile_name.split('_')
                self.tile_coords[tile_name] = (int(row_str), int(col_str))
            except ValueError:
                # This should not happen with standard floortile instances
                # print(f"Warning: Unexpected tile name format: {tile_name}")
                pass # Ignore tiles with unexpected names

        # 3. Identify all robots
        self.robots = set()
        # Scan static and initial state facts for robot names
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            if len(parts) > 1 and (parts[0] == 'robot-at' or parts[0] == 'robot-has'):
                self.robots.add(parts[1])
        self.robots = list(self.robots) # Convert to list for consistent iteration

        # 4. Identify available colors
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # 5. Build adjacency list
        self.adj_tiles = {} # {tile: [adjacent_tiles]}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # Predicate is (direction tile1 tile2) meaning tile1 is [direction] of tile2
                # This implies tile1 and tile2 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                self.adj_tiles.setdefault(tile1, []).append(tile2)
                self.adj_tiles.setdefault(tile2, []).append(tile1)

    def __call__(self, node):
        state = node.state

        # 2. Identify current painted tiles
        current_painted_tiles = {} # {tile: color}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted_tiles[tile] = color

        # 3. Identify current robot locations and colors
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif len(parts) == 3 and parts[0] == 'robot-has':
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color

        # 4. Create list of unpainted goal tiles and check for unsolvable states
        unpainted_goals = [] # [(tile, color)]
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            if goal_tile in current_painted_tiles:
                if current_painted_tiles[goal_tile] != goal_color:
                    # Goal tile is painted with the wrong color - unsolvable
                    return 1000000 # Large penalty
            else:
                # Tile is not painted with the goal color
                unpainted_goals.append((goal_tile, goal_color))

        # 5. If the state is the goal state, return 0
        if not unpainted_goals:
            return 0

        # 1. Initialize heuristic value
        h = 0

        # 6. Add cost for paint actions (1 per unpainted goal tile)
        h += len(unpainted_goals)

        # 7. Calculate color acquisition cost
        needed_colors = {color for (tile, color) in unpainted_goals}
        held_colors = {robot_colors.get(robot) for robot in self.robots if robot_colors.get(robot) is not None}
        colors_to_acquire = needed_colors - held_colors
        h += len(colors_to_acquire) # Add 1 for each needed color not currently held by any robot

        # 8. Calculate movement and clearing costs
        for goal_tile, goal_color in unpainted_goals:
            goal_tile_coord = self.tile_coords.get(goal_tile)
            if goal_tile_coord is None:
                 # This tile was in the goal but not found in static/initial facts?
                 # Should not happen in valid problems, but handle defensively.
                 continue

            # Cost for movement to an adjacent tile
            min_dist_to_adjacent = float('inf')
            adjacent_tiles = self.adj_tiles.get(goal_tile, [])

            # If a goal tile has no adjacent tiles, it's likely unsolvable
            # unless the robot starts on it and can paint from there (not allowed by domain)
            # We rely on the wrong-color check or search failure for unsolvability.
            # For heuristic, if no adjacent tiles, min_dist_to_adjacent remains inf,
            # which will not add to h unless we handle it explicitly. Let's assume
            # valid problems have connected grids.

            for robot in self.robots:
                robot_loc = robot_locations.get(robot)
                if robot_loc is None:
                    # Robot location unknown? Should not happen in valid state.
                    continue
                robot_coord = self.tile_coords.get(robot_loc)
                if robot_coord is None:
                    # Robot location tile not found in tile_coords? Should not happen.
                    continue

                dist_to_adjacent = float('inf')
                for adj_tile in adjacent_tiles:
                     adj_coord = self.tile_coords.get(adj_tile)
                     if adj_coord is None:
                         # Adjacent tile not found in tile_coords? Should not happen.
                         continue
                     # Manhattan distance
                     dist = abs(robot_coord[0] - adj_coord[0]) + abs(robot_coord[1] - adj_coord[1])
                     dist_to_adjacent = min(dist_to_adjacent, dist)

                min_dist_to_adjacent = min(min_dist_to_adjacent, dist_to_adjacent)

            # Add minimum Manhattan distance to an adjacent tile
            # Only add if a path was found (min_dist_to_adjacent is not inf)
            if min_dist_to_adjacent != float('inf'):
                h += min_dist_to_adjacent
            # else: The goal tile might be unreachable from any robot position.
            # The heuristic doesn't explicitly handle this as unsolvable,
            # relying on search to fail or the wrong-color check.

            # Cost for clearing the tile if a robot is on it
            if any(robot_locations.get(r) == goal_tile for r in self.robots):
                 h += 1 # Cost to move off the tile

        # 9. Return the final value of h
        return h
