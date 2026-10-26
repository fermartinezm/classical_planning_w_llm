from heuristics.heuristic_base import Heuristic

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is treated as a string
    fact_str = str(fact)
    # Remove leading/trailing parentheses and split by spaces
    return fact_str[1:-1].split()

# Helper function to parse tile name into coordinates
def get_coords(tile_name):
    """Parses a tile name like 'tile_row_col' into a tuple (row, col)."""
    try:
        # Assuming tile names are in the format 'tile_row_col' where row and col are integers.
        # The specific indexing (0-based or 1-based) doesn't affect Manhattan distance calculation,
        # as long as it's consistent. We parse them as integers directly.
        parts = tile_name.split('_')
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # If a tile name is not in the expected format, it indicates an issue
        # with the problem instance definition. Raising an error is appropriate.
        raise ValueError(f"Unexpected tile name format: {tile_name}")


# Helper function for Manhattan distance
def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two coordinate tuples (row, col)."""
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the estimated cost for each unpainted goal tile,
    considering the minimum movement distance for robot1 to reach a paintable location
    for that tile and the paint action itself, and adds a total estimated cost for
    color changes robot1 needs to make.

    # Assumptions
    - The heuristic calculates the cost assuming 'robot1' is responsible for painting
      all unpainted goal tiles. If multiple robots exist, this heuristic sums the
      estimated cost for robot1 to complete all remaining tasks, which might be
      a significant overestimate but provides a relative measure of state difficulty.
    - Tiles are arranged in a grid, and names like 'tile_row_col' can be parsed
      into coordinates.
    - Goal tiles that are already painted with the wrong color indicate an unsolvable
      state (heuristic should be infinity).
    - All required colors are available (`available-color`).
    - The grid structure (adjacency via up, down, left, right) is static and defines
      from which tile a robot can paint an adjacent tile.

    # Heuristic Initialization
    - Store the goal conditions (`(painted ?x ?c)` facts).
    - Build a map `paint_from_locations` where keys are tile names that can be painted,
      and values are sets of tile names where a robot must be located to paint the key tile.
      This map is constructed by parsing the static `up`, `down`, `left`, `right` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h = 0`.
    2. Identify all goal facts of the form `(painted ?tile ?color)`. These are stored
       in `self.goal_tiles_info` during initialization.
    3. Create a list `unpainted_goal_tiles_info` to store tuples `(tile_name, required_color)`
       for goal tiles that are not yet painted correctly.
    4. Iterate through the goal tiles and their required colors from `self.goal_tiles_info`:
        a. For each goal tile `T` and required color `C`, check if the fact `(painted T C)`
           is already true in the current state. If yes, this goal is satisfied for this tile;
           continue to the next goal tile.
        b. If the goal is not satisfied, check if the tile `T` is painted with a *different*
           color `C'` in the current state (`(painted T C')` where `C' != C`). This indicates
           an unsolvable state. If detected, return `float('inf')`.
        c. If the tile `T` is not painted correctly and not painted wrong, it needs painting.
           Add `(T, C)` to `unpainted_goal_tiles_info`.
    5. If `unpainted_goal_tiles_info` is empty, return `h = 0`. The goal is reached.
    6. Find the location `L_R` and color `C_R` of `robot1` from the current state. Iterate
       through state facts to find `(robot-at robot1 ?loc)` and `(robot-has robot1 ?color)`.
       If `robot1` is not found or its state is incomplete, return `float('inf')`.
    7. Parse `L_R` into coordinates `robot_coords` using `get_coords`.
    8. Identify the set of distinct colors `S_needed` required by the tiles in `unpainted_goal_tiles_info`.
    9. Calculate the total estimated color change cost for robot1:
       `color_cost = 0`
       If there are colors needed (`len(S_needed) > 0`):
           If robot1's current color `C_R` is one of the needed colors:
               `color_cost = len(S_needed) - 1` # Needs to change color for the other |S_needed|-1 colors
           Else (`C_R` is not a needed color):
               `color_cost = len(S_needed)` # Needs to change color to acquire the first needed color, and then for the remaining |S_needed|-1 colors
       Add `color_cost` to `h`.
    10. For each `(tile_name, required_color)` tuple in `unpainted_goal_tiles_info`:
        a. Add 1 to `h` for the paint action itself.
        b. Find the set of possible robot locations `PaintLocs` from which `tile_name` can be painted,
           using the `paint_from_locations` map built during initialization.
        c. Calculate the minimum Manhattan distance from `robot_coords` to any coordinate in `PaintLocs`.
           `min_dist = float('inf')`
           If `tile_name` is in `self.paint_from_locations`:
               paint_locations = self.paint_from_locations[tile_name]
               For each `paint_loc` in `paint_locations`:
                   Parse `paint_loc` into coordinates `paint_loc_coords`.
                   Calculate Manhattan distance `dist = manhattan_distance(robot_coords, paint_loc_coords)`.
                   Update `min_dist = min(min_dist, dist)`.
           If `min_dist` is still `float('inf')` (e.g., tile cannot be painted from anywhere, or paint_locations was empty), return `float('inf')`.
        d. Add `min_dist` to `h` as the estimated movement cost to get robot1 to a position where it can paint this tile.
    11. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the paint_from_locations map from static facts.
        """
        self.goals = task.goals  # Store goal conditions

        # Build the map: painted_tile -> set of robot_location tiles
        self.paint_from_locations = {}
        for fact in task.static:
            parts = get_parts(fact)
            # Check for adjacency facts that define paintable relations
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                painted_tile = parts[1] # The tile being painted (e.g., tile_1_1 in (up tile_1_1 tile_0_1))
                robot_location = parts[2] # The tile the robot must be AT (e.g., tile_0_1 in (up tile_1_1 tile_0_1))
                if painted_tile not in self.paint_from_locations:
                    self.paint_from_locations[painted_tile] = set()
                self.paint_from_locations[painted_tile].add(robot_location)

        # Store goal tiles and their required colors for quick access
        self.goal_tiles_info = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles_info[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        h = 0
        unpainted_goal_tiles_info = []
        needed_colors = set()

        # Convert state frozenset to a set of strings for efficient lookup
        state_facts_str = {str(fact) for fact in state}

        # Find robot1's current location and color
        current_robot_location = None
        current_robot_color = None
        robot_found = False
        # Iterate through state facts to find robot1's state
        for fact_str in state_facts_str:
            parts = get_parts(fact_str)
            if parts[0] == 'robot-at' and len(parts) == 3 and parts[1] == 'robot1':
                current_robot_location = parts[2]
                robot_found = True
            elif parts[0] == 'robot-has' and len(parts) == 3 and parts[1] == 'robot1':
                 current_robot_color = parts[2]

        # Check if robot1's state is fully known
        if not robot_found or current_robot_location is None or current_robot_color is None:
             # Robot1 state is incomplete or robot1 doesn't exist (unexpected in valid problems)
             # Indicate an unsolvable or invalid state
             return float('inf')

        # Get robot1's coordinates
        try:
            robot_coords = get_coords(current_robot_location)
        except ValueError:
            # Robot location name is not in expected format
            return float('inf') # Indicate problem

        # Identify unpainted goal tiles and check for unsolvable states
        for goal_tile, goal_color in self.goal_tiles_info.items():
            goal_fact_str = f'(painted {goal_tile} {goal_color})'

            # Check if goal is already satisfied for this tile
            if goal_fact_str in state_facts_str:
                continue

            # Check for unsolvable state (painted wrong color)
            is_painted_wrong = False
            # Iterate through state facts related to this tile's painted status
            for fact_str in state_facts_str:
                 parts = get_parts(fact_str)
                 if parts[0] == 'painted' and len(parts) == 3 and parts[1] == goal_tile:
                     # Tile is painted, but not with the goal color (checked above)
                     is_painted_wrong = True
                     break # Found painted state for this tile

            if is_painted_wrong:
                 # Goal tile is painted with the wrong color - unsolvable state
                 return float('inf')

            # If not painted correctly and not painted wrong, it needs painting
            unpainted_goal_tiles_info.append((goal_tile, goal_color))
            needed_colors.add(goal_color)


        # If no unpainted goal tiles, the goal is reached
        if not unpainted_goal_tiles_info:
            return 0

        # Calculate total estimated color change cost for robot1
        color_cost = 0
        if len(needed_colors) > 0:
            if current_robot_color in needed_colors:
                # Robot has one of the needed colors, needs to change for the others
                color_cost = len(needed_colors) - 1
            else:
                # Robot does not have any of the needed colors, needs to acquire one first
                # and then change for the others
                color_cost = len(needed_colors)
        h += color_cost

        # Calculate movement and paint cost for each unpainted tile
        for tile_name, required_color in unpainted_goal_tiles_info:
            # Add cost for the paint action itself (always 1)
            h += 1

            # Find the set of locations robot1 must be AT to paint this tile
            paint_locations = self.paint_from_locations.get(tile_name, set())

            if not paint_locations:
                 # This tile cannot be painted from any known location based on static facts.
                 # This indicates an issue with the domain/instance definition for a goal tile.
                 return float('inf')

            # Calculate the minimum Manhattan distance from robot1's current location
            # to any of the valid paint locations for this tile.
            min_dist = float('inf')
            for paint_loc in paint_locations:
                try:
                    paint_loc_coords = get_coords(paint_loc)
                    dist = manhattan_distance(robot_coords, paint_loc_coords)
                    min_dist = min(min_dist, dist)
                except ValueError:
                    # Should not happen if paint_loc names from static facts are valid tiles
                    return float('inf') # Indicate problem

            if min_dist == float('inf'):
                 # Could not find a reachable paint location (e.g., grid disconnected, or paint_locations was empty).
                 # Should not happen for valid goal tiles in connected grids.
                 return float('inf')

            h += min_dist # Add estimated movement cost

        return h
