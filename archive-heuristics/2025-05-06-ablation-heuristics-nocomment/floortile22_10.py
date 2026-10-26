# Need to import the base class
from heuristics.heuristic_base import Heuristic

# Helper functions (defined outside the class)
def get_parts(fact):
    """Helper to parse a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    # Handle potential empty strings after split
    return fact[1:-1].split() if fact and fact.startswith('(') and fact.endswith(')') else []

def parse_tile_name(tile_name):
    """Helper to parse 'tile_row_col' into (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a standard tile name format
    return None # Return None if parsing fails

def manhattan_distance(coords1, coords2):
    """Calculate Manhattan distance between two (row, col) tuples."""
    # Ensure inputs are valid tuples
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# The heuristic class
class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum costs
    required to paint each unsatisfied goal tile. For each goal tile
    that is not yet painted with the correct color, the heuristic calculates
    the minimum cost across all robots to paint that tile. This cost is
    estimated as the sum of:
    1. Cost to change the robot's color (1 if needed, 0 otherwise).
    2. Cost to move the robot to a tile adjacent to the goal tile
       (estimated using Manhattan distance to the nearest adjacent tile).
    3. Cost to perform the paint action (1).
    The heuristic returns infinity if any goal tile is painted with the
    wrong color, as this state is assumed to be a dead end.

    Assumptions:
    - Tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - The grid is connected based on 'up', 'down', 'left', 'right' predicates.
    - Robots always possess one color from the set of available colors.
    - Goal tiles are initially clear or correctly painted (no unpaint action).
    - The heuristic ignores the 'clear' predicate requirement for movement
      between tiles, using simple Manhattan distance.
    - The heuristic assumes that if a goal tile needs painting and is clear,
      there is at least one adjacent tile a robot *could* theoretically reach
      (i.i., the grid is not degenerate around goal tiles).

    Heuristic Initialization:
    The constructor processes the static information and goal state from the task.
    - It extracts the required color for each goal tile from the task's goals.
    - It identifies all tile objects mentioned in static adjacency facts or goals
      and maps their names ('tile_row_col') to (row, col) coordinates and vice-versa.
      This mapping is used for Manhattan distance calculations.
    - It extracts the set of available colors.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine the location and color of each robot,
       which tiles are painted (and with which color), and which tiles are clear.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each goal tile and its required color as defined in the task's goals.
    4. For the current goal tile `T` requiring color `C`:
       a. Check if `T` is already painted with color `C` in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. Check if `T` is painted with *any other* color `C'` in the current state. If yes, this state is considered a dead end (as there's no unpaint action); return `float('inf')`.
       c. If `T` is not painted correctly and not painted incorrectly (implying it must be clear, assuming valid states), this tile needs painting. Calculate the minimum cost to paint `T` across all robots:
          i. Initialize `min_robot_cost` to infinity.
          ii. Get the coordinates `(gr, gc)` of the goal tile `T` using the pre-calculated map. If the tile is not in the map, skip it or treat as unreachable.
          iii. Determine the coordinates of all potential adjacent tiles: `(gr+1, gc)`, `(gr-1, gc)`, `(gr, gc+1)`, `(gr, gc-1)`.
          iv. Filter these potential adjacent coordinates to keep only those that correspond to actual tile objects found in the pre-calculated map.
          v. For each robot `R` with current location `robot_tile` and color `robot_color`:
             - Get the coordinates `(rr, rc)` of `robot_tile` using the pre-calculated map. If the robot's tile is not in the map, skip this robot for this goal tile.
             - Calculate the `color_cost`: 1 if `robot_color` is not `C`, 0 otherwise.
             - Calculate the `move_cost`: Find the minimum Manhattan distance from `(rr, rc)` to the coordinates of any of the valid adjacent tiles found in step 4.c.iv. If there are no valid adjacent tiles or the robot's tile is invalid, the move_cost remains infinity for this robot/tile pair.
             - The `paint_cost` is 1.
             - The total cost for robot `R` to paint tile `T` is `color_cost + move_cost + paint_cost`.
             - Update `min_robot_cost = min(min_robot_cost, robot_cost)`.
          vi. Add `min_robot_cost` to the total heuristic value `h`. If `min_robot_cost` remained infinity (e.g., no robot could reach an adjacent tile), `h` will become infinity.
    5. After iterating through all goal tiles, return the accumulated heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        self.available_colors = {
            get_parts(fact)[1]
            for fact in self.static_facts
            if get_parts(fact) and get_parts(fact)[0] == "available-color" and len(get_parts(fact)) == 2
        }

        # Build tile coordinate mapping from all tiles mentioned in static adjacency facts or goals
        all_tile_names = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 all_tile_names.add(parts[1])
                 all_tile_names.add(parts[2])
        for tile in self.goal_paintings.keys():
             all_tile_names.add(tile)

        self.tile_coords = {}
        self.coords_to_tile = {}
        for tile_name in all_tile_names:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
                self.coords_to_tile[coords] = tile_name

        # Optional: Add tiles from initial state if not already captured
        # This requires access to task.initial_state, which is not standard for heuristic __init__
        # Relying on static facts and goals should cover all relevant tiles for the heuristic.

        # Check if we found any tiles that match the expected format
        if not self.tile_coords and (self.goal_paintings or any(get_parts(f) and get_parts(f)[0] in ["up", "down", "left", "right"] for f in self.static_facts)):
             # This warning might be too aggressive if the problem is just empty
             # but for typical floortile problems, tiles should be found.
             print("Warning: No tiles found with 'tile_row_col' format in static adjacency facts or goals.")


    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        current_paintings = {}
        robot_locations = {}
        robot_colors = {}

        # Using sets for faster lookup
        state_set = set(state)

        for fact in state_set:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    current_paintings[tile] = color
            elif predicate == "robot-at":
                 if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif predicate == "robot-has":
                 if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        # 2. Initialize heuristic
        h = 0

        # 3. Identify unpainted goal tiles and check for dead ends
        tiles_to_paint = []
        for goal_tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(goal_tile)

            if current_color == goal_color:
                # Already painted correctly
                continue
            elif current_color is not None and current_color != goal_color:
                # Painted incorrectly - dead end
                return float('inf')
            # If current_color is None, it's not painted. It needs painting.
            tiles_to_paint.append((goal_tile, goal_color))

        # 4. If all goal tiles are painted correctly, goal reached
        if not tiles_to_paint:
            return 0

        # 5. Calculate cost for each tile that needs painting
        for goal_tile, goal_color in tiles_to_paint:
            min_robot_cost = float('inf')

            goal_coords = self.tile_coords.get(goal_tile)
            if goal_coords is None:
                 # Goal tile name didn't match 'tile_r_c' format or wasn't in static/goals.
                 # Cannot calculate distance. Treat as unreachable.
                 return float('inf') # Problematic goal tile

            # Determine coordinates of potential adjacent tiles
            r, c = goal_coords
            potential_adj_coords = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]

            # Filter for valid adjacent tiles that exist in the grid map
            valid_adj_tiles = [
                self.coords_to_tile[coords]
                for coords in potential_adj_coords
                if coords in self.coords_to_tile
            ]

            if not valid_adj_tiles:
                 # Goal tile has no valid neighbors in the grid map.
                 # Cannot paint it from an adjacent tile. Unsolvable?
                 # This might happen for a 1x1 grid, but paint actions require adjacent tile.
                 # Treat as unreachable.
                 return float('inf') # Problematic grid structure around goal tile

            # Calculate min cost over all robots
            # Ensure we iterate over robots found in the current state
            for robot, robot_tile in robot_locations.items():
                robot_color = robot_colors.get(robot) # Get robot's current color

                # Cost to change color
                # Assuming robots always have a color, changing costs 1 if it's the wrong one.
                # If a robot somehow had no color (not possible by domain), this would need adjustment.
                color_cost = 0
                if robot_color != goal_color:
                    color_cost = 1

                # Cost to move to an adjacent tile
                move_cost = float('inf')
                robot_coords = self.tile_coords.get(robot_tile)

                if robot_coords is not None: # Check if robot's tile is in our map
                    min_dist_to_adj = float('inf')
                    for adj_tile in valid_adj_tiles:
                        adj_coords = self.tile_coords.get(adj_tile)
                        if adj_coords is not None: # Check if adjacent tile is in our map
                            dist = manhattan_distance(robot_coords, adj_coords)
                            min_dist_to_adj = min(min_dist_to_adj, dist)
                    move_cost = min_dist_to_adj # Will be inf if no valid_adj_tiles or robot_coords invalid

                # Cost to paint
                paint_cost = 1 # One paint action

                # Total cost for this robot to paint this tile
                if move_cost != float('inf'): # Only consider if robot can reach an adjacent tile
                    robot_cost = color_cost + move_cost + paint_cost
                    min_robot_cost = min(min_robot_cost, robot_cost)

            # Add the minimum cost for this tile to the total heuristic
            # If min_robot_cost is still inf, it means no robot can paint this tile
            # (e.g., no reachable adjacent tile exists). This indicates unsolvability.
            # Adding inf will make the total heuristic inf.
            h += min_robot_cost

        return h
