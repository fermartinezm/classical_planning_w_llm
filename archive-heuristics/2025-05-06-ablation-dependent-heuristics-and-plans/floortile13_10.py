from fnmatch import fnmatch
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(predicate)' or '(predicate arg1 arg2)'
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts, though unlikely with a proper parser
        return fact.split()
    return fact[1:-1].split()

# Note: The match function from examples is useful but not strictly needed
# if we directly parse parts and check predicate names and argument counts.
# Keeping it for consistency with examples.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a coordinate tuple (row, col).
    Assumes the format is strictly 'tile_int_int'.
    Returns None if parsing fails.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            return None
    except ValueError:
        return None

def manhattan_distance(coord1, coord2):
    """Calculates the Manhattan distance between two coordinates (r1, c1) and (r2, c2)."""
    if coord1 is None or coord2 is None:
        # This case should ideally be handled before calling this function
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated
    minimum cost for each unpainted goal tile. The estimated cost for a single
    tile includes the cost to make the tile clear (if occupied) and the minimum
    cost for any robot to move to an adjacent tile, change color if needed,
    and perform the paint action.

    # Assumptions
    - Goal tiles that need painting are initially clear or occupied by a robot.
    - Tiles painted with the wrong color in the initial state cannot be repainted
      (as the paint action requires the tile to be clear). Such states are
      considered unsolvable (heuristic returns infinity).
    - Tile names follow the format 'tile_row_col' where row and col are integers,
      and these names correspond to grid coordinates used for Manhattan distance.
      Row index increases upwards, column index increases rightwards.
    - All colors required in the goal are available.
    - The cost of any action (move, paint, change_color) is 1.

    # Heuristic Initialization
    - Extracts the set of goal predicates related to painted tiles.
    - Extracts all tile objects mentioned in the task and parses their names
      to build a mapping from tile name to grid coordinates (row, col).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `H` to 0.
    2. Identify the set of goal tiles that are not currently painted with the
       correct color in the state.
    3. Get the current location and color of each robot.
    4. Identify which tiles are currently occupied by robots.
    5. Identify which tiles are currently painted with any color.
    6. For each goal tile `T` that needs to be painted with color `C_T` and is
       not yet correctly painted:
       a. Check if `T` is currently painted with a color different from `C_T`.
          If yes, return `float('inf')` as the state is likely unsolvable.
       b. Calculate the cost component for this tile: `tile_cost = 0`.
       c. If `T` is currently occupied by a robot, add 1 to `tile_cost` (cost
          for the robot to move off).
       d. Calculate the minimum cost for any robot `R` to paint tile `T`, assuming
          `T` is clear and an adjacent tile is available.
          - Initialize `min_paint_cost_from_any_robot` to `float('inf')`.
          - Get coordinates for `T`. If coordinates are missing, return `float('inf')`.
          - For each robot `R`:
            - Get robot `R`'s current location `T_R` and color `C_R`.
            - Get coordinates for `T_R`. If coordinates are missing, skip this robot.
            - Calculate the Manhattan distance between `T_R` and `T`.
            - Moves cost for `R` to get adjacent to `T`: `max(0, distance - 1)`.
            - Color cost for `R`: 1 if `C_R` is not `C_T`, else 0.
            - Paint cost: 1.
            - Total estimated cost for `R` to paint `T` from its current state
              (assuming T is clear): Moves cost + Color cost + Paint cost.
            - Update `min_paint_cost_from_any_robot` with the minimum cost found
              across all robots.
       e. If `min_paint_cost_from_any_robot` is still `float('inf')` (e.g., no robots),
          return `float('inf')`.
       f. Add `min_paint_cost_from_any_robot` to `tile_cost`.
       g. Add `tile_cost` to the total heuristic `H`.
    7. Return the total heuristic cost `H`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract goal painted tiles: {(tile_name, color_name)}
        self.goal_painted_tiles = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                self.goal_painted_tiles.add((parts[1], parts[2]))

        # Extract tile objects and parse coordinates from all facts in the task
        self.tile_coords = {}
        # Collect all fact strings from initial state, goals, and static facts
        all_relevant_facts_strings = set(task.initial_state) | set(task.goals) | set(task.static)

        unique_tile_names = set()
        # Scan all fact strings for arguments that look like tile names
        for fact_str in all_relevant_facts_strings:
            parts = get_parts(fact_str)
            for part in parts:
                if part.startswith('tile_'):
                    unique_tile_names.add(part)

        # Try parsing coordinates for all found tile names
        for tile_name in unique_tile_names:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords

        # Note: If tile names don't follow the 'tile_r_c' format or coordinates
        # cannot be parsed, self.tile_coords will be incomplete. The heuristic
        # will return infinity if coordinates for a goal tile are missing.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        occupied_tiles = set() # {tile_name}
        current_painted_tiles_with_color = {} # {tile_name: color_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
                occupied_tiles.add(parts[2])
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == "painted" and len(parts) == 3:
                current_painted_tiles_with_color[parts[1]] = parts[2]


        # Identify unpainted goal tiles that are not correctly painted
        unpainted_goal_tiles_tasks = [] # List of (tile_name, goal_color)
        for goal_tile, goal_color in self.goal_painted_tiles:
            current_color = current_painted_tiles_with_color.get(goal_tile)
            if current_color != goal_color:
                # This tile needs painting (either clear or painted wrong)
                unpainted_goal_tiles_tasks.append((goal_tile, goal_color))

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goal_tiles_tasks:
            return 0

        total_heuristic = 0

        # Check if we have any robots to perform actions
        if not robot_locations:
             # If there are unpainted tiles but no robots, it's unsolvable
             return float('inf')

        # Process each unpainted goal tile task
        for tile, goal_color in unpainted_goal_tiles_tasks:
            # Check if the tile is painted with the wrong color
            # If it's in current_painted_tiles_with_color, its color is not goal_color
            # (because of how unpainted_goal_tiles_tasks was built).
            if tile in current_painted_tiles_with_color:
                 # Problem is unsolvable if a goal tile is painted with the wrong color
                 # and paint action requires clear tile.
                 return float('inf') # Return infinity

            # Calculate the cost component for this tile
            tile_cost = 0

            # Cost to make the tile clear (if occupied)
            is_occupied = tile in occupied_tiles
            if is_occupied:
                 # Cost to move the robot off the tile
                 tile_cost += 1

            # Minimum cost for a robot to paint T (assuming T is clear)
            min_paint_cost_from_any_robot = float('inf')

            tile_coord = self.tile_coords.get(tile)
            if tile_coord is None:
                 # If we can't find coordinates for a goal tile, we can't estimate.
                 return float('inf')

            for robot_name, robot_tile in robot_locations.items():
                robot_coord = self.tile_coords.get(robot_tile)
                robot_color = robot_colors.get(robot_name) # Should always have a color

                if robot_coord is None or robot_color is None:
                     # Should not happen if init processed correctly
                     continue

                # Estimated cost for this robot to paint this tile *once it's clear*
                # This involves moving to an adjacent tile, changing color, and painting.

                # Cost to move robot R to an adjacent tile of T
                # Manhattan distance from robot's tile to target tile T
                dist_to_tile = manhattan_distance(robot_coord, tile_coord)

                # Moves needed to get adjacent to T.
                # If dist is 0 (robot at T), it needs 1 move to get off, then another move to get adjacent (total 2 moves to get adjacent from T).
                # If dist is 1 (robot adjacent), it needs 0 moves to get adjacent.
                # If dist > 1, it needs dist - 1 moves to get adjacent.
                # A simple estimate: max(0, dist_to_tile - 1) moves to get adjacent.
                moves_cost = max(0, dist_to_tile - 1)

                # Cost to change color
                color_cost = 1 if robot_color != goal_color else 0

                # Paint action cost
                paint_cost = 1

                cost_for_this_robot = moves_cost + color_cost + paint_cost
                min_paint_cost_from_any_robot = min(min_paint_cost_from_any_robot, cost_for_this_robot)

            # If min_paint_cost_from_any_robot is still inf, it means no robot could be considered
            # (e.g., no robots found or coordinate parsing failed for all robots).
            if min_paint_cost_from_any_robot == float('inf'):
                 return float('inf')

            # Add the minimum cost to paint this specific tile (after it's clear)
            tile_cost += min_paint_cost_from_any_robot

            # Add the total cost for this tile to the overall heuristic
            total_heuristic += tile_cost

        return total_heuristic
