from heuristics.heuristic_base import Heuristic
# fnmatch is not strictly needed for this heuristic but is common in PDDL parsing helpers
# from fnmatch import fnmatch

# Define helper functions for PDDL fact parsing and grid calculations

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# match function is not strictly needed for this heuristic but can be kept
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # If parts are not integers, this tile doesn't fit the expected pattern
            # For this domain, we assume all relevant tiles follow tile_r_c
            # Raising an error is appropriate if the format is guaranteed.
            raise ValueError(f"Tile name '{tile_name}' does not have integer coordinates.")
    # If the name doesn't start with 'tile_' or doesn't have 3 parts
    raise ValueError(f"Unexpected tile name format: {tile_name}")


def dist(coords1, coords2):
    """Calculates Manhattan distance between two coordinate tuples (r, c)."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    by summing the minimum estimated cost for each unpainted goal tile independently.
    The cost for a single tile considers the movement cost for the closest robot
    to get adjacent to the tile, the cost for that robot to acquire the correct color,
    and the paint action itself.

    # Assumptions
    - Tiles needing painting are initially in a 'clear' state. If a tile is found
      to be painted with a color different from the goal, the state is considered
      unsolvable (heuristic returns infinity).
    - Tile names follow the format 'tile_r_c' where r and c are integers representing row and column.
    - Manhattan distance provides a reasonable estimate for movement cost on the grid,
      ignoring dynamic 'clear' constraints on intermediate tiles.
    - Robots always hold some color if the 'robot-has' predicate is present for them.
      If a robot is present but has no 'robot-has' predicate, it's assumed it needs
      to acquire a color (cost 1).

    # Heuristic Initialization
    - Stores the goal conditions to identify which tiles need to be painted and with which color.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted tile C)`.
    2. From the current state, determine which of these goal conditions are not yet satisfied.
       These are the 'unpainted goal tiles'. While doing this, check if any goal tile
       is painted with a *different* color than required by the goal; if so, return infinity.
    3. Extract the current location and held color for each robot from the state.
    4. Initialize the total heuristic value `h` to 0.
    5. For each unpainted goal tile `T` that needs color `C`:
       a. Determine the coordinates `(tr, tc)` of tile `T` by parsing its name using `get_coords`.
       b. Initialize a variable `min_cost_for_tile_T` to infinity.
       c. For each robot `R` found in the state:
          i. Get the robot's current location `R_loc_name` and held color `R_color`.
          ii. Determine the coordinates `(rr, rc)` of the robot's location using `get_coords`.
          iii. Calculate the cost for the robot to acquire the needed color `C`:
               - `cost_color = 1` if `R_color` is not `C` (or if `R_color` is None, implying no color held).
               - `cost_color = 0` if `R_color` is `C`.
          iv. Calculate the Manhattan distance `D` between the robot's coordinates `(rr, rc)` and the tile's coordinates `(tr, tc)` using `dist`.
          v. Calculate the estimated movement cost for the robot to reach a tile adjacent to `T` (where it can paint from):
             - If `D == 0` (robot is at `T`), the robot must move off `T` to an adjacent tile (1 move). Cost = 1.
             - If `D == 1` (robot is adjacent to `T`), the robot is already in position. Cost = 0.
             - If `D > 1`, the minimum moves to reach the closest adjacent tile is `D - 1`. Cost = `D - 1`.
             - This can be expressed as `cost_movement = (D == 0) * 1 + (D > 1) * (D - 1)`.
          vi. Calculate the total estimated cost for robot `R` to paint tile `T`:
              `cost_R_for_T = cost_movement + cost_color + 1` (the +1 is for the paint action itself).
          vii. Update `min_cost_for_tile_T = min(min_cost_for_tile_T, cost_R_for_T)`.
       d. Add `min_cost_for_tile_T` to the total heuristic value `h`.
    6. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # The set of facts that must hold in goal states.
        # We are interested in (painted tile color) goals.
        self.goals = task.goals
        # Static facts are not strictly needed for this heuristic's calculation
        # based on tile_r_c parsing and Manhattan distance.
        # task.static # Example of accessing static facts if needed

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Extract current state information
        robot_locations = {} # Map robot name to tile name
        robot_colors = {}    # Map robot name to color name
        painted_tiles = set() # Set of (tile_name, color_name) tuples that are painted

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue
            predicate = parts[0]
            if predicate == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == 'painted':
                tile, color = parts[1], parts[2]
                painted_tiles.add((tile, color))

        # Identify unpainted goal tiles and check for unsolvable states
        unpainted_goals = [] # List of (tile_name, needed_color) tuples
        goal_tiles_needing_paint = set() # Set of tile names that are goals for painting

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty or malformed goals
                continue
            predicate = parts[0]
            if predicate == 'painted':
                tile, needed_color = parts[1], parts[2]
                goal_tiles_needing_paint.add(tile)
                if (tile, needed_color) not in painted_tiles:
                    unpainted_goals.append((tile, needed_color))

        # Check if any goal tile is painted with the wrong color
        for p_tile, p_color in painted_tiles:
             if p_tile in goal_tiles_needing_paint:
                 # Find the needed color for this tile from goals
                 needed_color = None
                 for goal in self.goals:
                     parts = get_parts(goal)
                     if parts and parts[0] == 'painted' and parts[1] == p_tile:
                         needed_color = parts[2]
                         break
                 if needed_color is not None and p_color != needed_color:
                     # This state is likely a dead end in this domain (cannot unpaint/repaint)
                     # print(f"Debug: Tile {p_tile} painted with wrong color {p_color}, needs {needed_color}. Returning inf.")
                     return float('inf')


        # If all goals are painted, the heuristic is 0
        if not unpainted_goals:
            return 0

        total_heuristic_cost = 0

        # Calculate cost for each unpainted goal tile
        for tile, needed_color in unpainted_goals:
            try:
                tile_coords = get_coords(tile)
            except ValueError:
                 # Handle unexpected tile names gracefully, maybe return inf or a large number
                 # indicating a potentially problematic state/instance.
                 # For this problem, we assume valid tile names.
                 # print(f"Warning: Could not parse coordinates for tile '{tile}'. Returning infinity.")
                 return float('inf')


            min_cost_for_tile = float('inf')

            # Find the minimum cost among all robots to paint this tile
            if not robot_locations:
                 # No robots available to paint
                 # print("Debug: No robots found in state. Returning inf.")
                 return float('inf')

            for robot in robot_locations:
                robot_loc_name = robot_locations[robot]
                # Ensure robot_loc_name is a tile name that can be parsed
                if not robot_loc_name.startswith('tile_'):
                     # Robot might be in a vehicle in other domains, but not in floortile.
                     # Assuming robot_at always links robot to a tile.
                     # print(f"Warning: Robot '{robot}' at unexpected location format '{robot_loc_name}'. Skipping robot.")
                     continue # Skip this robot if location format is unexpected

                try:
                    robot_coords = get_coords(robot_loc_name)
                except ValueError:
                     # print(f"Warning: Could not parse coordinates for robot location '{robot_loc_name}'. Skipping robot.")
                     continue # Skip this robot if location coordinates cannot be parsed

                robot_color = robot_colors.get(robot) # Get color, None if robot has no color (e.g., free-color)

                # Cost to acquire the correct color
                # If robot_color is None, it needs to acquire *a* color first, then change if needed.
                # Assuming 1 action to get any color, then 1 to change if wrong.
                # Simplified: If robot_color is None or wrong, cost is 1.
                cost_color = 1 if robot_color != needed_color else 0

                # Manhattan distance between robot and tile
                D = dist(robot_coords, tile_coords)

                # Estimated movement cost to get adjacent to the tile
                # 1 if D=0 (at tile), 0 if D=1 (adjacent), D-1 if D>1
                cost_movement = (D == 0) * 1 + (D > 1) * (D - 1)

                # Total estimated cost for this robot to paint this specific tile
                cost_R_for_T = cost_movement + cost_color + 1 # +1 for the paint action

                min_cost_for_tile = min(min_cost_for_tile, cost_R_for_T)

            # If min_cost_for_tile is still infinity, it means no valid robot could be found
            # or reach this tile (e.g., parsing failed for all robots).
            if min_cost_for_tile == float('inf'):
                 # print("Debug: No valid robot could reach tile. Returning inf.")
                 return float('inf')


            total_heuristic_cost += min_cost_for_tile

        return total_heuristic_cost
