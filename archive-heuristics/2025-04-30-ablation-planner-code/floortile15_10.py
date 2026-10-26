# Ensure the base class is correctly imported. Assuming it's in a file named heuristic_base.py
# within a 'heuristics' directory relative to where this file will be placed.
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Simple split assuming standard PDDL fact format like '(predicate arg1 arg2)'
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # This might fail for nested structures, but PDDL facts in state/goal are usually flat.
    # Example state/static format confirms this simple structure.
    return fact[1:-1].split()

def parse_tile_name(tile_str):
    """Parses 'tile_r_c' into (r, c) integers."""
    try:
        parts = tile_str.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        return None # Return None for invalid tile names
    except (ValueError, IndexError):
        return None # Return None if parsing fails

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (r, c) coordinate tuples."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated minimum cost for each unpainted
    goal tile independently. The cost for a single tile considers the movement
    cost for the closest robot to reach a painting position, the cost for that
    robot to have the correct color, and the paint action itself.

    # Assumptions
    - The grid structure is regular and tile names follow the 'tile_r_c' format.
    - The 'clear' predicate is ignored for movement and painting costs (assumed
      paths/target tiles can be made clear).
    - If a goal tile is already painted with the wrong color, the problem is
      unsolvable (this heuristic implicitly assumes such states are not reached
      or handled by the search; it only counts tiles not painted correctly).
    - Robots always hold some color (no 'free-color' state needing a first color pick).
    - All colors required by the goal are available (predicate `available-color` holds).

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Determines the grid dimensions (max row and column) by parsing tile names
      from static facts and goal facts to check for valid adjacent tile positions
      during distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Goal Tiles:** Extract all goal conditions of the form `(painted tile_X color_Y)`
       to determine which tiles need to be painted and with what color. Store this mapping.
    2. **Determine Grid Dimensions:** Parse all tile names found in the static facts
       (connectivity predicates like `up`, `down`, `left`, `right`) and the goal
       conditions to find the maximum row and column indices. This defines the grid
       boundaries (0 to max_row, 0 to max_col).
    3. **Get Current State Information:** In the `__call__` method, parse the current
       state facts to find:
       - The current location (`robot-at`) and color held (`robot-has`) for each robot.
       - The current painted status (`painted`) for each tile.
    4. **Identify Unsatisfied Goal Tiles:** Compare the current painted status of tiles
       with the goal conditions. Create a list of goal tiles that are not currently
       painted with their required color.
    5. **Calculate Cost for Each Unsatisfied Tile:** For each tile `T` in the list of
       unsatisfied goals that needs color `C`:
       a. Parse the tile name `T` to get its grid coordinates `(r, c)`.
       b. Determine the coordinates of the four potential painting positions adjacent
          to `T`: `(r+1, c)`, `(r-1, c)`, `(r, c+1)`, `(r, c-1)`. Filter this list
          to include only those coordinates that fall within the determined grid boundaries
          (0 <= row < grid_rows, 0 <= col < grid_cols).
       c. Initialize a variable `min_cost_for_tile` to infinity.
       d. Iterate through each robot `R` with its current location `R_loc` and color `R_color`:
          i. Parse `R_loc` to get its grid coordinates `(r', c')`.
          ii. Calculate the minimum Manhattan distance from `(r', c')` to any of the
              valid painting position coordinates found in step 5b. This is the estimated
              movement cost for robot `R` to get into a position to paint tile `T`.
              If there are no valid paint positions, this remains infinity.
          iii. Calculate the color change cost for robot `R`: 1 if `R_color` is not `C`,
               otherwise 0.
          iv. The total estimated cost for robot `R` to paint tile `T` is the movement
              cost + the color change cost + 1 (for the paint action itself).
          v. Update `min_cost_for_tile` with the minimum of its current value and the
             cost calculated for robot `R`.
       e. Add `min_cost_for_tile` to the total heuristic value. If `min_cost_for_tile`
          remained infinity (meaning no robot can paint this tile), add a large penalty
          instead, indicating a likely unsolvable state or a state far from the goal.
    6. **Return Total Heuristic:** The final `total_heuristic` value is the sum of the
       minimum estimated costs for painting each unsatisfied goal tile. If there are
       no unsatisfied goals, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and grid dimensions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for each tile.
        # Mapping: tile_name -> required_color
        self.goal_tiles = {}
        # Collect all tile names mentioned in goals and static facts to determine grid size
        all_tile_names = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_tiles[tile] = color
                    all_tile_names.add(tile)

        # Determine grid dimensions (max row and column) from static facts and goal tiles.
        max_row = 0
        max_col = 0

        # Look for tile names in connectivity predicates in static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["up", "down", "left", "right"]:
                 # Connectivity facts are (pred tile_A tile_B)
                 if len(parts) == 3:
                     all_tile_names.add(parts[1])
                     all_tile_names.add(parts[2])

        # Parse coordinates from all collected tile names to find max row/col
        for tile_name in all_tile_names:
            coords = parse_tile_name(tile_name)
            if coords:
                r, c = coords
                max_row = max(max_row, r)
                max_col = max(max_col, c)

        # Store grid boundaries (inclusive)
        self.grid_rows = max_row + 1 # Rows are 0 to max_row
        self.grid_cols = max_col + 1 # Cols are 0 to max_col


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Get current robot locations and colors
        robot_info = {} # robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "robot-at":
                if len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['location'] = location
            elif parts[0] == "robot-has":
                 if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color

        # 2. Get currently painted tiles
        painted_tiles = {} # tile_name -> color_name
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted":
                 if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    painted_tiles[tile] = color

        # 3. Identify unsatisfied goal tiles
        unsatisfied_goals = {} # tile_name -> required_color
        for goal_tile, required_color in self.goal_tiles.items():
            current_color = painted_tiles.get(goal_tile)
            # A goal is unsatisfied if the tile is not painted with the required color.
            # This includes tiles that are not painted at all (current_color is None)
            # and potentially tiles painted with the wrong color (current_color != required_color).
            # Based on domain structure, painting wrong color seems terminal for that tile.
            # We only count tiles that need painting and are not currently painted correctly.
            if current_color != required_color:
                 unsatisfied_goals[goal_tile] = required_color


        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        total_heuristic = 0
        UNPAINTABLE_PENALTY = 1000 # Penalty for a goal tile that cannot be painted by any robot

        # 4. Calculate cost for each unsatisfied goal tile
        for goal_tile, required_color in unsatisfied_goals.items():
            goal_coords = parse_tile_name(goal_tile)
            if goal_coords is None:
                 # Should not happen in valid problems, but handle defensively
                 total_heuristic += UNPAINTABLE_PENALTY # Treat as unpaintable
                 continue

            min_cost_for_tile = float('inf')

            # Find possible paint positions (adjacent tiles) for this goal tile
            r, c = goal_coords
            potential_paint_positions_coords = []
            # Check UP position (robot must be DOWN)
            if r + 1 < self.grid_rows:
                 potential_paint_positions_coords.append((r + 1, c))
            # Check DOWN position (robot must be UP)
            if r - 1 >= 0:
                 potential_paint_positions_coords.append((r - 1, c))
            # Check LEFT position (robot must be RIGHT)
            if c + 1 < self.grid_cols:
                 potential_paint_positions_coords.append((r, c + 1))
            # Check RIGHT position (robot must be LEFT)
            if c - 1 >= 0:
                 potential_paint_positions_coords.append((r, c - 1))

            # If a tile has no adjacent tiles, it cannot be painted.
            if not potential_paint_positions_coords:
                 total_heuristic += UNPAINTABLE_PENALTY
                 continue

            # For each robot, calculate cost to paint this tile
            has_reachable_robot = False
            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot state is incomplete, skip
                    continue

                robot_coords = parse_tile_name(robot_location)
                if robot_coords is None:
                    continue

                # Calculate minimum movement cost for this robot to reach any paint position
                min_move_cost = float('inf')
                for paint_pos_coords in potential_paint_positions_coords:
                    dist = manhattan_distance(robot_coords, paint_pos_coords)
                    min_move_cost = min(min_move_cost, dist)

                # If min_move_cost is still inf, this robot cannot reach any paint position (e.g., disconnected grid)
                if min_move_cost == float('inf'):
                    continue # This robot cannot paint this tile

                has_reachable_robot = True # At least one robot can reach a paint position

                # Calculate color change cost
                color_cost = 0
                if robot_color != required_color:
                    color_cost = 1 # Assume change_color action costs 1

                # Total cost for this robot to paint this tile
                cost_this_robot = min_move_cost + color_cost + 1 # +1 for the paint action

                # Update minimum cost for this tile across all robots
                min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

            # Add the minimum cost for this tile to the total heuristic
            if min_cost_for_tile != float('inf'):
                 total_heuristic += min_cost_for_tile
            elif not has_reachable_robot:
                 # If min_cost_for_tile is still inf and there are robots, it means
                 # no robot can reach any paint position for this tile.
                 # If there are no robots at all, has_reachable_robot is false.
                 # In either case, if no robot can paint this tile, add penalty.
                 total_heuristic += UNPAINTABLE_PENALTY


        return total_heuristic
