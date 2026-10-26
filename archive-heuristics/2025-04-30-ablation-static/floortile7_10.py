# Assume Heuristic base class is available in the specified path
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Simple split is sufficient for this domain's predicates.
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses tile name 'tile_row_col' into (row, col) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a valid tile_row_col format
    return None # Or raise error

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated minimum cost for each
    unsatisfied goal tile independently, considering the closest robot that
    can paint it.

    # Assumptions
    - Tiles are named in the format 'tile_row_col' allowing coordinate extraction.
    - The grid structure implied by 'up', 'down', 'left', 'right' predicates
      corresponds to these row/column coordinates.
    - Manhattan distance is used as a proxy for movement cost, ignoring potential
      blockages by other robots or painted tiles.
    - Goal tiles are initially 'clear' or correctly 'painted'. If a goal tile
      is found to be painted with the wrong color in any state, the heuristic
      returns a large value indicating a likely unsolvable state.
    - All colors required by goal tiles are available in the domain (either
      initially held by a robot or listed as available-color).

    # Heuristic Initialization
    - Extracts the goal conditions (target color for each goal tile).
    - Parses all tile names found in the problem definition to build a mapping
      from tile name to (row, col) coordinates based on the 'tile_row_col' format.
    - Extracts the set of available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Initialize the total heuristic cost `h` to 0.
    2. Identify the current location and held color for each robot in the state.
    3. Iterate through each goal fact `(painted T C_goal)` specified in the problem's goal:
        a. Check if the goal fact `(painted T C_goal)` is already true in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
        b. Check if the tile `T` is currently painted with a *different* color `C_wrong`. Iterate through state facts looking for `(painted T C_wrong)` where `C_wrong != C_goal`. If found, this state is likely unsolvable in the Floortile domain (as there's no unpaint/repaint action). Return a very large heuristic value (e.g., 1,000,000).
        c. If the tile `T` is not correctly painted and not painted wrongly, determine its state: Is it `clear` or is a robot `R` currently at `T`? (It must be one of these in a valid state if not painted).
        d. If a robot `R` is currently at tile `T` (i.e., `(robot-at R T)` is in the state):
            - This tile needs painting. The robot must first move *off* the tile (1 move action) to an adjacent tile, and then paint `T` from the adjacent tile (1 paint action).
            - If the robot `R` does not have the required color `C_goal`, it needs to perform a `change_color` action (1 action). This is only possible if `C_goal` is an available color. Check if the robot `R` can potentially paint with `C_goal` (has the color or color is available). If not, this state is likely unsolvable for this tile (robot is blocking and cannot paint). Return a large value.
            - If the robot can paint, the cost for this tile is 1 (move off) + (1 if robot `R` needs to change color) + 1 (paint action). Add this cost to `h`.
        e. If tile `T` is `clear` (i.e., `(clear T)` is in the state):
            - This tile needs painting. Find the minimum cost for *any* robot `R` to paint tile `T` with color `C_goal`.
            - Initialize `min_cost_for_tile` to infinity.
            - For each robot `R` at `Loc_R` with color `Color_R`:
                - Check if this robot can potentially paint with the goal color `C_goal` (has the color or color is available).
                - If the robot `R` can paint with `C_goal`:
                    - Get the coordinates for `Loc_R` and `T` using the pre-calculated `self.tile_coords` map.
                    - Calculate the Manhattan distance `dist` between `Loc_R` and `T`.
                    - Calculate the cost to get the correct color: `color_cost = 1` if `Color_R != C_goal`, otherwise 0.
                    - Calculate the moves and paint cost: The robot needs `dist - 1` moves to reach an adjacent tile (assuming `dist >= 1` since the tile is clear). Then 1 paint action. Total moves and paint cost is `(dist - 1) + 1 = dist`.
                    - The total cost for robot `R` to paint tile `T` is `color_cost + dist`.
                    - Update `min_cost_for_tile = min(min_cost_for_tile, cost_R)`.
            - After checking all robots, if `min_cost_for_tile` is still infinity, it means no robot can paint this tile (e.g., goal color not available and no robot has it). Return a large value (e.g., 1,000,000) indicating a likely unsolvable state.
            - Otherwise, add `min_cost_for_tile` to `h`.

    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and available colors.
        """
        self.goals = task.goals  # Goal conditions (frozenset of facts)
        static_facts = task.static # Static facts (frozenset of facts)
        initial_state = task.initial_state # Initial state facts

        # 1. Parse tile coordinates
        self.tile_coords = {}
        # Collect all unique tile names mentioned in initial state, goals, and static facts
        all_tile_names = set()
        for fact in initial_state | goals | static_facts:
             parts = get_parts(fact)
             # Check all arguments in the fact
             for arg in parts[1:]:
                 if arg.startswith('tile_'):
                     all_tile_names.add(arg)

        # Parse coordinates for each collected tile name
        for tile_name in all_tile_names:
             coords = parse_tile_name(tile_name)
             if coords:
                 self.tile_coords[tile_name] = coords
             # Note: Tiles that don't fit the pattern will not have coordinates.
             # This is okay if the problem only uses tile_row_col format for grid tiles.

        # 2. Store available colors
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Store goal tiles and their required colors for quick lookup
        self.goal_tiles_info = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles_info[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        h = 0  # Initialize heuristic value
        UNSOLVABLE_COST = 1000000 # Use a large value for unsolvable states

        # Get robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]

        # Iterate through each goal tile
        for g_tile, g_color in self.goal_tiles_info.items():
            # Check the current state of the goal tile
            tile_state = None # 'satisfied', 'wrong_color', 'clear', 'robot_on_it'
            robot_on_tile = None

            # Check for painted state first
            is_painted = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'painted' and parts[1] == g_tile:
                    is_painted = True
                    if parts[2] == g_color:
                        tile_state = 'satisfied'
                    else:
                        tile_state = 'wrong_color'
                    break # Found painted status

            if tile_state is None: # Not painted
                 if (f'clear {g_tile}') in state:
                     tile_state = 'clear'
                 else:
                     # If not clear and not painted, a robot must be on it in a valid state
                     for robot, loc in robot_locations.items():
                         if loc == g_tile:
                             tile_state = 'robot_on_it'
                             robot_on_tile = robot
                             break
                     # If still None, the state is inconsistent
                     if tile_state is None:
                          return UNSOLVABLE_COST # Inconsistent state

            if tile_state == 'satisfied':
                continue # This goal tile is done

            if tile_state == 'wrong_color':
                # If a goal tile is painted with the wrong color, it's likely unsolvable
                return UNSOLVABLE_COST # Return a large value

            if tile_state == 'robot_on_it':
                robot = robot_on_tile
                robot_color = robot_colors.get(robot)

                # Check if this robot can paint the tile with the goal color
                # It can if it has the color OR the color is available to change into.
                can_paint = (robot_color == g_color) or (g_color in self.available_colors)

                if not can_paint:
                    # Robot is blocking the tile and cannot paint it with the required color.
                    # This state is likely a dead end for this tile.
                    return UNSOLVABLE_COST # Unsolvable

                # Cost: Move off (1) + Change color (if needed, 1) + Paint (1)
                color_cost = 1 if robot_color != g_color else 0
                moves_and_paint_cost = 1 + 1 # Move off + Paint

                h += color_cost + moves_and_paint_cost

            elif tile_state == 'clear':
                min_cost_for_tile = float('inf')
                tile_coords = self.tile_coords.get(g_tile)

                if tile_coords is None:
                     # Should not happen if tile names are consistent and parsed correctly
                     return UNSOLVABLE_COST # Cannot find coordinates

                for robot, loc_r in robot_locations.items():
                    robot_color = robot_colors.get(robot)
                    loc_r_coords = self.tile_coords.get(loc_r)

                    if loc_r_coords is None:
                         continue # Cannot find coordinates

                    # Check if this robot can potentially paint with the goal color
                    # It can if it has the color OR the color is available to change into.
                    can_paint = (robot_color == g_color) or (g_color in self.available_colors)

                    if can_paint:
                        dist = abs(tile_coords[0] - loc_r_coords[0]) + abs(tile_coords[1] - loc_r_coords[1])

                        # Cost to change color if needed
                        color_cost = 1 if robot_color != g_color else 0

                        # Moves needed to get adjacent to the tile + Paint action:
                        # Robot needs dist-1 moves to get adjacent (since dist >= 1 for clear tile).
                        # Then 1 paint action. Total = (dist - 1) + 1 = dist.
                        moves_and_paint_cost = dist

                        cost_R = color_cost + moves_and_paint_cost
                        min_cost_for_tile = min(min_cost_for_tile, cost_R)

                if min_cost_for_tile == float('inf'):
                    # This happens if no robot can paint the tile (e.g., goal color not available and no robot has it)
                    return UNSOLVABLE_COST # Unsolvable

                h += min_cost_for_tile

        return h
