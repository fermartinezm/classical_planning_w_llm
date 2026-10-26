# Required imports
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper functions
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name string 'tile_row_col' into a (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_0_1'
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Not in expected format
            return None
    except (ValueError, IndexError):
        # Error parsing integers or parts
        return None

def manhattan_distance(coord1, coord2):
    """Calculates the Manhattan distance between two grid coordinates (r1, c1) and (r2, c2)."""
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing up
        the estimated costs for each unpainted goal tile. The cost for a single
        unpainted goal tile is estimated as the sum of:
        1. Cost of the paint action (1).
        2. Cost of changing robot color if no robot currently holds the required color (1 per color needed).
        3. Cost of movement for a robot to reach a tile adjacent to the target tile. This is estimated
           as the minimum Manhattan distance from any robot's current location to the target tile, adjusted
           to represent reaching an *adjacent* tile.

    Assumptions:
        - Goal tiles are either clear or painted with the correct color in any reachable state.
          If a goal tile is found to be painted with a wrong color, or is not clear and not correctly painted,
          the state is considered unsolvable and assigned an infinite heuristic value.
        - The grid structure is implied by tile names in the format 'tile_row_col', where row and col
          can be parsed as integers representing grid coordinates.
        - Manhattan distance is a reasonable approximation for movement cost on the grid.
        - There is at least one robot in the problem.

    Heuristic Initialization:
        The constructor pre-calculates and stores:
        - A mapping from tile names (e.g., 'tile_0_1') to their grid coordinates (row, col) by parsing
          all object names that follow the 'tile_row_col' format found in the task definition (static, initial, goals).
        - The set of available colors from static facts.
        - A dictionary mapping goal tile names to their required paint colors from the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by verifying if all goal facts are present in the state. If yes, return 0.
        2. Initialize the heuristic value `h` to 0.
        3. Parse the current state to identify robot locations, robot held colors, and painted/clear tiles.
        4. Identify unpainted goal tiles and check for unsolvable states: Iterate through the stored goal tiles and their required colors. For each goal tile `T` that needs color `C`:
           - Check if `(painted T C)` is true in the current state. If yes, this goal is satisfied, continue to the next goal tile.
           - Check if `T` is painted with *any* color `C'` (i.e., `(painted T C')` is true for any `C'`). If yes, and it's not the required color `C`, the tile is painted incorrectly, which is assumed to make the goal unreachable in this domain. Return infinity.
           - Check if `(clear T)` is true in the current state. If `T` is neither painted correctly nor clear, it's in an unexpected state (e.g., occupied by a robot, or some other unhandled state). Assume this makes the goal unreachable. Return infinity.
           - If the tile `T` is clear and needs color `C`, add `(T, C)` to a list of unpainted goal tiles.
        5. Calculate cost components:
           - Add the number of unpainted goal tiles to `h`. This accounts for the `paint` action needed for each.
           - Determine the set of colors required by the unpainted goal tiles.
           - Determine the set of colors currently held by robots.
           - Add the number of required colors that are not currently held by any robot to `h`. This accounts
             for the `change_color` actions needed (assuming one change per color is sufficient if multiple robots need the same new color).
           - Calculate movement cost: For each unpainted goal tile `T`:
             - Retrieve its coordinates `(r_T, c_T)` using the pre-calculated map.
             - Initialize minimum moves to an adjacent tile for this goal tile to infinity.
             - For each robot `R`:
               - Retrieve its current coordinates `(r_R, c_R)`.
               - Calculate the Manhattan distance `dist` between `(r_R, c_R)` and `(r_T, c_T)`.
               - The minimum moves for robot `R` to reach a tile *adjacent* to `T` is 1 if `dist == 0` (robot is on the tile and must move off) or `dist - 1` if `dist > 0` (robot moves towards the tile until adjacent).
               - Update the minimum moves for tile `T` with the minimum found across all robots.
             - Add this minimum movement cost for tile `T` to a running total movement cost.
           - Add the total movement cost to `h`.
        6. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        # No call to super().__init__(task) based on example heuristics
        self.goals = task.goals
        self.tile_coords = {}
        self.available_colors = set()

        # Extract tile names and coordinates from all objects mentioned in the task
        # We assume objects of type 'tile' follow the 'tile_row_col' naming convention
        all_fact_parts = set()
        for fact in task.static | task.initial_state | task.goals:
             all_fact_parts.update(get_parts(fact))

        for part in all_fact_parts:
             coords = parse_tile_name(part)
             if coords is not None:
                 self.tile_coords[part] = coords

        # Extract available colors from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'available-color':
                if len(parts) == 2:
                    self.available_colors.add(parts[1])

        # Store goal tiles and their required colors
        self.goal_tiles_info = {} # {tile_name: color}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                # Goal fact is (painted tile_name color)
                if len(parts) == 3:
                    tile_name = parts[1]
                    color = parts[2]
                    self.goal_tiles_info[tile_name] = color
                # else: malformed goal fact, ignore or handle error

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        h = 0
        unpainted_goal_tiles = [] # List of (tile_name, color)
        current_robot_locations = {} # {robot_name: tile_name}
        current_robot_colors = {} # {robot_name: color}
        painted_tiles = {} # {tile_name: color}
        clear_tiles = set()

        # Parse current state to get relevant facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            if parts[0] == 'robot-at':
                if len(parts) == 3:
                    robot_name = parts[1]
                    tile_name = parts[2]
                    current_robot_locations[robot_name] = tile_name
            elif parts[0] == 'robot-has':
                 if len(parts) == 3:
                    robot_name = parts[1]
                    color = parts[2]
                    current_robot_colors[robot_name] = color
            elif parts[0] == 'painted':
                 if len(parts) == 3:
                    tile_name = parts[1]
                    color = parts[2]
                    painted_tiles[tile_name] = color
            elif parts[0] == 'clear':
                 if len(parts) == 2:
                    tile_name = parts[1]
                    clear_tiles.add(tile_name)


        # Identify unpainted goal tiles and check for unsolvable states
        for tile_name, required_color in self.goal_tiles_info.items():
            is_painted_correctly = False
            is_painted_wrongly = False

            if tile_name in painted_tiles:
                if painted_tiles[tile_name] == required_color:
                    is_painted_correctly = True
                else:
                    is_painted_wrongly = True

            if is_painted_correctly:
                # Goal satisfied for this tile
                pass
            elif is_painted_wrongly:
                # Painted with wrong color - unsolvable in this domain
                return float('inf') # Or a large number like 1e9
            else:
                # Tile is not painted. Check if it's clear.
                if tile_name not in clear_tiles:
                     # Tile is not painted, but also not clear.
                     # This state is likely unreachable or a dead end in valid problems.
                     # Assume unsolvable.
                     return float('inf') # Or a large number like 1e9

                # Tile is clear and needs painting
                unpainted_goal_tiles.append((tile_name, required_color))

        # If all goal tiles are painted correctly, h should be 0 (already checked at the start)

        # Heuristic Calculation
        # 1. Cost for paint actions
        h += len(unpainted_goal_tiles)

        # 2. Cost for change_color actions
        required_colors = {color for (tile, color) in unpainted_goal_tiles}
        colors_held = set(current_robot_colors.values())
        colors_to_change = required_colors - colors_held
        h += len(colors_to_change)

        # 3. Cost for movement
        movement_cost = 0
        if not current_robot_locations:
             # No robots? Cannot paint. Unsolvable.
             return float('inf') # Or a large number like 1e9

        for tile_name, required_color in unpainted_goal_tiles:
            # Ensure tile_name exists in pre-calculated coords (should if it's a goal tile)
            if tile_name not in self.tile_coords:
                 # This indicates a problem with parsing or problem definition
                 return float('inf') # Or a large number like 1e9 # Cannot calculate distance

            loc_T = self.tile_coords[tile_name]
            min_moves_to_robot_adj = float('inf')

            for robot_name, robot_tile in current_robot_locations.items():
                # Ensure robot_tile exists in pre-calculated coords
                if robot_tile not in self.tile_coords:
                     # This indicates a problem with parsing or state representation
                     return float('inf') # Or a large number like 1e9 # Cannot calculate distance

                loc_R = self.tile_coords[robot_tile]
                dist = manhattan_distance(loc_R, loc_T)

                # Minimum moves for robot R to reach a tile *adjacent* to T
                # If robot is at T (dist=0), it needs 1 move to get off to an adjacent tile.
                # If robot is adjacent to T (dist=1), it needs 0 moves.
                # If robot is further (dist > 1), it needs dist - 1 moves.
                moves_to_adjacent = 1 if dist == 0 else dist - 1

                min_moves_to_robot_adj = min(min_moves_to_robot_adj, moves_to_adjacent)

            if min_moves_to_robot_adj != float('inf'):
                 movement_cost += min_moves_to_robot_adj
            # else: This case implies no robots were found, handled above.


        h += movement_cost

        return h
