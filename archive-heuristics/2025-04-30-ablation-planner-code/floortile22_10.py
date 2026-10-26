from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_R_C' into a (row, col) tuple of integers.
    Assumes tile names follow this format. Returns None if parsing fails.
    """
    if not isinstance(tile_name, str):
        return None
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None
    return None


def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns float('inf') if parsing fails for either tile.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted. It sums the minimum cost for any robot to paint
    each individual unpainted goal tile, considering movement, color change, and the paint action itself.

    # Assumptions
    - Goal conditions primarily require painting tiles that are initially clear. The heuristic focuses only on goal tiles that are currently clear and need painting.
    - Robots always hold a color (no 'free-color' state that requires picking up a first color).
    - Tiles are arranged in a grid, and movement/adjacency corresponds to Manhattan distance derived from tile names like 'tile_R_C'.
    - The cost of achieving each unpainted goal fact is independent, and the minimum cost
      among robots for each fact can be summed up. This ignores potential positive
      interactions (like one color change enabling multiple paints or one movement step
      getting closer to multiple target tiles) and negative interactions (like robots
      blocking each other or needing to move off a tile that another robot wants to paint).

    # Heuristic Initialization
    - Stores the goal conditions to identify which tiles need to be painted and with which color.
    - Static facts are not explicitly needed for this heuristic calculation, as tile adjacency
      is inferred from tile names for Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions that require a tile to be painted with a specific color. Store these as a mapping from tile name to required color.
    2. From the current state, extract the current location and held color for each robot.
    3. From the current state, identify which tiles are currently painted and with what color, and which tiles are clear.
    4. Determine the set of "unpainted clear goal tiles". These are the tiles specified in the goal that need painting, are not currently painted correctly, and are currently clear in the state.
    5. Initialize the total heuristic cost to 0.
    6. If there are unpainted clear goal tiles but no robots, the state is unsolvable, return infinity.
    7. For each `tile_T` and its required `color_C` in the set of unpainted clear goal tiles:
       a. Calculate the minimum cost for *any* robot to paint this specific tile:
          i. Initialize a minimum preparation cost for this tile to infinity.
          ii. For each robot `R` with location `Loc_R` and held color `Color_R`:
             - Calculate the Manhattan distance `dist` between `Loc_R` and `tile_T`.
             - If distance calculation fails (e.g., invalid tile name), treat cost as infinity for this robot.
             - Calculate the number of moves required for robot `R` to get adjacent to `tile_T`. This is `dist - 1` if `dist > 0` (already adjacent or further away), and 1 if `dist == 0` (robot is on the tile and must move off to paint an adjacent tile).
             - Calculate the color change cost for robot `R`: 1 if `Color_R` is different from `color_C`, 0 otherwise.
             - The preparation cost for robot `R` for this tile is `moves_to_get_adjacent + color_change_cost`.
             - Update the minimum preparation cost for `tile_T` with the minimum found across all robots.
          iii. If a finite minimum preparation cost was found (meaning at least one robot can reach the tile), add 1 (for the paint action) plus this minimum preparation cost to the total heuristic cost.
          iv. If no robot could reach this tile (minimum preparation cost remained infinity), the state is likely unsolvable, return infinity.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions are facts that must be true.

        # Pre-process goal conditions to easily find required painted tiles
        self.goal_painted_tiles = {} # {tile_name: color_name}
        # Assuming task.goals is a frozenset of individual fact strings
        for goal_fact_str in self.goals:
             parts = get_parts(goal_fact_str)
             if parts and parts[0] == 'painted' and len(parts) == 3:
                 tile, color = parts[1], parts[2]
                 self.goal_painted_tiles[tile] = color
             # We ignore other types of goal conditions if any exist, as the heuristic
             # focuses on the painting task.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of fact strings.

        # Check if goal is reached (heuristic is 0)
        # A state is a goal state if all goal facts are in the state.
        if self.goals.issubset(state):
             return 0

        # Extract robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == 'robot-has' and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot not in robot_info:
                    robot_info[robot] = {}
                 robot_info[robot]['color'] = color

        # Identify currently painted tiles and clear tiles
        current_painted_tiles = {} # {tile_name: color_name}
        current_clear_tiles = set() # {tile_name}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted_tiles[tile] = color
            elif parts[0] == 'clear' and len(parts) == 2:
                 tile = parts[1]
                 current_clear_tiles.add(tile)


        # Identify unpainted goal tiles that need painting and are clear
        unpainted_clear_goal_tiles = [] # List of (tile_name, color_name) tuples
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the goal tile is not painted with the correct color
            is_painted_correctly = (goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] == goal_color)

            if not is_painted_correctly:
                 # This tile needs to be painted with goal_color.
                 # For a solvable problem, it must be clear to be paintable.
                 if goal_tile in current_clear_tiles:
                    unpainted_clear_goal_tiles.append((goal_tile, goal_color))
                 # else:
                     # This tile is a goal tile needing painting but is not clear.
                     # It must be painted with the wrong color. This state is likely
                     # unsolvable or on a path to failure in this domain.
                     # We ignore it for the heuristic calculation, assuming solvable problems.


        total_heuristic_cost = 0

        # If there are unpainted clear goal tiles but no robots, the state is unsolvable.
        if unpainted_clear_goal_tiles and not robot_info:
             return float('inf')


        for tile_T, color_C in unpainted_clear_goal_tiles:
            min_cost_for_this_tile = float('inf')

            for robot_name, info in robot_info.items():
                robot_loc = info.get('location')
                robot_color = info.get('color')

                if robot_loc is None or robot_color is None:
                    # Robot info is incomplete, skip this robot for this tile
                    continue

                # Calculate movement cost to get adjacent
                dist = manhattan_distance(robot_loc, tile_T)

                if dist == float('inf'):
                    # Cannot calculate distance (e.g., invalid tile name),
                    # or tile is unreachable in a disconnected grid (unlikely in floortile).
                    # Treat as infinite cost for this robot for this tile.
                    continue

                # Moves needed to get adjacent to tile_T
                # If robot is on tile_T (dist=0), needs 1 move to get off, then is adjacent.
                # If robot is adjacent (dist=1), needs 0 moves.
                # If robot is further (dist > 1), needs dist - 1 moves.
                moves_to_adjacent = dist - 1 if dist > 0 else 1

                # Calculate color change cost
                color_change_cost = 1 if robot_color != color_C else 0

                # Total preparation cost for this robot for this tile
                prep_cost = moves_to_adjacent + color_change_cost

                min_cost_for_this_tile = min(min_cost_for_this_tile, prep_cost)

            # Add cost for this tile: 1 (paint action) + minimum preparation cost
            if min_cost_for_this_tile != float('inf'):
                 total_heuristic_cost += 1 + min_cost_for_this_tile
            else:
                 # This tile is a clear goal tile needing painting, but no robot
                 # could reach it (min_cost_for_this_tile remained inf). State is unsolvable.
                 return float('inf')


        return total_heuristic_cost
