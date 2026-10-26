from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Utility functions from example Logistics heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return [] # Return empty list for unexpected format
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific utility for parsing tile names
def parse_tile_name(tile_str):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    parts = tile_str.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # If parts are not integers, it's not a standard tile name
            return None
    # If format is not 'tile_x_y', it's not a tile we can parse grid coords for
    return None

# Domain-specific utility for Manhattan distance
def manhattan_distance(tile1_str, tile2_str):
    """Calculates Manhattan distance between two tiles based on parsed names."""
    p1 = parse_tile_name(tile1_str)
    p2 = parse_tile_name(tile2_str)

    # If either tile name cannot be parsed, we cannot calculate grid distance.
    # This implies the objects are not on the grid in the expected format.
    # For heuristic purposes, assume they are infinitely far apart for grid movement.
    if p1 is None or p2 is None:
        # Exception: If the strings are identical, distance is 0 even if not parsable grid tiles.
        # This handles cases like comparing 'robot1' to 'robot1', although our use case
        # is comparing tile names.
        if tile1_str == tile2_str:
             return 0
        return float('inf') # Cannot calculate grid distance

    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums three components:
    1. The number of tiles that still need to be painted according to the goal.
    2. The number of colors required by the unpainted goal tiles that are not
       currently held by any robot.
    3. An estimate of the movement cost, calculated as the sum, over all tiles
       that need painting, of the minimum Manhattan distance required for any
       robot to reach a tile adjacent to that unpainted tile.

    # Assumptions
    - The goal only specifies tiles that should be painted with a specific color.
    - Tiles that need painting according to the goal are currently 'clear'.
    - If a tile is painted with a color different from the goal color, the problem
      instance is considered unsolvable (as there's no unpaint action).
    - Tile names follow the format 'tile_r_c' allowing Manhattan distance calculation.
    - All movement actions (up, down, left, right) and color change actions have a cost of 1.
    - Paint actions (paint_up, paint_down, etc.) have a cost of 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which color.
    - Extracts static facts to identify available colors (although not strictly needed for this version)
      and potentially build an adjacency list (not used in this version which relies on Manhattan distance).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set `U` of `(tile, color)` pairs from the goal that are not
       currently satisfied in the state (i.e., the tile is not painted with the
       correct color).
    2. Check for unsolvable states: If any tile in `U` is currently painted with
       a *different* color, return infinity.
    3. Initialize heuristic value `h` to 0.
    4. Add the number of unpainted goal tiles (`len(U)`) to `h`. This accounts
       for the minimum number of paint actions required.
    5. Identify the set of colors `Needed_Colors` required by tiles in `U`.
    6. Identify the set of colors `Held_Colors` currently held by robots.
    7. Add the number of colors in `Needed_Colors` that are not in `Held_Colors`
       (`len(Needed_Colors - Held_Colors)`) to `h`. This accounts for the minimum
       number of color change actions required.
    8. Calculate the movement cost component:
       - Initialize `movement_cost` to 0.
       - For each `(tile_to_paint, required_color)` in `U`:
         - Find the minimum number of moves required for *any* robot to reach
           a tile adjacent to `tile_to_paint`. This minimum is calculated as
           `ManhattanDistance(robot_location, tile_to_paint) - 1`
           over all robots (since the distance will be at least 1 for a clear tile).
         - Add this minimum number of moves to `movement_cost`.
       - Add `movement_cost` to `h`.
    9. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal tiles and their required colors.
        self.goal_tiles = {} # {tile: color}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

        # Store available colors (not strictly needed for this heuristic calculation)
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}

        # Adjacency list is not strictly required for the Manhattan distance calculation
        # used in this heuristic, but can be built from static facts like 'up', 'down', etc.
        # for potential future use or verification.
        # self.adjacency_list = {} # {tile: {adjacent_tiles}}
        # for fact in static_facts:
        #     parts = get_parts(fact)
        #     if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
        #          _, tile1, tile2 = parts
        #          # Adjacency is symmetric
        #          self.adjacency_list.setdefault(tile1, set()).add(tile2)
        #          self.adjacency_list.setdefault(tile2, set()).add(tile1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract relevant information from the current state
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}
        painted_tiles = set() # {(tile, color)}
        # clear_tiles = set() # {tile} # Not explicitly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts if any
            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles.add((tile, color))
            # elif predicate == "clear" and len(parts) == 2:
            #     tile = parts[1]
            #     clear_tiles.add(tile)

        # Identify tiles that need painting
        # U = {(T, C) | (T, C) is a goal and (T, C) is not in state}
        U = set()
        for tile, color in self.goal_tiles.items():
            if (tile, color) not in painted_tiles:
                U.add((tile, color))

        # Check for unsolvable state (tile painted with wrong color)
        # If a tile T is in U (needs color C) but is currently painted with C' != C, it's unsolvable.
        for tile_to_paint, required_color in U:
             for painted_tile, painted_color in painted_tiles:
                 if painted_tile == tile_to_paint and painted_color != required_color:
                     # Tile is painted with the wrong color, and cannot be repainted (requires clear)
                     return float('inf') # Unsolvable

        # Heuristic calculation starts
        h = 0

        # 1. Cost for paint actions: Each tile in U needs one paint action.
        h += len(U)

        # If U is empty, all goal painted facts are satisfied.
        if not U:
            return h # Goal reached, heuristic is 0

        # 2. Cost for color changes: For each color needed by tiles in U, if no robot has it.
        needed_colors = {C for (T, C) in U}
        held_colors = set(robot_colors.values())
        h += len(needed_colors - held_colors)

        # 3. Cost for movement: Estimate moves needed to get robots adjacent to tiles in U.
        movement_cost = 0
        # For each tile that needs painting...
        for tile_to_paint, required_color in U:
            min_moves_to_adj_tile = float('inf')

            # Find the minimum moves for any robot to reach a tile adjacent to tile_to_paint.
            # A robot at Loc_R needs dist(Loc_R, Adj_T) moves to reach Adj_T adjacent to T.
            # The minimum dist(Loc_R, Adj_T) over all Adj_T adjacent to T is dist(Loc_R, T) - 1 (if dist >= 1).
            # Since tile_to_paint is in U, it's not painted correctly, and assumed clear.
            # If it's clear, no robot is currently AT tile_to_paint. So dist(Loc_R, tile_to_paint) >= 1.
            # Thus, moves needed for robot R to reach adjacent to tile_to_paint is dist(Loc_R, tile_to_paint) - 1.
            # We take the minimum over all robots.
            
            # Ensure there is at least one robot to calculate distance from
            if not robot_locations:
                 # If there are tiles to paint but no robots, it's unsolvable
                 return float('inf')

            for robot, robot_loc in robot_locations.items():
                dist_R_to_T = manhattan_distance(robot_loc, tile_to_paint)
                
                # If dist_R_to_T is inf (e.g., unparsable tile name), moves_needed will be inf.
                # If dist_R_to_T is 0, it means robot is at the tile. This shouldn't happen for a clear tile.
                # If it somehow happens, moves_needed = 0 - 1 = -1, which is wrong.
                # Let's stick to the logic: robot must move to an ADJACENT tile.
                # If dist(Loc_R, T) is 1, robot is already adjacent, moves_needed = 0.
                # If dist(Loc_R, T) is 2, robot needs 1 move to get adjacent.
                # If dist(Loc_R, T) is k, robot needs k-1 moves to get adjacent.
                # This is simply dist(Loc_R, T) - 1, assuming dist >= 1.
                # Manhattan distance is always >= 0. If dist is 0, robot is at T.
                # If robot is at T, T cannot be clear. If T is in U, it must be clear.
                # So dist(Loc_R, T) must be >= 1 for any robot R if T is in U.
                # Therefore, moves_needed = dist_R_to_T - 1 is correct here.

                # Handle the case where dist_R_to_T is inf (e.g., unparsable tile name)
                if dist_R_to_T == float('inf'):
                    moves_needed = float('inf')
                else:
                    moves_needed = dist_R_to_T - 1

                min_moves_to_adj_tile = min(min_moves_to_adj_tile, moves_needed)

            # Add the minimum movement cost for this specific tile to the total movement cost.
            # This assumes different robots can work on different tiles simultaneously.
            # If min_moves_to_adj_tile is still inf, it means no robot location was parsable,
            # or the tile_to_paint was not parsable, or all robots are infinitely far.
            # This indicates a problem, but adding inf to the heuristic correctly reflects
            # an effectively unreachable state (or a state far from goal).
            if min_moves_to_adj_tile == float('inf'):
                 return float('inf') # Cannot reach this tile

            movement_cost += min_moves_to_adj_tile


        h += movement_cost

        return h
