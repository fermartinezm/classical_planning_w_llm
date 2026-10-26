from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # print(f"Warning: Malformed fact string: {fact_str}") # Optional: add logging
        return []
    return fact_str[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_R_C' into a tuple (R, C)."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format
            # print(f"Warning: Unexpected tile name format: {tile_name}") # Optional: add logging
            return None
    except (ValueError, IndexError):
        # Handle parsing errors
        # print(f"Warning: Error parsing tile name: {tile_name}") # Optional: add logging
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Handle invalid tile names - cannot calculate distance
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are not
    yet painted correctly. It sums the estimated costs for each unpainted
    goal tile. The cost for a tile depends on whether it needs to be cleared
    (if occupied by a robot) and the cost for a robot to reach an adjacent
    tile, acquire the correct color (if needed), and paint it.

    # Assumptions
    - The grid structure is implied by 'up', 'down', 'left', 'right' predicates
      and tile names like 'tile_R_C'.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - A robot must be on a tile adjacent to the target tile to paint it.
    - A robot must have the correct color to paint a tile.
    - A tile must be 'clear' to be painted or moved onto.
    - If a goal tile is painted with the wrong color, the instance is
      considered unsolvable (heuristic returns infinity).
    - If a goal tile is not clear and not painted correctly, it is assumed
      to be occupied by a robot and requires at least one move action to clear it.
    - The heuristic sums costs for each unpainted goal tile independently,
      which is non-admissible but aims to guide greedy search.

    # Heuristic Initialization
    - Extract the set of goal facts `(painted tile color)`.
    - Extract the set of available colors from static facts.
    - Build a map `paint_positions` where `paint_positions[target_tile]` is
      the set of tiles from which a robot can paint `target_tile`. This is
      derived from the 'up', 'down', 'left', 'right' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`.
    2. Initialize `total_cost = 0`.
    3. Identify robot locations and colors in the current state.
    4. For each goal fact `(painted tile color)`:
       a. Check if `(painted tile color)` is true in the current state. If yes, continue to the next goal fact (cost is 0 for this tile).
       b. If `(painted tile color)` is false:
          i. Check if `(clear tile)` is false.
             - If `(clear tile)` is false, check if the tile is painted with *any* color `C'` where `C' != color`. If yes, the instance is unsolvable, return `float('inf')`.
             - If `(clear tile)` is false and the tile is *not* painted with any color, the tile must be occupied by a robot. Add 1 to `total_cost` for this tile (representing the move action needed to clear it).
          ii. Calculate the cost to paint this tile (assuming it is or will become clear).
              - Find the set of tiles `Adj_Tiles` from which a robot can paint `tile` using the precomputed `paint_positions` map.
              - Calculate the minimum Manhattan distance from any robot's current location to any tile in `Adj_Tiles`. Let this be `min_dist_to_adj`. If no robots or no reachable paint positions, return `float('inf')`.
              - Determine the color cost: Check if any robot currently has `(robot-has R color)`. If yes, `color_cost = 0`. If no, check if `(available-color color)` is in the available colors set. If yes, `color_cost = 1` (assuming one robot will change color). If no, return `float('inf')` (color needed but not available/held).
              - Add `min_dist_to_adj + color_cost + 1` to `total_cost` (where +1 is for the paint action itself).
    5. Return `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes paint positions and available colors.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal painted tiles: {(tile, color)}
        self.goal_painted_tiles = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    self.goal_painted_tiles.add((parts[1], parts[2]))
                # else: Warning handled in get_parts if needed

        # Store available colors
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "available-color":
                 if len(parts) == 2:
                    self.available_colors.add(parts[1])
                 # else: Warning handled in get_parts if needed

        # Build map: target_tile -> {tiles_robot_can_paint_from}
        # A robot at tile X can paint tile Y if (up Y X), (down Y X), (left Y X), or (right Y X)
        # So, for a target tile Y, the paint positions are the tiles X such that Y is related to X
        self.paint_positions = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["up", "down", "left", "right"]:
                if len(parts) == 3:
                    target_tile = parts[1] # Y in (up Y X)
                    paint_from_tile = parts[2] # X in (up Y X)
                    if target_tile not in self.paint_positions:
                        self.paint_positions[target_tile] = set()
                    self.paint_positions[target_tile].add(paint_from_tile)
                # else: Warning handled in get_parts if needed


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # If no robots exist, the problem is unsolvable
        if not robot_locations:
             return float('inf')

        total_cost = 0

        # Iterate through all goal painted tiles
        for tile, color in self.goal_painted_tiles:
            painted_fact = f"(painted {tile} {color})"

            # If the tile is already painted correctly, no cost for this tile
            if painted_fact in state:
                continue

            # Tile needs painting
            clear_fact = f"(clear {tile})"

            # Check if painted with wrong color first
            is_painted_wrong = False
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "painted" and len(parts) == 3 and parts[1] == tile and parts[2] != color:
                    is_painted_wrong = True
                    break

            if is_painted_wrong:
                # Tile is painted with the wrong color - unsolvable
                return float('inf')

            # If not painted correctly and not painted wrong, it needs painting.
            # It might be clear or occupied.

            # If not clear, it must be occupied by a robot. Add cost to clear it.
            if clear_fact not in state:
                 # Add 1 for the move action needed to clear the tile.
                 total_cost += 1
                 # Now, calculate the cost to paint it *after* it's cleared.
                 # This cost includes travel, color change, and paint action.
                 # We add this cost regardless of whether it was initially clear
                 # or needed clearing, as the painting steps are always required
                 # once the tile is clear.

            # Calculate the cost to paint this tile (assuming it is or will become clear)

            # Find tiles from which this tile can be painted
            paint_from_tiles = self.paint_positions.get(tile, set())

            if not paint_from_tiles:
                # Cannot paint this tile
                # print(f"Warning: Goal tile {tile} has no defined paint positions.")
                return float('inf')

            # Find min distance from any robot to any paint position for this tile
            min_dist_to_adj = float('inf')
            for robot, r_loc in robot_locations.items():
                for adj_tile in paint_from_tiles:
                    dist = manhattan_distance(r_loc, adj_tile)
                    min_dist_to_adj = min(min_dist_to_adj, dist)

            if min_dist_to_adj == float('inf'):
                 # No reachable paint position for this tile
                 return float('inf')

            # Determine color cost
            has_color = any(c == color for r, c in robot_colors.items())
            color_cost = 0
            if not has_color:
                if color in self.available_colors:
                    color_cost = 1 # Need to change color
                else:
                    # Color needed is not available and no robot has it
                    # Unsolvable
                    # print(f"Warning: Needed color {color} for tile {tile} is not available and no robot has it.")
                    return float('inf')

            # Add travel cost, color cost (if needed), and paint action cost
            total_cost += min_dist_to_adj + color_cost + 1

        return total_cost
