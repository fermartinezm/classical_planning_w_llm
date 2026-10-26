from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Use zip to handle cases where parts might be longer than args (with trailing wildcard)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes tile names follow this format.
    """
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_0_1' or 'tile_12_34'
        if len(parts) >= 3 and parts[0] == 'tile':
            row = int(parts[-2])
            col = int(parts[-1])
            return (row, col)
        else:
            # Handle unexpected formats gracefully, though problem implies standard naming
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except ValueError:
        print(f"Warning: Could not parse coordinates from tile name: {tile_name}")
        return None

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles based on their names.
    Returns infinity if coordinates cannot be parsed for either tile.
    """
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)

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
    with the correct color. It considers:
    1. Tiles painted with the wrong color (need clearing and repainting).
    2. Tiles that are clear but need painting.
    3. The colors required for the unpainted/wrongly-painted tiles.
    4. The movement cost for robots to reach the tiles.

    # Assumptions:
    - Tile names follow the format 'tile_row_col'.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates,
      allowing Manhattan distance to be a reasonable movement estimate.
    - Each robot starts with a color.
    - Clearing a wrongly painted tile requires a robot to move onto it and off it
      (abstracted as a fixed cost).
    - Painting a tile requires the robot to be adjacent, have the correct color,
      and the tile to be clear.

    # Heuristic Initialization
    - Extract the goal conditions, specifically the required color for each tile.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify all tiles that need to be painted according to the goal and their target colors.
    2. For each goal tile, check its current state:
       - Is it already painted with the correct color? (Cost = 0 for this tile).
       - Is it painted with the wrong color? (Requires clearing + repainting).
       - Is it clear? (Requires painting).
    3. Count the number of tiles that are wrongly painted. Each such tile needs to be cleared (abstracted cost 1) and then painted (cost 1). Add 2 to the heuristic for each.
    4. Count the number of tiles that are clear but need painting. Each such tile needs to be painted (cost 1). Add 1 to the heuristic for each.
    5. Identify the set of *distinct* colors required for all tiles that are currently unpainted or wrongly painted.
    6. Identify the set of colors currently held by the robots.
    7. For each color required that no robot currently holds, add 1 to the heuristic (cost of a color change action).
    8. For each tile that needs painting (either clear or wrongly painted), estimate the movement cost:
       - Find the minimum Manhattan distance from any robot's current location to this tile.
       - Add this minimum distance to the heuristic. (This is an overestimate for adjacency but serves as a simple proxy for movement effort).
    9. The total heuristic value is the sum of costs from steps 3, 4, 7, and 8.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Store goal locations and required colors for each tile
        self.goal_painted_color = {}
        for goal in self.goals:
            # Goal facts are like '(painted tile_1_1 white)'
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_color[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        h = 0  # Initialize heuristic value

        # Track tiles that are not painted correctly and the colors needed
        tiles_needing_paint = set()
        wrongly_painted_tiles = set()
        colors_needed = set()

        # 1. Identify tiles that need painting and their status
        for tile, goal_color in self.goal_painted_color.items():
            # Check if the tile is already painted correctly
            if f"(painted {tile} {goal_color})" in state:
                continue # This goal is satisfied for this tile

            # This tile needs painting. Add it to the set.
            tiles_needing_paint.add(tile)
            colors_needed.add(goal_color)

            # Check if the tile is painted with the wrong color
            is_wrongly_painted = False
            for fact in state:
                # Look for any painted fact about this tile
                if match(fact, "painted", tile, "*"):
                    current_color = get_parts(fact)[2]
                    if current_color != goal_color:
                        wrongly_painted_tiles.add(tile)
                        is_wrongly_painted = True
                    # Found a painted fact, no need to check further for this tile's color status
                    break

            # If not painted at all, it must be clear (implicitly, as per domain)
            # We don't need to explicitly check for (clear tile) because if it's
            # not painted correctly and not wrongly painted, it must be clear
            # (or the problem is ill-formed, which we assume is not the case).

        # 2. Add cost for clearing wrongly painted tiles and painting
        # Each wrongly painted tile needs clearing (abstract cost 1) + painting (cost 1)
        h += len(wrongly_painted_tiles) * 2
        # Each clear tile needing paint needs painting (cost 1)
        h += (len(tiles_needing_paint) - len(wrongly_painted_tiles)) * 1

        # 3. Add cost for color changes
        robot_colors = set()
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                robot_colors.add(robot_color)

        # Colors needed that no robot currently has
        colors_to_acquire = colors_needed - robot_colors
        h += len(colors_to_acquire) # Cost to change color for each missing required color

        # 4. Add movement cost
        robot_locations = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_name, tile_name = get_parts(fact)[1], get_parts(fact)[2]
                robot_locations[robot_name] = tile_name

        # For each tile that needs painting, add the minimum distance from any robot
        for tile in tiles_needing_paint:
            min_dist_to_tile = float('inf')
            for robot_loc in robot_locations.values():
                dist = manhattan_distance(robot_loc, tile)
                min_dist_to_tile = min(min_dist_to_tile, dist)

            if min_dist_to_tile != float('inf'):
                 h += min_dist_to_tile
            # If min_dist_to_tile is inf, it means tile coordinates couldn't be parsed.
            # This shouldn't happen in valid instances but good to be aware.
            # If there are no robots, min_dist_to_tile will remain inf.
            # If there are tiles needing paint but no robots, the problem is unsolvable.
            # A large heuristic value is appropriate in that case, but adding inf
            # might cause issues. Let's assume solvable problems with robots.

        return h

