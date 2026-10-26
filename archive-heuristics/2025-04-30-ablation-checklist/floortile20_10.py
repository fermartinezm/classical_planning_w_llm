from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra parameters in future PDDL)
    # fnmatch handles the wildcard '*'
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse tile names like "tile_row_col"
def parse_tile_name(tile_name):
    """
    Parses a tile name string (e.g., "tile_2_5") into its row and column coordinates.
    Assumes the format is "tile_row_col".
    Returns a tuple (row, col) of integers.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe log a warning or raise an error
            # For this heuristic, we'll assume the format is consistent
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None # Or raise ValueError
    except (ValueError, IndexError):
        print(f"Warning: Could not parse tile name coordinates: {tile_name}")
        return None # Or raise ValueError

# Helper function to calculate Manhattan distance between two tiles
def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns the distance as an integer. Returns infinity if parsing fails.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        return math.inf # Cannot calculate distance

    r1, c1 = coords1
    r2, c2 = coords2

    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It considers the paint action itself, the movement
    needed to get a robot adjacent to the tile, and the color changes needed.

    # Assumptions:
    - Tiles are arranged in a grid and named "tile_row_col".
    - Adjacency predicates define movement on this grid.
    - A robot must be adjacent to a tile to paint it.
    - A robot must have the correct color to paint a tile.
    - A tile must be clear to be painted.
    - If a goal tile is painted with the wrong color, the state is considered unsolvable
      by this heuristic (returns a large value).

    # Heuristic Initialization
    - Stores the goal conditions, specifically which tiles need which color.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are not yet painted correctly.
    2. For any goal tile that is currently painted with the *wrong* color, the state
       is likely unsolvable with the given actions (no unpaint action). Return a
       very large heuristic value.
    3. For each goal tile that is currently `clear` (and needs painting):
       a. Add 1 to the heuristic for the paint action itself.
       b. Estimate the movement cost: Find the minimum Manhattan distance from any
          robot's current location to *any* tile adjacent to the target tile. Add
          this minimum distance to the heuristic. The distance to an adjacent tile
          from a tile at distance D is max(0, D-1). If a robot is at the tile (D=0),
          it needs 1 move to get off to an adjacent clear tile before painting.
          So, if min_dist_to_tile is D, moves needed is 1 if D=0, and D-1 if D>0.
       c. Estimate the color change cost: This is tricky with multiple robots.
          A simple approach is to count, for all unpainted goal tiles, the set of
          colors needed. Then, count the set of colors currently held by robots.
          For each color needed that no robot has, add 1 to the heuristic, assuming
          one `change_color` action can make that color available to *some* robot.
          This is a relaxation.
    4. Sum up the costs from steps 3a, 3b, and 3c for all unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal requirements: {tile_name: color_name}
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Define a large value to represent unsolvable states
        self.UNSOLVABLE_COST = 1000000

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Track robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, location = get_parts(fact)
                robot_locations[robot] = location
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        total_cost = 0  # Initialize heuristic cost.
        tiles_to_paint = [] # List of (tile_name, color_name) that need painting
        needed_colors = set() # Set of colors needed for tiles_to_paint

        # First pass: Identify tiles needing paint and check for unsolvability
        for tile, goal_color in self.goal_tiles.items():
            is_painted_correctly = False
            is_painted_wrongly = False
            is_clear = False

            # Check current state for the tile's status
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    is_painted_correctly = True
                    break # Found correct paint, no need to check other facts for this tile status
                elif match(fact, "painted", tile, "*"):
                    # Check if it's painted with *any* color (which implies not clear)
                    # And if that color is different from the goal color
                    _, painted_tile, current_color = get_parts(fact)
                    if painted_tile == tile and current_color != goal_color:
                         is_painted_wrongly = True
                         break # Found wrong paint, state is likely unsolvable
                elif match(fact, "clear", tile):
                    is_clear = True
                    # Don't break, might find painted fact later (though state should be consistent)

            if is_painted_wrongly:
                # If a goal tile is painted with the wrong color, it cannot be repainted
                # according to the domain rules (paint requires clear).
                return self.UNSOLVABLE_COST

            if not is_painted_correctly and is_clear:
                # This tile needs to be painted with the goal_color
                tiles_to_paint.append((tile, goal_color))
                needed_colors.add(goal_color)

        # If we found unsolvable state during the loop, return the large cost
        if total_cost >= self.UNSOLVABLE_COST:
             return total_cost

        # Calculate color acquisition cost
        available_colors = set(robot_colors.values())
        colors_to_acquire = needed_colors - available_colors
        total_cost += len(colors_to_acquire) # Add 1 for each color that no robot has

        # Calculate paint and movement cost for each tile needing paint
        for tile, goal_color in tiles_to_paint:
            # Cost for the paint action itself
            total_cost += 1

            # Estimate movement cost: minimum moves for any robot to get adjacent
            min_dist_to_adjacent = math.inf

            for robot, robot_location in robot_locations.items():
                dist_to_tile = manhattan_distance(robot_location, tile)
                if dist_to_tile == math.inf:
                    # Cannot reach this tile from this robot's location (e.g., parsing failed)
                    continue

                # Moves needed to get adjacent:
                # If robot is at the tile (dist=0), it needs 1 move to get off.
                # If robot is adjacent (dist=1), it needs 0 moves to get adjacent.
                # If robot is further (dist>1), it needs dist-1 moves to get adjacent.
                moves_to_adjacent = 1 if dist_to_tile == 0 else max(0, dist_to_tile - 1)

                min_dist_to_adjacent = min(min_dist_to_adjacent, moves_to_adjacent)

            # Add the minimum movement cost to the heuristic
            if min_dist_to_adjacent != math.inf:
                 total_cost += min_dist_to_adjacent
            else:
                 # If no robot can reach the tile, state is unsolvable
                 return self.UNSOLVABLE_COST


        return total_cost

