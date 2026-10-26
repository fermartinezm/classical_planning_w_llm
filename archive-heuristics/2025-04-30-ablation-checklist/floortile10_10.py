from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Utility functions (copied from example heuristics)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra parameters)
    # or args might contain trailing wildcards. fnmatch handles the wildcard logic.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific utility functions
def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_R_C' into a (row, column) tuple of integers.
    Assumes tile names follow the format 'tile_row_col'.
    """
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen with valid tile names

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        return float('inf') # Should not happen with valid tile names
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color. It considers the number of tiles
    that need painting, the movement cost for robots to reach those tiles,
    and the cost to acquire the necessary colors.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the format 'tile_R_C'.
    - The grid structure (up, down, left, right predicates) is consistent
      with Manhattan distance.
    - Robots can only paint tiles adjacent to them.
    - A tile must be clear to be painted or moved onto. The heuristic
      simplifies by ignoring the 'clear' precondition for adjacent tiles
      and assuming the target tile is clear if not painted incorrectly.
    - The cost of changing color is 1.
    - The cost of moving is 1 per step (Manhattan distance).
    - The cost of painting is 1.

    # Heuristic Initialization
    - Extracts all tile names from the task facts to validate adjacency.
    - Extracts the goal painting requirements (tile and color).

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify all goal tiles that are not currently painted with the correct color.
    2. For any goal tile that is currently painted with the *wrong* color, the state
       is likely a dead end in this domain (no unpaint action). Return a very large value (infinity).
    3. Count the number of tiles that are currently 'clear' but need to be painted
       according to the goal. Let this be `num_tiles_to_paint`.
    4. The heuristic starts with `num_tiles_to_paint` (representing the paint action for each).
    5. Identify the set of *colors* needed for these `clear` goal tiles.
    6. Identify the set of *colors* currently held by robots.
    7. For each needed color that no robot currently possesses, add 1 to the heuristic.
       This estimates the cost to acquire each necessary color at least once.
    8. For each tile that needs painting (is clear and a goal tile):
       a. Find all actual tiles adjacent to it based on the grid structure (potential
          neighbors derived from name parsing, filtered by existence in the problem).
       b. Calculate the minimum Manhattan distance from *any* robot's current location
          to *any* of these adjacent tiles.
       c. Add this minimum distance to the heuristic. This estimates the movement cost
          for *some* robot to get into position to paint this tile. This step
          overestimates movement if one robot paints multiple tiles, but is acceptable
          for a non-admissible heuristic.
    9. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals

        # Extract all tile names from task facts for quick lookup
        self.all_tile_names = set()
        for fact in task.facts:
            parts = get_parts(fact)
            # Check predicates that involve tiles
            if parts[0] in ['robot-at', 'clear', 'painted', 'up', 'down', 'left', 'right']:
                 for part in parts[1:]:
                     if part.startswith('tile_'):
                         self.all_tile_names.add(part)

        # Store goal locations and colors for tiles
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current state information
        current_robot_locations = {} # {robot_name: tile_name}
        current_robot_colors = {}    # {robot_name: color}
        current_paintings = {}       # {tile_name: color}
        current_clear_tiles = set()  # {tile_name}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                current_robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                current_robot_colors[robot] = color
            elif parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
            elif parts[0] == 'clear':
                tile = parts[1]
                current_clear_tiles.add(tile)

        total_cost = 0
        tiles_to_paint = [] # List of (tile, needed_color) tuples

        # Identify tiles that need painting according to the goal
        for goal_tile, needed_color in self.goal_paintings.items():
            if goal_tile in current_paintings:
                current_color = current_paintings[goal_tile]
                if current_color != needed_color:
                    # Tile is painted with the wrong color - likely a dead end
                    return float('inf')
                # Tile is painted correctly, no cost
            else:
                # Tile is not painted. It must be clear or occupied by a robot.
                # If occupied, a robot needs to move off, making it clear.
                # We simplify and assume it will become clear.
                # This tile needs to be painted.
                tiles_to_paint.append((goal_tile, needed_color))

        # If all goal tiles are painted correctly, the heuristic is 0
        if not tiles_to_paint:
            return 0

        # --- Heuristic Calculation ---

        # Base cost: 1 action for painting each tile that needs it
        total_cost += len(tiles_to_paint)

        # Color acquisition cost: 1 for each needed color that no robot currently has
        needed_colors_set = {color for _, color in tiles_to_paint}
        robot_colors_set = set(current_robot_colors.values())
        colors_to_acquire_at_least_once = needed_colors_set - robot_colors_set
        total_cost += len(colors_to_acquire_at_least_once)

        # Movement cost: Estimate movement for each tile needing paint
        for tile, needed_color in tiles_to_paint:
            tile_coords = parse_tile_name(tile)
            if tile_coords is None:
                 # Should not happen with valid tile names, but handle defensively
                 return float('inf')

            r, c = tile_coords
            # Potential adjacent tile coordinates
            potential_adj_coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

            # Convert potential coordinates to tile names and filter for actual tiles in the problem
            actual_adj_tiles = []
            for pr, pc in potential_adj_coords:
                potential_name = f'tile_{pr}_{pc}'
                if potential_name in self.all_tile_names:
                    actual_adj_tiles.append(potential_name)

            # If a tile needs painting but has no adjacent tiles in the problem, it's impossible
            if not actual_adj_tiles:
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            min_move_cost_for_this_tile = float('inf')

            if not current_robot_locations:
                 # No robots available to paint
                 return float('inf')

            for robot_loc in current_robot_locations.values():
                min_dist_from_robot = float('inf')
                for adj_tile in actual_adj_tiles:
                    dist = manhattan_distance(robot_loc, adj_tile)
                    min_dist_from_robot = min(min_dist_from_robot, dist)
                min_move_cost_for_this_tile = min(min_move_cost_for_this_tile, min_dist_from_robot)

            # Add the minimum movement cost required to get a robot near this tile
            total_cost += min_move_cost_for_this_tile

        return total_cost

