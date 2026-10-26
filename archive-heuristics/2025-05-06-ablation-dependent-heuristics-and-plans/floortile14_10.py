from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Remove parentheses and split
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """
    Parses a tile name string (e.g., 'tile_3_2') into integer coordinates (row, col).
    Assumes tile names follow the format 'tile_row_col'.
    """
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_row_col'
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe return None or raise error
            # For this domain, the format seems consistent.
            return None
    except (ValueError, IndexError):
        # Handle cases where conversion to int fails or parts are missing
        return None

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Assumes tile names can be parsed into coordinates using get_coords.
    """
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance if coordinates are invalid
        return float('inf') # Or some large value indicating unreachable

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing three components: the number of tiles that still need to be painted,
    the cost to acquire necessary colors, and the estimated movement cost for robots
    to reach the tiles. It is designed for greedy best-first search and is not admissible.

    # Assumptions
    - The tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - The grid structure implies Manhattan distance is a reasonable lower bound for movement cost between tiles.
    - Goal tiles that are not painted correctly in the current state are assumed to be 'clear',
      as there is no action to unpaint a tile, and the problem is assumed solvable.
    - Robots can change to any available color.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the required color for each goal tile.
    - Static facts defining the grid structure (up, down, left, right) are implicitly handled
      by the coordinate-based Manhattan distance calculation, assuming a standard grid layout
      corresponding to the tile naming convention. Available colors are not explicitly needed
      in the constructor but are relevant during the heuristic calculation based on the state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles and their required colors from the task's goal conditions.
    2. Identify which of these goal tiles are *not* currently painted with the correct color in the current state. These are the "unsatisfied goal tiles".
    3. Calculate the base cost: Add 1 to the heuristic for each unsatisfied goal tile. This represents the minimum one 'paint' action needed for each such tile.
    4. Calculate the color cost:
       - Determine the set of colors required by the unsatisfied goal tiles.
       - Determine the set of colors currently held by all robots in the state.
       - Count the number of colors in the required set that are *not* present in the set of colors currently held by any robot. Add this count to the heuristic. This estimates the cost to make each newly needed color available to at least one robot.
    5. Calculate the movement cost:
       - For each unsatisfied goal tile:
         - Find the minimum Manhattan distance from *any* robot's current location to this tile.
         - Add this minimum distance to the heuristic. This estimates the movement needed to get a robot close to the tile, summing this minimum over all unsatisfied tiles. This component can overestimate movement but provides a gradient towards goal tiles.
    6. The total heuristic value is the sum of the base cost, color cost, and movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and required colors for each tile.
        # Assuming goal facts are of the form (painted tile_name color_name)
        self.goal_tiles = {} # { tile_name: color_name }
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts).

        # 1. Identify unsatisfied goal tiles
        painted_tiles_state = {} # { tile_name: color_name }
        robot_locations = {} # { robot_name: tile_name }
        robot_colors = {} # { robot_name: color_name }

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                painted_tiles_state[tile] = color
            elif parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            # We don't strictly need 'clear' facts based on the assumption
            # that unpainted goal tiles are clear.

        unsatisfied_goals = {} # { tile_name: required_color }
        for tile, required_color in self.goal_tiles.items():
            if tile not in painted_tiles_state or painted_tiles_state[tile] != required_color:
                unsatisfied_goals[tile] = required_color

        # If all goals are satisfied, the heuristic is 0.
        if not unsatisfied_goals:
            return 0

        h = 0

        # 3. Base cost: One paint action per unsatisfied tile
        h += len(unsatisfied_goals)

        # 4. Color cost: Cost to acquire needed colors
        colors_needed = set(unsatisfied_goals.values())
        colors_held = set(robot_colors.values())

        # Count colors needed that no robot currently holds
        new_colors_to_acquire = colors_needed - colors_held
        h += len(new_colors_to_acquire) # Add 1 for each color that needs to be introduced

        # 5. Movement cost: Sum of minimum distances for each unsatisfied tile
        for tile, required_color in unsatisfied_goals.items():
            min_dist_to_tile = float('inf')

            # Find the minimum distance from any robot to this tile
            for robot, loc in robot_locations.items():
                dist = manhattan_distance(loc, tile)
                min_dist_to_tile = min(min_dist_to_tile, dist)

            # Add the minimum distance found for this tile
            if min_dist_to_tile != float('inf'):
                 h += min_dist_to_tile
            # Note: If a tile is unreachable by any robot, min_dist_to_tile remains inf.
            # Adding inf to h would make it inf, correctly indicating an unsolvable state
            # from this perspective. However, standard heuristic practice often assumes
            # solvable problems and finite heuristic values. If float('inf') is added,
            # the search will prune this branch.

        return h

