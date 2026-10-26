import re
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base.Heuristic is available in the environment
# from heuristics.heuristic_base import Heuristic

# If Heuristic base class is not provided externally, define a minimal one:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError


# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (r, c) tuple."""
    match = re.match(r"tile_(\d+)_(\d+)", tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen for valid tile names

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (r, c) coordinate tuples."""
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of tiles that still need to be painted correctly.
    2. The estimated number of color changes required.
    3. The estimated movement cost to reach the vicinity of the first tile needing paint.

    # Assumptions
    - The goal state specifies the required color for certain tiles using the `(painted tile color)` predicate.
    - Tiles not mentioned in the goal `painted` facts do not need a specific color (or their current state is acceptable).
    - The robot always holds some color (no `free-color` state requiring an initial color acquisition action).
    - Movement between adjacent tiles takes 1 action.
    - Changing color takes 1 action.
    - Painting an adjacent tile takes 1 action.
    - The grid structure allows Manhattan distance calculation as a lower bound on movement cost between tiles.
    - The `clear` predicate constraint is relaxed for movement and painting in the heuristic calculation.
    - Tile names follow the format 'tile_row_col' where row and column are integers.

    # Heuristic Initialization
    - Extract the goal painting requirements: a dictionary mapping tile names to their required goal color.
    - Parse tile names from static facts (specifically connectivity facts like up/down/left/right) to create a mapping from tile name string to (row, column) integer coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Tiles: Iterate through the goal facts. For each goal fact `(painted tile color)`, check if the current state contains the same fact. Collect all tile names for which the goal painting fact is not present in the current state into a set `unpainted_tiles`.
    2. Base Cost (Painting): The minimum number of paint actions required is equal to the number of tiles that need painting. Add `len(unpainted_tiles)` to the total heuristic cost. If `unpainted_tiles` is empty, the heuristic is 0 (goal state).
    3. Color Change Cost:
       - Determine the set of unique colors `needed_colors` required for the tiles in `unpainted_tiles` based on the goal paintings.
       - Find the robot's current color from the state facts `(robot-has robot color)`.
       - If `needed_colors` is not empty:
         - If the robot's current color is not in `needed_colors`, the robot must change color at least once to get a needed color, and potentially change between other needed colors. The minimum number of color changes is `len(needed_colors)`. Add this to the cost.
         - If the robot's current color *is* in `needed_colors` and there is more than one needed color (`len(needed_colors) > 1`), the robot must change color at least `len(needed_colors) - 1` times to paint tiles requiring different colors. Add this to the cost.
       - If `needed_colors` is empty, the color change cost is 0.
    4. Movement Cost:
       - Find the robot's current location tile from the state fact `(robot-at robot tile)`. Convert this tile name to coordinates using the precomputed mapping.
       - The robot needs to move to a tile adjacent to each tile in `unpainted_tiles`.
       - Calculate the Manhattan distance from the robot's current coordinates to the coordinates of each tile in `unpainted_tiles`. The minimum number of moves to get *adjacent* to a tile is the Manhattan distance minus 1 (clamped at 0).
       - Find the minimum of these "adjacent distances" over all tiles in `unpainted_tiles`. This represents a lower bound on the movement cost to reach the vicinity of the first tile that needs painting. Add this minimum distance to the total cost.
    5. Total Heuristic Value: The final heuristic value is the sum of the costs calculated in steps 2, 3, and 4.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Store goal paintings: {tile_name: color}
        self.goal_paintings = {}
        for goal in task.goals:
            # Goal facts are typically positive literals
            if match(goal, "painted", "*", "*"):
                _, tile_name, color = get_parts(goal)
                self.goal_paintings[tile_name] = color

        # Extract tile coordinates: {tile_name: (row, col)}
        self.tile_coords = {}
        # Collect all tile names from static connectivity facts
        tile_names = set()
        for fact in task.static:
             parts = get_parts(fact)
             if len(parts) > 2 and parts[0] in ["up", "down", "left", "right"]:
                 tile_names.add(parts[1])
                 tile_names.add(parts[2])

        # Parse coordinates from tile names like 'tile_r_c'
        for tile_name in tile_names:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Information from the current state
        current_paintings = {} # {tile_name: color}
        robot_loc_name = None
        robot_color = None

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "painted", "*", "*"):
                _, tile_name, color = parts
                current_paintings[tile_name] = color
            elif match(fact, "robot-at", "*", "*"):
                 _, robot_name, loc_name = parts
                 robot_loc_name = loc_name
            elif match(fact, "robot-has", "*", "*"):
                 _, robot_name, color = parts
                 robot_color = color # Assuming only one robot or we care about the color of the robot-at

        # 1. Identify Unpainted Tiles (that need painting according to goal)
        unpainted_tiles = set()
        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile not in current_paintings or current_paintings[goal_tile] != goal_color:
                 unpainted_tiles.add(goal_tile)

        # If all painting goals are met, heuristic is 0
        if not unpainted_tiles:
            return 0

        # Initialize heuristic cost
        cost = 0

        # Cost Component 1: Painting actions
        cost += len(unpainted_tiles)

        # Cost Component 2: Color changes
        needed_colors = set(self.goal_paintings[tile] for tile in unpainted_tiles)

        # Assuming robot always has a color based on domain structure and examples
        if robot_color is None:
             # This case is unexpected but handled defensively.
             # If robot has no color but needs to paint, it must acquire one.
             # The domain doesn't have an action for this. Assume it starts with one.
             # If it truly has no color and needs colors, add cost to get the first one.
             if needed_colors:
                 # Cost to get the first color + changes between others
                 cost += len(needed_colors)
        elif robot_color not in needed_colors:
             # Robot has a color, but not one of the needed ones. Needs to change to a needed color.
             # Cost to change to first needed + changes between others
             cost += len(needed_colors)
        elif len(needed_colors) > 1:
             # Robot has one of the needed colors, but needs others too.
             # Cost to change between the other needed colors
             cost += len(needed_colors) - 1
        # If robot_color is in needed_colors and len(needed_colors) == 1, cost += 0. Correct.


        # Cost Component 3: Movement
        # Find minimum distance from robot to any tile adjacent to an unpainted tile
        robot_loc_coords = self.tile_coords.get(robot_loc_name) # Get robot coords

        if robot_loc_coords is not None: # Robot location is known
            min_dist_to_adjacent_unpainted = float('inf')
            for tile_name in unpainted_tiles:
                tile_coords = self.tile_coords.get(tile_name)
                if tile_coords is not None:
                    # Minimum moves to get adjacent to tile_coords is Manhattan(robot_loc_coords, tile_coords) - 1
                    # Need to handle case where robot is already adjacent or on the tile
                    dist = manhattan_distance(robot_loc_coords, tile_coords)
                    dist_to_adjacent = max(0, dist - 1) # Cannot have negative distance
                    min_dist_to_adjacent_unpainted = min(min_dist_to_adjacent_unpainted, dist_to_adjacent)

            if min_dist_to_adjacent_unpainted != float('inf'):
                 cost += min_dist_to_adjacent_unpainted
            # else: This case implies unpainted tiles have no coordinates, which shouldn't happen
            # if tile names are consistently 'tile_r_c' and present in static facts.

        # The heuristic value is the sum of the components.
        # It is non-negative and 0 iff unpainted_tiles is empty.
        return cost
