from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

# Helper functions (can be outside the class or inside)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It considers the number of tiles needing
    painting, the colors required, and the movement cost for robots to reach
    those tiles.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the format 'tile_R_C'.
    - The only way to satisfy a '(painted tile color)' goal is by using the paint
      action on a clear tile.
    - If a goal tile is painted with the wrong color, the problem is unsolvable
      (as there's no action to unpaint or clear a painted tile). The heuristic
      returns infinity in this case.
    - The cost of any action (move, paint, change_color) is 1.

    # Heuristic Initialization
    - Extracts the set of goal painting facts (tile, color) pairs from the task goals.
    - Does not need to pre-process grid structure as tile coordinates can be
      parsed from names on the fly.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal painting facts `(painted tile color)` from the task goals.
    2. Determine which of these goal facts are *not* true in the current state.
       These are the "unmet painting goals", stored as a set of `(tile, color)` tuples.
    3. Identify the set of unique tiles involved in the unmet painting goals.
    4. For each tile involved in an unmet painting goal, check if it is `clear`
       in the current state. If any such tile is *not* clear, it must be
       painted with the wrong color (since it's not painted correctly either,
       as per step 2). In this case, the problem is unsolvable from this state,
       so return `float('inf')`.
    5. If all tiles involved in unmet painting goals are clear:
       a. Count the number of unmet painting goals. This contributes 1 for each
          required paint action. Let this be `paint_cost`.
       b. Identify the set of unique colors required by the unmet painting goals.
       c. Identify the set of unique colors currently held by *any* robot in the state.
       d. Count the number of required colors (from 5b) that are not held by any robot (from 5c).
          This estimates the minimum number of `change_color` actions needed
          across all robots to acquire the necessary colors. Let this be `color_cost`.
       e. Identify the current location of each robot.
       f. For each robot, calculate the minimum Manhattan distance from its
          current location to a tile *adjacent* to any of the tiles that need
          painting (identified in step 3). The distance from location `loc_r`
          to a tile `t` is `self.distance(loc_r, t)`. The minimum moves to reach
          a tile adjacent to `t` is `max(0, self.distance(loc_r, t) - 1)`.
          Sum these minimum distances (one minimum per robot) over all robots.
          This estimates the total movement cost for robots to get close to the
          work area. Let this be `movement_cost`.
       g. The total heuristic value is the sum: `paint_cost + color_cost + movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painting facts.
        """
        self.goals = task.goals
        # Store goal painting facts as a set of (tile_name, color_name) tuples
        self.goal_paintings = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) == 3:
                self.goal_paintings.add((parts[1], parts[2]))

    @staticmethod
    def parse_tile_name(tile_name):
        """Parses a tile name like 'tile_R_C' into (row, col) integers."""
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None # Not a valid tile name format
        return None # Not a valid tile name format

    @staticmethod
    def distance(tile1_name, tile2_name):
        """Calculates the Manhattan distance between two tiles."""
        coords1 = floortileHeuristic.parse_tile_name(tile1_name)
        coords2 = floortileHeuristic.parse_tile_name(tile2_name)
        if coords1 is None or coords2 is None:
            # This indicates an issue if tile names are expected to be in the format tile_R_C
            # For robustness, could potentially fall back to BFS distance if needed,
            # but assuming standard format based on examples.
            # Returning infinity signals an error or impossible distance.
            return float('inf')
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1 & 2. Identify unmet painting goals
        unmet_paintings = set()
        for tile, color in self.goal_paintings:
            if f'(painted {tile} {color})' not in state:
                unmet_paintings.add((tile, color))

        # If all goal paintings are met, the heuristic is 0
        if not unmet_paintings:
            return 0

        # 3. Identify tiles involved in unmet painting goals
        unpainted_goal_tiles = {tile for tile, color in unmet_paintings}

        # 4. Check for unsolvable state (goal tile painted wrongly)
        for tile in unpainted_goal_tiles:
            # If a tile that needs painting is not clear, it must be painted
            # with the wrong color, making the problem unsolvable.
            if f'(clear {tile})' not in state:
                 return float('inf')

        # 5a. Count paint actions needed
        paint_cost = len(unmet_paintings)

        # 5b & 5c. Identify required colors and colors held by robots
        required_colors = {color for tile, color in unmet_paintings}
        held_colors = set()
        robot_locations = {} # Map robot name to tile name
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-has' and len(parts) == 3:
                # Fact is like (robot-has robot1 black)
                held_colors.add(parts[2])
            elif parts[0] == 'robot-at' and len(parts) == 3:
                 # Fact is like (robot-at robot1 tile_0_4)
                 robot_locations[parts[1]] = parts[2]

        # 5d. Calculate color change cost
        # Count how many required colors are not held by *any* robot
        color_cost = sum(1 for color in required_colors if color not in held_colors)

        # 5e, 5f. Calculate movement cost
        movement_cost = 0
        if unpainted_goal_tiles: # Only calculate if there are tiles to paint
            for robot, robot_loc in robot_locations.items():
                # Find the minimum distance from the robot's current location
                # to a tile adjacent to any unpainted goal tile.
                min_dist_to_adjacent = float('inf')
                for target_tile in unpainted_goal_tiles:
                    dist = self.distance(robot_loc, target_tile)
                    # Distance to an adjacent tile is max(0, dist - 1)
                    dist_to_adj = max(0, dist - 1)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist_to_adj)

                # Add the minimum distance for this robot to the total movement cost
                if min_dist_to_adjacent != float('inf'):
                     movement_cost += min_dist_to_adjacent
                # Note: If min_dist_to_adjacent is inf for a robot (e.g., tile name parsing failed),
                # adding inf to movement_cost is correct as it indicates an issue or unreachable tile.


        # 5g. Total heuristic value
        total_cost = paint_cost + color_cost + movement_cost

        return total_cost
