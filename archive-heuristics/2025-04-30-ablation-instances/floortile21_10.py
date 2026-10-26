from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses tile name 'tile_r_c' into row and column integers."""
    # Assuming tile names are always in the format 'tile_row_col'
    parts = tile_name.split('_')
    if len(parts) != 3 or parts[0] != 'tile':
        # Handle unexpected format, maybe raise error or return default
        # For this problem, assuming standard format is safe.
        raise ValueError(f"Unexpected tile name format: {tile_name}")
    try:
        row = int(parts[1])
        col = int(parts[2])
        return row, col
    except ValueError:
        raise ValueError(f"Could not parse row/col from tile name: {tile_name}")


def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    r1, c1 = parse_tile_name(tile1_name)
    r2, c2 = parse_tile_name(tile2_name)
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently clear but need a specific color. It sums the estimated cost
    for each such tile independently. The cost for a single tile is estimated as
    the cost to change the robot's color (if needed), plus the Manhattan distance
    to the tile, plus the cost to paint the tile.

    # Assumptions
    - The goal only requires painting tiles that are currently clear or become clear.
      Tiles painted with the wrong color are considered unsolvable or irrelevant
      to the goal satisfaction by this heuristic.
    - The grid is structured such that Manhattan distance is a reasonable approximation
      of movement cost.
    - There is only one robot.
    - The goal is primarily defined by `(painted tile color)` facts.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
      This information is stored in `self.goal_painted_tiles`. Static facts
      (like grid structure or available colors) are not explicitly stored as
      Manhattan distance is used for movement cost approximation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the robot's current location and the color it is holding from the current state.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each tile and its required color stored during initialization (`self.goal_painted_tiles`).
    4. For the current tile `t` and required color `c`:
       a. Check if the state already contains the fact `(painted t c)`. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. Check if the state contains the fact `(clear t)`. If not, the tile is not clear (likely painted with a different color or occupied). Based on domain rules, it cannot be painted. This heuristic assumes such tiles do not contribute to the solvable part of the problem; continue to the next goal tile.
       c. If the state contains `(clear t)` and `(painted t c)` is a goal:
          i. Calculate the cost to get the correct color: If the robot's current color is not `c`, add 1 to the cost (for a `change_color` action). If the robot has color `c`, add 0.
          ii. Calculate the cost to move the robot to tile `t`: Find the robot's current location. Calculate the Manhattan distance between the robot's location and tile `t`. Add this distance to the cost.
          iii. Add the cost to paint the tile: Add 1 to the cost (for a `paint` action).
          iv. Add the calculated cost for this tile to the `total_cost`.
    5. Return the `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Store goal tiles and their required colors
        self.goal_painted_tiles = {}
        # task.goals is a frozenset of fact strings
        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                _, tile, color = get_parts(goal_fact)
                self.goal_painted_tiles[tile] = color

        # Static facts are available in task.static but not explicitly needed
        # for this heuristic's calculation using Manhattan distance.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot's current location and color
        robot_location = None
        robot_color = None
        # Assuming only one robot and it is always at a location and has a color
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                # Assuming the second argument is the robot name, third is location
                parts = get_parts(fact)
                if len(parts) == 3:
                    robot_location = parts[2]
            elif match(fact, "robot-has", "*", "*"):
                 # Assuming the second argument is the robot name, third is color
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    robot_color = parts[2]

        # If robot info is missing, return a high value or handle as error
        # Assuming valid states always contain robot-at and robot-has for the robot.
        if robot_location is None or robot_color is None:
             # This state is likely malformed or represents an impossible scenario
             # Return a high value to discourage search down this path, or 0 if
             # we assume it's an unpaintable dead end that doesn't contribute.
             # Returning 0 is safer for greedy search if it's not a goal state.
             # However, if it's truly unsolvable from here, a high value is better.
             # Let's assume valid states always have robot info.
             pass # Continue assuming robot_location and robot_color are found

        total_cost = 0  # Initialize action cost counter.

        for tile, required_color in self.goal_painted_tiles.items():
            # Check if the tile is already painted correctly
            if f"(painted {tile} {required_color})" in state:
                continue # This goal is satisfied for this tile

            # Check if the tile is clear (can be painted)
            if f"(clear {tile})" not in state:
                 # Tile is not clear (likely painted with wrong color or occupied).
                 # Cannot paint it according to domain rules (precondition clear ?y).
                 # This heuristic assumes such tiles do not contribute to the solvable
                 # part of the problem; skip this tile.
                 continue

            # Tile needs painting and is clear. Calculate cost for this tile independently.
            tile_cost = 0

            # Cost to get the correct color
            if robot_color != required_color:
                tile_cost += 1 # Change color action cost

            # Cost to move robot to the tile
            # Need robot_location, which we found earlier
            if robot_location != tile: # Only add move cost if not already at the tile
                 try:
                     dist = manhattan_distance(robot_location, tile)
                     tile_cost += dist
                 except ValueError:
                     # Handle cases where tile names are not in expected format
                     # This state might be invalid or represent an unsolvable part.
                     # Skipping this tile means it adds 0 cost, which is reasonable
                     # for a non-admissible heuristic on potentially unsolvable parts.
                     continue # Skip this tile calculation

            # Cost to paint the tile
            # This assumes the robot is now at the tile with the correct color
            tile_cost += 1 # Paint action cost

            total_cost += tile_cost

        # The heuristic is 0 if and only if total_cost is 0.
        # total_cost is 0 if and only if all tiles in self.goal_painted_tiles
        # are already painted with the required color AND are not clear.
        # This matches the goal state under the assumption that the goal is
        # solely defined by these painted facts.

        return total_cost

