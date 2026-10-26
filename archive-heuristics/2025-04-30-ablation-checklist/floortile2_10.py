from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (e.g., extra parameters in fact)
    # fnmatch handles wildcards correctly.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_r_c' into integer coordinates (row, column).
    Assumes tile names follow this format.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe return None or raise error
            # For typical benchmarks, this format is consistent.
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except (ValueError, IndexError):
        print(f"Error parsing tile coordinates for: {tile_name}")
        return None

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles based on their names.
    Assumes tile names can be parsed into coordinates.
    """
    coords1 = get_tile_coords(tile1_name)
    coords2 = get_tile_coords(tile2_name)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed
        return float('inf')

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with their target colors. It sums the estimated cost for
    each unpainted goal tile independently. The estimated cost for a single
    unpainted tile is 1 (for the paint action) plus the minimum cost for
    any robot to reach an adjacent tile with the correct color.

    # Assumptions:
    - Tiles are arranged in a grid, and tile names follow the format 'tile_row_col'.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1 action.
    - The 'clear' precondition for painting is assumed to be met or resolvable
      via movement already accounted for, unless the tile is painted with the
      wrong color (which is treated as an unsolvable state for that tile).
    - The heuristic does not account for coordination between robots or
      potential blocking by other robots or wrongly painted tiles (beyond
      marking the tile as unsolvable).

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify all goal tiles that are not yet painted with their required color.
    2. For each such unpainted goal tile T with target color C:
       a. Check if the tile T is currently painted with a *different* color. If so,
          the problem is likely unsolvable in this domain (no unpaint action),
          return infinity.
       b. Calculate the minimum cost for *any* robot to paint tile T with color C.
          This minimum cost is calculated over all robots R:
          - Cost = 1 (for the paint action itself)
          - + Estimated movement cost for robot R to reach a tile adjacent to T.
            A simple estimate is the Manhattan distance between the robot's current
            tile and the target tile T. (Note: Robot paints from an *adjacent* tile,
            but Manhattan distance to T is a reasonable proxy for distance to an
            adjacent tile in a grid).
          - + Estimated color change cost for robot R: 1 if robot R does not
            currently have color C, and 0 otherwise.
       c. The minimum cost for this specific unpainted goal tile is 1 + min(movement_cost + color_cost)
          over all robots.
    3. The total heuristic value is the sum of these minimum costs for all
       unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal requirements: map tile name to required color.
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # We might also want to know all available colors for the unsolvable check
        self.available_colors = set()
        for fact in task.static:
             if match(fact, "available-color", "*"):
                 self.available_colors.add(get_parts(fact)[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Track robot locations and current colors.
        robot_info = {} # {robot_name: {'loc': tile_name, 'color': color_name}, ...}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['loc'] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        total_cost = 0  # Initialize action cost counter.

        # Iterate through the goal tiles and their required colors.
        for tile_T, color_C in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color.
            if f"(painted {tile_T} {color_C})" in state:
                continue # This goal is already satisfied for this tile.

            # Check if the tile is painted with the *wrong* color.
            # In this domain, there's no unpaint action, so this is likely unsolvable.
            is_wrongly_painted = False
            for color_other in self.available_colors:
                if color_other != color_C and f"(painted {tile_T} {color_other})" in state:
                    is_wrongly_painted = True
                    break

            if is_wrongly_painted:
                 # This tile cannot be painted with the correct color. Problem unsolvable.
                 return float('inf')

            # This tile needs to be painted with color_C.
            # Estimate the minimum cost for any robot to do this.
            min_robot_cost_for_this_tile = float('inf')

            if not robot_info: # No robots available
                 return float('inf') # Cannot paint if no robots

            for robot_R, info in robot_info.items():
                robot_loc = info.get('loc')
                robot_color = info.get('color') # Robot might not have a color initially? Domain says robot-has in init.

                if robot_loc is None or robot_color is None:
                    # Robot state is incomplete, skip or handle appropriately.
                    # Assuming well-formed states where robots always have location and color.
                    continue

                # Estimate movement cost for robot_R to get adjacent to tile_T.
                # Using Manhattan distance to tile_T as a proxy.
                move_cost = manhattan_distance(robot_loc, tile_T)

                # Estimate color change cost for robot_R.
                color_cost = 0
                if robot_color != color_C:
                    color_cost = 1 # Cost to change color

                # Total estimated cost for this robot to paint this tile (excluding the paint action itself).
                robot_cost = move_cost + color_cost

                # Update the minimum cost found across all robots for this tile.
                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_cost)

            # Add the cost for this unpainted tile to the total heuristic value.
            # This includes 1 action for the paint itself, plus the minimum robot cost
            # to get into a state where the paint action is possible (adjacent + correct color).
            # If min_robot_cost_for_this_tile is still infinity, it means no robot can reach it,
            # which implies the problem is unsolvable (e.g., isolated tile).
            if min_robot_cost_for_this_tile == float('inf'):
                 return float('inf')

            total_cost += 1 + min_robot_cost_for_this_tile

        return total_cost

