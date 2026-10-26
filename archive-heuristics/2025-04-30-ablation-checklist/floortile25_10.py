from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A more robust check might be needed for complex patterns, but this is sufficient for this domain
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_str):
    """
    Parses a tile name string like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes the format is strictly 'tile_R_C' where R and C are integers.
    """
    try:
        parts = tile_str.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, though problem description implies standard naming
            raise ValueError(f"Unexpected tile name format: {tile_str}")
    except (ValueError, IndexError) as e:
        print(f"Error parsing tile name {tile_str}: {e}")
        # Return a value that indicates an invalid tile, maybe (-1, -1) or raise error
        # For this heuristic, let's assume valid tile names and potentially crash if not,
        # as malformed names would indicate a problem instance error.
        raise e


def manhattan_distance(tile1_str, tile2_str):
    """
    Calculates the Manhattan distance between two tiles given their string names.
    """
    r1, c1 = parse_tile_name(tile1_str)
    r2, c2 = parse_tile_name(tile2_str)
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the minimum estimated cost for each unpainted goal tile. The minimum
    cost for a single tile is estimated by considering the best robot
    (the one requiring the fewest actions) to paint that specific tile,
    ignoring interactions between painting different tiles or coordination
    between robots.

    # Assumptions:
    - Tiles that need to be painted according to the goal are either currently
      `(clear)` or painted with the wrong color.
    - The domain rules only allow painting `(clear)` tiles. Therefore,
      if a goal tile is currently painted with the wrong color, it cannot
      be repainted according to the domain definition provided. We assume
      solvable instances only require painting tiles that are currently clear.
    - Robot names and tile names follow the 'robotX' and 'tile_R_C' format
      allowing coordinate extraction for distance calculation.
    - Robots always have a color initially and can change it if `available-color`
      predicate holds for the target color.

    # Heuristic Initialization
    - Stores the goal conditions from the task.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted ?tile ?color)`.
    2. For each such goal condition `(painted T C)`:
       a. Check if this fact `(painted T C)` is already true in the current state. If yes, this goal is met for this tile; contribute 0 to the heuristic for this tile.
       b. If the goal is not met, check if the tile `T` is currently `(clear T)` in the state. If it's not clear (meaning it's painted, likely the wrong color based on assumption), it cannot be painted according to the domain rules. For solvable instances, this tile should not need painting or indicates an issue. We assume unpainted goal tiles are clear. If it's not clear, it doesn't contribute to the heuristic *as a paintable tile*.
       c. If the goal `(painted T C)` is not met AND the tile `T` is `(clear T)`, then this tile needs to be painted with color `C`.
       d. For this tile `T` needing color `C`, calculate the minimum cost for *any* robot to paint it.
          i. For each robot `R`:
             - Find its current location `X` (`(robot-at R X)`).
             - Find its current color `C_R` (`(robot-has R C_R)`).
             - Calculate the Manhattan distance between the robot's current tile `X` and the target tile `T`.
             - The movement cost for the robot to get *adjacent* to `T` is `max(0, distance(X, T) - 1)`. (If distance is 0, robot is on T, needs 1 move to adjacent. If distance is 1, robot is adjacent, needs 0 moves).
             - The color change cost is 1 if the robot's current color `C_R` is not the required color `C`, otherwise 0.
             - The paint action cost is 1.
             - The total estimated cost for robot `R` to paint tile `T` is `move_cost + color_cost + paint_cost`.
          ii. Find the minimum of these costs over all robots. This is the estimated cost for this specific tile.
       e. Add this minimum estimated cost for tile `T` to the total heuristic value.
    3. The total heuristic value is the sum of the minimum estimated costs for all unpainted, clear goal tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract robot positions and colors
        robot_pos = {}
        robot_color = {}
        clear_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_pos[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif parts[0] == 'clear':
                tile = parts[1]
                clear_tiles.add(tile)

        total_heuristic = 0

        # Iterate through goal conditions
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)

            # We only care about painted goals
            if goal_parts[0] == 'painted':
                goal_tile = goal_parts[1]
                goal_color = goal_parts[2]

                # Check if the goal for this tile is already met
                if goal_fact_str in state:
                    continue # Tile is already painted correctly

                # Check if the tile is clear (required to paint)
                if goal_tile not in clear_tiles:
                    # Tile is not clear, cannot be painted according to domain rules.
                    # Assuming solvable instances don't require repainting.
                    continue

                # This tile needs painting (it's a goal, not met, and is clear)
                min_cost_for_this_tile = float('inf')

                # Find the minimum cost for any robot to paint this tile
                for robot_name, current_robot_tile in robot_pos.items():
                    current_robot_color = robot_color.get(robot_name) # Get color, default None if not found (shouldn't happen)

                    # If robot has no color (unexpected), it can't paint this tile yet
                    if current_robot_color is None:
                         continue # Or assign a very high cost? Let's assume robots always have a color.

                    # Calculate movement cost to get adjacent to the goal tile
                    dist_to_tile = manhattan_distance(current_robot_tile, goal_tile)
                    # Cost to reach a tile adjacent to the goal tile
                    # If dist is 0 (robot on tile), needs 1 move to adjacent.
                    # If dist is 1 (robot adjacent), needs 0 moves.
                    # If dist > 1, needs dist - 1 moves to get adjacent.
                    move_cost = max(0, dist_to_tile - 1)

                    # Calculate color change cost
                    color_cost = 0
                    if current_robot_color != goal_color:
                        color_cost = 1 # Needs one change_color action

                    # Paint action cost
                    paint_cost = 1

                    # Total estimated cost for this robot to paint this tile
                    cost_for_this_robot = move_cost + color_cost + paint_cost

                    # Update minimum cost for this tile
                    min_cost_for_this_tile = min(min_cost_for_this_tile, cost_for_this_robot)

                # Add the minimum cost for this tile to the total heuristic
                # If min_cost_for_this_tile is still infinity, it means there are no robots,
                # or robots cannot reach/paint the tile (e.g., no available color).
                # In a solvable instance, there should be robots and available colors.
                # If it's infinity, the problem is likely unsolvable from this state,
                # and returning infinity (or a very large number) is appropriate.
                if min_cost_for_this_tile != float('inf'):
                    total_heuristic += min_cost_for_this_tile
                # else:
                    # Problem might be unsolvable, heuristic remains high or becomes infinity.
                    # For greedy search, a large number is fine.

        return total_heuristic

