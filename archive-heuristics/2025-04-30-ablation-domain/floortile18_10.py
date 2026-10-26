# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# If running standalone, uncomment this mock class:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch
import math # For abs

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments unless args contains wildcards
    # A simpler check is just to zip and check all parts match args pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_X_Y' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format - return None or raise error
            # print(f"Warning: Unexpected tile name format: {tile_name}") # Avoid printing in heuristic
            return None
    except (ValueError, IndexError):
        # print(f"Error parsing tile name {tile_name}: {e}") # Avoid printing in heuristic
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed
        return float('inf') # Return infinity if tile names are unparseable

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up
    independent costs for each unpainted goal tile. The cost for a single
    unpainted goal tile is estimated as the minimum cost for any robot
    to reach that tile with the required color and paint it.

    # Assumptions
    - Tiles are named in the format 'tile_row_col', allowing Manhattan distance calculation.
    - Required colors are available for pickup.
    - Tiles needing painting are currently clear (or can be treated as such for heuristic purposes).
    - Actions have a cost of 1.
    - Robots can drop and pickup colors anywhere.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which colors.
    - Static facts are not explicitly used for grid structure (relies on tile naming) or color availability (assumes availability).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile_T color_C)`.
    2. Identify which of these goal facts are *not* currently true in the state. Let this set of unsatisfied goals be `U`.
    3. If `U` is empty, the state is a goal state, and the heuristic is 0.
    4. Initialize the total heuristic value `h` to be the number of unsatisfied goal tiles (`|U|`). This accounts for the paint action needed for each tile.
    5. For each unsatisfied goal `(target_tile, target_color)` in `U`:
        a. Find the current location and held color for each robot in the state.
        b. Initialize a minimum cost for this specific tile `min_robot_cost_for_this_tile` to infinity.
        c. For each robot `R` found to have a location in the state:
            i. Determine the robot's current tile `current_tile`.
            ii. Calculate the Manhattan distance between `current_tile` and `target_tile`.
            iii. Calculate the estimated cost for robot `R` to acquire `target_color` if it doesn't have it:
                - 0 if `R` already has `target_color`.
                - 1 if `R` has no color.
                - 2 if `R` has a different color (drop current, pickup target).
            iv. The estimated cost for robot `R` to paint this specific tile (excluding the paint action itself, which is counted in step 4) is `distance + color_acquisition_cost`.
            v. Update `min_robot_cost_for_this_tile` with the minimum cost found so far across all robots for this specific tile.
        d. Add `min_robot_cost_for_this_tile` to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions. We are interested in (painted tile color) goals.
        self.goal_painted_tiles = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_painted_tiles[tile] = color
                # else: # Avoid printing warnings in production heuristic
                #      print(f"Warning: Unexpected goal fact format: {goal}")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify robot locations and held colors
        robot_locations = {}
        robot_colors = {}
        # We only need robots whose location is known to calculate distance
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # If no robots are found with a location, and there are goals, heuristic is infinity
        # as goals cannot be reached.
        if not robot_locations and self.goal_painted_tiles:
             return float('inf')


        # 2. Identify unpainted goal tiles
        unpainted_goals = [] # List of (tile, color) tuples
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is already painted with the correct color
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state
            # As per assumptions, we ignore the 'clear' predicate state for simplicity.
            if not is_painted_correctly:
                 unpainted_goals.append((goal_tile, goal_color))

        # 3. If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # 4. Initialize heuristic with the number of paint actions needed
        # This counts the 'paint' action itself for each required tile.
        h = len(unpainted_goals)

        # 5. Estimate movement and color acquisition cost for each unpainted goal tile
        for target_tile, target_color in unpainted_goals:
            min_robot_cost_for_this_tile = float('inf')

            # Consider each robot whose location is known
            for robot_name, current_tile in robot_locations.items():

                # Calculate distance to the target tile
                dist = manhattan_distance(current_tile, target_tile)
                if dist == float('inf'):
                    # If distance cannot be calculated (e.g., invalid tile name), this path is impossible/problematic
                    # For this tile, the cost via this robot is infinite.
                    continue

                # Calculate cost for this robot to get the target color
                color_cost = 0
                if robot_name not in robot_colors:
                    # Robot has no color, needs to pick up target_color
                    color_cost = 1
                elif robot_colors[robot_name] != target_color:
                    # Robot has wrong color, needs to drop and pickup
                    color_cost = 2
                # If robot_colors[robot_name] == target_color, color_cost is 0

                # Total estimated cost for this robot to paint this specific tile (excluding paint action)
                # This is the cost to get the robot to the tile with the right color.
                robot_total_cost_for_this_tile = dist + color_cost

                # Update minimum cost for this tile across all robots
                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_total_cost_for_this_tile)

            # Add the minimum estimated cost for this tile to the total heuristic
            # If min_robot_cost_for_this_tile is still inf, it means no robot could reach it
            # (e.g., due to invalid tile names or no robots found with location).
            # This tile might be unreachable.
            # Adding infinity would make the total heuristic infinity, which is fine for unreachable goals.
            # However, for solvable problems, this should be finite.
            if min_robot_cost_for_this_tile == float('inf'):
                 # This suggests an issue with the problem instance or state representation
                 # where a goal tile is unreachable by any robot with a known location.
                 return float('inf') # Indicate this branch is likely unsolvable

            h += min_robot_cost_for_this_tile

        return h
