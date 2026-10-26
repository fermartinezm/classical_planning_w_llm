from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_R_C' into integer coordinates (R, C).
    Assumes tile names follow the format 'tile_row_col'.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers, though unlikely in valid PDDL
            return None
    return None # Return None for unexpected formats

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles based on their names.
    Returns infinity if tile names cannot be parsed.
    """
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)

    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid names

    row1, col1 = coords1
    row2, col2 = coords2

    return abs(row1 - row2) + abs(col1 - col2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all required tiles with the
    correct color. It sums the minimum estimated cost for each unpainted
    goal tile, considering robot locations and colors.

    # Assumptions
    - Tiles are named in the format 'tile_row_col'.
    - A tile painted with the wrong color cannot be repainted (or requires
      complex actions not modeled here, so we treat it as an unsolvable state
      for that tile).
    - The 'clear' precondition for painting is implicitly handled by assuming
      a robot can eventually reach an adjacent tile and paint it. The cost
      of making a tile clear (e.g., by moving a robot off it) is not explicitly
      modeled but is assumed to be small or included in movement.
    - The heuristic ignores the 'clear' precondition for movement when calculating
      Manhattan distance, assuming paths are generally traversable.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically which tiles need to be painted
      and with which color.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify all goal facts of the form `(painted tile color)`. Store these
       as a map from tile name to required color.
    2. Identify the current location and color of each robot from the state.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each goal tile and its required color from the goal map.
    5. For the current goal tile `tile_g` and required color `color_g`:
       a. Check if `(painted tile_g color_g)` is already true in the current state.
          If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. Check if `(painted tile_g color_w)` is true for any color `color_w`
          that is *not* `color_g`. If yes, the tile is painted incorrectly.
          Based on the domain, this tile cannot be repainted. This state is likely
          a dead end for this goal tile. Return a very large value (infinity)
          to indicate a poor state.
       c. If the tile is not painted correctly (and not painted incorrectly),
          it needs to be painted. Calculate the minimum cost for *any* robot
          to paint this tile.
          i. Initialize `min_robot_cost` to infinity.
          ii. For each robot `r` at `tile_r` with color `color_r`:
              - Calculate the estimated cost for robot `r` to paint `tile_g`:
                - Movement cost: Manhattan distance from `tile_r` to `tile_g`.
                  This is a relaxation, ignoring the need to move to an *adjacent*
                  tile and the `clear` precondition for movement.
                - Color change cost: 1 if `color_r` is not `color_g`, otherwise 0.
                - Paint action cost: 1.
                - Total cost for robot `r` for this tile = Movement cost + Color change cost + Paint action cost.
              - Update `min_robot_cost = min(min_robot_cost, Total cost for robot r)`.
          iii. Add `min_robot_cost` to the `total_cost`.
    6. Return the `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal colors for each tile that needs painting.
        # Map: tile_name -> required_color
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

        # Extract all possible colors from available-color facts (useful for checking wrong color)
        self.available_colors = set()
        for fact in task.static:
             parts = get_parts(fact)
             if parts and parts[0] == "available-color" and len(parts) == 2:
                 self.available_colors.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current robot locations and colors
        robot_info = {} # Map: robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['location'] = location
                elif parts[0] == "robot-has" and len(parts) == 3:
                     robot, color = parts[1], parts[2]
                     if robot not in robot_info:
                         robot_info[robot] = {}
                     robot_info[robot]['color'] = color
                # Note: 'free-color' is ignored as it's not used in actions

        total_cost = 0  # Initialize action cost counter.

        # Check each goal tile
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            is_painted_correctly = False
            is_painted_wrong = False

            # Check current state for the goal tile's paint status
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "painted" and len(parts) == 3:
                    painted_tile, painted_color = parts[1], parts[2]
                    if painted_tile == goal_tile:
                        if painted_color == goal_color:
                            is_painted_correctly = True
                            break # Found correct paint, no need to check other colors for this tile
                        else:
                            is_painted_wrong = True
                            # Don't break yet, might find the correct color later (though unlikely in valid states)

            if is_painted_correctly:
                continue # This goal tile is satisfied

            if is_painted_wrong:
                 # Tile is painted with the wrong color. Assuming this is a dead end.
                 # Return a large value to penalize this state heavily.
                 return float('inf')

            # If we reach here, the tile needs to be painted with goal_color
            # Calculate the minimum cost for any robot to paint this tile
            min_robot_cost_for_this_tile = float('inf')

            if not robot_info:
                 # No robots available, cannot paint. Problem likely unsolvable.
                 # Or maybe the state representation doesn't include robots if they are not at their initial location?
                 # Assuming robots are always represented if they exist.
                 # If no robots, this tile cannot be painted.
                 return float('inf')


            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color') # Robot might not have a color initially?

                if robot_location is None:
                    # Robot location unknown, cannot calculate cost
                    continue # Skip this robot

                # Cost to move robot to the goal tile's location (relaxation)
                move_cost = manhattan_distance(robot_location, goal_tile)
                if move_cost == float('inf'):
                     # Cannot parse tile names, cannot calculate distance
                     continue # Skip this robot

                # Cost to change color if needed
                # If robot_color is None, it needs to get a color first.
                # Assuming getting a color involves change_color from 'nothing' to 'color'.
                # The domain doesn't have 'free-color' in actions, only 'robot-has'.
                # 'change_color' requires 'robot-has ?c'. So a robot must have *some* color to change it.
                # Let's assume robots always have a color or can get one (e.g., from a dispenser, not modeled).
                # Simplification: Assume color change is always possible if the target color is available.
                # Cost is 1 if current color is wrong or None, 0 if correct.
                color_change_cost = 0
                if robot_color != goal_color:
                     color_change_cost = 1 # Needs to change color

                # Cost of the paint action itself
                paint_cost = 1

                # Total estimated cost for this robot to paint this tile
                current_robot_cost = move_cost + color_change_cost + paint_cost

                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, current_robot_cost)

            # If no robot could potentially paint this tile (e.g., no robots, or invalid tile names)
            if min_robot_cost_for_this_tile == float('inf'):
                 return float('inf') # Problem likely unsolvable from this state

            total_cost += min_robot_cost_for_this_tile

        return total_cost

