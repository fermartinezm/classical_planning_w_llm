from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Use math.inf for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """Parses a tile name like 'tile_row_col' into (row, col) integer coordinates."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
    except (ValueError, IndexError):
        # Handle cases where tile name format is unexpected
        # print(f"Warning: Unexpected tile name format: {tile_name}") # Optional warning
        return None # Indicate invalid format

def manhattan_distance(tile1_name, tile2_name):
    """Calculates the Manhattan distance between two tiles given their names."""
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if coordinates are invalid
        return float('inf') # Indicate unreachable

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with the correct colors. It calculates the minimum cost for the
    closest robot to paint each unpainted goal tile independently and sums these costs.
    The cost for a robot to paint a tile includes movement to a paintable adjacent
    location, changing color if necessary, and the paint action itself.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' static facts.
    - Tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - Goal tiles that need painting are initially 'clear' or become 'clear'
      (i.e., the problem does not require repainting tiles that are already
      painted with the wrong color, as there's no action for that).
    - All goal tiles specified in the goal are paintable from at least one
      adjacent tile defined by the static facts.
    - There is at least one robot.

    # Heuristic Initialization
    - Store the goal conditions.
    - Parse static facts to build a map (`paintable_from`) where keys are tiles
      that can be painted, and values are lists of tuples `(paint_location, action_name)`,
      indicating that the tile can be painted from `paint_location` using `action_name`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal predicates of the form `(painted tile_x_y color)`.
    2. Determine which of these goal predicates are NOT satisfied in the current state.
       These are the 'unsatisfied goal tiles'.
    3. For each robot, find its current location and the color it is holding from the state.
    4. Initialize the total heuristic value to 0.
    5. For each unsatisfied goal tile `T` that needs color `C`:
        a. Find all possible adjacent locations `PaintLoc` from which tile `T` can be painted,
           using the `paintable_from` map built during initialization.
        b. Initialize the minimum cost for this tile (`min_cost_for_tile_T`) to infinity.
        c. For each robot `R`:
            i. Get the robot's current location `Loc_R` and color `C_R`.
            ii. Calculate the minimum Manhattan distance from `Loc_R` to any of the
                valid `PaintLoc`s for tile `T`. This is the estimated movement cost.
                If no valid `PaintLoc` exists for `T`, this distance remains infinity.
            iii. Calculate the color change cost: 1 if `C_R` is different from the
                 required color `C`, otherwise 0.
            iv. The cost for robot `R` to paint tile `T` is the movement cost + color change cost + 1 (for the paint action itself).
            v. Update `min_cost_for_tile_T` with the minimum cost found so far across all robots for tile `T`.
        d. Add `min_cost_for_tile_T` to the total heuristic value.
    6. Return the total heuristic value. If the value is infinity, it indicates an
       unsolvable state (e.g., a goal tile cannot be painted by any robot).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the paintable_from map from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build map from tile Y to list of (tile X, action) where (dir Y X) is true
        # This means tile Y can be painted from tile X using the corresponding action.
        self.paintable_from = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                direction, tile_y, tile_x = parts
                action_name = f'paint_{direction}'
                if tile_y not in self.paintable_from:
                    self.paintable_from[tile_y] = []
                self.paintable_from[tile_y].append((tile_x, action_name))

        # Store goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color
            # Assuming goals only involve 'painted' predicates for this domain

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif len(parts) == 3 and parts[0] == "robot-has":
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color

        # Extract currently painted tiles and their colors from the current state
        current_painted = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_painted[tile] = color

        # Identify unsatisfied goal tiles
        unsatisfied_goals = {
            tile: color
            for tile, color in self.goal_tiles.items()
            if tile not in current_painted or current_painted[tile] != color
        }

        # If all goals are satisfied, the heuristic is 0
        if not unsatisfied_goals:
            return 0

        total_heuristic = 0

        # Calculate cost for each unsatisfied goal tile
        for tile_T, required_color_C in unsatisfied_goals.items():
            min_cost_for_tile_T = float('inf')

            # Get possible locations from which tile_T can be painted
            # These are tiles X such that (dir tile_T X) is a static fact.
            possible_paint_locations = [
                paint_loc for paint_loc, _ in self.paintable_from.get(tile_T, [])
            ]

            # If a goal tile cannot be painted from anywhere (based on static facts),
            # it implies an unsolvable state or problem definition issue.
            # The min_cost_for_tile_T will remain inf, and summing it will result in inf.
            # This is appropriate for an unsolvable state.

            # For each robot, calculate the cost to paint this tile
            for robot_R, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_R) # Get robot's current color

                # Find the minimum distance from the robot's current location
                # to any of the valid paint locations for tile_T.
                min_dist_to_paint_loc = float('inf')
                if possible_paint_locations: # Avoid min over empty set
                    min_dist_to_paint_loc = min(
                        manhattan_distance(robot_loc, paint_loc)
                        for paint_loc in possible_paint_locations
                    )

                # If min_dist_to_paint_loc is still inf, this robot cannot reach
                # any paintable location for this tile (e.g., invalid tile names,
                # or no paintable locations defined).

                # Calculate color change cost
                color_change_cost = 1 if robot_color != required_color_C else 0

                # Total cost for this robot to paint this tile
                # Includes movement, color change, and the paint action itself (cost 1)
                # If min_dist_to_paint_loc is inf, cost_R_for_T will also be inf.
                cost_R_for_T = min_dist_to_paint_loc + color_change_cost + 1

                # Update the minimum cost for this tile across all robots
                min_cost_for_tile_T = min(min_cost_for_tile_T, cost_R_for_T)

            # Add the minimum cost for this tile to the total heuristic
            # If min_cost_for_tile_T is inf, it means no robot can paint this tile.
            # Summing inf will result in inf, indicating an unsolvable state.
            total_heuristic += min_cost_for_tile_T

        # Return the calculated heuristic value
        # If total_heuristic is inf, it means at least one goal tile is unreachable/unpaintable.
        # This is a valid heuristic value indicating a high cost or unsolvable state.
        return total_heuristic

