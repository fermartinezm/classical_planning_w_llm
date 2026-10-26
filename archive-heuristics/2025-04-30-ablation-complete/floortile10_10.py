# Need to import the base class and fnmatch
from fnmatch import fnmatch
# Assuming heuristic_base.py is in a 'heuristics' directory relative to where this file is run
# If the structure is different, the import path might need adjustment.
# Based on the problem description, the import path 'heuristics.heuristic_base' is expected.
from heuristics.heuristic_base import Heuristic
import re

# Utility functions needed by the heuristic class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure we don't zip unequal length lists
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """Parses 'tile_row_col' and returns (row, col)."""
    # Ensure input is a string before regex
    if not isinstance(tile_name, str):
        return None
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Return None if the name doesn't match the expected tile format
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    coords1 = get_tile_coords(tile1_name)
    coords2 = get_tile_coords(tile2_name)
    # If either object is not a tile or malformed, distance is effectively infinite
    if coords1 is None or coords2 is None:
        # This indicates an issue, maybe trying to calculate distance to a non-tile object?
        # Or a tile name doesn't match the expected pattern.
        # Return a large number to represent unreachable/very far.
        return 1000000 # Consistent with unsolvable case

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It considers the number of tiles needing painting,
    the color changes required, and the movement needed for robots to reach
    paintable tiles.

    # Assumptions
    - Tiles needing painting are initially clear or become clear.
    - Tiles painted with the wrong color make the problem unsolvable.
    - Robots always hold exactly one color.
    - The tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - Movement cost is estimated using Manhattan distance on the grid.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and with which color.
    - Static facts (like adjacency relations) are not explicitly used to build a graph,
      but the tile naming convention is used to infer grid coordinates for distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal facts of the form `(painted tile_X_Y color)`. Store these as the target state for relevant tiles.
    2. Extract the current state information: robot locations, robot held colors, which tiles are painted and with what color, and which tiles are clear.
    3. Determine the set of "unpainted goal tiles": these are tiles specified in the goal that are not currently painted with the correct color in the state.
    4. Check if any goal tile is currently painted with a *different* color than its goal color. If such a tile exists, the problem is assumed unsolvable in this domain (as there's no unpaint action), and a very large heuristic value is returned.
    5. Filter the unpainted goal tiles to include only those that are currently `clear`. According to the domain rules, only clear tiles can be painted. These `ClearUnpaintedGoals` are the tiles that can potentially be painted next.
    6. The base heuristic cost is the number of tiles in `ClearUnpaintedGoals`. Each of these tiles requires at least one `paint` action.
    7. Calculate the color cost: Determine the set of distinct colors required by the tiles in `ClearUnpaintedGoals`. For each required color, check if *any* robot currently holds that color. If a required color is not held by any robot, add 1 to the heuristic. This estimates the minimum number of `change_color` actions needed to make all required colors available to at least one robot.
    8. Calculate the movement cost: For each robot, identify the subset of `ClearUnpaintedGoals` that require the color the robot currently holds. If this subset is not empty, the robot needs to move to be adjacent to one of these tiles to perform a paint action. Calculate the minimum Manhattan distance from the robot's current location to an *adjacent* tile of any tile in this subset. The distance to an adjacent tile `AdjT` of a target tile `T` from a robot location `R_loc` is estimated as `max(0, Manhattan_Dist(R_loc, T) - 1)`. Sum these minimum distances over all robots that have a color needed by at least one clear unpainted goal tile. Add this sum to the heuristic. This estimates the minimum travel needed for each capable robot to reach its first potential painting task.
    9. The total heuristic value is the sum of the base cost (paint actions), color cost (color changes), and movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Static facts are implicitly used by the tile naming convention.
        """
        self.goals = task.goals

        # Store goal locations and colors for quick lookup
        self.goal_tiles = {} # {tile_name: color}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Ensure goal fact has correct structure before accessing args
                if len(args) == 2:
                    tile, color = args
                    self.goal_tiles[tile] = color
                # else: ignore malformed goal fact

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 2. Extract current state information
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color}
        painted_tiles_state = {} # {tile_name: color}
        clear_tiles_state = set() # {tile_name}

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, robot, location = parts
                    robot_locations[robot] = location
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, robot, color = parts
                    robot_colors[robot] = color
            elif match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, tile, color = parts
                    painted_tiles_state[tile] = color
            elif match(fact, "clear", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, tile = parts
                    clear_tiles_state.add(tile)

        # 3. Determine the set of "unpainted goal tiles"
        unpainted_goals = {} # {tile_name: color}
        for tile, goal_color in self.goal_tiles.items():
            if tile not in painted_tiles_state or painted_tiles_state[tile] != goal_color:
                 unpainted_goals[tile] = goal_color

        # Check if the goal is reached
        if not unpainted_goals:
             return 0 # Goal state

        # 4. Check for unsolvable cases (tile painted with wrong color)
        for tile, goal_color in unpainted_goals.items():
             if tile in painted_tiles_state and painted_tiles_state[tile] != goal_color:
                 # This tile is painted the wrong color. Assuming unsolvable.
                 return 1000000 # A large number

        # 5. Filter unpainted goals to only include clear tiles
        # These are the tiles that can actually be painted in the current state
        clear_unpainted_goals = {
            tile: color for tile, color in unpainted_goals.items()
            if tile in clear_tiles_state
        }

        # 6. Base cost: Each clear unpainted goal tile needs a paint action.
        h = len(clear_unpainted_goals)

        # 7. Color cost: Count distinct colors needed for clear unpainted tiles
        # that no robot currently has.
        needed_colors = set(clear_unpainted_goals.values())
        current_robot_colors = set(robot_colors.values())
        missing_colors = needed_colors - current_robot_colors
        h += len(missing_colors)

        # 8. Movement cost: For each robot, calculate the minimum distance
        # to an adjacent tile of *any* clear unpainted goal tile it *could* paint
        # (i.e., a tile needing its current color). Sum these minimum distances.
        movement_cost = 0
        for robot, r_loc in robot_locations.items():
            r_color = robot_colors.get(robot) # Get robot's current color

            if r_color is None:
                 # This robot doesn't have a color fact in the state.
                 # Based on domain, robots always have a color initially and after change_color.
                 # This case might indicate an unexpected state representation or initial state.
                 # For robustness, skip this robot for movement calculation.
                 continue

            # Find tiles this robot *could* paint with its current color
            paintable_tiles_for_R = {
                tile for tile, color in clear_unpainted_goals.items()
                if color == r_color
            }

            min_dist_R = float('inf')

            if paintable_tiles_for_R:
                # Robot needs to reach an adjacent tile of one of these paintable tiles
                for target_tile in paintable_tiles_for_R:
                    dist = manhattan_distance(r_loc, target_tile)
                    # Distance to be adjacent is distance to target minus 1, but not less than 0
                    dist_to_adj = max(0, dist - 1)
                    min_dist_R = min(min_dist_R, dist_to_adj)

                # Add the minimum distance for this robot to reach its *first* task location
                # Only add if a reachable paintable tile was found
                if min_dist_R != float('inf'):
                     movement_cost += min_dist_R

        h += movement_cost

        return h
