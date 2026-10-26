import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_R_C' into a (row, col) tuple."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen with valid PDDL instances for this domain

def manhattan_distance(tile1_name, tile2_name):
    """Calculates the Manhattan distance between two tiles given their names."""
    coord1 = parse_tile_name(tile1_name)
    coord2 = parse_tile_name(tile2_name)
    if coord1 is None or coord2 is None:
        # Should not happen with valid PDDL instances for this domain
        return float('inf')
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def is_adjacent(tile1_name, tile2_name):
    """Checks if two tiles are adjacent based on Manhattan distance."""
    return manhattan_distance(tile1_name, tile2_name) == 1


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unpainted goal tile
    independently, considering the minimum cost for any robot to reach an adjacent
    tile with the correct color and perform the paint action.

    # Assumptions
    - Tiles are arranged in a grid and named 'tile_R_C'.
    - Manhattan distance is a reasonable approximation for movement cost.
    - If a goal tile is not painted correctly and is not clear, the state is likely
      unsolvable (as there's no unpaint action). The heuristic returns infinity in this case.
    - Robots always start with a color (no 'free-color' state that prevents getting a color).
    - All tile names follow the 'tile_R_C' pattern.

    # Heuristic Initialization
    - Extracts the goal conditions (`painted` facts).
    - Static facts are available in task.static but not explicitly needed
      for this Manhattan distance based heuristic, as grid structure is
      inferred from tile names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Extract the current location and held color for each robot from the state.
    3. Iterate through each goal condition `(painted T C)` stored during initialization:
        a. Check if the goal `(painted T C)` is already satisfied in the current state. If yes, continue to the next goal tile.
        b. If the goal is not satisfied, check the state of tile `T`.
           - If `(clear T)` is NOT in the state, and `(painted T C)` is NOT in the state, it implies `T` is painted with the wrong color (or occupied by a robot, which prevents painting). Since there's no unpaint action, this is an unsolvable state for this tile. Return infinity.
           - If `(clear T)` IS in the state (and `(painted T C)` is not), the tile needs to be painted.
        c. If `T` is clear and needs to be painted with color `C`:
            i. Calculate the minimum cost for any robot to reach a tile adjacent to `T` and be ready to paint it with color `C`. Initialize minimum robot cost for this tile to infinity.
            ii. For each robot `r` at `Loc_r` holding `Color_r`:
                - Calculate the Manhattan distance from `Loc_r` to `T`: `dist_to_T`.
                - Calculate the estimated moves needed to reach a tile adjacent to `T`:
                    - If `dist_to_T == 0` (robot is currently AT the goal tile): `moves = 1` (needs 1 move to get to an adjacent tile to paint FROM).
                    - If `dist_to_T == 1` (robot is currently ADJACENT to the goal tile): `moves = 0` (no moves needed to be adjacent).
                    - If `dist_to_T > 1`: `moves = dist_to_T - 1` (moves towards the tile until adjacent).
                - Calculate the color change cost: `color_cost = 1` if `Color_r != C`, else `0`.
                - Total cost for robot `r` to get ready = `moves + color_cost`.
                - Update minimum robot cost for tile `T`: `min_robot_cost_for_T = min(min_robot_cost_for_T, moves + color_cost)`.
            iii. Add the minimum robot cost for tile `T` (to get ready) plus the paint action cost (1) to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions, specifically the required painted state for tiles.
        # We expect goals like (painted tile_X_Y color)
        self.goal_painted_tiles = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                if len(args) == 2:
                    tile, color = args
                    self.goal_painted_tiles[tile] = color
                # Handle potential other goal types if necessary, but domain suggests only 'painted'

        # Static facts are available in task.static but not explicitly needed
        # for this Manhattan distance based heuristic, as grid structure is
        # inferred from tile names.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Extract robot locations and colors from the current state
        robots_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robots_info:
                    robots_info[robot] = {}
                robots_info[robot]['location'] = location
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                if robot not in robots_info:
                    robots_info[robot] = {}
                robots_info[robot]['color'] = color
            # Assuming robots always have a location and a color in valid states

        total_heuristic_cost = 0

        # Check each goal painted tile
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the goal for this tile is already satisfied
            goal_satisfied = False
            is_clear = False
            is_painted_wrong = False

            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "painted" and len(parts) == 3 and parts[1] == goal_tile:
                    if parts[2] == goal_color:
                        goal_satisfied = True
                        break # Goal for this tile is met
                    else:
                        is_painted_wrong = True
                        break # Painted with wrong color
                elif parts and parts[0] == "clear" and len(parts) == 2 and parts[1] == goal_tile:
                    is_clear = True

            if goal_satisfied:
                continue # This tile is done

            # If not satisfied, check if it's in an unsolvable state (painted wrong color or occupied)
            # If it's not clear, and not painted correctly (checked above), it's painted wrong or occupied.
            # Painting requires the tile to be clear. If it's not clear and not painted correctly,
            # it cannot be painted correctly without an unpaint/clear action (which doesn't exist).
            # If a robot is on it, it also cannot be painted.
            if not is_clear:
                 # This state is likely unsolvable for this goal tile.
                 # Return infinity to prune this branch.
                 return float('inf')

            # If the tile is clear and needs painting:
            min_robot_cost_for_tile = float('inf')

            # If there are no robots, the problem is likely unsolvable if there are unpainted tiles
            if not robots_info:
                 return float('inf')

            for robot_name, robot_data in robots_info.items():
                robot_location = robot_data.get('location')
                robot_color = robot_data.get('color')

                if robot_location is None or robot_color is None:
                    # Should not happen in valid states, but handle defensively
                    continue # Skip this robot if its state is incomplete

                # Calculate moves needed to get adjacent to the goal tile
                dist_to_goal_tile = manhattan_distance(robot_location, goal_tile)

                # Moves to get to a tile adjacent to the goal tile
                if dist_to_goal_tile == 0: # Robot is currently AT the goal tile
                    moves_to_adjacent = 1 # Needs 1 move to get to an adjacent tile to paint FROM
                elif dist_to_goal_tile == 1: # Robot is currently ADJACENT to the goal tile
                    moves_to_adjacent = 0 # No moves needed to be adjacent
                else: # Robot is further than 1 step away
                    moves_to_adjacent = dist_to_goal_tile - 1 # Moves towards the tile until adjacent

                # Calculate color change cost
                color_cost = 0
                if robot_color != goal_color:
                    color_cost = 1 # Needs one change_color action

                # Total cost for this robot to get ready to paint this tile
                cost_for_this_robot = moves_to_adjacent + color_cost

                # Update minimum cost for this tile across all robots
                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_this_robot)

            # Add the minimum cost to get a robot ready + the paint action cost (1)
            # If min_robot_cost_for_tile is still infinity, it means no robot could reach it,
            # which implies an issue (e.g., no robots, or tile name parsing failed).
            # Given valid problems, this should be finite if robots exist.
            if min_robot_cost_for_tile == float('inf'):
                 # This case should ideally not be reached in solvable problems with robots
                 # if tile names are parsed correctly and robots exist.
                 # Returning infinity is a safe fallback.
                 return float('inf')


            total_heuristic_cost += min_robot_cost_for_tile + 1 # +1 for the paint action

        return total_heuristic_cost
