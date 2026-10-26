from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid fact format

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """
    Extract row and column from tile name like 'tile_r_c'.
    Assumes tile names follow the format 'tile_<row>_<col>'.
    """
    try:
        parts = tile_name.split('_')
        # PDDL names are case-insensitive, but Python strings are not.
        # Ensure consistent case if needed, but typically names match exactly.
        # Assuming format is always tile_row_col
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle cases where tile name format is unexpected
        print(f"Warning: Unexpected tile name format: {tile_name}")
        return None # Or raise an error

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculate Manhattan distance between two tiles based on their names.
    Returns infinity if tile names are not in the expected format.
    """
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid tile names

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

def dist_to_adjacent(loc_tile_name, target_tile_name):
    """
    Calculate minimum moves from loc_tile to a tile adjacent to target_tile
    using Manhattan distance on the grid.
    Returns infinity if tile names are not in the expected format.
    """
    d = manhattan_distance(loc_tile_name, target_tile_name)

    if d == float('inf'):
        return float('inf')

    if d == 0:
        # Robot is at the target tile, needs 1 move to get to an adjacent tile
        return 1
    else:
        # Robot is not at the target tile. Needs d-1 moves to reach a tile at distance 1.
        return d - 1


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal
    tiles that are currently unpainted and clear. For each such tile, it calculates
    the minimum cost for any robot to reach an adjacent tile with the correct
    color and perform the paint action. The total heuristic is the sum of these
    minimum costs over all unpainted, clear goal tiles.

    # Assumptions
    - Tiles needing painting according to the goal are initially clear.
    - If a goal tile is found painted with a color different from the goal color
      in a reachable state, the problem is considered unsolvable from that state
      (heuristic returns infinity).
    - Movement cost between adjacent tiles is 1. Manhattan distance provides a
      reasonable lower bound estimate for movement on the grid.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.

    # Heuristic Initialization
    - Extracts the target color for each goal tile from the task's goal conditions.
    - Extracts available colors (though not strictly necessary for this heuristic's logic).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted tile_x_y color)`. Store
       these as a mapping from tile name to the required color.
    2. In the `__call__` method, parse the current state to find:
       - The current location of each robot (`robot-at`).
       - The current color held by each robot (`robot-has`).
       - The set of tiles that are currently clear (`clear`).
       - The set of tiles that are currently painted and their color (`painted`).
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each goal tile and its target color identified in `__init__`.
    5. For a goal tile `T` needing color `C`:
       - Check if the goal `(painted T C)` is already satisfied in the current state. If yes, continue to the next goal tile.
       - If the goal is not satisfied, check the current state of tile `T`.
       - If `T` is not `clear`:
         - Check if `T` is painted with a *different* color. If yes, the state is likely unsolvable in this domain (no unpaint action). Return `float('inf')`.
         - If `T` is not clear and not painted with the wrong color, this state is unexpected for a goal tile that started clear. Return `float('inf')`.
       - If `T` *is* `clear`:
         - This tile needs to be painted. The cost for this tile will be 1 (for the paint action) plus the minimum cost for *some* robot to get into position with the correct color.
         - Calculate the minimum cost over all robots `R` to achieve the state where `R` is adjacent to `T` and has color `C`.
         - For a robot `R` at `Loc_R` with color `Color_R`:
           - Cost to get the correct color: 1 if `Color_R` is not `C`, else 0.
           - Cost to move from `Loc_R` to a tile adjacent to `T`: Estimate this using Manhattan distance. If `Loc_R` is `T`, it takes 1 move to get adjacent. If `Loc_R` is already adjacent to `T`, it takes 0 moves. If `Loc_R` is further, it takes `Manhattan(Loc_R, T) - 1` moves. This is calculated by the `dist_to_adjacent` helper function.
           - The total "preparation" cost for robot `R` for this tile is (color cost) + (movement cost).
         - Find the minimum preparation cost among all robots.
         - Add this minimum preparation cost plus the paint action cost (1) to the total heuristic cost.
    6. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # static_facts = task.static # Static facts are not strictly needed for this heuristic's core logic

        # Store goal locations and required colors for each tile
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Fact is (painted tile color)
                if len(parts) == 3:
                    _, tile, color = parts
                    self.goal_tiles[tile] = color
                else:
                     print(f"Warning: Unexpected goal fact format: {goal}")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Parse current state facts
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles = set() # Store as set of (tile, color) tuples

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                _, robot, tile = parts
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                _, robot, color = parts
                robot_colors[robot] = color
            elif predicate == "clear" and len(parts) == 2:
                _, tile = parts
                clear_tiles.add(tile)
            elif predicate == "painted" and len(parts) == 3:
                _, tile, color = parts
                painted_tiles.add((tile, color))
            # Ignore other predicates like up, down, left, right, available-color, free-color

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each goal tile and its required color
        for target_tile, target_color in self.goal_tiles.items():

            # Check if the goal for this tile is already satisfied
            if (target_tile, target_color) in painted_tiles:
                continue # This goal is met, no cost

            # If the goal is not met, check the current state of the tile
            if target_tile not in clear_tiles:
                # The tile is not clear. It must be painted with the wrong color
                # or in some other unexpected state (e.g., occupied by robot).
                # Assuming solvable problems only have goal tiles that are clear
                # or correctly painted, this indicates an unsolvable state.
                # We can check explicitly if it's painted with the wrong color.
                is_painted_wrong_color = False
                for (painted_tile, painted_color) in painted_tiles:
                    if painted_tile == target_tile and painted_color != target_color:
                        is_painted_wrong_color = True
                        break

                if is_painted_wrong_color:
                     # Goal tile painted with the wrong color - likely unsolvable
                     return float('inf')
                else:
                     # Goal tile is not clear, but not painted with the wrong color.
                     # This shouldn't happen for goal tiles that start clear.
                     # Treat as unsolvable.
                     return float('inf')


            # If the tile is clear, it needs to be painted.
            # Calculate the minimum cost for any robot to paint this tile.
            min_robot_prep_cost_for_this_tile = float('inf')

            # Iterate through all robots
            for robot_name, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get robot's current color

                # Cost to get the correct color
                color_cost = 0
                if robot_color != target_color:
                    # Assumes target_color is always available if needed
                    color_cost = 1 # Cost of change_color action

                # Cost to move to a tile adjacent to the target tile
                move_cost = dist_to_adjacent(robot_location, target_tile)

                # Total preparation cost for this robot for this tile
                robot_prep_cost = color_cost + move_cost

                # Update minimum preparation cost found so far for this tile
                min_robot_prep_cost_for_this_tile = min(min_robot_prep_cost_for_this_tile, robot_prep_cost)

            # Add the minimum preparation cost + paint action cost (1) to the total
            # Check if min_robot_prep_cost_for_this_tile is still infinity (e.g., invalid tile names)
            if min_robot_prep_cost_for_this_tile == float('inf'):
                 return float('inf') # Propagate infinity if distance calculation failed

            total_cost += min_robot_prep_cost_for_this_tile + 1 # Add 1 for the paint action

        return total_cost

