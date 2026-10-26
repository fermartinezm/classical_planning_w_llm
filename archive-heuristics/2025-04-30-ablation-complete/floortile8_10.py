import math # For float('inf')

# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Return empty list or handle error for malformed facts if necessary
    return []

def parse_tile(tile_name):
    """Parses a tile name like 'tile_row_col' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_0_1'
        if len(parts) == 3 and parts[0] == 'tile':
            # Convert row and column parts to integers
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        # Return None or raise error for unexpected formats
        return None
    except (ValueError, IndexError):
        # Handle cases where parts are not integers or index is out of bounds
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coords1 = parse_tile(tile1_name)
    coords2 = parse_tile(tile2_name)

    # If parsing failed for either tile name, distance is effectively infinite
    if coords1 is None or coords2 is None:
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2

    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of goal tiles that are not yet painted correctly (each requires a paint action).
    2. The number of colors required by unpainted goal tiles that are not currently held by any robot (each requires a color change action).
    3. An estimate of the movement cost, calculated as the sum, over all unpainted goal tiles, of the minimum moves required for the closest robot to reach a tile adjacent to the goal tile.

    # Assumptions
    - Goal conditions only involve `(painted ?x ?c)` facts.
    - Tiles specified in goal conditions are initially `clear` if not already painted correctly. Repainting is not possible or needed within the scope of achieving the goal state facts.
    - The grid structure implied by `up`, `down`, `left`, `right` predicates corresponds to a rectangular grid where tiles are named `tile_row_col`.
    - All actions have a cost of 1.
    - Any available color can be acquired by a robot via `change_color`.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which color. Store this as a dictionary mapping tile names to required colors.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unsatisfied Goal Tiles:
       - Iterate through the goal conditions (`self.goals`).
       - For each goal fact `(painted tile_name color)`, check if this exact fact exists in the current state (`node.state`).
       - Collect all goal tiles for which the corresponding `(painted tile_name color)` fact is *not* in the current state into a dictionary `unsatisfied_goals = {tile_name: color}`.

    2. Calculate Paint Action Cost:
       - The number of unsatisfied goal tiles is a lower bound on the number of paint actions required (each tile needs one paint action).
       - Add `len(unsatisfied_goals)` to the total heuristic cost.

    3. Calculate Color Change Cost:
       - Determine the set of colors required by the unsatisfied goal tiles (`needed_colors`).
       - Determine the set of colors currently held by any robot (`robot_current_colors`).
       - For each color in `needed_colors` that is not in `robot_current_colors`, at least one robot must perform a `change_color` action to acquire that color. Add 1 for each such color to the total heuristic cost. This is a lower bound on the number of *new* colors that need to be introduced into the robot fleet's hands.

    4. Calculate Movement Cost:
       - Identify the current location of each robot from the state facts `(robot-at robot_name tile_name)`. Store this as `robot_locations = {robot_name: tile_name}`.
       - For each tile `t` in `unsatisfied_goals`:
         - Find the minimum Manhattan distance from any robot's current location to tile `t`. Let this be `min_dist_to_tile`.
         - To paint tile `t`, a robot must reach a tile adjacent to `t`. The minimum number of moves to get from a robot's location `loc_r` to *any* tile adjacent to `t` is `max(0, manhattan_distance(loc_r, t) - 1)`.
         - Find the minimum of this value over all robots: `min_moves_to_adjacent = min_{r} max(0, manhattan_distance(robot_locations[r], t) - 1)`.
         - Add `min_moves_to_adjacent` to the total heuristic cost. This sums the minimum independent movement cost for each unpainted tile, assuming the closest robot handles it.

    5. Sum Components:
       - The total heuristic value is the sum of the costs from steps 2, 3, and 4.
       - If `unsatisfied_goals` is empty, the state is a goal state, and the heuristic should be 0. The calculation naturally results in 0 in this case.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # Store goal conditions, mapping tile to required color for painted goals
        self.goal_tiles_colors = {}

        # task.goals is a frozenset of goal facts
        for goal_fact_str in task.goals:
            parts = get_parts(goal_fact_str)
            # Assuming goal facts are directly (painted tile color)
            if parts and parts[0] == "painted":
                 if len(parts) == 3:
                      tile, color = parts[1], parts[2]
                      self.goal_tiles_colors[tile] = color
            # Ignore other types of goal facts if any (like 'and' if it were present as a fact)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # 1. Identify Unsatisfied Goal Tiles
        unsatisfied_goals = {} # {tile_name: required_color}
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color}

        # Parse current state to find robot locations and robot colors
        # We don't need to parse all painted tiles, just check if the goal painted fact exists.
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == "robot-at":
                 if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif predicate == "robot-has":
                 if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        # Compare current state with goals to find unsatisfied tiles
        for goal_tile, goal_color in self.goal_tiles_colors.items():
            # Goal is (painted goal_tile goal_color)
            # Check if the exact goal fact is in the current state
            goal_fact_str = f"(painted {goal_tile} {goal_color})"
            if goal_fact_str not in state:
                 unsatisfied_goals[goal_tile] = goal_color


        # If all goals are satisfied, the heuristic is 0
        if not unsatisfied_goals:
            return 0

        # Initialize heuristic components
        paint_cost = 0
        color_change_cost = 0
        movement_cost = 0

        # 2. Calculate Paint Action Cost
        paint_cost = len(unsatisfied_goals)

        # 3. Calculate Color Change Cost
        needed_colors = set(unsatisfied_goals.values())
        robot_current_colors = set(robot_colors.values())

        # Count colors needed by unsatisfied goals that no robot currently has
        for color in needed_colors:
            if color not in robot_current_colors:
                color_change_cost += 1

        # 4. Calculate Movement Cost
        if not robot_locations:
             # Should not happen in valid floortile problems, but handle defensively
             # If no robots, problem is likely unsolvable unless already goal state (handled above)
             # Return infinity or a very large number?
             return float('inf')

        for tile_to_paint, required_color in unsatisfied_goals.items():
            min_dist_to_tile = float('inf')
            for robot, robot_loc in robot_locations.items():
                dist = manhattan_distance(robot_loc, tile_to_paint)
                if dist is not None and dist < min_dist_to_tile:
                    min_dist_to_tile = dist

            # Minimum moves for the closest robot to reach a tile adjacent to the goal tile.
            # A robot at distance D needs D-1 moves to reach an adjacent tile (if D > 0).
            # If D=0 (robot is at the tile), it needs 0 moves to reach an adjacent tile.
            # If D=1 (robot is adjacent), it needs 0 moves to reach an adjacent tile.
            # So, moves needed is max(0, D - 1).
            min_moves_to_adjacent = max(0, min_dist_to_tile - 1) if min_dist_to_tile != float('inf') else float('inf')

            movement_cost += min_moves_to_adjacent

        # 5. Sum Components
        total_heuristic = paint_cost + color_change_cost + movement_cost

        return total_heuristic
