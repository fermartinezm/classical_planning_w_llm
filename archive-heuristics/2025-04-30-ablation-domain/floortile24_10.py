from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        # This should not happen with valid PDDL tile names in this domain,
        # but return None to indicate failure if format is unexpected.
        pass
    return None # Indicate parsing failure

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # If tile names cannot be parsed into coordinates,
        # we cannot calculate a meaningful distance.
        # Returning 0 might underestimate, returning infinity might overestimate.
        # Assuming all relevant tiles follow the 'tile_r_c' format.
        # If not, this heuristic might be incomplete.
        # For robustness, return a large value if parsing fails, implying unknown distance.
        # However, given the domain structure, parsing should succeed for tiles.
        # Let's return 0 as a fallback, assuming non-grid objects won't be involved in distance.
        return 0
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums up the estimated cost for each unsatisfied
    goal tile independently. The cost for a single tile includes:
    1. Cleaning cost (if needed).
    2. Painting cost.
    3. Estimated cost for the "best" robot to reach the tile and acquire the correct color.

    # Assumptions
    - All tiles relevant to goals and robot locations follow the 'tile_r_c' naming
      convention allowing Manhattan distance calculation.
    - Pickup/drop color actions are possible at any tile a robot is located.
    - Action costs are uniform (cost 1 per action).
    - The heuristic assumes that the minimum cost for each goal tile can be summed
      independently, ignoring potential conflicts or synergies between goals
      (e.g., multiple tiles needing the same robot or color). This makes it
      non-admissible but potentially effective for greedy search.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically which tiles need to be painted
      and with which color.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Identify the current location for each robot using `(robot-at robot tile)` facts.
       - Identify the color currently held by each robot using `(robot-has robot color)` facts.
       - Identify the current state of each tile (painted with a specific color using `(painted tile color)` or clear using `(clear tile)`). If a tile is neither painted nor clear, assume it is dirty.

    2. Initialize Total Heuristic Cost:
       - Set `total_cost = 0`.

    3. Iterate Through Goal Tiles:
       - For each goal condition `(painted goal_tile goal_color)` identified during initialization:
         - Check if this specific goal fact `(painted goal_tile goal_color)` is present in the current state. If it is, the goal for this tile is satisfied, and we add 0 cost for this tile; continue to the next goal tile.

    4. Calculate Cost for Unsatisfied Goal Tiles:
       - If the goal for `goal_tile` is NOT satisfied:
         - Calculate the base action cost for the tile itself:
           - Check the current state of `goal_tile`:
             - If `(painted goal_tile C_current)` is in the state where `C_current != goal_color`, the tile is painted wrong. It needs a `clean` action and a `paint` action. Add 2 to the cost.
             - If `(clear goal_tile)` is in the state, the tile is clear. It needs a `paint` action. Add 1 to the cost.
             - If the tile is neither `painted` nor `clear` in the state, assume it is dirty. It needs a `clean` action and a `paint` action. Add 2 to the cost.
         - Estimate the minimum cost for any robot to become "ready" to paint this `goal_tile` with `goal_color`. Readiness means the robot is at `goal_tile` and holds `goal_color`.
           - Initialize `min_readiness_cost = infinity`.
           - For each robot:
             - Get the robot's current location (`robot_location`) and color (`robot_color`).
             - If the robot's location is unknown, this robot cannot be used for estimation for this tile; skip it.
             - Calculate the estimated movement cost: Manhattan distance between `robot_location` and `goal_tile`.
             - Calculate the estimated color acquisition cost for this robot to get `goal_color`:
               - 0 if `robot_color` is already `goal_color`.
               - 1 if the robot has no color (needs `pickup goal_color`).
               - 2 if the robot has a different color (`robot_color != goal_color` and `robot_color` is not None) (needs `drop robot_color` then `pickup goal_color`).
             - The readiness cost for this specific robot is `movement_cost + color_acquisition_cost`.
             - Update `min_readiness_cost` with the minimum readiness cost found so far across all robots.
           - If, after checking all robots, `min_readiness_cost` is still infinity (meaning no robot location was known), the state is likely unsolvable; return infinity.
           - Add `min_readiness_cost` to the `total_cost`.

    5. Return the `total_cost`. This value is 0 if and only if all goal tiles were already satisfied in the initial state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # task.static contains static facts like spatial relations, available colors.
        # We don't strictly need to process static facts if we assume grid structure
        # and uniform action costs for this non-admissible heuristic.

        # Store goal tiles and their required colors
        self.goal_tiles = {} # {tile_name: color_name}
        for goal in self.goals:
            # Assuming goal is always (painted tile color)
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract robot locations and colors from the current state
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name or None}}
        # Find all robots first
        all_robots = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "robot-at" and len(parts) == 3:
                 all_robots.add(parts[1])
             elif parts and parts[0] == "robot-has" and len(parts) == 3:
                 all_robots.add(parts[1])

        # Initialize robot info
        for robot in all_robots:
             robot_info[robot] = {'location': None, 'color': None}

        # Populate robot info from state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot in robot_info: # Should always be true now
                    robot_info[robot]['location'] = location
            elif parts[0] == "robot-has" and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot in robot_info: # Should always be true now
                    robot_info[robot]['color'] = color

        # Extract current tile states (painted or clear) for relevant tiles (goal tiles)
        tile_current_state = {} # {tile_name: {'painted': color} or {'clear': True}}
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue

             if parts[0] == "painted" and len(parts) == 3:
                 tile, color = parts[1], parts[2]
                 # Only store if it's a goal tile or if we need state for non-goal tiles?
                 # The heuristic only cares about goal tiles' current state.
                 if tile in self.goal_tiles:
                     tile_current_state[tile] = {'painted': color}
             elif parts[0] == "clear" and len(parts) == 2:
                 tile = parts[1]
                 # Only store if it's a goal tile and not already marked as painted
                 if tile in self.goal_tiles and tile not in tile_current_state:
                     tile_current_state[tile] = {'clear': True}

        total_cost = 0

        # Iterate through each goal tile and its required color
        for goal_tile, goal_color in self.goal_tiles.items():
            current_tile_state = tile_current_state.get(goal_tile)

            # Check if the goal for this tile is already satisfied
            if current_tile_state and 'painted' in current_tile_state and current_tile_state['painted'] == goal_color:
                continue # Goal met for this tile, cost is 0 for this tile

            # Goal is not satisfied. Calculate base actions needed for the tile itself (clean/paint)
            base_tile_action_cost = 0
            if current_tile_state and 'painted' in current_tile_state:
                 # Tile is painted, but with the wrong color
                 base_tile_action_cost = 2 # Needs clean + paint
            elif current_tile_state and 'clear' in current_tile_state:
                 # Tile is clear, needs paint
                 base_tile_action_cost = 1 # Needs paint
            else:
                 # Tile is neither painted nor clear (among goal tiles checked) -> assume dirty
                 base_tile_action_cost = 2 # Needs clean + paint

            total_cost += base_tile_action_cost

            # Estimate minimum readiness cost for this tile across all robots
            min_readiness_cost = float('inf')

            # Ensure there are robots to perform actions
            if not robot_info:
                 # No robots found, cannot paint. Problem unsolvable from here.
                 return float('inf')

            for robot, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None:
                     # Robot location unknown, cannot use this robot for estimate
                     continue

                # 1. Movement cost
                move_cost = manhattan_distance(robot_location, goal_tile)

                # 2. Color acquisition cost
                color_cost = 0
                if robot_color != goal_color:
                    if robot_color is not None:
                        color_cost += 1 # Cost for drop action
                    color_cost += 1 # Cost for pickup action

                readiness_cost = move_cost + color_cost
                min_readiness_cost = min(min_readiness_cost, readiness_cost)

            # If min_readiness_cost is still inf, it means no robot location was found,
            # which should be caught by the check before the robot loop.
            # If somehow it's inf here, it indicates an issue, return inf.
            if min_readiness_cost == float('inf'):
                 return float('inf')

            total_cost += min_readiness_cost

        # The total_cost calculated is the sum of (tile_base_cost + min_robot_readiness_cost)
        # for each unsatisfied goal tile.
        # If all goals were satisfied, the loop body was skipped, and total_cost is 0.
        # This matches the requirement that h=0 only for goal states.

        return total_cost

