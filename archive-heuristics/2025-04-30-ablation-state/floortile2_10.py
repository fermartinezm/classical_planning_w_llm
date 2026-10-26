# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, this code will require a mock or actual Heuristic class definition.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a minimal Heuristic base class if the import fails
    # This is just for standalone testing purposes if the framework is not present.
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError("Heuristic __call__ method not implemented.")
    # print("Warning: Could not import Heuristic from heuristics.heuristic_base. Using minimal definition.")


from fnmatch import fnmatch
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_0_1', so parts should be ['tile', '0', '1']
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected format, return None or raise error
            return None
    except (ValueError, IndexError):
        # Handle parsing errors
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed
        # Return a large value to indicate this path is likely not useful
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by summing the estimated minimum cost for each tile that needs to be
    painted according to the goal but is not yet painted correctly in the
    current state. The estimated minimum cost for a single tile considers the
    closest robot, the cost for that robot to move to an adjacent tile, the
    cost for that robot to acquire the correct color, and the paint action itself.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates allows
      movement between adjacent tiles with cost 1 (if the target tile is clear).
    - Tile names are in the format 'tile_row_col' where row and col are integers.
    - Goal states require specific tiles to be painted with specific colors.
    - If a goal tile is not painted with the correct color, it is assumed to be
      'clear' (as there's no action to unpaint or change color of an already
      painted tile in the domain). Solvable problems will not require repainting.
    - The heuristic uses Manhattan distance as a lower bound for movement cost
      on the grid, ignoring potential blockages by painted tiles.
    - The heuristic sums costs for individual tiles independently, which may
      overestimate the total cost but aims to guide greedy search effectively.

    # Heuristic Initialization
    - Stores the set of goal facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. Extract the current location of each robot from the state facts. Store this in a dictionary mapping robot name to tile name.
    3. Extract the current color held by each robot from the state facts. Store this in a dictionary mapping robot name to color name.
    4. Iterate through each goal fact provided in the task definition.
    5. For each goal fact `(painted ?t ?c)`:
       a. Check if this goal fact is already present in the current state. If yes, this tile is already painted correctly, so continue to the next goal fact.
       b. If the goal fact is not in the current state, this tile `?t` needs to be painted with color `?c`.
       c. Initialize a variable `min_cost_for_tile` to infinity. This will track the minimum estimated cost for *any* robot to paint this specific tile.
       d. Iterate through each robot `r` identified in step 2.
       e. For the current robot `r`:
          i. Get its current location `loc_r` from the dictionary created in step 2.
          ii. Get its current color `color_r` from the dictionary created in step 3.
          iii. Calculate the estimated movement cost for robot `r` to reach a tile adjacent to `?t`. This is estimated as `max(0, ManhattanDistance(loc_r, ?t) - 1)`. We subtract 1 because the robot only needs to reach an *adjacent* tile, not the tile itself.
          iv. Calculate the color change cost for robot `r`. This is 1 if `color_r` is not equal to the required color `?c`, and 0 otherwise.
          v. The estimated cost for robot `r` to paint tile `?t` is the sum of the movement cost, the color change cost, and 1 for the paint action itself.
          vi. Update `min_cost_for_tile` with the minimum of its current value and the cost calculated in the previous step (e.v).
       f. After checking all robots, add the final `min_cost_for_tile` to the `total_heuristic`.
    6. Return the `total_heuristic` value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state facts (frozenset of strings)

        # Extract robot locations and colors from the current state
        robot_locs = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                robot_locs[robot] = location
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_heuristic = 0

        # Iterate through goal conditions to find unpainted tiles
        for goal in self.goals:
            # Check if the goal is a painted fact
            if match(goal, "painted", "*", "*"):
                goal_parts = get_parts(goal)
                tile_to_paint = goal_parts[1]
                required_color = goal_parts[2]

                # Check if the tile is already painted correctly in the current state
                if goal in state:
                    continue # Tile is already painted correctly

                # This tile needs to be painted. Calculate the minimum cost for any robot.
                loc_t = tile_to_paint

                min_cost_for_tile = float('inf')

                # Find the minimum cost for any robot to paint this tile
                for robot, loc_r in robot_locs.items():
                    color_r = robot_colors.get(robot) # Get robot's current color

                    # If robot has no color (shouldn't happen based on domain, but safe check)
                    if color_r is None:
                         continue # This robot cannot paint

                    # Calculate movement cost to an adjacent tile
                    # Manhattan distance from robot location to tile location
                    dist_to_tile = manhattan_distance(loc_r, loc_t)

                    # If distance calculation failed (e.g., invalid tile name format), skip this robot for this tile
                    if dist_to_tile == float('inf'):
                        continue

                    # Need to reach an adjacent tile, which is 1 step closer
                    move_cost = max(0, dist_to_tile - 1)

                    # Calculate color change cost
                    color_change_cost = 1 if color_r != required_color else 0

                    # Total estimated cost for this robot to paint this tile
                    # Cost = movement + color change + paint action (cost 1)
                    cost_this_robot = move_cost + color_change_cost + 1

                    min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

                # Add the minimum cost found for this tile to the total heuristic
                # If min_cost_for_tile is still infinity, it means no robot was found
                # (e.g., no robots exist), which might indicate an unsolvable state.
                # In a solvable state with robots, this will be finite.
                if min_cost_for_tile != float('inf'):
                    total_heuristic += min_cost_for_tile
                # else: If no robots exist or tile name parsing failed for all robots,
                # min_cost_for_tile remains inf. Adding 0 is safe and indicates
                # this specific goal tile contributes nothing if it can't be reached/painted.


        return total_heuristic
