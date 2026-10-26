import fnmatch
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic class for standalone testing if needed (remove for final submission)
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            return None # Indicate parsing failure
    return None # Indicate parsing failure


def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    p1 = parse_tile_name(tile1_name)
    p2 = parse_tile_name(tile2_name)
    if p1 is None or p2 is None:
        # Cannot calculate distance if parsing failed
        return float('inf')
    r1, c1 = p1
    r2, c2 = p2
    return abs(r1 - r2) + abs(c1 - c2)

# The Heuristic class definition
# from heuristics.heuristic_base import Heuristic # Keep this in mind for the final output block

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums up the estimated cost for each unpainted goal tile
    independently. The estimated cost for a single tile includes the paint action itself,
    the minimum movement cost for any robot to reach the tile, and the minimum color
    acquisition/change cost for any robot to get the required color.

    # Assumptions
    - Tiles are named in the format 'tile_row_col'.
    - The cost of moving between adjacent tiles is 1.
    - The cost of pickup and drop actions is 1.
    - If a tile is not painted with its goal color, it is considered 'clear'
      and ready to be painted. The heuristic does not explicitly check the 'clear' predicate,
      assuming unpainted goal tiles are always clear in valid problem instances.
    - The heuristic sums costs independently for each unpainted goal tile,
      which is non-admissible but aims to guide greedy search by prioritizing
      states where goal conditions are "closer" to being met.
    - Color acquisition/change cost for a robot needing color C is estimated as 0
      if it already has C, 1 if it has no color (pickup C), and 2 if it has
      a different color (drop current, pickup C). This is calculated per tile
      and robot, and the minimum is taken.
    - Movement cost for a tile is estimated as the minimum Manhattan distance
      from any robot to that tile.

    # Heuristic Initialization
    - Extract the goal conditions (`painted` facts) to identify which tiles
      need to be painted and with which colors. Store these as a set of
      `(tile, color)` tuples.
    - Extract available colors from static facts. (Note: This isn't strictly
      used in the current heuristic logic, but is good practice to extract
      static info).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of goal facts of the form `(painted tile color)`.
    2. Identify the set of `(painted tile color)` facts currently true in the state.
    3. Determine the set of `unpainted_goals`: `(tile, color)` pairs from the goal
       that are not present in the current state.
    4. If `unpainted_goals` is empty, the state is a goal state, and the heuristic is 0.
    5. Initialize the total heuristic cost `h` to 0.
    6. Identify the current location of each robot (`robot-at` facts).
    7. Identify the current color held by each robot (`robot-has` facts).
    8. Handle case with no robots: If there are unpainted goals but no robots, it's unsolvable. Return infinity.
    9. For each `(goal_tile, goal_color)` pair in `unpainted_goals`:
        a. Add 1 to `h` (representing the `paint` action needed for this tile).
        b. Calculate the minimum preparation cost for this tile across all robots:
           Initialize `min_robot_prep_cost` to infinity.

           # Check if the goal_tile name is parsable
           if parse_tile_name(goal_tile) is None:
                # If the goal tile name is malformed, this problem is likely invalid or unsolvable
                return float('inf')

           for robot, robot_loc in robot_locations.items():
                # Ensure robot_loc is also parsable before calculating distance
                if parse_tile_name(robot_loc) is None:
                    # If a robot location is malformed, skip this robot
                    continue

                # i. Movement cost
                move_cost = manhattan_distance(robot_loc, goal_tile)

                # ii. Color acquisition/change cost
                current_color = robot_colors.get(robot)
                color_cost = 0
                if current_color != goal_color:
                    if current_color is not None: # Has wrong color
                        color_cost += 1 # Cost to drop current color
                    color_cost += 1 # Cost to pickup the goal color

                # iii. Total preparation cost for this robot for this tile
                total_preparation_cost = move_cost + color_cost

                # iv. Update minimum
                min_robot_prep_cost = min(min_robot_prep_cost, total_preparation_cost)

           # c. Add minimum preparation cost to heuristic
           # If min_robot_prep_cost is still infinity, it means no robot could reach/prepare for this tile
           # (e.g., all robot locations were unparsable, or no robots existed, handled above).
           # This tile is effectively unreachable.
           if min_robot_prep_cost == float('inf'):
                return float('inf')

           h += min_robot_prep_cost

    # 10. Return total cost
    return h
