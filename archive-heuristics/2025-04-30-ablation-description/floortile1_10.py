# Required imports
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

# Helper functions (copied from example heuristics)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

# Domain-specific helper function
def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected format, maybe return None or raise error
            return None
    except ValueError:
        # Handle cases where r or c are not integers
        return None

# Domain-specific helper function
def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance for invalid tile names
        return float('inf') # Or some large number
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles with their target colors. It sums the estimated cost for
    each individual unpainted goal tile, considering the cost for the closest
    robot to reach a tile adjacent to the target tile and change color if needed.

    # Assumptions
    - The grid structure of tiles (tile_r_c) allows Manhattan distance calculation.
    - The primary costs are changing color (1 action), moving between adjacent
      tiles (1 action), and painting an adjacent tile (1 action).
    - The heuristic assumes a robot can move to any tile (ignoring the 'clear'
      precondition for movement for simplicity in distance calculation, which
      makes it an underestimate).
    - It assumes that if a tile is painted with the wrong color, the goal is
      unreachable (infinite cost).

    # Heuristic Initialization
    - Extract the goal conditions from the task, specifically identifying
      which tiles need to be painted and with which colors.
    - Static facts (like adjacency relations) are not explicitly stored or
      used for graph traversal, as the grid structure implied by tile names
      is used for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify all goal conditions of the form `(painted ?tile ?color)`. Store
       these target tiles and their required colors.
    3. Iterate through the identified goal tiles. For each goal `(painted T C)`:
       a. Check the current state for the status of tile `T`.
       b. If `(painted T C)` is already true in the state, the goal for this tile is met. Continue to the next goal tile.
       c. If `(painted T C')` is true in the state where `C' != C`, the tile is painted the wrong color. This state represents a dead end for this goal tile, as there's no action to unpaint a tile. Return a very large number (representing infinity) immediately.
       d. If the goal for tile T is not met and it's not painted the wrong color (meaning it's either unpainted or painted the correct color, which is handled in step 3b):
          i. Find the current location and color of every robot present in the state.
          ii. Initialize a variable `min_robot_cost_for_tile` to infinity.
          iii. If there are no robots, this tile cannot be painted. Return infinity.
          iv. For each robot `R` at location `Loc_R` with color `Color_R`:
              - Calculate the estimated cost for this robot to paint tile `T`:
                  - `cost_for_this_robot = 0`
                  - If `Color_R != C`: Add 1 to `cost_for_this_robot` (for the `change_color` action).
                  - Calculate the Manhattan distance `dist_to_tile` between `Loc_R` and `T`.
                  - If distance calculation fails (e.g., invalid tile name format), this robot cannot reach. Set `cost_for_this_robot` to infinity.
                  - Otherwise, calculate the estimated cost for movement and painting:
                      - If `dist_to_tile == 0` (robot is currently on tile `T`): The robot must move off `T` (1 action) to an adjacent tile, and then paint `T` from that adjacent tile (1 action). Total moves+paint cost = 2.
                      - If `dist_to_tile > 0` (robot is not on tile `T`): The robot needs `max(0, dist_to_tile - 1)` moves to reach a tile adjacent to `T`, and then 1 paint action. Total moves+paint cost = `max(0, dist_to_tile - 1) + 1`.
                      - Add this moves+paint cost to `cost_for_this_robot`.
              - Update `min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_this_robot)`.
          v. If after checking all robots, `min_robot_cost_for_tile` is still infinity, it means no robot can paint this tile (e.g., due to invalid tile names). Return infinity.
          vi. Add `min_robot_cost_for_tile` to the `total_cost`.
    4. After iterating through all goal tiles, return the `total_cost`. If all goals were met, the cost will be 0. If some goals are unmet but reachable, the cost will be positive. If any dead end was found, infinity was returned earlier.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions.
        # Extract goal tiles and their required colors.
        # Goals are a frozenset of facts like '(painted tile_1_1 white)'
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Static facts are not directly used for distance calculation,
        # as we assume a grid structure from tile names.
        # self.static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Check for dead ends first (tile painted wrong color)
        for tile, required_color in self.goal_tiles.items():
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == 'painted' and len(parts) == 3 and parts[1] == tile:
                    painted_color = parts[2]
                    if painted_color != required_color:
                        # Tile is painted with the wrong color - unreachable goal
                        return float('inf')
                    # If painted with the correct color, the goal for this tile is met, no dead end.
                    break # Found painted status for this tile

        total_cost = 0  # Initialize action cost counter.

        # Find robot locations and colors
        robots_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robots_info:
                    robots_info[robot] = {'location': None, 'color': None}
                robots_info[robot]['location'] = location
            elif parts and parts[0] == 'robot-has' and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot not in robots_info:
                    robots_info[robot] = {'location': None, 'color': None}
                 robots_info[robot]['color'] = color

        # Calculate cost for each unmet goal tile
        for tile, required_color in self.goal_tiles.items():
            # Check if this specific goal (painted tile with required_color) is met
            goal_met_for_tile = False
            for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'painted' and len(parts) == 3 and parts[1] == tile and parts[2] == required_color:
                     goal_met_for_tile = True
                     break

            if goal_met_for_tile:
                continue # This goal is already satisfied

            # If not met, calculate the minimum cost to paint it
            min_robot_cost_for_tile = float('inf')

            if not robots_info:
                 # No robots available to paint any remaining goal tiles
                 return float('inf')

            for robot_name, info in robots_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Should not happen in valid states, but handle defensively
                    continue

                cost_for_this_robot = 0

                # Cost to change color if needed
                if robot_color != required_color:
                    cost_for_this_robot += 1 # change_color action

                # Cost for movement and painting
                dist_to_tile = manhattan_distance(robot_location, tile)

                if dist_to_tile == float('inf'):
                     # Cannot calculate distance (e.g., invalid tile name)
                     cost_for_this_robot = float('inf') # This robot cannot reach
                else:
                    # Robot needs to be at a tile adjacent to 'tile' to paint it.
                    # Minimum moves to get from robot_location to a tile adjacent to 'tile'.
                    # If robot is at tile (dist=0), it must move off (1 action) to an adjacent tile,
                    # and then paint from that adjacent tile (1 action). Total moves+paint = 2.
                    # If robot is not at tile (dist>0), it needs `max(0, dist - 1)` moves to get adjacent,
                    # then 1 paint action. Total moves+paint = `max(0, dist - 1) + 1`.
                    if dist_to_tile == 0:
                         moves_and_paint_cost = 2
                    else:
                         moves_and_paint_cost = max(0, dist_to_tile - 1) + 1

                    cost_for_this_robot += moves_and_paint_cost

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_this_robot)

            if min_robot_cost_for_tile == float('inf'):
                 # If no robot could calculate a finite cost for this tile
                 return float('inf')

            total_cost += min_robot_cost_for_tile

        # If the loop finishes, all goal tiles are either met or were handled
        # by adding their minimum cost. If total_cost is 0, all goals were met.
        # If total_cost is > 0, there are unmet goals.
        # If any dead end was detected, float('inf') would have been returned earlier.

        return total_cost
