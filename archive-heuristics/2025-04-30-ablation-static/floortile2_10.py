# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for string format
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts
        return []

    # Remove parentheses and split by whitespace
    parts = fact[1:-1].split()
    return parts

def parse_tile_coords(tile_name):
    """Parse tile name 'tile_r_c' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        # Expecting format 'tile_row_col'
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Return None for unexpected tile name format
            return None
    except (ValueError, IndexError):
        # Return None for parsing errors (e.g., non-integer row/col)
        return None


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles that are not yet painted correctly. For each unpainted goal tile,
    it calculates the minimum cost for any robot to reach a tile adjacent to the
    goal tile, change color if necessary, and perform the paint action. The total
    heuristic value is the sum of these minimum costs over all unpainted goal tiles.

    # Assumptions
    - The grid structure is implied by tile names like 'tile_row_col'. The heuristic
      uses Manhattan distance based on these parsed coordinates as a proxy for
      movement cost, ignoring potential obstacles (non-clear tiles) for movement.
    - Robots always have a color (`robot-has`) and can change it if the target
      color is available. The `free-color` predicate is not considered.
    - Goal tiles that are not painted with the required color are assumed to be
      `clear` and thus paintable. The heuristic does not explicitly handle cases
      where a goal tile is painted with the wrong color (assuming such states
      do not occur in solvable instances or are handled by other search mechanisms).
    - The cost of each action (move, change_color, paint) is assumed to be 1.
    - The heuristic sums costs independently for each unpainted goal tile, ignoring
      robot coordination, conflicts, or the fact that a robot can only perform
      one action at a time.

    # Heuristic Initialization
    - The constructor extracts the goal conditions, specifically identifying which
      tiles need to be painted and with which color. This information is stored
      in `self.goal_tiles`. Static facts are not strictly needed for this version
      of the heuristic, as grid structure is inferred from tile names and distances
      are calculated using Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify all goal tiles that are not currently painted with the required color.
       Iterate through the stored `self.goal_tiles`. For each goal `(painted T C)`,
       check if this exact fact exists in the current state. If not, consider `T`
       as an unpainted goal tile needing color `C`.
    3. For each identified unpainted goal tile `T` needing color `C`:
       a. Parse the tile name `T` to get its coordinates `(r, c)`. If parsing fails,
          assign a large penalty for this tile and continue.
       b. Determine the minimum cost for *any* robot to paint this tile. Initialize
          `min_robot_cost_for_tile` to infinity.
       c. Extract current robot locations and colors from the state facts.
       d. If no robots are found, assign a large penalty for this tile and continue.
       e. Iterate through all robots found:
          i. Get the robot's current location `robot_tile` and current color `robot_color`.
          ii. Parse the robot's tile name `robot_tile` to get its coordinates `(rr, rc)`.
              If parsing fails, skip this robot for this tile.
          iii. Calculate the Manhattan distance `d = abs(rr - r) + abs(rc - c)` between
               the robot's location and the goal tile's location.
          iv. Calculate the estimated number of *move* actions required for the robot
              to reach a tile adjacent to the goal tile `T`.
              - If the robot is currently *at* the goal tile (`d == 0`), it needs 1 move
                to step off `T` (making it clear). After this move, the robot is at a tile
                adjacent to `T`.
              - If the robot is already adjacent to the goal tile (`d == 1`), it needs 0 moves.
              - If the robot is further away (`d > 1`), it needs `d - 1` moves to reach
                an adjacent tile using a Manhattan path.
              - Moves cost = 1 if `d == 0` else `d - 1`.
          v. Calculate the estimated cost to change the robot's color if needed.
              - Color cost = 1 if `robot_color` is not `C`, else 0.
          vi. The cost for this specific robot to paint tile `T` is:
              Moves cost + Color cost + 1 (for the paint action itself).
          vii. Update `min_robot_cost_for_tile = min(min_robot_cost_for_tile, current_robot_cost)`.
       f. If `min_robot_cost_for_tile` is still infinity (should not happen if robots were found),
          assign a large penalty for this tile. Otherwise, add `min_robot_cost_for_tile` to
          the total heuristic value `h`.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).

        # Store goal locations and required colors for each tile.
        # Map: tile_name -> required_color
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile_name = parts[1]
                color = parts[2]
                self.goal_tiles[tile_name] = color
            # Note: Other goal types are ignored by this heuristic.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Extract current robot locations and colors
        robot_locations = {} # Map: robot_name -> tile_name
        robot_colors = {}    # Map: robot_name -> color
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "robot-at" and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                robot_locations[robot_name] = tile_name
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_name, color = parts[1], parts[2]
                robot_colors[robot_name] = color

        total_cost = 0  # Initialize action cost counter.
        large_penalty = 1000 # Penalty for unparseable tiles or no robots

        # Iterate through all goal tiles
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if this goal tile is already painted correctly
            is_painted_correctly = f"(painted {goal_tile} {required_color})" in state

            if is_painted_correctly:
                 continue # This goal is met for this tile

            # If not painted correctly, it needs painting.
            # We assume it is clear if not painted correctly.
            # A more robust heuristic might check for (clear goal_tile)
            # and handle (painted goal_tile wrong_color) cases.

            # Tile needs painting. Calculate min cost for any robot.
            goal_r_c = parse_tile_coords(goal_tile)
            if goal_r_c is None:
                 # Cannot parse tile coordinates, assign large penalty for this tile
                 total_cost += large_penalty
                 continue

            min_robot_cost_for_tile = math.inf

            # Consider each robot
            if not robot_locations:
                 # No robots available to paint this tile
                 min_robot_cost_for_tile = large_penalty
            else:
                for robot_name, robot_tile in robot_locations.items():
                    robot_color = robot_colors.get(robot_name)

                    if robot_color is None:
                        # Robot doesn't have a color? Should not happen based on domain.
                        # Skip this robot for this tile.
                        continue

                    robot_rr_rc = parse_tile_coords(robot_tile)
                    if robot_rr_rc is None:
                        # Cannot parse robot tile coordinates, skip this robot
                        continue

                    # Calculate Manhattan distance
                    d = abs(robot_rr_rc[0] - goal_r_c[0]) + abs(robot_rr_rc[1] - goal_r_c[1])

                    # Calculate moves cost to get adjacent to the goal tile
                    # If robot is at the goal tile (d=0), needs 1 move off.
                    # If robot is adjacent (d=1), needs 0 moves.
                    # If robot is further (d>1), needs d-1 moves.
                    moves_cost = 1 if d == 0 else max(0, d - 1)

                    # Calculate color change cost
                    color_cost = 1 if robot_color != required_color else 0

                    # Total cost for this specific robot to paint tile T
                    # Moves to get adjacent + Color change + Paint action (cost 1)
                    current_robot_cost = moves_cost + color_cost + 1

                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, current_robot_cost)

            # Add the minimum cost for this tile to the total heuristic
            if min_robot_cost_for_tile != math.inf:
                 total_cost += min_robot_cost_for_tile
            else:
                 # This happens if robot_locations was empty, or if all robots
                 # had unparseable locations.
                 total_cost += large_penalty


        return total_cost
