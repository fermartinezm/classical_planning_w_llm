from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For infinity

# Define helper functions outside the class if they are general utilities
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and inner spaces in arguments if any (though unlikely in floortile)
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated minimum cost for each unpainted
    goal tile independently, considering the cost of changing color and moving
    the nearest robot to an adjacent tile.

    # Assumptions
    - Tiles are arranged in a grid, and tile names like 'tile_R_C' indicate
      row R and column C, allowing Manhattan distance calculation.
    - The grid is connected for solvable problems.
    - Robots must have a color specified by the `robot-has` predicate to use
      the `change_color` action and subsequently paint. Robots without `robot-has`
      are assumed unable to paint.
    - Tiles painted with the wrong color cannot be repainted (as paint requires `clear`).
      Such states are considered dead ends.
    - The cost of each action (move, paint, change_color) is 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.
    - Extracts static facts (`up`, `down`, `left`, `right`) to build a map
      of adjacent tiles for each tile.
    - Parses tile names to extract grid coordinates for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that are specified in the goal state with a target color.
       Store these as `goal_painted = {(tile, color), ...}`.
    2. Identify the set of tiles that *need* painting in the current state.
       A tile `T` needs painting with color `C_goal` if `(T, C_goal)` is in `goal_painted`
       but `(painted T C_goal)` is not in the current state.
       Let this set be `needs_paint = {(tile, color), ...}`.
    3. Check for dead ends: For every tile `T` that is in `goal_painted` with color `C_goal`,
       check if the current state contains `(painted T C_wrong)` where `C_wrong != C_goal`.
       If such a fact exists, the tile is painted incorrectly and cannot be repainted.
       Return a very large value (infinity) as the state is likely unsolvable.
    4. If `needs_paint` is empty, the goal is reached (for painted tiles), return 0.
    5. Extract the current location of each robot: `robot_locations = {robot: tile, ...}`.
    6. Extract the current color held by each robot: `robot_colors = {robot: color, ...}`.
       Also, identify all robots present in the state.
    7. If there are no robots but tiles need painting, return infinity.
    8. Initialize `total_heuristic = 0`.
    9. For each `(tile, goal_color)` in `needs_paint`:
       - Calculate the minimum cost for *any* robot that *can* paint this specific tile
         with the required color.
       - Initialize `min_robot_cost_for_tile = infinity`.
       - For each robot `R` in the state:
         - Get the robot's current location `R_loc`. If unknown (`None`), this robot cannot paint this tile.
         - Get the robot's current color `R_color`. If the robot has no `robot-has` predicate (`None`),
           assume it cannot paint using `change_color`.
         - If `R_loc is not None and R_color is not None`: # Robot has location and color
           # Check if robot location has valid coordinates. If not, this robot cannot paint this tile.
           if r_loc in self.tile_coords:
             r_loc_coords = self.tile_coords[r_loc]

             # Find minimum distance from robot to any adjacent tile
             current_min_dist = float('inf')
             can_reach_adj = False
             adjacent_tiles_of_target = self.adjacent_tiles.get(tile, []) # Use .get for safety

             if adjacent_tiles_of_target:
                for adj_tile in adjacent_tiles_of_target:
                    if adj_tile in self.tile_coords: # Ensure adjacent tile has coords
                        adj_coords = self.tile_coords[adj_tile]
                        dist = abs(r_loc_coords[0] - adj_coords[0]) + abs(r_loc_coords[1] - adj_coords[1])
                        current_min_dist = min(current_min_dist, dist)
                        can_reach_adj = True

             if can_reach_adj:
                min_dist_to_adj = current_min_dist
                cost_for_r = min_dist_to_adj  # Movement cost

                # Add color change cost if needed
                if r_color != goal_color:
                     cost_for_r += 1 # Change color action

                cost_for_r += 1 # Paint action

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_r)
             # else: Target tile has no reachable adjacent tiles with coordinates, this robot cannot paint it.
           # else: Robot location has no coordinates, this robot cannot paint it.
         # else: Robot has no location or no color, cannot paint this tile.

       - If `min_robot_cost_for_tile` is still infinity after checking all robots,
         it means no robot could paint the tile. Return infinity.
       - Add `min_robot_cost_for_tile` to `total_heuristic`.

        return total_heuristic
