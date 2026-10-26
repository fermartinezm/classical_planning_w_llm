# Assuming heuristics.heuristic_base exists and defines a Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper functions specific to this domain/heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            # Handle cases where the object is not a valid tile name format
            return None, None
    # Handle cases where the object name doesn't follow the tile_x_y format
    return None, None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    r1, c1 = parse_tile_name(tile1_name)
    r2, c2 = parse_tile_name(tile2_name)
    if r1 is not None and r2 is not None:
        return abs(r1 - r2) + abs(c1 - c2)
    # If either is not a valid tile name (e.g., robot location is not a tile, which shouldn't happen),
    # return a large value indicating it's not a valid path.
    return float('inf')


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unpainted goal tile
    independently. The cost for a single tile includes the paint action itself,
    the cost to move a robot to the tile, and the cost for that robot to acquire
    the correct color.

    # Assumptions
    - Each unpainted goal tile must be painted by a robot.
    - The cost to paint a tile is 1 action.
    - The cost to move between adjacent tiles is 1 action. Manhattan distance is used
      as an estimate for movement cost, assuming a regular grid structure based on
      tile naming (tile_row_col).
    - The cost to pick up a color is 1 action.
    - The cost to drop a color is 1 action.
    - Colors are assumed to be available for pickup if needed (ignoring the state of
      `available-color` and who holds which color beyond the robot needing it).
    - The heuristic for a single tile considers either using a robot already at the tile
      or moving the closest robot to the tile, and adds the cost for that robot to
      acquire the correct color. It picks the minimum of these options.
    - The total heuristic is the sum of costs for each unpainted goal tile, ignoring
      potential synergies (like one robot painting multiple tiles, or multiple robots
      cooperating efficiently) and negative interactions (like robots blocking each other).

    # Heuristic Initialization
    - Extracts the goal conditions (`(painted tile color)`) from the task to identify
      which tiles need to be painted and with which color. Stores this in `self.goal_paintings`.
    - Static facts are not explicitly parsed into separate data structures beyond
      what the base `Task` class provides, as the Manhattan distance calculation
      relies directly on tile name parsing.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost `total_cost` to 0.
    2. Identify the current location and held color for each robot by iterating
       through the facts in the current state. Store these in `robot_locs` and `robot_colors` dictionaries.
    3. Iterate through each goal tile and its required color stored in `self.goal_paintings`.
    4. For the current goal tile `T` and required color `C`:
       a. Check if the fact `(painted T C)` is present in the current state.
       b. If `(painted T C)` is present, the goal for this tile is met. Do nothing and proceed to the next goal tile.
       c. If `(painted T C)` is *not* present:
          i. This tile needs painting. Add 1 to `total_cost` (representing the `paint` action).
          ii. Calculate the minimum cost required to get a robot to tile `T` with color `C`, ready to paint. Initialize `min_prep_cost` to infinity.
          iii. Consider Scenario A: Using a robot already at tile `T`.
              - Iterate through all robots. If a robot `R` is at `T`:
                  - Calculate the cost for `R` to acquire color `C`:
                      - If `R` has a different color, cost is 1 (drop) + 1 (pickup) = 2.
                      - If `R` has no color, cost is 1 (pickup).
                      - If `R` already has color `C`, cost is 0.
                  - Update `min_prep_cost` with the minimum acquisition cost found among all robots at `T`.
          iv. Consider Scenario B: Moving the closest robot to tile `T`.
              - Find the robot `R_closest` that is closest to `T` using `manhattan_distance`.
              - If a robot is found:
                  - Calculate the cost to move `R_closest` to `T` (`move_cost = manhattan_distance(R_closest_loc, T)`).
                  - Calculate the cost for `R_closest` to acquire color `C` *after* reaching `T` (same logic as step c.iii, but applied to `R_closest`). Let this be `acquisition_cost`.
                  - The total cost for Scenario B is `move_cost + acquisition_cost`.
                  - Update `min_prep_cost` with the minimum of its current value and the cost for Scenario B.
          v. If `min_prep_cost` is still infinity (meaning no robots exist or can reach the tile), the problem is likely unsolvable from this state. Add `min_prep_cost` to `total_cost`. Note that adding infinity results in infinity.
    5. After iterating through all goal tiles, return the final `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Extract goal conditions: {tile: color} mapping for tiles that need painting.
        self.goal_paintings = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Static facts are available in task.static but not explicitly
        # parsed into separate data structures for this heuristic, as
        # Manhattan distance relies on tile name parsing.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Extract current robot locations and held colors
        robot_locs = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, loc = parts[1], parts[2]
                robot_locs[robot] = loc
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_cost = 0

        # Iterate through each tile that needs to be painted according to the goal
        for goal_tile, goal_color in self.goal_paintings.items():
            # Check if the tile is already painted with the correct color in the current state
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state

            if not is_painted_correctly:
                # This tile needs painting. Add cost for the paint action.
                total_cost += 1

                # Calculate the minimum cost to get a robot to this tile with the correct color.
                min_prep_cost = float('inf')

                # Scenario A: Use a robot already at the goal tile.
                robot_at_goal_tile_found = False
                for robot, loc in robot_locs.items():
                    if loc == goal_tile:
                        robot_at_goal_tile_found = True
                        # Calculate cost for this robot to get the correct color
                        cost_to_get_color = 0
                        if robot in robot_colors and robot_colors[robot] != goal_color:
                            cost_to_get_color += 1 # drop wrong color
                        # If robot doesn't have the color, it needs to pick it up
                        if robot not in robot_colors or robot_colors[robot] != goal_color:
                             cost_to_get_color += 1 # pickup correct color
                        min_prep_cost = min(min_prep_cost, cost_to_get_color)

                # Scenario B: Move the closest robot to the goal tile.
                min_move_cost_to_tile = float('inf')
                closest_robot = None
                for robot, loc in robot_locs.items():
                    dist = manhattan_distance(loc, goal_tile)
                    if dist < min_move_cost_to_tile:
                        min_move_cost_to_tile = dist
                        closest_robot = robot

                cost_if_move_robot = float('inf')
                if closest_robot:
                    # Cost to move the robot
                    cost_if_move_robot = min_move_cost_to_tile
                    # Cost for this robot to get the correct color *after* moving
                    cost_to_get_color_at_tile = 0
                    if closest_robot in robot_colors and robot_colors[closest_robot] != goal_color:
                        cost_to_get_color_at_tile += 1 # drop wrong color
                    # If robot doesn't have the color, it needs to pick it up
                    if closest_robot not in robot_colors or robot_colors[closest_robot] != goal_color:
                        cost_to_get_color_at_tile += 1 # pickup correct color
                    cost_if_move_robot += cost_to_get_color_at_tile

                # Choose the minimum preparation cost between the two scenarios
                # If Scenario A was never possible (no robot at tile), min_prep_cost remains inf from init.
                # If Scenario B was never possible (no robots at all), cost_if_move_robot remains inf.
                min_prep_cost = min(min_prep_cost, cost_if_move_robot)

                # Add the minimum preparation cost to the total cost
                # This will be inf if no robots exist or can reach the tile, correctly reflecting unsolvability.
                if min_prep_cost != float('inf'):
                    total_cost += min_prep_cost
                else:
                    # If a goal tile is unreachable by any robot, the problem is unsolvable.
                    # Return infinity in this case.
                    return float('inf')


        # The heuristic is 0 only if total_cost is 0, which happens only if
        # the loop finds no unpainted goal tiles.
        return total_cost
