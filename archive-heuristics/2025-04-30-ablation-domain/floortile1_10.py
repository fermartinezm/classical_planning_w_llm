from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where tile name format is unexpected
            return None
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue with tile naming convention or parsing.
        # Return infinity to discourage paths involving invalid tiles.
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated cost for each unmet goal tile
    independently, considering the tile's current state (clear or wrongly painted),
    the cost for a robot to reach the tile, and the cost for a robot to have the
    correct color.

    # Assumptions
    - Each goal specifies a tile and the required color it must be painted with.
    - Tiles are arranged in a grid, and movement cost between adjacent tiles is 1.
      Manhattan distance is used as a lower bound estimate for movement.
    - Robots can change color at any tile (cost 1).
    - A tile painted with the wrong color must be cleared before being repainted (cost 1 for clear, 1 for paint).
    - A clear tile needs only painting (cost 1 for paint).
    - The heuristic assumes the best robot (closest with minimal color change) is used for each individual goal tile, ignoring resource contention or optimal path planning for multiple tiles/robots.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify the current location and color of each robot from the state facts. Store these in dictionaries (robot_locations, robot_colors).
    3. For each tile and its required color specified in the task's goals (stored in self.goal_conditions):
       a. Check if the tile is already painted with the required color in the current state. This is done by checking for the presence of the fact `(painted tile required_color)`. If this fact exists, this goal is met, and we continue to the next goal tile.
       b. If the goal is not met, determine the base cost for painting/clearing the tile:
          - Check if the tile is currently painted with *any* color (i.e., if any fact `(painted tile any_color)` exists in the state). If it is painted, it must be painted with the wrong color (since the correct color check failed). The base cost is 2 (1 for the `clear` action, 1 for the `paint` action).
          - If the tile is not painted at all (i.e., no fact `(painted tile any_color)` exists), it is assumed to be clear. The base cost is 1 (1 for the `paint` action).
       c. Estimate the minimum cost for a robot to get into a position to paint this tile:
          - Initialize `min_robot_cost` to infinity.
          - For each robot (identified in step 2):
             - Get the robot's current location and color.
             - Calculate the Manhattan distance from the robot's current location to the goal tile's location using the `manhattan_distance` helper function.
             - Determine the color change cost: Add 1 if the robot's current color is different from the required color for the tile, otherwise add 0.
             - The cost for this specific robot to paint this tile is the calculated distance plus the color change cost.
             - Update `min_robot_cost` with the minimum of the current `min_robot_cost` and the cost for this robot.
       d. Add the base tile cost (from step 3b) and the `min_robot_cost` (from step 3c) for this tile to the total heuristic cost.
    4. Return the final `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions: map tile name to required color
        self.goal_conditions = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_conditions[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Extract robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]

        # Iterate through each goal tile and its required color
        for tile, required_color in self.goal_conditions.items():
            # Check if the goal is already met for this tile
            if f"(painted {tile} {required_color})" in state:
                continue # This tile is already painted correctly

            # Goal is not met. Determine the base cost for actions at the tile.
            tile_cost = 0
            is_painted_at_all = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'painted' and parts[1] == tile:
                    is_painted_at_all = True
                    break # Found painted status

            if is_painted_at_all:
                tile_cost += 2 # Needs clear (1) + paint (1)
            else:
                tile_cost += 1 # Needs paint (1) - implies it's clear if not painted

            # Estimate the minimum cost for a robot to reach this tile and have the right color
            min_robot_cost = float('inf')
            
            # If there are no robots, this goal is unreachable (or requires creating a robot, which is not standard PDDL)
            # For solvable problems, there will be robots.
            if not robot_locations:
                 # If no robots, this goal cannot be achieved. Heuristic should reflect this,
                 # but for solvable problems, this branch might not be strictly necessary.
                 # Returning infinity would indicate unsolvability from this state,
                 # but the problem implies solvable instances. Let's assume robots exist.
                 pass # Should not happen in valid solvable states

            for robot_name, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get robot's current color

                dist = manhattan_distance(robot_loc, tile)

                color_change_cost = 0
                if robot_color != required_color:
                    color_change_cost = 1 # Cost to change color

                current_robot_cost = dist + color_change_cost
                min_robot_cost = min(min_robot_cost, current_robot_cost)

            # Add the cost for this unmet goal tile to the total
            # If min_robot_cost is still inf (e.g., no robots), adding it keeps total_cost as inf
            total_cost += tile_cost + min_robot_cost

        return total_cost
