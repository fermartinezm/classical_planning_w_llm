from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(robot-at robot1 tile_0_4)" -> ["robot-at", "robot1", "tile_0_4"]
    # Example: "(clear tile_1_5)" -> ["clear", "tile_1_5"]
    # Example: "(painted tile_1_2 black)" -> ["painted", "tile_1_2", "black"]
    return fact[1:-1].split()

def get_coords(tile_name):
    """Extract (row, col) integer coordinates from tile name string like 'tile_R_C'."""
    # Example: "tile_0_1" -> (0, 1)
    try:
        parts = tile_name.split('_')
        # Expecting format 'tile_row_col'
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # This should not happen in valid floortile instances, but handle defensively
            print(f"Warning: Unexpected tile name format: {tile_name}")
            # Return dummy coordinates or raise error
            # Returning large coordinates will make its distance large, reflecting an issue
            return (-1000, -1000) # Or raise ValueError
    except (ValueError, IndexError) as e:
        print(f"Error parsing tile coordinates for {tile_name}: {e}")
        # Return dummy coordinates or raise error
        return (-1000, -1000) # Or raise ValueError


def dist(tile1_name, tile2_name):
    """Calculate Manhattan distance between two tiles."""
    r1, c1 = get_coords(tile1_name)
    r2, c2 = get_coords(tile2_name)
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted and clear. It calculates the minimum cost for each
    unpainted goal tile independently, considering the distance of each robot to the
    tile and the color the robot currently holds. The total heuristic is the sum
    of these minimum costs.

    # Assumptions
    - Tiles are arranged in a grid, and movement/painting is based on adjacency
      defined by up/down/left/right predicates, which correspond to grid
      movements. Manhattan distance is used as a proxy for move cost.
    - Goal tiles are either clear or correctly painted in the initial state.
      If a goal tile is found to be painted with the wrong color in any state,
      the heuristic returns infinity, assuming the goal is unreachable.
    - Each robot can paint any clear tile adjacent to it, provided it has the
      correct color.
    - The cost of changing color is 1 action.
    - The cost of moving between adjacent tiles is 1 action.
    - The cost of painting an adjacent tile is 1 action.
    - The heuristic sums the minimum cost for each unpainted goal tile independently,
      ignoring potential conflicts or shared costs (non-admissible).

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to identify:
       - The current location of each robot.
       - The color each robot currently holds.
       - Which tiles are currently painted and with what color.
       - Which tiles are currently clear.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal tile and its required color as extracted during initialization.
    4. For the current goal tile:
       a. Check its status in the current state:
          - If it is already painted with the correct color, it contributes 0 to the heuristic.
          - If it is painted with a *different* color, the goal is assumed unreachable from this state; return infinity.
          - If it is clear, it needs to be painted.
       b. If the goal tile is clear, calculate the minimum cost for *any* robot to paint it:
          - Initialize minimum cost for this tile to infinity.
          - For each robot:
             - Get the robot's current location and color.
             - Calculate the Manhattan distance from the robot's location to the goal tile's location.
             - Calculate the number of moves required for the robot to reach a tile adjacent to the goal tile. This is 1 if the robot is currently at the goal tile (needs to move away then paint), and `distance - 1` if the robot is further away (distance > 0).
             - Calculate the cost to change the robot's color if it doesn't hold the required color (1 if different, 0 if same).
             - The total cost for this robot to paint this tile is `moves_to_adjacent + color_change_cost + paint_action_cost` (where paint_action_cost is 1).
             - Update the minimum cost for this tile with the minimum cost found among all robots.
          - Add the minimum cost for this tile to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings)
        # Static facts are not directly used in the heuristic calculation per state,
        # but implicitly define the grid structure used by the distance function
        # through the tile naming convention.

        # Store goal locations and required colors for each tile.
        # Format: {tile_name: required_color}
        self.goal_tiles = {}
        for goal_fact_str in self.goals:
            # Assuming goal facts are of the form '(painted tile_X_Y color)'
            parts = get_parts(goal_fact_str)
            if len(parts) == 3 and parts[0] == 'painted':
                 tile_name, color_name = parts[1], parts[2]
                 self.goal_tiles[tile_name] = color_name
            # Ignore other potential goal predicates if any

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Parse the current state
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        painted_tiles = {}   # {tile_name: color_name}
        clear_tiles = set()  # {tile_name}

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                robot_locations[robot_name] = tile_name
            elif predicate == 'robot-has' and len(parts) == 3:
                robot_name, color_name = parts[1], parts[2]
                robot_colors[robot_name] = color_name
            elif predicate == 'painted' and len(parts) == 3:
                tile_name, color_name = parts[1], parts[2]
                painted_tiles[tile_name] = color_name
            elif predicate == 'clear' and len(parts) == 2:
                tile_name = parts[1]
                clear_tiles.add(tile_name)
            # Ignore other predicates like up, down, left, right, available-color, free-color

        total_cost = 0

        # Iterate through goal tiles and calculate cost for unsatisfied ones
        for tile_Y, color_C in self.goal_tiles.items():
            if tile_Y in painted_tiles:
                # Check if it's painted with the correct color
                if painted_tiles[tile_Y] != color_C:
                    # Wrongly painted goal tile - likely unreachable
                    return math.inf # Use math.inf for infinity
                else:
                    # Correctly painted - this goal is satisfied
                    continue # Cost is 0 for this tile

            elif tile_Y in clear_tiles:
                # This tile needs to be painted with color_C
                min_robot_cost_for_tile = math.inf # Use math.inf

                # Calculate the minimum cost for any robot to paint this tile
                for robot_r, robot_loc in robot_locations.items():
                    robot_color = robot_colors.get(robot_r) # Use .get for safety

                    # Calculate Manhattan distance
                    d = dist(robot_loc, tile_Y)

                    # Moves needed to reach a tile adjacent to tile_Y
                    # If robot is at tile_Y (d=0), needs 1 move away.
                    # If robot is adjacent (d=1), needs 0 moves.
                    # If robot is further (d>1), needs d-1 moves.
                    moves_to_adjacent = (1 if d == 0 else max(0, d - 1))

                    # Cost to change color if needed
                    color_change_cost = (1 if robot_color != color_C else 0)

                    # Total cost for this robot to paint this tile
                    # = moves to adjacent + color change cost + paint action cost (1)
                    cost_for_this_robot = moves_to_adjacent + color_change_cost + 1

                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_this_robot)

                # Add the minimum cost for this tile to the total heuristic
                # If there are no robots, min_robot_cost_for_tile remains inf,
                # which is correct as the goal is unreachable.
                total_cost += min_robot_cost_for_tile

            # If tile_Y is neither painted nor clear, it's an unexpected state.
            # Based on domain, tiles are either clear or painted.
            # If a goal tile is missing from state predicates, it's likely an issue.
            # We assume goal tiles are always represented either as clear or painted.
            # If it's not in goal_tiles, we ignore it (it's not a goal).
            # If it's in goal_tiles but not in painted_tiles or clear_tiles, something is wrong.
            # For robustness, we could return inf, but let's assume valid states.


        # Heuristic is 0 if and only if all goal tiles are correctly painted.
        # Our loop structure ensures this:
        # - If all goal tiles are correctly painted, they are skipped by the first 'if' condition.
        # - No tiles are in 'clear_tiles' that are also goal tiles.
        # - total_cost remains 0.
        # If any goal tile is not correctly painted:
        # - If wrongly painted, returns inf.
        # - If clear, min_robot_cost_for_tile will be >= 1, so total_cost will be > 0.

        return total_cost

