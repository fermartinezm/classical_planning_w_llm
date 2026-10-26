from heuristics.heuristic_base import Heuristic
# No other imports like fnmatch are strictly needed for this implementation.

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to parse tile name into row and column
def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            pass # Not a standard tile name format
    return None, None # Indicate parsing failure

# Helper function to calculate Manhattan distance between two tiles
def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    r1, c1 = parse_tile_name(tile1_name)
    r2, c2 = parse_tile_name(tile2_name)
    if r1 is not None and r2 is not None:
        return abs(r1 - r2) + abs(c1 - c2)
    return float('inf') # Cannot calculate distance if parsing fails

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The number of tiles that still need to be painted correctly.
    2. The estimated number of color changes required.
    3. The estimated movement cost to reach the vicinity of the nearest unpainted tile.

    # Assumptions
    - Tiles are arranged in a grid and named 'tile_row_col'.
    - Manhattan distance is a reasonable estimate for movement cost on the grid.
    - Tiles that need painting are initially 'clear' or painted incorrectly. The domain implies
      painting requires a 'clear' tile, so we assume tiles needing painting are currently clear.
      If a tile is painted with the wrong color, it cannot be repainted according to the domain,
      making the state likely unsolvable (heuristic returns infinity or a large value implicitly
      if such a tile is a goal). We assume valid instances where tiles needing painting are clear.
    - The robot always holds a color.

    # Heuristic Initialization
    - Extract the goal conditions, specifically the required color for each tile that must be painted.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of tiles that are specified in the goal as `(painted tile color)` but are not currently in the state as `(painted tile color)`. Let this set be `UnpaintedGoalTiles`.
    2. If `UnpaintedGoalTiles` is empty, the state is a goal state, and the heuristic is 0.
    3. Count the number of tiles in `UnpaintedGoalTiles`. This count represents a lower bound on the number of paint actions required. Add this count to the total heuristic value.
    4. Identify the set of distinct colors required for the tiles in `UnpaintedGoalTiles`. Let this be `NeededColors`.
    5. Determine the robot's current location (`RobotTile`) and the color it currently holds (`RobotColor`) by examining the state facts.
    6. Estimate the color change cost: The robot needs to acquire each color in `NeededColors` at least once. If the robot's current color (`RobotColor`) is one of the `NeededColors`, it effectively already has one needed color. The estimated number of color changes is `len(NeededColors)` if `RobotColor` is not in `NeededColors`, and `len(NeededColors) - 1` if `RobotColor` is in `NeededColors`. Add this estimated cost to the total heuristic value. Ensure the cost is non-negative.
    7. Estimate the movement cost: Calculate the Manhattan distance from the robot's current tile (`RobotTile`) to each tile `t` in `UnpaintedGoalTiles`. Find the minimum of these distances. This minimum distance is a simple estimate of the initial movement required to get close to *any* remaining painting task. Add this minimum distance to the total heuristic value. Handle cases where tile names cannot be parsed.
    8. The total heuristic value is the sum of the counts/estimates from steps 3, 6, and 7.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals  # Goal conditions.

        # Store goal colors for each tile that needs to be painted.
        # Mapping: tile_name -> goal_color
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles
        unpainted_goal_tiles = {} # tile_name -> goal_color
        current_painted_facts = {fact for fact in state if get_parts(fact)[0] == "painted"}
        current_clear_facts = {fact for fact in state if get_parts(fact)[0] == "clear"}

        for tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is NOT painted with the goal color in the current state
            goal_fact = f"(painted {tile} {goal_color})"
            if goal_fact not in current_painted_facts:
                 # According to the domain, painting requires the tile to be clear.
                 # If the tile is painted with the wrong color, it cannot be painted.
                 # We assume valid instances don't require repainting wrong colors.
                 # So, if it's not painted correctly, it must be clear and needs painting.
                 # We can optionally check if it's clear, but for valid instances,
                 # if it's not painted correctly and is a goal tile, it must be clear.
                 # Let's add a check for robustness, though it might not be strictly needed
                 # for valid problem instances. If it's not clear and not painted correctly,
                 # it's an unsolvable part of the goal.
                 if f"(clear {tile})" in current_clear_facts:
                     unpainted_goal_tiles[tile] = goal_color
                 else:
                     # Tile is not clear and not painted with the goal color.
                     # This state might be unsolvable or requires actions not modeled (like unpainting).
                     # Return a large heuristic value to indicate difficulty/impossibility.
                     # print(f"Warning: Tile {tile} needs {goal_color} but is not clear and not painted correctly.")
                     return float('inf')


        # 2. If no tiles need painting, goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 3. Count unpainted tiles (proxy for paint actions)
        num_unpainted_tiles = len(unpainted_goal_tiles)

        # 4. Identify needed colors
        needed_colors = set(unpainted_goal_tiles.values())

        # 5. Find robot location and color
        robot_tile = None
        robot_color = None
        # Assuming one robot named 'robot1' as is standard in floortile benchmarks
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at" and parts[1] == "robot1":
                robot_tile = parts[2]
            elif parts[0] == "robot-has" and parts[1] == "robot1":
                 robot_color = parts[2]

        if robot_tile is None or robot_color is None:
             # Should not happen in a valid state, but handle defensively
             # print("Warning: Robot location or color not found in state.")
             return float('inf') # Indicate invalid or unsolvable state

        # 6. Estimate color change cost
        # Number of distinct colors needed. If robot already has one, maybe one less change needed.
        color_change_cost = len(needed_colors)
        if robot_color in needed_colors:
             color_change_cost -= 1
        # Ensure cost is not negative
        color_change_cost = max(0, color_change_cost)


        # 7. Estimate movement cost
        # Minimum Manhattan distance from robot to any unpainted tile
        min_move_cost = float('inf')
        robot_row, robot_col = parse_tile_name(robot_tile)

        if robot_row is not None: # Proceed only if robot tile name is parsable
            for tile_name in unpainted_goal_tiles.keys():
                tile_row, tile_col = parse_tile_name(tile_name)
                if tile_row is not None: # Proceed only if unpainted tile name is parsable
                    dist = abs(robot_row - tile_row) + abs(robot_col - tile_col)
                    # Use the distance to the tile itself as a simple proxy for movement effort.
                    # The robot needs to get *adjacent* to paint, but this simple distance
                    # is a reasonable non-admissible estimate of how "far" the closest task is.
                    min_move_cost = min(min_move_cost, dist)

        # If min_move_cost is still inf, it means either robot tile or all unpainted tiles
        # had unparsable names. This shouldn't happen with standard floortile instances
        # and unpainted_goal_tiles was not empty.
        if min_move_cost == float('inf') and unpainted_goal_tiles:
             # print("Warning: Could not calculate minimum movement cost.")
             return float('inf') # Indicate potential issue or unsolvable part

        # If unpainted_goal_tiles was empty, we returned 0 already.
        # If it was not empty, min_move_cost should be finite unless parsing failed.
        # If parsing failed, we returned inf. So if we reach here, min_move_cost is finite.


        # 8. Sum components
        total_cost = num_unpainted_tiles + color_change_cost + min_move_cost

        return total_cost
