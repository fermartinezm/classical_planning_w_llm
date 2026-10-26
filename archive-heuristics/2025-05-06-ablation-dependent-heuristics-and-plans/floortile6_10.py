# Add necessary imports
from fnmatch import fnmatch # Although not strictly used in the final logic, included from example structure
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Helper function from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper functions for tile coordinates and distance
def parse_tile_name(tile_name):
    """Parses 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where tile name format is unexpected
            return None
    return None # Not a tile name in expected format

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        # Cannot calculate distance for invalid tile names.
        # This indicates an issue with the problem instance or state representation.
        # Return a large value to penalize states with unexpected tile names.
        # In a well-formed floortile instance, this case should not be reached for tiles.
        # However, the domain definition implies robot-at takes a tile, so robot_loc
        # should always be a tile name. Target_tile comes from goal facts, also tiles.
        return float('inf') # Should not happen with tile_r_c format

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

def movement_cost_to_be_adjacent(robot_loc, target_tile):
    """
    Calculates minimum moves for robot at robot_loc to reach a tile adjacent to target_tile.
    - If robot is at target_tile (Manhattan distance 0), needs 1 move to an adjacent tile
      to make the target tile clear and be in a position to paint.
    - If robot is adjacent to target_tile (Manhattan distance 1), needs 0 moves.
    - If robot is further (Manhattan distance > 1), needs Manhattan distance - 1 moves.
    """
    d = manhattan_distance(robot_loc, target_tile)
    if d == 0:
        return 1
    elif d == 1:
        return 0
    else: # d > 1
        return d - 1

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums three components: the number of tiles needing
    painting, the number of required colors not currently held by any robot, and
    an estimate of the total movement cost for robots to get into position. States
    where a goal tile is painted with the wrong color are considered unsolvable
    with the given actions and assigned infinite cost.

    # Assumptions
    - Tiles are arranged in a grid structure where adjacency corresponds to
      Manhattan distance 1, and tile names follow the 'tile_row_col' format.
    - Robot movement cost between adjacent tiles is 1.
    - Color change cost is 1.
    - Painting a tile costs 1 action and requires the robot to be at a tile
      adjacent to the target tile, with the correct color, and the target tile
      to be clear.
    - The domain does not include actions to unpaint a tile.

    # Heuristic Initialization
    - Extracts the goal conditions from the task to identify which tiles need
      to be painted and with which colors, storing them in `self.goal_painted_tiles`.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is calculated as the sum of three
    estimated costs:

    1.  **Painting Cost (h_paint):** This is the number of goal tiles that are
        not yet painted with their required color according to the goal. Each such
        tile requires at least one `paint_...` action. This component is simply
        the count of such tiles.

    2.  **Color Acquisition Cost (h_color_change):** This estimates the cost
        related to robots needing to have the correct colors. It counts the number
        of distinct colors required by the unpainted goal tiles that are *not*
        currently held by any robot. Each such missing color must be acquired
        by at least one robot via a `change_color` action (cost 1). This is a
        lower bound on color changes needed just to make the required colors
        available among the robots.

    3.  **Movement Cost (h_movement):** This estimates the total movement needed
        to bring robots into positions where they can paint the unpainted goal
        tiles. For each goal tile `T` that needs painting, a robot must move to
        a tile adjacent to it. We estimate the minimum number of moves required
        for *any* robot `R` to reach a tile adjacent to `T`. This minimum movement
        cost for a single robot `R` to get adjacent to `T` is determined by the
        Manhattan distance `d` between `R`'s location and `T`:
        - If `d == 0` (robot is at `T`), it needs 1 move to get off `T` and onto an adjacent tile.
        - If `d == 1` (robot is already adjacent to `T`), it needs 0 moves.
        - If `d > 1`, it needs `d - 1` moves to reach an adjacent tile.
        The `h_movement` component is the sum of these minimum movement costs,
        calculated independently for each unpainted goal tile. This component
        overestimates if a single robot movement helps satisfy the adjacency
        requirement for multiple unpainted tiles.

    Additionally, if any goal tile is found to be painted with a color different
    from its required goal color, the heuristic returns infinity, as this state
    is considered unsolvable with the available actions.

    The total heuristic value is `h_paint + h_color_change + h_movement`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Extract goal painted tiles: {tile_name: color_name}
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify current state of painted tiles and robot states
        current_painted_tiles = {}
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                current_painted_tiles[parts[1]] = parts[2]
            elif parts[0] == "robot-at":
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]

        robots = list(robot_locations.keys()) # Get list of robot names

        # Identify unpainted goal tiles and check for wrong colors
        unpainted_goal_tiles = {} # {tile: required_color}
        required_colors = set()

        for goal_tile, goal_color in self.goal_painted_tiles.items():
            current_color = current_painted_tiles.get(goal_tile)

            if current_color is None:
                 # Tile is not painted, needs painting
                 unpainted_goal_tiles[goal_tile] = goal_color
                 required_colors.add(goal_color)
            elif current_color != goal_color:
                 # Tile is painted with the wrong color. Unsolvable state with these actions.
                 return float('inf')
            # Else: current_color == goal_color, tile is painted correctly.

        # If there are no unpainted goal tiles, the goal is reached.
        if not unpainted_goal_tiles:
            return 0

        # 3. Calculate heuristic components

        # Component 1: Painting Cost
        h_paint = len(unpainted_goal_tiles)

        # Component 2: Color Acquisition Cost
        current_robot_colors = set(robot_colors.values())
        # Count how many required colors are not currently held by any robot
        missing_colors_count = len(required_colors - current_robot_colors)
        h_color_change = missing_colors_count

        # Component 3: Movement Cost
        h_movement = 0
        for target_tile, required_color in unpainted_goal_tiles.items():
            min_moves_to_adjacent = float('inf')
            for robot in robots:
                robot_loc = robot_locations[robot]
                # Calculate moves for this robot to get adjacent to target_tile
                moves = movement_cost_to_be_adjacent(robot_loc, target_tile)
                min_moves_to_adjacent = min(min_moves_to_adjacent, moves)
            if min_moves_to_adjacent != float('inf'): # Should always find a path in connected grid
                 h_movement += min_moves_to_adjacent
            # Note: This sums the minimum movement cost for *each* tile independently,
            # which overestimates if one movement helps paint multiple tiles.

        # Total heuristic is the sum of components
        total_heuristic = h_paint + h_color_change + h_movement

        return total_heuristic
