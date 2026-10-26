# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_tile(tile_name):
    """Parses a tile name like 'tile_R_C' into a tuple (R, C)."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format gracefully
            return None
    except (ValueError, IndexError):
        # Handle cases where parts are not integers or indices are out of bounds
        return None

def manhattan_distance_coords(coords1, coords2):
    """Calculates the Manhattan distance between two coordinate tuples (R, C)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


# Ensure the base class is available
from heuristics.heuristic_base import Heuristic


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated cost
    for each goal tile that is not yet painted with the correct color. The estimated cost
    for a single unpainted goal tile is the minimum cost for any robot to paint that tile,
    considering the cost of changing color, moving to an adjacent tile, and the paint action itself.

    # Assumptions
    - All actions have a unit cost of 1.
    - Tiles in the goal that are not currently painted with the correct color are assumed to be clear.
    - The grid structure and adjacency are defined by the 'up', 'down', 'left', 'right' static predicates.
    - Tile names follow the format 'tile_R_C' where R and C are integers representing row and column.
    - Solvable problems do not require unpainting or dealing with tiles painted with the wrong color.
    - Movement cost is estimated using Manhattan distance on the grid, ignoring potential blockages by other objects/robots.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which colors.
    - Extracts static facts ('up', 'down', 'left', 'right') to build a mapping from a tile to the set of coordinates
      where a robot must be located to paint it (i.e., its adjacent tiles according to the painting actions).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted ?tile ?color)`. Store these target tiles and their required colors in `self.goal_tiles`.
    2. Pre-process static facts to create a mapping `self.painting_adj_coords` where `self.painting_adj_coords[T]` is the set of (row, col) coordinates of tiles `X`
       such that a robot at `X` can paint tile `T` (i.e., `(up T X)`, `(down T X)`, `(left T X)`, or `(right T X)` is true).
    3. In the `__call__` method, for the given state:
       a. Identify the current location name and coordinates, and the color held by each robot from the state facts. Store this information in `robot_info`.
       b. Initialize the total heuristic cost `h = 0`.
       c. Iterate through each target tile name `target_tile_name` and its required color `required_color` stored in `self.goal_tiles`.
       d. Check if the fact `(painted target_tile_name required_color)` is already true in the current state. If yes, this goal tile is satisfied; continue to the next goal tile.
       e. If `(painted target_tile_name required_color)` is not true, this tile needs painting. Calculate the minimum cost for *any* robot to paint this tile:
          i. Initialize `min_cost_for_tile = float('inf')`.
          ii. Get the set of required robot coordinates `required_robot_coords` from `self.painting_adj_coords.get(target_tile_name, set())`. If this set is empty, this tile cannot be painted from any known adjacent location; skip it (it might indicate an invalid or unsolvable state, but for greedy search, skipping is fine).
          iii. For each robot in the `robot_info` dictionary:
              - Get the robot's current location coordinates `robot_loc_coords` and color `robot_color`.
              - If robot location coordinates are not available (e.g., tile name unparsable) or color is missing, skip this robot for this tile calculation.
              - Calculate the estimated cost for this specific robot to paint tile `target_tile_name` with color `required_color`:
                - Base cost = 1 (for the paint action).
                - Color cost = 1 if `robot_color` is not equal to `required_color`, otherwise 0 (for the `change_color` action, assuming it's available).
                - Movement cost: Find the minimum Manhattan distance from the robot's current coordinates `robot_loc_coords` to any coordinate tuple in the set `required_robot_coords`.
                - Total robot cost = Base cost + Color cost + Movement cost.
              - Update `min_cost_for_tile = min(min_cost_for_tile, Total robot cost)`.
          iv. If `min_cost_for_tile` is not `infinity` (meaning at least one robot can potentially paint this tile), add `min_cost_for_tile` to the total heuristic cost `h`.
    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Store goal tiles and their required colors: {tile_name: color_name}
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Store painting adjacency: {tile_to_paint_name: set_of_robot_location_coords_to_paint_from}
        # This maps a tile that needs painting to the coordinates of the tiles where a robot must stand
        # to perform the paint action (e.g., the tile directly above, below, left, or right).
        self.painting_adj_coords = {}

        for fact in task.static:
            parts = get_parts(fact)
            if parts and len(parts) == 3:
                predicate, tile_y, tile_x = parts
                # Predicates like (up Y X) mean Y is above X. To paint Y, robot must be at X.
                # So, X is a required robot location to paint Y.
                if predicate in ["up", "down", "left", "right"]:
                    # tile_y is the tile being painted, tile_x is the robot location
                    coords_x = parse_tile(tile_x)
                    if coords_x is not None:
                        if tile_y not in self.painting_adj_coords:
                            self.painting_adj_coords[tile_y] = set()
                        self.painting_adj_coords[tile_y].add(coords_x)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get robot locations and colors from the current state
        robot_info = {} # {robot_name: {'loc_name': tile_name, 'loc_coords': (R, C), 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, loc_name = parts[1], parts[2]
                    loc_coords = parse_tile(loc_name)
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['loc_name'] = loc_name
                    robot_info[robot]['loc_coords'] = loc_coords
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color

        total_cost = 0

        # Iterate through goal tiles that need painting
        for target_tile_name, required_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly in the current state
            if f"(painted {target_tile_name} {required_color})" in state:
                continue # This goal is already satisfied

            # This tile needs painting. Calculate the minimum cost among all robots to paint it.
            min_cost_for_tile = float('inf')

            # Get the coordinates a robot needs to be at to paint this tile
            required_robot_coords = self.painting_adj_coords.get(target_tile_name, set())

            # If there are no known locations to paint this tile from (should not happen for goal tiles in valid problems)
            if not required_robot_coords:
                 # This tile cannot be painted based on static adjacency.
                 # It might indicate an unsolvable problem or a state that is a dead end.
                 # For a greedy heuristic, we can effectively assign infinite cost by skipping.
                 continue

            # Consider each robot and find the minimum cost for any robot to paint this tile
            for robot_name, info in robot_info.items():
                robot_loc_coords = info.get('loc_coords')
                robot_color = info.get('color')

                # Ensure robot has valid location coordinates and color information
                if robot_loc_coords is None or robot_color is None:
                    continue # Skip this robot if its information is incomplete

                # Calculate the estimated cost for this specific robot to paint the target tile
                current_robot_cost = 0

                # 1. Cost for the paint action itself
                current_robot_cost += 1

                # 2. Cost for color change (if the robot doesn't have the required color)
                if robot_color != required_color:
                    current_robot_cost += 1 # Assumes change_color action is available and costs 1

                # 3. Cost for movement to a painting location adjacent to the target tile
                min_dist_to_paint_location = float('inf')
                for required_coords in required_robot_coords:
                    dist = manhattan_distance_coords(robot_loc_coords, required_coords)
                    min_dist_to_paint_location = min(min_dist_to_paint_location, dist)

                # If a valid movement distance was found (i.e., required_robot_coords was not empty and coords were valid)
                if min_dist_to_paint_location != float('inf'):
                    current_robot_cost += min_dist_to_paint_location
                    # Update the minimum cost found among all robots for this tile
                    min_cost_for_tile = min(min_cost_for_tile, current_robot_cost)

            # Add the minimum cost found among all robots for this tile to the total heuristic cost
            if min_cost_for_tile != float('inf'):
                total_cost += min_cost_for_tile
            # Else: No robot could reach a painting location for this tile. It contributes effectively infinite cost.

        return total_cost
