from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

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

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted and clear. It sums the cost components for each
    such tile: the paint action itself, the cost to acquire the necessary color
    if no robot has it, and the estimated movement cost for the closest robot
    to reach the tile.

    # Assumptions
    - Tiles are arranged in a grid and named like 'tile_row_col'.
    - 'up', 'down', 'left', 'right' predicates define adjacency based on grid coordinates.
    - Goal tiles that are not yet painted are currently 'clear'. Tiles painted with
      the wrong color are not considered (assuming valid problem instances where
      goal tiles are initially clear or correctly painted).
    - The cost of changing color is 1 action per color needed that no robot currently holds.
      This is a simplification as one robot changing color can satisfy the need for that color.
    - The movement cost for a tile is estimated by the minimum Manhattan distance
      from any robot's current location to the tile's location. This is an
      underestimate of the moves needed to reach an *adjacent* tile, but
      provides a simple distance metric.

    # Heuristic Initialization
    - Extracts the coordinates (row, col) for each tile from the static 'up',
      'down', 'left', 'right' facts by parsing tile names like 'tile_row_col'.
    - Stores the goal conditions, specifically mapping goal tiles to their
      required colors.
    - Stores the set of available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify all goal tiles that are currently unpainted AND clear in the state.
       Store these tiles and their required colors.
    3. If there are no such tiles, the heuristic is 0 (goal reached for painting).
    4. Add 1 to the total cost for each identified unpainted clear goal tile. This accounts for the paint action.
    5. Determine the set of colors required by the identified unpainted clear goal tiles.
    6. Determine the set of colors currently held by the robots in the state.
    7. Identify the colors that are required but not currently held by any robot.
    8. Add the number of such colors to the total cost. This estimates the cost of changing colors.
    9. For each identified unpainted clear goal tile:
       a. Find its coordinate (row, col) using the precomputed map.
       b. Find the current location (tile) of each robot.
       c. For each robot, calculate the Manhattan distance from its current location
          to the goal tile's location using the precomputed coordinates.
       d. Find the minimum Manhattan distance among all robots for this specific goal tile.
       e. Add this minimum distance to the total cost. This estimates the movement cost.
    10. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting tile coordinates, goal colors,
        and available colors from the task definition.
        """
        super().__init__(task)

        self.tile_coords = {}  # Map tile name to (row, col)

        # Collect all tile names mentioned in static facts (adjacency) and goal facts
        all_tiles = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                 all_tiles.add(parts[1])
                 all_tiles.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'painted':
                  all_tiles.add(parts[1])

        # Parse coordinates from tile names like 'tile_row_col'
        for tile_name in all_tiles:
            try:
                # Assuming tile names are like 'tile_row_col'
                _, r_str, c_str = tile_name.split('_')
                row, col = int(r_str), int(c_str)
                self.tile_coords[tile_name] = (row, col)
            except ValueError:
                # If a tile name doesn't match the expected format,
                # we cannot determine its coordinates for the Manhattan distance.
                # This might indicate an issue with the problem instance naming,
                # or a need for a more complex coordinate extraction method
                # (e.g., BFS from a known origin tile using adjacency facts).
                # For this heuristic, we'll simply skip such tiles when calculating
                # movement cost, but they are still considered for paint/color costs
                # if they are goal tiles.
                pass


        # Store goal tiles and their required colors.
        self.goal_tiles = {}  # Map tile name to required color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Store available colors (not strictly used in the current calculation,
        # but good practice).
        self.available_colors = {get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles that are currently unpainted and clear.
        """
        state = node.state  # Current world state.

        # Find robot locations and colors
        robot_locations = {}  # Map robot name to tile name
        robot_colors = {}  # Map robot name to color
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts and parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify unpainted goal tiles that are clear
        unpainted_clear_goal_tiles = {}  # Map tile name to required color
        for tile, color in self.goal_tiles.items():
            # Check if the tile is NOT painted with the correct color
            is_painted_correctly = f"(painted {tile} {color})" in state
            # Check if the tile is clear (needed for painting)
            is_clear = f"(clear {tile})" in state

            # We only need to paint tiles that are goals and are currently clear.
            # Tiles painted with the wrong color are assumed not to be goal tiles
            # in valid instances, or represent dead ends not handled by this heuristic.
            if not is_painted_correctly and is_clear:
                 unpainted_clear_goal_tiles[tile] = color

        # If all goal tiles are painted correctly (or are not clear, which implies
        # they are painted, possibly incorrectly, but we assume valid instances),
        # the heuristic is 0.
        if not unpainted_clear_goal_tiles:
            return 0

        # Heuristic calculation
        total_cost = 0

        # Cost 1: Paint action for each tile
        # Each unpainted clear goal tile needs one paint action.
        total_cost += len(unpainted_clear_goal_tiles)

        # Cost 2: Acquire needed colors
        # Identify all colors required by the unpainted clear goal tiles.
        needed_colors_set = set(unpainted_clear_goal_tiles.values())
        # Identify colors currently held by robots.
        current_robot_colors = set(robot_colors.values())

        # For each color needed that no robot currently has, assume at least one
        # change_color action is required somewhere.
        colors_to_acquire = needed_colors_set - current_robot_colors
        total_cost += len(colors_to_acquire)

        # Cost 3: Movement for each tile
        # For each unpainted clear goal tile, estimate the movement cost.
        # We use the minimum Manhattan distance from any robot's current location to the tile.
        # This is a relaxation ignoring adjacency requirement and clear path.
        for tile, color in unpainted_clear_goal_tiles.items():
            min_dist_to_tile = float('inf')
            tile_coord = self.tile_coords.get(tile)

            # If tile coordinate wasn't parsed (e.g., malformed name), skip movement cost for this tile.
            # This shouldn't happen with standard instances, but prevents errors.
            if tile_coord is None:
                 continue

            # Find the closest robot to this tile
            for robot, robot_loc in robot_locations.items():
                robot_coord = self.tile_coords.get(robot_loc)

                # If robot location coordinate wasn't parsed, skip this robot.
                if robot_coord is None:
                     continue

                # Calculate Manhattan distance
                dist = abs(tile_coord[0] - robot_coord[0]) + abs(tile_coord[1] - robot_coord[1])
                min_dist_to_tile = min(min_dist_to_tile, dist)

            # Add the minimum distance found for this tile to the total cost.
            # If no robots were found or had valid locations, min_dist_to_tile remains inf.
            # In a valid problem instance with unpainted tiles, there must be robots.
            # If min_dist_to_tile is still inf, it implies no robots could be located,
            # which might suggest an unsolvable state or parsing issue. We add 0
            # in this case, effectively ignoring movement cost if robot location is unknown.
            if min_dist_to_tile != float('inf'):
                total_cost += min_dist_to_tile


        return total_cost
