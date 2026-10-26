from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the number of tiles that need painting,
    the cost to acquire the necessary colors, and the estimated movement cost
    for robots to reach painting positions.

    # Assumptions
    - Tiles are arranged in a grid, and their names follow the pattern 'tile_R_C'
      where R is the row and C is the column (0-indexed).
    - Movement cost between adjacent tiles is 1. Manhattan distance is used
      as an estimate for movement cost, ignoring obstacles (painted tiles).
    - Acquiring a color costs 1 action (`change_color`), assuming an available
      robot and the color is available in the domain.
    - A tile painted with the wrong color makes the state unsolvable.

    # Heuristic Initialization
    - Parses goal conditions to identify target colors for specific tiles.
    - Parses static facts (`up`, `down`, `left`, `right`) to build a map
      from tile names to (row, col) coordinates and an adjacency list
      for the grid structure.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine robot locations, robot held colors,
       and the current state of each tile (clear or painted with a specific color).
    2. Identify the set of goal tiles that are currently 'clear' but need painting
       according to the goal. These are the 'unpainted goal tiles'.
    3. Check if any goal tile is currently painted with a color different from
       its goal color. If so, the state is considered unsolvable, and a very
       large heuristic value (infinity) is returned.
    4. Initialize the heuristic value `h`.
    5. Add the number of unpainted goal tiles to `h`. This accounts for the
       minimum number of 'paint' actions required.
    6. Identify the set of distinct colors required by the unpainted goal tiles.
       Compare this set with the colors currently held by robots. For each
       required color that no robot currently holds, add 1 to `h`. This
       estimates the minimum number of 'change_color' actions needed across
       all robots.
    7. Estimate the movement cost. For each robot, calculate the minimum
       Manhattan distance from its current location to *any* tile that is
       adjacent to *any* of the unpainted goal tiles. Sum these minimum
       distances over all robots and add the total to `h`. This estimates
       the total movement effort for robots to get into position to paint.
    8. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for each tile
        self.goal_paintings = {}
        for goal in self.goals:
            # Goal facts are typically (painted tile_X_Y color)
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build tile coordinates and adjacency list from static facts
        self.tile_coords = {}
        self.adjacent_tiles = {} # {tile_name: [adj_tile_name, ...]}

        # Helper to parse tile name into (row, col)
        def parse_tile_name(tile_name):
            try:
                parts = tile_name.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    return (int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass # Handle unexpected tile names if necessary
            return None # Indicate parsing failed

        # Populate tile_coords and build initial adjacency structure
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                all_tiles.add(tile1)
                all_tiles.add(tile2)

                coord1 = parse_tile_name(tile1)
                coord2 = parse_tile_name(tile2)

                if coord1: self.tile_coords[tile1] = coord1
                if coord2: self.tile_coords[tile2] = coord2

                # Add adjacency (undirected graph)
                self.adjacent_tiles.setdefault(tile1, []).append(tile2)
                self.adjacent_tiles.setdefault(tile2, []).append(tile1)

        # Remove duplicates from adjacency lists
        for tile in self.adjacent_tiles:
            self.adjacent_tiles[tile] = list(set(self.adjacent_tiles[tile]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        robot_locations = {}
        robot_colors = {}
        tile_states = {} # {tile_name: 'clear' or 'painted_color'}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "clear":
                tile = parts[1]
                tile_states[tile] = 'clear'
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                tile_states[tile] = f'painted_{color}'

        # Identify unpainted goal tiles and check for wrongly painted tiles
        unpainted_goals = [] # List of (tile, goal_color) tuples
        wrongly_painted = False

        for tile, goal_color in self.goal_paintings.items():
            current_state = tile_states.get(tile) # None if tile state is not in current facts

            if current_state == 'clear':
                unpainted_goals.append((tile, goal_color))
            elif current_state is not None and current_state != f'painted_{goal_color}':
                 # Tile is painted, but not with the correct goal color
                 wrongly_painted = True
                 break # Unsolvable state

        # If any tile is painted with the wrong color, return infinity
        if wrongly_painted:
            return math.inf # Use math.inf for a large number indicating unsolvable

        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goals:
            return 0

        # --- Heuristic Calculation ---
        h = 0

        # 1. Cost for paint actions: Each unpainted goal tile needs one paint action.
        h += len(unpainted_goals)

        # 2. Cost for color changes: Estimate the minimum color changes needed.
        # Count distinct colors required by unpainted tiles that no robot currently has.
        required_colors = {color for tile, color in unpainted_goals}
        available_robot_colors = set(robot_colors.values())
        colors_to_acquire = required_colors - available_robot_colors
        h += len(colors_to_acquire) # Simplified: assume 1 action per color needed across all robots

        # 3. Cost for movement: Estimate the movement effort for robots.
        # For each robot, find the minimum distance to *any* tile adjacent to *any* unpainted goal tile.
        total_movement_cost = 0
        for robot, robot_loc in robot_locations.items():
            min_dist_for_this_robot = math.inf
            robot_coord = self.tile_coords.get(robot_loc) # Get robot's coordinates

            if robot_coord is None:
                 # Robot location not found in tile_coords - should not happen in valid instances
                 continue # Skip this robot or handle error

            # Find minimum distance from robot to any tile adjacent to any unpainted tile
            for tile, goal_color in unpainted_goals:
                # Check if the tile exists in our coordinate map (should always for unpainted goals)
                if tile not in self.tile_coords:
                    continue

                # Iterate through adjacent tiles of the current unpainted goal tile
                for adj_tile in self.adjacent_tiles.get(tile, []):
                    adj_coord = self.tile_coords.get(adj_tile) # Get adjacent tile's coordinates

                    if adj_coord is None:
                        # Adjacent tile coordinates not found - should not happen
                        continue # Skip this adjacent tile

                    # Calculate Manhattan distance
                    dist = abs(robot_coord[0] - adj_coord[0]) + abs(robot_coord[1] - adj_coord[1])
                    min_dist_for_this_robot = min(min_dist_for_this_robot, dist)

            # Add the minimum distance found for this robot to the total movement cost
            if min_dist_for_this_robot != math.inf:
                total_movement_cost += min_dist_for_this_robot

        h += total_movement_cost

        return h
