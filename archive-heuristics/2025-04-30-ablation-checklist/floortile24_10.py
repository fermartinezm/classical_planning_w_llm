from fnmatch import fnmatch
# Assuming Heuristic base class is available in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class for standalone testing if needed
# In a real planning system, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

# Utility functions to parse PDDL facts (copied from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific utility for tile names and coordinates
def parse_tile_name(tile_name):
    """Parses 'tile_row_col' into (row, col) integers."""
    try:
        # Tile names are expected to be in the format 'tile_row_col'
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row_str, col_str = parts[1], parts[2]
            return int(row_str), int(col_str)
    except ValueError:
        # Handle cases where the object is not a tile or has unexpected format
        pass # Return None below
    return None, None

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are missing
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost by summing the minimum cost for each unpainted goal tile.
    The minimum cost for a tile considers the closest robot, the need to change
    color, movement cost (approximated by Manhattan distance to adjacency),
    and the paint action.

    Returns infinity if any goal tile is painted the wrong color or if any
    tile adjacent to a goal tile is painted (making it impossible to paint
    the goal tile).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and adjacency information from static facts.
        """
        super().__init__(task)

        # Store goal requirements: { tile_name: color_name }
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Goal fact is (painted tile color)
                if len(args) == 2:
                    tile, color = args
                    self.goal_tiles[tile] = color

        # Build tile coordinate map: { tile_name: (row, col) }
        self.tile_coords = {}
        # Build adjacency map: { tile_name: [adjacent_tile1, ...] }
        self.adj_map = {}

        # Collect all tile objects mentioned in the problem
        all_tiles = set()
        # Look in initial state and static facts for objects that look like tiles
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tiles.add(part)
        # Also add tiles from goals just in case (though they should be in init/static)
        all_tiles.update(self.goal_tiles.keys())


        # Parse coordinates for all identified tiles
        for tile in all_tiles:
            coords = parse_tile_name(tile)
            if coords != (None, None):
                self.tile_coords[tile] = coords
                self.adj_map[tile] = [] # Initialize adjacency list

        # Build adjacency map from static facts (up, down, left, right)
        for fact in task.static:
            parts = get_parts(fact)
            # Adjacency facts are like (direction tile1 tile2)
            # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1
            # This implies tile_1_1 and tile_0_1 are adjacent.
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                dir_pred, tile1, tile2 = parts
                # Add bidirectional adjacency if both tiles are in our map
                if tile1 in self.adj_map and tile2 in self.adj_map:
                     if tile2 not in self.adj_map[tile1]:
                         self.adj_map[tile1].append(tile2)
                     if tile1 not in self.adj_map[tile2]:
                         self.adj_map[tile2].append(tile1)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Parse current state information
        current_robot_locations = {} # { robot_name: tile_name }
        current_robot_colors = {}    # { robot_name: color_name }
        current_painted_tiles = {}   # { tile_name: color_name }
        current_clear_tiles = set()  # { tile_name }

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at": # e.g. (robot-at robot1 tile_0_4)
                robot, tile = parts[1], parts[2]
                current_robot_locations[robot] = tile
            elif len(parts) == 3 and parts[0] == "robot-has": # e.g. (robot-has robot1 white)
                 robot, color = parts[1], parts[2]
                 current_robot_colors[robot] = color
            elif len(parts) == 2 and parts[0] == "clear": # e.g. (clear tile_1_5)
                tile = parts[1]
                current_clear_tiles.add(tile)
            elif len(parts) == 3 and parts[0] == "painted": # e.g. (painted tile_1_2 black)
                tile, color = parts[1], parts[2]
                current_painted_tiles[tile] = color

        # Check for unsolvability first
        # If any goal tile is painted the wrong color, it's unsolvable
        # If any tile adjacent to a goal tile is painted, it's unsolvable
        # (because a robot needs to move to a clear adjacent tile to paint)
        for goal_tile, goal_color in self.goal_tiles.items():
            # 1. Is the goal tile painted the wrong color?
            if goal_tile in current_painted_tiles:
                if current_painted_tiles[goal_tile] != goal_color:
                    return float('inf') # Unsolvable

            # 2. Is any tile adjacent to this goal tile painted?
            # If an adjacent tile is painted, a robot cannot move onto it,
            # which is required to paint the goal tile from that position.
            # Assuming painted tiles cannot become clear.
            if goal_tile in self.adj_map: # Ensure tile exists in our map
                for adjacent_tile in self.adj_map[goal_tile]:
                    if adjacent_tile in current_painted_tiles:
                        return float('inf') # Unsolvable

        total_cost = 0

        # Calculate cost for each unpainted goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # If the tile is already painted correctly, cost is 0 for this tile
            if goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] == goal_color:
                continue

            # Tile needs painting (it must be clear if not painted the wrong color)
            # Find the minimum cost for any robot to paint this tile
            min_robot_cost_for_this_tile = float('inf')

            # If there are no robots, this is unsolvable (or cost is inf)
            if not current_robot_locations:
                 return float('inf')

            for robot, robot_location in current_robot_locations.items():
                robot_color = current_robot_colors.get(robot) # Get robot's current color

                # Cost to get the correct color: 1 action if different, 0 if same
                # Assumes the target color is available (checked in domain definition, not state)
                color_cost = 0 if robot_color == goal_color else 1

                # Cost to move adjacent to the goal tile
                # Relaxed: Manhattan distance from robot's current tile to the goal tile, minus 1 (for adjacency)
                # This ignores the 'clear' precondition for intermediate tiles and the target adjacent tile itself,
                # but we already checked that adjacent tiles aren't *painted*.
                robot_coords = self.tile_coords.get(robot_location)
                goal_coords = self.tile_coords.get(goal_tile)

                if robot_coords is None or goal_coords is None:
                     # This indicates a problem with tile parsing or map building
                     move_cost = float('inf')
                else:
                     dist = manhattan_distance(robot_coords, goal_coords)
                     # Need to move *to* an adjacent tile, not the tile itself.
                     # Min moves to get adjacent is distance - 1.
                     # If distance is 0 (robot is on the tile), it cannot paint it (needs to be adjacent).
                     # If distance is 1 (robot is adjacent), move cost is 0.
                     # If distance > 1, min moves is dist - 1.
                     move_cost = max(0, dist - 1)


                # Total cost for this robot to paint this specific tile
                # color_cost + move_cost + paint_action_cost
                # Each paint action costs 1.
                robot_cost = color_cost + move_cost + 1

                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_cost)

            # Add the minimum cost for this tile to the total heuristic value
            # If min_robot_cost_for_this_tile is still inf, it means no robot can reach it (e.g., no tiles parsed),
            # which implies unsolvability.
            if min_robot_cost_for_this_tile == float('inf'):
                 return float('inf')

            total_cost += min_robot_cost_for_this_tile

        return total_cost

