from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not a valid tile name format
    return None # Does not start with 'tile_'

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # If parsing fails for any tile, they are likely not grid tiles
        # or the naming convention is unexpected. Assume infinite distance
        # if they are not the same tile.
        if tile1_name == tile2_name:
             return 0
        return float('inf')

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

# The main heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing,
    for each unpainted goal tile, the minimum cost for any robot to paint that tile.
    The cost for a robot to paint a tile includes:
    1. Movement cost: Manhattan distance from the robot's current location to the nearest tile adjacent to the target tile.
    2. Color change cost: 1 if the robot does not have the required color, 0 otherwise.
    3. Painting cost: 1 action.
    If any goal tile is painted with the wrong color, the heuristic returns infinity.

    # Assumptions
    - Tiles are named in the format 'tile_row_col'.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates in static facts.
    - Robots always have a color.
    - All goal tiles that are not currently painted with the correct color need to be painted.
    - Tiles that are painted with the wrong color cannot be unpainted (based on domain actions).
      Therefore, if a goal tile is painted with the wrong color, the state is considered unsolvable.
    - The grid defined by adjacency predicates is connected, or at least all goal tiles and reachable tiles form connected components.

    # Heuristic Initialization
    - Extract goal painting requirements: map each goal tile to its required color.
    - Build an adjacency map for tiles based on 'up', 'down', 'left', 'right' static facts.
    - Store all tile names mentioned in static facts or goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the current location and color for each robot from the state facts.
    3. Identify the current painted status and color for each tile from the state facts. (We don't need to explicitly track clear tiles, as a tile is either painted or clear).
    4. Identify tiles that are goal tiles but are not painted with the correct color in the current state.
    5. For each tile identified in step 4:
        a. If the tile is currently painted with *any* color (which must be the wrong color, based on step 4), the state is unsolvable. Return infinity.
        b. If the tile is currently clear (this is the only remaining case for tiles identified in step 4), it needs to be painted. Collect these tiles along with their required color.
    6. Initialize the total heuristic value to 0.
    7. For each goal tile T that is currently clear and needs painting with color C_T (collected in step 5b):
        a. Initialize the minimum cost to paint this tile (by any robot) to infinity.
        b. Find the set of tiles adjacent to T (where a robot can stand to paint T) using the pre-calculated adjacency map.
        c. If T has no adjacent tiles defined in the problem, it cannot be painted. The state is unsolvable. Return infinity.
        d. For each robot R:
            i. Get R's current location Loc_R and color Color_R.
            ii. Calculate the minimum Manhattan distance from Loc_R to any of T's adjacent tiles. Let this be min_dist.
            iii. If min_dist is infinity (robot cannot reach any adjacent tile, e.g., disconnected components, or robot location is invalid), this robot cannot paint T. Continue to the next robot.
            iv. Calculate the color change cost: 1 if Color_R is not C_T, else 0.
            v. The cost for robot R to paint T is min_dist + color_change_cost + 1 (for the paint action).
            vi. Update the minimum cost for tile T with the cost for robot R if it's lower.
        e. If after checking all robots, the minimum cost for tile T is still infinity, it means no robot can paint this tile. The state is likely unsolvable. Return infinity.
        f. Add the minimum cost for tile T to the total heuristic value.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal painting requirements: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build adjacency map: {tile_name: set(adjacent_tile_names)}
        self.adj_map = {}
        self.all_tiles = set() # Keep track of all tile names mentioned in static facts or goals

        # Helper to add adjacency in both directions
        def add_adjacency(tile1, tile2):
            self.adj_map.setdefault(tile1, set()).add(tile2)
            self.adj_map.setdefault(tile2, set()).add(tile1)
            self.all_tiles.add(tile1)
            self.all_tiles.add(tile2)

        # Populate adj_map and all_tiles from static adjacency facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Predicate is (direction tile_y tile_x) meaning tile_y is adjacent to tile_x
                # So tile_x and tile_y are adjacent to each other
                tile_y, tile_x = parts[1], parts[2]
                add_adjacency(tile_x, tile_y)

        # Add any tiles mentioned in goals that might not have appeared in adjacency facts
        for tile in self.goal_paintings:
             self.all_tiles.add(tile)

        # Note: Tiles mentioned only in the initial state (e.g., clear tiles)
        # but not in static adjacency facts or goals will not be in self.all_tiles.
        # This is acceptable as the heuristic focuses on goal tiles and their reachability
        # via the defined grid.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        # Assuming self.goals contains all conditions for the goal state
        if self.goals <= state:
            return 0

        # Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # Identify current tile states (painted or clear)
        painted_tiles = {} # {tile: color}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        # Identify goal tiles that are not painted correctly and check for unsolvability
        tiles_needing_painting = [] # List of (tile, required_color)

        for tile, required_color in self.goal_paintings.items():
            current_painted_color = painted_tiles.get(tile)

            if current_painted_color is None:
                # Tile is not painted. It needs painting if it's a goal tile.
                tiles_needing_painting.append((tile, required_color))
            elif current_painted_color != required_color:
                # Tile is painted, but with the wrong color. Unsolvable.
                return float('inf')
            # Else: Tile is painted with the correct color. It's satisfied.

        # If no tiles need painting, but goal wasn't reached, it implies the goal
        # contains predicates other than 'painted' which this heuristic ignores.
        # However, the problem description and examples suggest the goal is only painted facts.
        # If there were other goal facts, this heuristic would be an underestimate
        # for those parts, but still valid for the painting part.
        # Since we checked `self.goals <= state` at the start, if we reach here
        # and `tiles_needing_painting` is empty, it means all painted goals are met,
        # and the initial goal check should have returned 0.

        total_heuristic = 0

        # Calculate cost for each clear goal tile that needs painting
        for tile, required_color in tiles_needing_painting:
            min_cost_for_tile = float('inf')

            # Find adjacent tiles that exist in the problem grid
            adjacent_tiles = self.adj_map.get(tile, set())
            existing_adjacent_tiles = [adj for adj in adjacent_tiles if adj in self.all_tiles]

            if not existing_adjacent_tiles:
                 # A goal tile has no neighbors defined in the grid. Unpaintable.
                 # This state is likely unsolvable.
                 return float('inf')

            # Calculate min cost for any robot to paint this tile
            for robot, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot) # Get robot's current color

                # Calculate min distance from robot to any adjacent tile of the target tile
                min_dist_to_adj = float('inf')
                # Check if robot_location is a valid tile name before calculating distance
                if parse_tile_coords(robot_location) is not None:
                    for adj_tile in existing_adjacent_tiles:
                        dist = manhattan_distance(robot_location, adj_tile)
                        min_dist_to_adj = min(min_dist_to_adj, dist)
                # else: robot_location is not a tile? (e.g. robot-at predicate missing or malformed)
                # This state is likely invalid or unsolvable. min_dist_to_adj remains inf.


                # If min_dist_to_adj is still inf, it means the robot cannot reach any adjacent tile
                # (e.g., disconnected grid components, or robot location is invalid).
                # This robot cannot paint this tile.
                if min_dist_to_adj == float('inf'):
                    continue # Try next robot

                # Calculate color change cost
                # Assumes required_color is always available if it's in the goal
                color_change_cost = 1 if robot_color != required_color else 0

                # Total cost for this robot to paint this tile
                cost_for_R = min_dist_to_adj + color_change_cost + 1 # +1 for the paint action

                min_cost_for_tile = min(min_cost_for_tile, cost_for_R)

            # If min_cost_for_tile is still infinity after checking all robots,
            # it means this unpainted goal tile is unreachable/unpaintable by any robot.
            # This state is likely unsolvable.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            total_heuristic += min_cost_for_tile

        return total_heuristic
