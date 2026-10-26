from fnmatch import fnmatch
import math # For float('inf')
# Assuming heuristic_base is available in the specified path
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    with the correct colors. It considers the number of tiles that still need
    painting, the colors that need to be acquired by robots, and the movement
    cost for robots to reach tiles needing painting.

    # Assumptions
    - The grid structure is defined by up/down/left/right predicates.
    - Tiles are named in the format 'tile_row_col'.
    - Robots always hold one color from the available colors.
    - Problems are well-formed: tiles specified in the goal as painted are initially clear,
      and tiles not in the goal as painted are initially clear or correctly painted.
      (If these assumptions are violated, the heuristic returns infinity).
    - Movement cost is estimated using Manhattan distance, ignoring the 'clear'
      precondition for intermediate tiles during movement.

    # Heuristic Initialization
    - Extract the goal painted states (tile and required color).
    - Build the tile grid adjacency graph and map tile names to coordinates (row, col)
      from the static up/down/left/right predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that are required to be painted in the goal state but are
       currently clear in the current state. These are the tiles that still need painting.
    2. Check if any tile is painted with the wrong color or is painted when it
       should not be according to the goal. If so, the problem is likely unsolvable,
       return infinity.
    3. If the set of tiles needing painting is empty, the goal is reached, return 0.
    4. Determine the set of colors required for the tiles needing painting.
    5. Determine the set of colors currently held by the robots.
    6. Calculate the number of distinct colors needed that are not currently held by any robot.
       This contributes to the heuristic cost (1 action per color to acquire).
    7. For each tile that needs painting, calculate the minimum Manhattan distance
       from any robot's current location to any tile adjacent to the tile needing painting.
       This estimates the movement cost for that specific tile painting task, assuming
       the closest robot is used and movement is unrestricted by 'clear' tiles.
    8. Sum the minimum Manhattan distances calculated in step 7 over all tiles
       needing painting.
    9. The total heuristic value is the sum of:
       - The number of tiles needing painting (each needs a paint action).
       - The number of colors that need to be acquired.
       - The sum of minimum Manhattan distances for movement.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # Call the base class constructor (optional, but good practice)
        super().__init__(task)

        self.goals = task.goals
        static_facts = task.static

        # Store goal locations and colors for painted tiles
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Goal fact is (painted tile_name color_name)
                if len(parts) == 3:
                    _, tile, color = parts
                    self.goal_painted[tile] = color
                # else: malformed goal fact, ignore or handle error

        # Build adjacency graph and tile coordinates from static facts
        self.adjacency_graph = {}
        self.tile_coords = {} # tile_name -> (row, col)

        # Helper to parse tile name like 'tile_1_5' into (1, 5)
        def parse_tile_name(tile_name):
            try:
                parts = tile_name.split('_')
                # Expecting format like 'tile_row_col'
                if len(parts) == 3 and parts[0] == 'tile':
                    # Convert row and column parts to integers
                    row = int(parts[1])
                    col = int(parts[2])
                    return (row, col)
            except (ValueError, IndexError):
                # Handle cases where the tile name format is unexpected
                # print(f"Warning: Could not parse tile name '{tile_name}'") # Debugging
                pass
            return None # Indicate parsing failed

        # Iterate through static facts to build the grid structure
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            # Look for predicates defining adjacency
            if pred in ["up", "down", "left", "right"]:
                # Fact is (pred tile1 tile2) meaning tile1 is pred of tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies tile_0_1 and tile_1_1 are adjacent.
                if len(parts) == 3:
                    tile1, tile2 = parts[1], parts[2]

                    # Add to adjacency graph (undirected edges)
                    self.adjacency_graph.setdefault(tile1, set()).add(tile2)
                    self.adjacency_graph.setdefault(tile2, set()).add(tile1)

                    # Store coordinates if not already known
                    if tile1 not in self.tile_coords:
                        coords1 = parse_tile_name(tile1)
                        if coords1 is not None:
                            self.tile_coords[tile1] = coords1
                    if tile2 not in self.tile_coords:
                        coords2 = parse_tile_name(tile2)
                        if coords2 is not None:
                            self.tile_coords[tile2] = coords2
                # else: malformed adjacency fact, ignore or handle error


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # --- Parse current state ---
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        current_painted_map = {} # tile -> color
        # We don't strictly need current_clear_tiles set if we assume
        # any tile not in current_painted_map is clear. Let's rely on that assumption
        # based on typical PDDL domain structure unless proven otherwise.
        # current_clear_tiles = set() # set of clear tile names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "robot-at":
                if len(parts) == 3:
                    _, robot, tile = parts
                    robot_locations[robot] = tile
            elif pred == "robot-has":
                 if len(parts) == 3:
                    _, robot, color = parts
                    robot_colors[robot] = color
            elif pred == "painted":
                 if len(parts) == 3:
                    _, tile, color = parts
                    current_painted_map[tile] = color
            # elif pred == "clear":
            #      if len(parts) == 2:
            #         _, tile = parts
            #         current_clear_tiles.add(tile)
            # else: ignore other predicates in state for heuristic calculation


        # --- Identify tiles needing painting and check for unsolvable states ---
        tiles_to_paint = set()
        colors_needed = set()

        # Check goal tiles: For each tile that *should* be painted...
        for goal_tile, goal_color in self.goal_painted.items():
            if goal_tile in current_painted_map:
                # The tile is currently painted.
                if current_painted_map[goal_tile] != goal_color:
                    # It's painted with the wrong color. This state is likely unsolvable
                    # as there's no action to unpaint or change a tile's color directly.
                    # print(f"Heuristic returning inf: Tile {goal_tile} painted {current_painted_map[goal_tile]}, goal is {goal_color}") # Debug
                    return float('inf')
                # Else: The tile is painted with the correct color. Goal met for this tile.
            else:
                # The tile is not currently painted (assumed to be clear).
                # It needs to be painted with the goal color.
                tiles_to_paint.add(goal_tile)
                colors_needed.add(goal_color)

        # Check tiles painted that shouldn't be: For each tile that *is* painted...
        for painted_tile in current_painted_map:
            if painted_tile not in self.goal_painted:
                # This tile is painted, but the goal does not require it to be painted
                # (or requires it to be clear, or painted a different color, handled above).
                # Since there's no unpaint action, this state is likely unsolvable.
                # print(f"Heuristic returning inf: Tile {painted_tile} painted but not in goal_painted list.") # Debug
                return float('inf')

        # If no tiles need painting (and no unsolvable states found), the goal must be reached.
        # This check is logically covered by the initial `self.goals <= state` check
        # given the structure of the floortile domain goal.
        # if not tiles_to_paint:
        #      return 0 # Redundant but harmless if logic is perfect


        # --- Calculate heuristic components ---

        # Component 1: Number of paint actions needed. Each tile needing paint requires one paint action.
        h_paint = len(tiles_to_paint)

        # Component 2: Number of colors to acquire.
        # Count distinct colors needed for painting that no robot currently holds.
        colors_held = set(robot_colors.values())
        colors_to_acquire = colors_needed - colors_held
        h_color_change = len(colors_to_acquire)

        # Component 3: Movement cost (sum of minimum Manhattan distances).
        # For each tile needing paint, estimate the cost for the closest robot
        # to reach an adjacent tile from which it can paint.
        h_movement = 0
        for tile in tiles_to_paint:
            # We need to find the minimum distance from any robot to any adjacent tile of 'tile'.
            min_dist_to_tile = float('inf')

            # Get the set of tiles adjacent to the current 'tile' from the pre-built graph.
            adjacent_tiles = self.adjacency_graph.get(tile, set())

            # If a tile has no adjacent tiles defined in the static facts, it's unreachable
            # for painting actions that require being on an adjacent tile.
            if not adjacent_tiles:
                 # print(f"Heuristic returning inf: Tile {tile} has no adjacent tiles defined.") # Debug
                 return float('inf')

            # Iterate through each robot to find the minimum distance for this tile.
            for robot, robot_loc in robot_locations.items():
                # Calculate the minimum distance from the current robot's location
                # to any of the adjacent tiles of the tile needing paint.
                dist_from_robot = float('inf')

                # Get robot coordinates. If not found, something is wrong.
                robot_coords = self.tile_coords.get(robot_loc)
                if robot_coords is None:
                     # print(f"Heuristic returning inf: Robot {robot} at unknown tile {robot_loc}") # Debug
                     return float('inf') # Robot location is not a recognized tile?

                # Calculate distance to each adjacent tile and find the minimum.
                for adj_tile in adjacent_tiles:
                    adj_coords = self.tile_coords.get(adj_tile)
                    if adj_coords is None:
                         # print(f"Heuristic returning inf: Adjacent tile {adj_tile} has unknown coordinates") # Debug
                         return float('inf') # Adjacent tile is not a recognized tile?

                    # Calculate Manhattan distance between the robot's location and the adjacent tile.
                    manhattan_dist = abs(robot_coords[0] - adj_coords[0]) + abs(robot_coords[1] - adj_coords[1])
                    dist_from_robot = min(dist_from_robot, manhattan_dist)

                # Update the minimum distance considering all robots.
                min_dist_to_tile = min(min_dist_to_tile, dist_from_robot)

            # If after checking all robots, the minimum distance is still infinity,
            # it means no robot can reach any adjacent tile of this tile. Unsolvable.
            if min_dist_to_tile == float('inf'):
                 # print(f"Heuristic returning inf: Tile {tile} adjacent tiles unreachable by any robot.") # Debug
                 return float('inf')

            # Add the minimum movement cost for this tile to the total movement heuristic.
            h_movement += min_dist_to_tile

        # The total heuristic is the sum of the three components.
        total_heuristic = h_paint + h_color_change + h_movement

        return total_heuristic
