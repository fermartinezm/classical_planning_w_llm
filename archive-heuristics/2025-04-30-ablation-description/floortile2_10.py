from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parse tile name 'tile_R_C' into (R, C) tuple."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        pass # Handle error if needed, or assume valid format
    return None # Or raise error

def manhattan_distance(tile1, tile2, tile_coords_map):
    """Calculate Manhattan distance between two tiles using their coordinates map."""
    coords1 = tile_coords_map.get(tile1)
    coords2 = tile_coords_map.get(tile2)
    if coords1 is not None and coords2 is not None:
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
    return float('inf') # Cannot calculate distance if coordinates are missing

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles with the correct color. It considers the number of tiles
    that still need painting, the cost to get the correct colors to robots,
    and the movement cost for robots to reach tiles adjacent to those needing paint.

    # Assumptions
    - Goal tiles are initially either clear or already painted with the correct color.
      If a goal tile is found to be painted with the wrong color, the state is
      considered unsolvable (heuristic returns infinity).
    - Tile names follow the pattern 'tile_R_C' allowing coordinate extraction
      for Manhattan distance calculation.
    - The grid defined by up/down/left/right predicates is consistent with
      Manhattan distance on the extracted coordinates.
    - Each robot can hold at most one color at a time (inferred from typical
      domain structure, although predicate definition doesn't strictly enforce this).

    # Heuristic Initialization
    - Extracts all tile names from the initial state, goals, and static facts.
    - Parses coordinates (row, column) from tile names like 'tile_R_C'.
    - Builds a map from tile name to coordinates and vice-versa.
    - Builds an adjacency map based on the static up/down/left/right facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted T C)`.
    2. Iterate through these goal facts to find which ones are not satisfied in the current state.
    3. For each unsatisfied goal `(painted T C)`:
       - Check if the tile `T` is currently painted with *any* color other than `C`. To do this, check if `(painted T C)` is not in the state, but some other fact `(painted T C')` is in the state. If yes, the state is likely unsolvable; return infinity.
       - If the tile `T` is not painted with the goal color (and not painted wrong), it must be `clear` (assuming well-formed states where goal tiles are either clear or correctly painted initially). Add this tile and its required color `(T, C)` to a list of unpainted goal tiles. Collect all required colors `C` for these tiles into a set `NeededColors`.
    4. If there are no unpainted goal tiles, the heuristic is 0 (goal state).
    5. Extract the current location and color of each robot from the state.
    6. Calculate the heuristic value:
       - Start with the number of unpainted goal tiles (each needs at least one paint action).
       - Add a cost for colors: Count the number of distinct colors in `NeededColors` that are *not* currently held by *any* robot. Add this count to the heuristic. This is a lower bound on `change_color` actions needed across all robots to acquire the necessary colors.
       - Add a cost for movement: For each unpainted goal tile `T`, calculate the minimum Manhattan distance from *any* robot's current location to *any* tile adjacent to `T`. Sum these minimum distances. This estimates the minimum total movement required to bring robots into position to paint.
    7. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        initial_state_facts = task.initial_state # Need initial state too to get all objects

        # Collect all object names from initial state, goals, and static facts
        all_objects = set()
        for fact in self.static_facts | self.goals | initial_state_facts:
            parts = get_parts(fact)
            # Add all arguments as potential objects
            for obj in parts[1:]:
                all_objects.add(obj)

        self.TileCoords = {} # {tile: (row, col)}
        self.CoordsToTile = {} # {(row, col): tile}
        self.Adjacency = {} # {tile: {adjacent_tiles}}

        # Identify tiles and parse coordinates
        all_tiles = {obj for obj in all_objects if obj.startswith('tile_')}

        for tile in all_tiles:
            coords = get_coords(tile)
            if coords:
                self.TileCoords[tile] = coords
                self.CoordsToTile[coords] = tile

        # Build adjacency from static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                self.Adjacency.setdefault(t1, set()).add(t2)
                self.Adjacency.setdefault(t2, set()).add(t1) # Grid is bidirectional


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        UnpaintedGoalTiles = set() # Stores (tile, color) tuples
        NeededColors = set() # Stores colors needed for clear goal tiles
        RobotLocations = {} # Stores {robot: tile}
        RobotColors = {} # Stores {robot: color} # Assuming one color per robot

        # Extract info from state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, loc = parts[1], parts[2]
                RobotLocations[robot] = loc
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                RobotColors[robot] = color # Assuming one color per robot

        # Identify unpainted goal tiles and needed colors
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                goal_fact_str = f'(painted {tile} {color})'

                if goal_fact_str not in state:
                    # Goal fact is not satisfied. Check if the tile is painted at all.
                    is_painted_at_all = False
                    for state_fact in state:
                        state_parts = get_parts(state_fact)
                        if state_parts[0] == 'painted' and state_parts[1] == tile:
                            # Found a painted fact for this tile. Since it's not the goal color, it's wrong.
                            return float('inf') # Unsolvable

                    # If we reach here, the tile is not painted at all. It must be clear.
                    # (Assuming problem instances are well-formed where goal tiles are either clear or correctly painted initially)
                    # Let's explicitly check for clear just to be safe, although the logic implies it must be clear if not painted.
                    if f'(clear {tile})' in state:
                         UnpaintedGoalTiles.add((tile, color))
                         NeededColors.add(color)
                    # If it's not painted and not clear, something is wrong with the state representation or domain.
                    # Assuming valid states, this case shouldn't be reachable for goal tiles that are not satisfied.


        # If all goal tiles are painted correctly, the heuristic is 0.
        if not UnpaintedGoalTiles:
            return 0

        # Heuristic Calculation:

        # 1. Cost for painting each clear goal tile: 1 per tile.
        # This is a lower bound as each paint action satisfies one such requirement.
        total_cost += len(UnpaintedGoalTiles)

        # 2. Cost for getting necessary colors:
        # Count the number of distinct colors needed for unpainted goal tiles
        # that are not currently held by *any* robot.
        # This is a lower bound on the number of 'change_color' actions needed
        # across all robots to acquire the necessary colors.
        current_robot_colors_set = set(RobotColors.values())
        missing_needed_colors = NeededColors - current_robot_colors_set
        total_cost += len(missing_needed_colors)


        # 3. Cost for movement:
        # For each unpainted goal tile (T, C), find the minimum Manhattan distance
        # from any robot's current location to *any* tile adjacent to T.
        # Sum these minimum distances. This estimates the minimum total movement
        # required to bring robots into position to paint.
        for tile, color in UnpaintedGoalTiles:
            min_move_cost_for_tile = float('inf')
            
            # Find adjacent tiles using the pre-calculated Adjacency map
            adjacent_tiles = self.Adjacency.get(tile, set())

            if not adjacent_tiles:
                # Tile has no adjacent tiles? Problematic grid or tile.
                # This shouldn't happen for tiles that need painting in a solvable problem.
                 return float('inf')

            # Find minimum distance from any robot to any adjacent tile
            for robot, robot_loc in RobotLocations.items():
                # A robot can only paint if it's at a tile adjacent to the target tile.
                # The move actions require the destination tile to be clear.
                # Our distance calculation doesn't consider clearness of intermediate tiles.
                # This is a simplification, but standard for Manhattan distance heuristics.
                # The distance is from the robot's current tile to an adjacent tile of the goal tile.
                
                for adj_tile in adjacent_tiles:
                    dist = manhattan_distance(robot_loc, adj_tile, self.TileCoords)
                    min_move_cost_for_tile = min(min_move_cost_for_tile, dist)

            if min_move_cost_for_tile == float('inf'):
                # No robot could reach an adjacent tile? Unsolvable.
                return float('inf')

            total_cost += min_move_cost_for_tile

        # The heuristic value is the sum of paint costs, color acquisition costs, and movement costs.
        return total_cost
