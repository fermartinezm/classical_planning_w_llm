# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to parse tile names into grid coordinates
def parse_tile_name(tile_name):
    """Parses tile name 'tile_r_c' into (r, c) tuple."""
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_0_1', 'tile_1_1', 'tile_5_6'
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        # Handle potential variations if necessary, but stick to example format
        return None # Indicate parsing failed
    except (ValueError, IndexError):
        return None # Indicate parsing failed

# Helper function to calculate Manhattan distance
def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two coordinates (r1, c1) and (r2, c2)."""
    if coord1 is None or coord2 is None:
        return float('inf') # Cannot calculate distance if coordinates are unknown
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total cost to paint all goal tiles that are
    not yet painted correctly. The cost for each unpainted goal tile is estimated
    independently and summed up. The estimate for a single tile includes:
    1. The cost of the paint action (1).
    2. If a robot is currently occupying the goal tile, the cost to move it off (1).
    3. The minimum cost for any robot to reach a tile adjacent to the goal tile
       while holding the required color. This minimum cost is calculated as the
       Manhattan distance from the robot's current location to the closest adjacent
       tile, plus 1 if the robot needs to change color.

    # Assumptions
    - Goal tiles are initially clear or become clear if a robot moves off them.
    - Tiles painted with the wrong color cannot be repainted (unsolvable state).
    - Tile names follow the format 'tile_r_c' where r is the row and c is the column,
      allowing Manhattan distance calculation.
    - The grid defined by adjacency predicates is connected.
    - All necessary colors are available.

    # Heuristic Initialization
    - Parses static facts to identify all tiles and infer their grid coordinates
      assuming the 'tile_r_c' naming convention.
    - Builds a map of adjacent tiles for each tile based on inferred coordinates.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted T C)`.
    2. Filter these goals to find the set of unpainted goal tiles `U = {(T, C)}`
       where `(painted T C)` is a goal but is not true in the current state.
    3. If `U` is empty, the heuristic is 0 (goal state).
    4. Initialize the total heuristic value `h` to 0.
    5. For each unpainted goal tile `(T, C)` in `U`:
        a. Check the current state of tile `T`.
        b. If `T` is painted with a color `C'` different from `C`, the state is
           considered unsolvable, and a large heuristic value is returned.
        c. Add 1 to `h` for the paint action required for tile `T`.
        d. Check if any robot is currently located at tile `T`. If yes, add 1 to `h`
           for the move action required to clear the tile.
        e. Calculate the minimum cost for *any* robot to get into a position
           to paint tile `T` with color `C`. This involves finding the minimum
           over all robots `R`:
           - The Manhattan distance from `R`'s current location to the closest
             tile adjacent to `T`.
           - Plus 1 if `R`'s current color is not `C` (cost of `change_color`).
        f. Add this minimum robot cost (movement + color change) to `h`.
        g. If no robot can reach an adjacent tile (e.g., no adjacent tiles exist
           or all paths are blocked - though we don't model path blocking explicitly
           in this simplified heuristic, we check if adjacent tiles exist),
           return a large value indicating an unreachable goal tile.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Infers tile coordinates and adjacency based on tile names.
        """
        self.goals = task.goals
        static_facts = task.static

        self.tile_coords = {} # tile_name -> (r, c)
        self.tile_adjacents = {} # tile_name -> set of adjacent tile_names
        self.colors = set() # Available colors

        all_tiles = set()
        # Collect all tile names and available colors from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] in ['up', 'down', 'left', 'right']:
                # These facts define adjacency, but we will infer grid from names
                # Collect tile names here to ensure we get all of them
                if len(parts) >= 3:
                    all_tiles.add(parts[1])
                    all_tiles.add(parts[2])
            elif parts[0] == 'available-color':
                if len(parts) >= 2:
                    self.colors.add(parts[1])

        # Infer coordinates from tile names assuming 'tile_r_c' format
        # And build the adjacency map based on these coordinates
        coord_to_tile_name = {}
        for tile_name in all_tiles:
             coord = parse_tile_name(tile_name)
             if coord is not None:
                 self.tile_coords[tile_name] = coord
                 coord_to_tile_name[coord] = tile_name
             # else: print(f"Warning: Could not parse tile name {tile_name}") # Optional debug

        # Build tile_adjacents based on inferred coordinates
        for tile_name, (r, c) in self.tile_coords.items():
            self.tile_adjacents[tile_name] = set()
            # Check potential neighbors based on standard grid moves
            potential_neighbors = [(r+1, c), (r-1, c), (r, c-1), (r, c+1)]
            for neighbor_coord in potential_neighbors:
                if neighbor_coord in coord_to_tile_name:
                    self.tile_adjacents[tile_name].add(coord_to_tile_name[neighbor_coord])

        # Note: We don't explicitly check if all goal tiles are in self.tile_coords.
        # Assuming well-formed problems where goal tiles are part of the grid defined by static facts.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goals = {} # {tile_name: color_name}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                if goal not in state:
                     unpainted_goals[tile] = color

        if not unpainted_goals:
            return 0 # Goal reached

        h = 0 # Initialize heuristic value

        # Get current state info: robot locations/colors, tile states
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        clear_tiles = set()
        painted_tiles = {} # {tile_name: color_name} # Stores color if painted

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'robot-at' and len(parts) >= 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has' and len(parts) >= 3:
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == 'clear' and len(parts) >= 2:
                clear_tiles.add(parts[1])
            elif parts[0] == 'painted' and len(parts) >= 3:
                painted_tiles[parts[1]] = parts[2]

        # Calculate heuristic based on unpainted goals and robot states
        for target_tile, target_color in unpainted_goals.items():
            # Check if the tile is painted with the wrong color
            if target_tile in painted_tiles and painted_tiles[target_tile] != target_color:
                # Unsolvable state (cannot repaint). Return large number.
                return 1000000 # Effectively infinity

            # Cost for this tile is at least 1 (paint action)
            tile_cost = 1

            # Check if a robot is currently occupying the goal tile (making it not clear)
            robot_on_tile = None
            # A tile is not clear if it's painted OR if a robot is on it.
            # We only care if a robot is on it *and* it's a goal tile needing paint,
            # because that robot must move off first.
            if target_tile not in clear_tiles:
                 # If it's not clear, is a robot on it?
                 for robot_name, loc in robot_locations.items():
                     if loc == target_tile:
                         robot_on_tile = robot_name
                         break

            if robot_on_tile:
                # Robot is on the tile, must move off. Add 1 move cost.
                tile_cost += 1
                # After moving off, the tile becomes clear.

            # Calculate min effort for any robot to get ready to paint this tile
            # This effort is needed *after* the tile is clear (if it wasn't).
            min_robot_effort_for_this_tile = float('inf')

            # Find tiles adjacent to the target tile
            adjacent_tiles = self.tile_adjacents.get(target_tile, set())
            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles in the grid. Unreachable.
                 return 1000000 # Effectively infinity

            # Calculate min effort for any robot to reach an adjacent tile with the correct color
            for robot_name, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # Get robot's current color (can be None if robot-has is missing)

                # Cost to change color if needed
                # Assumes target_color is available if it's a goal color.
                color_change_cost = 0 if robot_color == target_color else 1

                # Cost to move from robot_location to an adjacent tile of target_tile
                min_move_cost = float('inf')
                robot_coord = self.tile_coords.get(robot_location)

                if robot_coord is not None: # Ensure robot location is mapped
                    for adj_tile in adjacent_tiles:
                        adj_coord = self.tile_coords.get(adj_tile)
                        if adj_coord is not None: # Ensure adjacent tile is mapped
                            dist = manhattan_distance(robot_coord, adj_coord)
                            min_move_cost = min(min_move_cost, dist)

                # Total effort for this robot for this tile (movement + color change)
                if min_move_cost != float('inf'):
                    robot_effort = min_move_cost + color_change_cost
                    min_robot_effort_for_this_tile = min(min_robot_effort_for_this_tile, robot_effort)

            # Add the minimum robot effort required for this tile
            if min_robot_effort_for_this_tile == float('inf'):
                 # No robot can reach an adjacent tile. Unsolvable.
                 return 1000000 # Effectively infinity

            tile_cost += min_robot_effort_for_this_tile
            h += tile_cost

        return h
