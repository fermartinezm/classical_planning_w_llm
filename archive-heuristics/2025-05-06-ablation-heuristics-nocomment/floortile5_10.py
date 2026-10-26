from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        # Assuming tile names are always in the format tile_row_col
        # and row/col are integers.
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # This should not happen with valid PDDL tile names like tile_r_c
        # Returning None indicates an issue parsing the name
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if tile names are not in expected format
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing up costs for:
    1. Painting each tile that needs to be painted according to the goal.
    2. Acquiring the necessary colors that no robot currently holds.
    3. Moving robots close to the tiles that need painting.
    A large penalty is added if any tile is painted with the wrong color,
    as this state is likely unsolvable given the domain actions.

    Assumptions:
    - Tile names are in the format 'tile_row_col' where row and col are integers,
      allowing Manhattan distance calculation.
    - Solvable problems do not require unpainting or repainting tiles.
      If a tile is painted with color C' in the state, and the goal requires
      it to be painted with color C (C' != C), the problem is considered
      unsolvable in practice, and a large heuristic value is returned.
    - Robots always start with a color and available colors are always available
      for change_color action.
    - Tiles are either clear, painted, or occupied by a robot. If a tile needing
      painting is not clear and not wrongly painted, it's assumed a robot is on it,
      requiring a move-off action (implicitly handled by movement cost).

    Heuristic Initialization:
    - Stores the goal facts.
    - Parses the goal facts to identify which tiles need to be painted and with which color.
      This information is stored in `self.goal_painted_status`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to find:
       - Which tiles are clear.
       - Which tiles are painted and with what color.
       - The current location of each robot.
       - The current color held by each robot.
    2. Identify tiles that are not in their goal painted state.
       - Iterate through the `self.goal_painted_status`. For each tile T and required color C:
         - Check if the fact `(painted T C)` exists in the current state.
         - If not, this tile needs attention.
    3. Separate tiles needing attention into two categories:
       - Tiles that are currently `painted` with a color `C' != C`. These are wrongly painted tiles.
       - The remaining tiles needing attention are those that are not painted with the required color and are not wrongly painted. These must be either `clear` or occupied by a robot. These are the tiles that require a paint action.
    4. Calculate heuristic components:
       - **Wrongly Painted Penalty**: If any tile is painted with the wrong color, return a large constant value multiplied by the number of such tiles. This guides the search away from these states.
       - **Paint Actions**: Count the number of tiles that require a paint action (those needing attention that are not wrongly painted). Add this count to the heuristic.
       - **Color Changes**: Identify the set of colors required for the tiles that require a paint action. Identify the set of colors currently held by robots. Count the number of required colors that are not held by any robot. Add this count to the heuristic.
       - **Movement Cost**: For each tile that requires a paint action, calculate the minimum Manhattan distance from any robot's current location to that tile's location. Sum these minimum distances. Add this sum to the heuristic. This estimates the total movement effort needed.
    5. Return the sum of the components. If the set of tiles needing painting is empty and there are no wrongly painted tiles, the goal is reached, and the heuristic is 0.
    """
    def __init__(self, task):
        self.goals = task.goals
        # Pre-process goal facts to know which tiles need which color
        self.goal_painted_status = {} # { tile: color }
        for goal_fact in self.goals:
            if goal_fact.startswith('(painted '):
                parts = get_parts(goal_fact)
                tile, color = parts[1], parts[2]
                self.goal_painted_status[tile] = color

    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        current_painted = {} # { tile: color }
        current_clear = set()
        robot_locations = {} # { robot: tile }
        robot_colors = {} # { robot: color }
        # robot_tiles = set() # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                current_clear.add(parts[1])
            elif parts[0] == 'robot-at':
                robot_name, tile_name = parts[1], parts[2]
                robot_locations[robot_name] = tile_name
                # robot_tiles.add(tile_name) # Not strictly needed
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]

        # 2. & 3. Identify tiles needing attention and categorize them
        tiles_to_paint = {} # { tile: required_color } - tiles that need painting action
        wrongly_painted_tiles_count = 0

        for tile, required_color in self.goal_painted_status.items():
            current_color = current_painted.get(tile)

            if current_color != required_color:
                # This tile is not in its goal painted state
                if current_color is not None:
                    # It's painted with the wrong color
                    wrongly_painted_tiles_count += 1
                else:
                    # It's not painted with the required color and is not wrongly painted.
                    # It must be either clear or occupied by a robot.
                    # In either case, it needs a paint action (and potentially a move-off).
                    tiles_to_paint[tile] = required_color

        # 4. Calculate heuristic components

        # Component 1: Wrongly Painted Penalty
        # If any tile is painted with the wrong color, this state is likely unsolvable.
        # Return a large value to guide the search away from these states.
        if wrongly_painted_tiles_count > 0:
            return 1000 * wrongly_painted_tiles_count # Large penalty per wrongly painted tile

        # If no tiles need painting, the goal is reached.
        if not tiles_to_paint:
            return 0

        h = 0

        # Component 2: Painting actions
        # Each tile needing painting requires one paint action.
        h += len(tiles_to_paint)

        # Component 3: Color changes
        # Identify colors required for the tiles needing painting.
        colors_required = set(tiles_to_paint.values())
        # Identify colors held by robots.
        colors_held = set(robot_colors.values())
        # Number of required colors not currently held by any robot.
        # Each such color must be acquired by at least one robot.
        # This is a lower bound on color changes needed across all robots.
        h += len(colors_required - colors_held)

        # Component 4: Movement cost
        # For each tile T needing painting, a robot must move adjacent to it.
        # Estimate the movement cost for each tile independently.
        # For tile T, find the minimum Manhattan distance from any robot's current location to that tile's location.
        # Sum these minimum distances. This is an optimistic estimate ignoring clear path constraints.
        h_move = 0
        if robot_locations: # Ensure there is at least one robot
            for tile_to_paint in tiles_to_paint:
                min_dist_to_tile = float('inf')
                tile_coords = get_coords(tile_to_paint)
                if tile_coords is None: # Should not happen with valid names like tile_r_c
                     min_dist_to_tile = 1000 # Large penalty for invalid tile name
                else:
                    for robot_loc in robot_locations.values():
                        robot_coords = get_coords(robot_loc)
                        if robot_coords is not None: # Should not happen with valid names
                            dist = abs(tile_coords[0] - robot_coords[0]) + abs(tile_coords[1] - robot_coords[1])
                            min_dist_to_tile = min(min_dist_to_tile, dist)

                if min_dist_to_tile != float('inf'):
                    h_move += min_dist_to_tile
                else:
                    # This case implies no robots or invalid tile names, add a large penalty
                    h_move += 1000 # Large penalty

        h += h_move

        return h
