from fnmatch import fnmatch
from collections import defaultdict
# Assuming heuristic_base is available in the environment path
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or return empty list
        return []
    return fact[1:-1].split()

# Utility function to match PDDL facts against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for number of parts if no wildcard is used in args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip up to the length of the shorter sequence to avoid errors
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Tile parsing function
def parse_tile_name(tile_name):
    """Parses tile name 'tile_row_col' into (row, col) tuple."""
    parts = tile_name.split('_')
    # Expecting format like 'tile_1_1', 'tile_0_5', 'tile_12_34'
    if len(parts) >= 3 and parts[0] == 'tile':
        try:
            # The row and column are typically the last two parts
            row = int(parts[-2])
            col = int(parts[-1])
            return (row, col)
        except ValueError:
            # Handle cases like tile_abc_def or tile_1_abc where conversion fails
            return None
    return None # Not in expected format

# Manhattan distance calculation
def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coord1 = parse_tile_name(tile1_name)
    coord2 = parse_tile_name(tile2_name)
    if coord1 is None or coord2 is None:
        # This indicates an issue with tile naming convention or input.
        # Return a large value to signify that distance cannot be calculated,
        # effectively making paths involving these tiles very expensive.
        return 1000000 # A large finite number

    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    minimum cost for each unpainted goal tile that is currently clear. The cost
    for a single tile is the minimum over all robots of the cost to get the
    correct color, move to a valid painting position, and perform the paint action.
    A penalty is added for unpainted goal tiles that are not currently clear.

    # Assumptions
    - Tile names follow the format 'tile_row_col' allowing coordinate parsing.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      corresponds to these coordinates.
    - Solvable instances do not require unpainting tiles or clearing tiles
      that are wrongly painted. Unpainted goal tiles that are not clear are
      assumed to be temporarily blocked (e.g., by a robot) or represent a
      difficult state.
    - The cost of moving between tiles is estimated by Manhattan distance,
      ignoring the 'clear' precondition for intermediate tiles on the path.
      The 'clear' precondition is only considered for the tile being painted.

    # Heuristic Initialization
    - Extracts goal conditions to identify target tiles and colors.
    - Parses static facts ('up', 'down', 'left', 'right') to precompute,
      for each tile, the set of adjacent tiles from which it can be painted.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state. If yes, the heuristic is 0.
    2.  Identify all goal predicates of the form `(painted ?tile ?color)`.
    3.  Identify which of these goal predicates are not satisfied in the current state. Let this set be `UnsatisfiedGoals`.
    4.  Extract the current location and color held by each robot from the state.
    5.  Identify all tiles that are currently `clear` from the state.
    6.  Initialize the total heuristic cost to 0.
    7.  For each unsatisfied goal `(painted T C)` in `UnsatisfiedGoals`:
        a.  Check if the tile `T` is currently `clear`.
        b.  If `T` is NOT `clear` (it's blocked, e.g., occupied or wrongly painted), add a fixed penalty (`BLOCKED_TILE_PENALTY`) to the total cost and continue to the next unsatisfied goal tile. (This handles tiles that cannot be painted immediately).
        c.  If `T` IS `clear`:
            i.  Find the set of potential robot positions `PaintingPositions[T]` from which tile `T` can be painted (precomputed during initialization based on adjacency).
            ii. Initialize `min_cost_for_tile_T` to infinity.
            iii. For each robot `R`:
                - Get `R`'s current location `R_loc` and color `R_color`.
                - Calculate `color_cost`: 1 if `R_color` is not `C`, otherwise 0.
                - Initialize `min_moves_to_painting_pos` to infinity.
                - For each `T_robot_pos` in `PaintingPositions[T]` (or empty set if T has no painting positions):
                    - Calculate the estimated movement cost as the Manhattan distance between `R_loc` and `T_robot_pos`.
                    - Update `min_moves_to_painting_pos` with the minimum distance found so far.
                - If `min_moves_to_painting_pos` is finite (meaning there's at least one reachable painting position):
                    - Calculate the total cost for robot `R` to paint tile `T`: `cost_by_R = color_cost + min_moves_to_painting_pos + 1` (where 1 is the cost of the paint action).
                    - Update `min_cost_for_tile_T` with the minimum `cost_by_R` found over all robots.
            iv. If `min_cost_for_tile_T` is finite (meaning at least one robot can potentially paint this clear tile), add `min_cost_for_tile_T` to the total cost.
            v.  If `min_cost_for_tile_T` is still infinity (meaning no robot can reach any painting position for this clear tile - unlikely in solvable instances but handled), add a large penalty (`UNREACHABLE_PAINT_POS_PENALTY`).
    8.  Return the calculated total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and precomputing
        painting positions from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations and colors for painted tiles
        self.goal_painted = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                self.goal_painted.add((parts[1], parts[2]))

        # Precompute painting positions: map tile -> set of tiles from which it can be painted
        # If (dir tile_y tile_x) is true, robot at tile_x can paint tile_y
        # So tile_x is a painting position for tile_y
        self.painting_positions = defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                tile_y, tile_x = parts[1], parts[2]
                self.painting_positions[tile_y].add(tile_x)

        # Store available colors (not strictly used in this heuristic calculation, but good practice)
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if get_parts(fact)[0] == 'available-color'}

        # Define penalties
        self.BLOCKED_TILE_PENALTY = 500
        self.UNREACHABLE_PAINT_POS_PENALTY = 1000 # Should be larger than max possible cost for one tile


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals.issubset(state):
            return 0

        # Extract dynamic information from the current state
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        current_painted = set()

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot, location = parts[1], parts[2]
                    robot_locations[robot] = location
                elif parts[0] == 'robot-has':
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                elif parts[0] == 'painted':
                    current_painted.add((parts[1], parts[2]))
            elif len(parts) == 2 and parts[0] == 'clear':
                 clear_tiles.add(parts[1])


        # Identify unsatisfied goal tiles
        unsatisfied_goals = [(T, C) for (T, C) in self.goal_painted if (T, C) not in current_painted]

        total_cost = 0

        # Calculate cost for each unsatisfied goal tile
        for tile_to_paint, target_color in unsatisfied_goals:
            # Check if the tile is clear (can be painted)
            if tile_to_paint not in clear_tiles:
                # Tile is not clear (e.g., occupied or wrongly painted) - add penalty
                total_cost += self.BLOCKED_TILE_PENALTY
                continue # Move to the next unsatisfied goal tile

            # Tile is clear and needs painting - find minimum cost to paint it
            min_cost_for_tile = float('inf')

            # Iterate through all robots to find the best one for this tile
            for robot in robot_locations:
                robot_loc = robot_locations[robot]
                robot_color = robot_colors.get(robot) # Get robot's current color

                # Cost to get the right color (1 if different, 0 if same)
                # Assuming robot always has a color based on domain init/actions
                color_cost = 1 if robot_color != target_color else 0

                # Find minimum moves for this robot to any valid painting position for this tile
                min_moves_to_painting_pos = float('inf')

                # Get potential painting positions for this tile (precomputed)
                potential_painting_positions = self.painting_positions.get(tile_to_paint, set())

                # Calculate distance to each potential painting position
                for painting_pos in potential_painting_positions:
                     # Estimate movement cost using Manhattan distance (ignoring intermediate clear constraints)
                     moves = manhattan_distance(robot_loc, painting_pos)
                     min_moves_to_painting_pos = min(min_moves_to_painting_pos, moves)

                # Calculate cost for this robot to paint this tile if a painting position is reachable
                if min_moves_to_painting_pos != float('inf'):
                    # Total cost for this robot: color change + movement + paint action (cost 1)
                    cost_by_robot = color_cost + min_moves_to_painting_pos + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_by_robot)

            # Add the minimum cost found for this tile to the total heuristic
            if min_cost_for_tile != float('inf'):
                total_cost += min_cost_for_tile
            else:
                # No robot can reach any painting position for this clear tile (unlikely in solvable)
                # Add a penalty indicating this tile is currently very hard/impossible to paint
                total_cost += self.UNREACHABLE_PAINT_POS_PENALTY


        return total_cost
