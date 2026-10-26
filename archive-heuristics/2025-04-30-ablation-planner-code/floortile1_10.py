# Utility function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(robot-at robot1 tile_0_4)" -> ["robot-at", "robot1", "tile_0_4"]
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts or non-string input
        return []
    return fact[1:-1].split()

# Utility function to parse tile name into coordinates
def parse_tile_coords(tile_name):
    """Parses a tile name like 'tile_R_C' into a tuple (R, C)."""
    # Example: "tile_3_2" -> (3, 2)
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle unexpected tile name format (e.g., non-integer row/col)
            return None
    return None

# Utility function to calculate Manhattan distance between two tile coordinates
def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two coordinate tuples (r1, c1) and (r2, c2)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# Import the base Heuristic class
# Assuming heuristics.heuristic_base is in the Python path
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Provide a dummy base class if the actual one is not available,
    # allowing the code structure to be checked.
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found")


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the required number of actions to reach the goal state
    by summing the estimated cost for each individual tile that needs to be painted
    correctly. The cost for painting a single tile is estimated as the minimum
    cost for any robot to get into a position to paint it (adjacent or move off if on the tile),
    change color if necessary, and perform the paint action.

    # Assumptions
    - Tiles are arranged in a grid, and tile names 'tile_R_C' correspond to coordinates (R, C).
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - Tiles that need painting are assumed to be paintable (e.g., clear) when a robot arrives.
      If a robot is currently on a tile that needs painting, it must first move off (cost 1).
    - The heuristic sums the minimum cost for each unsatisfied goal tile independently,
      which is an overestimate but provides a greedy estimate.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the required color for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task definition.
    2. In the current state, identify the location and current color of each robot.
    3. Identify which goal tiles are not currently painted with the correct color. These are the unsatisfied goals.
    4. If there are no unsatisfied goals, the heuristic value is 0.
    5. For each unsatisfied goal tile (T, C) (tile T needs color C):
        a. Calculate the minimum cost for *any* robot to paint this specific tile.
        b. The cost for a robot R to paint tile T with color C is estimated based on the Manhattan distance (dist) between R's location and T:
           - Moves cost: The number of moves for the robot to get into a position to paint T. If the robot is on T (dist=0), it costs 1 move to get off. If the robot is at distance D > 0, it costs D-1 moves to get adjacent. This is `dist - 1` if `dist > 0` else `1`.
           - Color Change Cost: 1 if robot R does not currently have color C, otherwise 0.
           - Paint Action Cost: 1.
           - Total cost for robot R = moves_cost + color_change_cost + paint_cost.
        c. Find the minimum of this cost over all robots.
    6. The total heuristic value is the sum of these minimum costs for each unsatisfied goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Store goal tiles and their required colors.
        self.goal_tiles = {}
        # task.goals is a list of goal literals (strings)
        for goal_literal in self.goals:
            parts = get_parts(goal_literal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile_name = parts[1]
                color_name = parts[2]
                self.goal_tiles[tile_name] = color_name
            # Assuming task.goals is a list of literals. If it's a complex structure
            # like (and (...)(...)), the planner interface should flatten it or
            # additional parsing logic would be needed here.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state # Current world state (frozenset of strings).

        # Extract current state information
        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                robot_locations[robot_name] = tile_name
            elif predicate == "robot-has" and len(parts) == 3:
                robot_name, color_name = parts[1], parts[2]
                robot_colors[robot_name] = color_name
            elif predicate == "painted" and len(parts) == 3:
                tile_name, color_name = parts[1], parts[2]
                painted_tiles[tile_name] = color_name

        # Identify unsatisfied goal tiles
        unsatisfied_goals = []
        for goal_tile, required_color in self.goal_tiles.items():
            if goal_tile not in painted_tiles or painted_tiles[goal_tile] != required_color:
                unsatisfied_goals.append((goal_tile, required_color))

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        total_heuristic_cost = 0

        # Calculate cost for each unsatisfied goal tile independently
        for tile_to_paint, required_color in unsatisfied_goals:
            target_coords = parse_tile_coords(tile_to_paint)
            if target_coords is None:
                 # Should not happen in valid problems, but handle defensively.
                 # If a goal tile name is unparseable, it's likely unreachable.
                 total_heuristic_cost += 1000 # Add a large penalty
                 continue

            min_cost_for_this_tile = float('inf')

            # Find the best robot to paint this tile
            if not robot_locations:
                 # No robots exist, problem is unsolvable
                 return float('inf') # Return infinity if no robots can perform tasks

            for robot_name, robot_tile in robot_locations.items():
                robot_coords = parse_tile_coords(robot_tile)
                if robot_coords is None:
                    # Should not happen, but handle defensively
                    continue

                dist_to_tile = manhattan_distance(robot_coords, target_coords)

                # Calculate moves needed to get into painting position (adjacent)
                # If robot is on the tile (dist=0), it needs 1 move off.
                # If robot is adjacent (dist=1), it needs 0 moves.
                # If robot is further (dist>1), it needs dist-1 moves.
                moves_cost = dist_to_tile - 1 if dist_to_tile > 0 else 1

                # Calculate color change cost
                current_robot_color = robot_colors.get(robot_name)
                color_change_cost = 1 if current_robot_color != required_color else 0

                # Total cost for this robot to paint this specific tile
                # Moves to get adjacent + Color Change (if needed) + Paint Action
                cost_for_this_robot = moves_cost + color_change_cost + 1

                min_cost_for_this_tile = min(min_cost_for_this_tile, cost_for_this_robot)

            # Add the minimum cost found for this tile to the total heuristic
            # If min_cost_for_this_tile is still inf, it means no robot could reach it (e.g. no robots exist)
            total_heuristic_cost += min_cost_for_this_tile if min_cost_for_this_tile != float('inf') else 1000 # Use a large number

        return total_heuristic_cost
