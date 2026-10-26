import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_R_C' into (row, column) integer coordinates.
    Assumes the format 'tile_<row>_<col>'.
    Returns (row, col) or None if parsing fails.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not valid integers
    return None # Doesn't match expected format

def manhattan_distance(coords1, coords2):
    """
    Calculates the Manhattan distance between two tile coordinates (r1, c1) and (r2, c2).
    """
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated minimum cost for each unpainted
    goal tile, considering the best robot to paint it.

    # Assumptions:
    - Tiles are named in the format 'tile_R_C' where R is the row and C is the column.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates,
      which correspond to adjacent tiles in a grid.
    - Goal tiles are initially either 'clear' or already painted correctly.
      (The domain does not provide an action to unpaint or repaint a wrongly
       painted tile, implying such states might be unsolvable or not part of
       standard problem instances).
    - All actions (move, paint, change_color) have a cost of 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state, the heuristic estimates the cost as follows:

    1. Identify all goal tiles and their required colors from the task definition.
    2. Identify the current location and color of each robot in the current state.
    3. Determine which goal tiles are *not* currently painted with the correct color.
       These are the "unpainted goal tiles".
    4. If there are no unpainted goal tiles, the heuristic is 0 (goal reached).
    5. If there are unpainted goal tiles, iterate through each one:
       a. For the current unpainted goal tile `T` requiring color `C`:
       b. Calculate the minimum cost for *any* robot to paint this specific tile `T`.
          This minimum cost for a robot `r` is estimated as:
          - Movement cost: The minimum number of moves for robot `r` to reach *any* tile adjacent to `T`. This is `max(0, ManhattanDistance(r.location, T) - 1)`.
          - Color change cost: 1 if robot `r` does not currently have color `C`, otherwise 0.
          - Paint action cost: 1 (to perform the paint action).
          - Total estimated cost for robot `r` to paint `T`: `moves + color_cost + paint_cost`.
       c. Find the minimum of these costs over all robots.
       d. Add this minimum cost for tile `T` to the total heuristic value.
    6. The total heuristic value is the sum of the minimum estimated costs for each
       unpainted goal tile.

    This heuristic is non-admissible because it sums costs for individual goal
    tiles independently, ignoring potential synergies (e.g., one move bringing
    the robot closer to multiple tiles, one color change enabling painting
    multiple tiles of the same color). However, it provides a more informed
    estimate than just counting unpainted tiles by considering location and color
    requirements.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Map goal tiles to their required colors
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Fact is (painted tile color)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_painted_tiles[tile] = color
                else:
                    # Handle unexpected goal format, though unlikely in valid PDDL
                    print(f"Warning: Unexpected goal format: {goal}")


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # If no robots found (shouldn't happen in valid problems), return infinity
        if not robot_locations:
             return float('inf')

        # 2. Identify unpainted goal tiles
        unpainted_goal_tiles = {} # {tile: needed_color}
        for goal_tile, needed_color in self.goal_painted_tiles.items():
            # Check if the tile is already painted with the correct color
            is_painted_correctly = f"(painted {goal_tile} {needed_color})" in state

            # If not painted correctly, it's an unpainted goal tile
            if not is_painted_correctly:
                 # We assume it must be clear if not painted correctly and is a goal tile
                 # (based on domain structure and typical problem instances where
                 # goal tiles are initially clear or correctly painted).
                 # If it were painted the wrong color, the problem might be unsolvable
                 # with the given actions, but we don't explicitly check for that here.
                 unpainted_goal_tiles[goal_tile] = needed_color

        # 3. If no unpainted goal tiles, goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 4. Calculate total heuristic cost by summing minimum costs per unpainted tile
        total_cost = 0

        for tile, needed_color in unpainted_goal_tiles.items():
            tile_coords = parse_tile_coords(tile)
            if tile_coords is None:
                 # Cannot parse tile coordinates, heuristic is invalid
                 return float('inf')

            min_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot, robot_loc_tile in robot_locations.items():
                robot_coords = parse_tile_coords(robot_loc_tile)
                if robot_coords is None:
                    # Cannot parse robot location coordinates, skip this robot or return inf
                    # For simplicity, let's assume valid tile names for robot locations
                    continue # Or return float('inf') if any location is invalid

                # Cost component 1: Movement to get adjacent to the tile
                dist = manhattan_distance(robot_coords, tile_coords)
                # Need to reach a tile adjacent to the target tile.
                # Min moves to adjacent is max(0, dist - 1).
                moves_to_adjacent = max(0, dist - 1)

                # Cost component 2: Color change if needed
                current_color = robot_colors.get(robot) # Get robot's current color
                color_change_cost = 0
                if current_color != needed_color:
                    # Check if the needed color is available (should be static)
                    # We assume available-color predicates are in static facts or initial state
                    # and don't change. If the needed color isn't available, problem is unsolvable.
                    # A robust heuristic might check this, but for simplicity, we assume availability.
                    color_change_cost = 1 # Cost to change color

                # Cost component 3: Paint action
                paint_cost = 1

                # Total estimated cost for this robot to paint this tile
                cost_for_robot = moves_to_adjacent + color_change_cost + paint_cost

                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # If min_cost_for_tile is still infinity, it means no robot could paint it
            # (e.g., no robots, or invalid tile names).
            if min_cost_for_tile == float('inf'):
                 return float('inf') # Problem likely unsolvable or state is invalid

            total_cost += min_cost_for_tile

        return total_cost

