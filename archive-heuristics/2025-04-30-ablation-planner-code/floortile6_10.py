from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """
    Parses a tile name string (e.g., 'tile_3_5') into integer coordinates (row, col).
    Assumes tile names are in the format 'tile_row_col'.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format, maybe return None or raise error
            return None
    except (ValueError, IndexError):
        # Handle cases where parts are not integers or list index is out of bounds
        return None

def manhattan_distance(coords1, coords2):
    """
    Calculates the Manhattan distance between two coordinate pairs (r1, c1) and (r2, c2).
    """
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    by summing three components: the number of tiles that still need painting,
    the cost to acquire necessary colors, and an estimate of the movement cost
    for robots to reach positions where they can paint the required tiles.

    # Assumptions
    - Tiles that need to be painted according to the goal are initially clear
      and remain clear until painted.
    - The grid structure allows movement between adjacent tiles, and Manhattan
      distance provides a reasonable estimate for movement cost, ignoring
      dynamic obstacles (like other robots or painted tiles) on the path,
      but considering the 'clear' requirement for the tile being painted
      (implicitly handled because unpainted goal tiles are assumed clear).
    - Robots always hold one color.
    - All colors required in the goal are available colors.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which colors. This information is stored in `self.goal_paintings`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:

    1.  **Identify Unpainted Goal Tiles:** Iterate through the goal conditions
        (`(painted tile color)`). For each such goal, check if the corresponding
        fact is present in the current state. Collect all goal `(tile, color)`
        pairs that are not satisfied in the current state.

    2.  **Calculate Paint Cost:** The minimum number of paint actions required
        is equal to the number of unpainted goal tiles. Add this count to the
        total heuristic value. If there are no unpainted goal tiles, the state
        is a goal state, and the heuristic is 0.

    3.  **Calculate Color Change Cost:** Identify the set of distinct colors
        required by the unpainted goal tiles. Check which colors are currently
        held by the robots in the current state. For each required color that
        is not held by *any* robot, at least one `change_color` action is
        needed for that color to become available on a robot. Add the count
        of such colors to the total heuristic value.

    4.  **Calculate Movement Cost:** For each unpainted goal tile `T` located
        at coordinates `(Tr, Tc)`:
        - Find the minimum Manhattan distance from *any* robot's current location
          `(Rr, Rc)` to *any* tile adjacent to `T`. A tile `AdjT` is adjacent
          to `T` if their Manhattan distance is 1. The minimum Manhattan distance
          from `(Rr, Rc)` to a tile adjacent to `(Tr, Tc)` is `max(0, Manhattan((Rr, Rc), (Tr, Tc)) - 1)`.
        - Sum these minimum distances over all unpainted goal tiles. Add this sum
          to the total heuristic value. This estimates the total movement effort
          needed, ignoring robot coordination and dynamic obstacles.

    5.  **Sum Costs:** The total heuristic value is the sum of the Paint Cost,
        Color Change Cost, and Movement Cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and required colors for each tile.
        # Format: {tile_name: color_name, ...}
        self.goal_paintings = {}
        for goal in self.goals:
            # Assuming goal predicates are always (painted tile color)
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color
            # Add other goal types if necessary, though 'painted' is the main one here

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state as a frozenset of strings.

        # 1. Identify Unpainted Goal Tiles
        unpainted_tasks = [] # List of (tile, color) tuples
        for tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted with the correct color
            if f"(painted {tile} {required_color})" not in state:
                 # Check if it's painted with the wrong color (shouldn't happen for clear goal tiles)
                 # We assume instances are well-formed such that goal tiles needing paint
                 # are initially clear and not painted with the wrong color.
                 # If it's not painted correctly, it needs painting.
                 unpainted_tasks.append((tile, required_color))

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_tasks:
            return 0

        # 2. Calculate Paint Cost
        paint_cost = len(unpainted_tasks)

        # 3. Calculate Color Change Cost
        needed_colors = {color for tile, color in unpainted_tasks}
        
        # Find colors currently held by robots
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-has" and len(parts) == 2:
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color # Assuming one color per robot

        held_colors = set(robot_colors.values())

        # Count colors needed for unpainted tiles that no robot currently holds
        colors_not_held = needed_colors - held_colors
        color_cost = len(colors_not_held)

        # 4. Calculate Movement Cost
        movement_cost = 0

        # Get current robot locations
        robot_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 2:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile

        # Calculate movement cost for each unpainted tile
        for tile, required_color in unpainted_tasks:
            tile_coords = parse_tile_coords(tile)
            if tile_coords is None:
                 # Cannot calculate distance for invalid tile name, skip or assign high cost
                 # For simplicity, we assume valid tile names
                 continue

            min_dist_to_adjacent = float('inf')

            # Find the minimum distance from any robot to a tile adjacent to the target tile
            for robot, robot_tile in robot_locations.items():
                robot_coords = parse_tile_coords(robot_tile)
                if robot_coords is None:
                     continue # Skip robot with invalid location

                dist_m = manhattan_distance(robot_coords, tile_coords)

                # Distance to an adjacent tile is max(0, dist_m - 1)
                dist_to_adj = max(0, dist_m - 1)

                min_dist_to_adjacent = min(min_dist_to_adjacent, dist_to_adj)

            # Add the minimum movement cost for this tile to the total
            if min_dist_to_adjacent != float('inf'):
                 movement_cost += min_dist_to_adjacent
            # else: the tile is unreachable by any robot (e.g., no robots or invalid tile names)
            # In a solvable problem, this shouldn't result in infinity unless the state is a dead end.
            # For a non-admissible heuristic, returning a large number or infinity for dead ends is fine.
            # However, the problem asks for finite values for solvable states.
            # If min_dist_to_adjacent is inf, it means no robot location was valid, which implies
            # the problem setup is likely invalid or the state is a dead end. We'll proceed assuming
            # valid robot locations exist if unpainted tasks exist.

        # 5. Sum Costs
        total_heuristic = paint_cost + color_cost + movement_cost

        return total_heuristic

