from fnmatch import fnmatch
# Assuming Heuristic base class is available in 'heuristics.heuristic_base'
# If running this code standalone, you might need a placeholder Heuristic class
# from heuristics.heuristic_base import Heuristic

# Placeholder Heuristic class for standalone testing if needed
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#
#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential non-string input or invalid format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't match if the fact has fewer parts than the pattern args,
    # unless the pattern explicitly allows it (e.g., trailing wildcards).
    # A simple zip and all check works if the number of parts must match args exactly,
    # or if fnmatch handles partial matching across zipped elements as intended.
    # Let's rely on the original logic from the example heuristics.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_str):
    """Parses a tile string like 'tile_r_c' into integer coordinates (r, c)."""
    try:
        parts = tile_str.split('_')
        # Expecting format 'tile_row_col'
        if len(parts) == 3 and parts[0] == 'tile':
            # Convert row and column to integers
            return int(parts[1]), int(parts[2])
        else:
            # Handle unexpected format gracefully
            # print(f"Warning: Unexpected tile format: {tile_str}") # Optional warning
            return None, None
    except ValueError:
        # Handle cases where row/col are not valid integers
        # print(f"Warning: Could not parse tile coordinates from: {tile_str}") # Optional warning
        return None, None

def get_robot_location_str(state):
    """Finds the robot's current tile string from the state."""
    for fact in state:
        # Match fact like '(robot-at robot1 tile_0_4)'
        if match(fact, "robot-at", "*", "*"):
            # Assuming there is exactly one robot and its location is always specified
            parts = get_parts(fact)
            if len(parts) == 3:
                return parts[2] # The third part is the tile object name
    return None # Should ideally not happen in a valid problem state

def get_robot_color(state):
    """Finds the color the robot is currently holding from the state."""
    for fact in state:
        # Match fact like '(robot-has robot1 white)'
        if match(fact, "robot-has", "*", "*"):
            # Assuming there is exactly one robot and its color is always specified
            parts = get_parts(fact)
            if len(parts) == 3:
                return parts[2] # The third part is the color object name
    return None # Should ideally not happen in a valid problem state

# The Heuristic class definition should be available from the planning framework
# If not, uncomment the placeholder definition above.

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles
    that are not currently painted with their goal color. It considers the
    number of paint actions, the estimated number of color changes, and
    an estimate of the movement cost.

    # Assumptions
    - The goal is to paint a specific set of tiles with specific colors.
    - The robot can only paint tiles directly above or below its current position. Specifically, to paint tile `tile_i_j`, the robot must be located at `tile_{i+1}_j` (to use the `paint_up` action) or `tile_{i-1}_j` (to use the `paint_down` action), assuming tile names are 'tile_row_col'.
    - All actions have a cost of 1.
    - The grid structure (up/down/left/right relations) is consistent and forms a grid.
    - Tile names are in the format 'tile_row_col'.
    - The robot always has exactly one color.

    # Heuristic Initialization
    - Extracts the goal conditions to determine which tiles need which colors.
      Stores this as a mapping from tile name string to goal color string.
    - Static facts are used implicitly via helper functions like `parse_tile_coords`
      which rely on the tile naming convention derived from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unpainted Tiles:** Iterate through the goal conditions. For each goal
        fact `(painted tile_i_j color_k)`, check if the current state contains this
        exact fact. If not, `tile_i_j` needs painting with `color_k`. Collect all
        such tiles and the set of distinct colors required for them.
    2.  **Goal Check:** If no tiles need painting, the current state satisfies all
        `painted` goals, so it is a goal state. Return 0.
    3.  **Calculate Heuristic Components:** If tiles need painting, compute the
        heuristic value as the sum of three estimated costs:
        a.  **Paint Cost:** Add 1 for each tile identified in step 1. This represents
            the minimum number of `paint` actions required.
        b.  **Color Change Cost:** Determine the robot's current color. Estimate the
            number of `change_color` actions. If the robot's current color is one
            of the needed colors, the estimated changes are `N - 1`, where `N` is
            the number of distinct needed colors. If the robot's current color is
            not needed, the estimated changes are `N`. Add this non-negative cost.
            This assumes an optimal sequence of painting all tiles of one color
            before switching.
        c.  **Movement Cost:** Estimate the total movement required. For each tile
            `tile_i_j` needing painting, the robot must eventually be at a tile
            `tile_{i+1}_j` or `tile_{i-1}_j` to paint it. Calculate the minimum
            Manhattan distance from the robot's current location `tile_r_c` to
            *either* of these two adjacent locations: `abs(c - j) + max(0, abs(r - i) - 1)`.
            Sum this minimum distance for *each* tile needing painting. This sum
            overestimates the true movement cost (as one move can reduce distance
            to multiple tiles) but provides a simple, non-zero estimate that
            correlates with the spatial distribution and distance of tasks.
    4.  **Total Heuristic:** Return the sum of the Paint Cost, Color Change Cost,
        and Movement Cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions as a mapping from tile string to its goal color string.
        self.goal_colors = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                # Assuming goal facts are always in the format (painted tile_name color_name)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_colors[tile] = color
                # else: Handle unexpected goal format? Assume valid PDDL goals.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # 1. Identify tiles needing painting and needed colors
        tiles_needing_painting = set()
        needed_colors = set()

        # Find robot's current location and color
        robot_loc_str = get_robot_location_str(state)
        robot_color = get_robot_color(state)

        # Parse robot coordinates
        R_robot, C_robot = parse_tile_coords(robot_loc_str)
        # If parsing fails (unexpected tile name format), return infinity.
        # This indicates an invalid state representation.
        if R_robot is None or C_robot is None:
             return float('inf')

        # Iterate through all tiles that have a goal color specified
        for tile_str, goal_color in self.goal_colors.items():
            # Check if the goal predicate for this tile is satisfied in the current state
            # A tile needs painting if the goal is (painted T C_goal) and the state does NOT contain (painted T C_goal).
            if f'(painted {tile_str} {goal_color})' not in state:
                 # The tile is not painted with the goal color. It needs painting.
                 tiles_needing_painting.add(tile_str)
                 needed_colors.add(goal_color)

        # 2. Goal Check: If no tiles need painting, it's a goal state
        if not tiles_needing_painting:
            return 0

        heuristic_value = 0

        # Heuristic Component 1: Paint actions
        # Each tile needing painting requires one paint action.
        heuristic_value += len(tiles_needing_painting)

        # Heuristic Component 2: Color change actions
        # Estimate changes needed based on distinct colors required.
        num_needed_colors = len(needed_colors)
        color_change_cost = 0
        if num_needed_colors > 0:
            if robot_color in needed_colors:
                # Robot has one of the needed colors, needs changes for the others
                color_change_cost = num_needed_colors - 1
            else:
                # Robot has a color not needed, must change to one of the needed colors, then potentially others
                color_change_cost = num_needed_colors
        heuristic_value += max(0, color_change_cost) # Ensure cost is non-negative

        # Heuristic Component 3: Movement actions
        # Estimate movement cost by summing minimum distances to paintable locations.
        total_movement_cost = 0
        for tile_str in tiles_needing_painting:
            I, J = parse_tile_coords(tile_str)
            # If parsing fails for a tile needing painting, something is wrong.
            # Return infinity or skip? Skipping might underestimate. Let's skip for robustness.
            if I is None or J is None:
                # print(f"Warning: Skipping tile with invalid name format: {tile_str}") # Optional warning
                continue

            # Calculate min Manhattan distance from robot_loc (R_robot, C_robot)
            # to a tile adjacent (up/down) to tile_i_j (I, J).
            # Adjacent tiles are in rows I-1 and I+1, same column J.
            # Distance to (I-1, J): abs(R_robot - (I-1)) + abs(C_robot - J)
            # Distance to (I+1, J): abs(R_robot - (I+1)) + abs(C_robot - J)
            # Min vertical distance to be adjacent to row I is max(0, abs(R_robot - I) - 1)
            # Total min distance = horizontal distance + min vertical distance to be adjacent
            dist_to_adj = abs(C_robot - J) + max(0, abs(R_robot - I) - 1)
            total_movement_cost += dist_to_adj

        heuristic_value += total_movement_cost

        return heuristic_value
