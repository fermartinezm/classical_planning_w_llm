# Remove fnmatch import as it's not used
# from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected format, maybe return empty list or raise error
        # For this heuristic, we'll assume valid fact strings from the state/goal
        return []
    return fact[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color and ensure tiles required to be clear
    are clear. It considers the number of tiles needing painting, the
    colors the robot needs to acquire, and the minimum movement cost to
    reach a painting location.

    # Assumptions
    - Tile names follow the format 'tile_row_col' where row and col are integers, allowing Manhattan distance calculation.
    - The grid connectivity allows movement between adjacent tiles (up, down, left, right), supporting Manhattan distance as a movement cost estimate.
    - There is only one robot, named 'robot1'.
    - The robot holds at most one color at a time.
    - If a tile is required to be clear in the goal, and it is currently painted, the problem is considered unsolvable from that state.

    # Heuristic Initialization
    - Extract the goal conditions from the task, separating them into tiles that need to be painted with a specific color (`goal_painted`) and a set of tiles that must be clear (`goal_clear`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal facts provided during initialization to create a mapping of tiles to their required goal color (`goal_painted`) and a set of tiles that must be clear (`goal_clear`).
    2. Parse the current state facts to determine the robot's current location (`state_robot_at`), the color it currently holds (`state_robot_has`), and the current painted status of tiles (`state_painted`).
    3. Check for unsolvable states: Iterate through the tiles in `goal_clear`. If any of these tiles are found in `state_painted`, the state is considered unsolvable under typical Floortile problem assumptions (no unpainting action), and the heuristic returns infinity.
    4. Identify unpainted goal tiles: Iterate through the `goal_painted` mapping. For each tile and its required color, check if the tile is painted in the current state and if it has the correct color. Collect all tiles that are not painted correctly into a set `unpainted_goals`, storing `(tile_str, goal_color_str)`.
    5. Identify needed colors: From the `unpainted_goals` set, collect all the distinct colors required.
    6. Initialize the heuristic value `h` to 0.
    7. Add the number of unpainted goal tiles (`len(unpainted_goals)`) to `h`. This accounts for the paint action needed for each tile.
    8. Calculate color change cost: Create a copy of the `needed_colors` set. If the robot's current color (`state_robot_has`) is in this set, remove it. Add the size of the remaining set (`len(colors_to_get)`) to `h`. This estimates the number of `change_color` actions required to get the necessary colors.
        *Note: This assumes changing from color A to B costs 1, regardless of A.*
    9. Calculate movement cost: If there are any `unpainted_goals` and the robot's location is known:
       - Define a helper function `parse_tile_coords` to convert tile strings like 'tile_row_col' into integer coordinates `(row, col)`.
       - Parse the robot's current tile string (`state_robot_at`) to get its coordinates. Handle potential parsing errors gracefully.
       - Initialize `min_dist_to_adj` to infinity.
       - For each `(tile_str, _)` in `unpainted_goals`:
         - Parse the tile string (`tile_str`) to get its coordinates. Handle potential parsing errors.
         - Calculate the Manhattan distance between the robot's coordinates and the target tile's coordinates.
         - The robot needs to reach a tile adjacent to the target tile to paint it. The minimum moves required to get from a distance `D` to an adjacent distance (distance 1) is `max(0, D - 1)`. Calculate this minimum distance to an adjacent tile.
         - Update `min_dist_to_adj` with the minimum found so far across all unpainted goal tiles.
       - If `min_dist_to_adj` is not infinity (meaning there was at least one unpainted goal and robot location was parsed), add `min_dist_to_adj` to `h`. This estimates the minimum movement cost to get into a position to paint *at least one* of the required tiles.
        *Note: This is a simplification; it doesn't account for the total path length to paint all tiles.*
    10. Add cost for robot blocking a clear goal tile: If the robot's current location (`state_robot_at`) is in the `goal_clear` set, add 1 to `h`. This accounts for the action needed to move the robot off a tile that must be clear according to the goal. (Note: The check for painted clear goal tiles already handles the unsolvable case).
    11. Return the final calculated value of `h`. The heuristic is 0 if and only if all goal conditions are met (no unpainted goal tiles, no needed colors, robot not on a clear goal tile, and no clear goal tile is painted).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Pre-process goals for faster lookup
        self.goal_painted = {} # Map tile_str -> color_str
        self.goal_clear = set() # Set of tile_str

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            if parts[0] == "painted":
                # Goal format: (painted tile_X_Y color_C)
                if len(parts) == 3:
                    self.goal_painted[parts[1]] = parts[2]
                # else: ignore malformed goal fact
            elif parts[0] == "clear":
                 # Goal format: (clear tile_X_Y)
                if len(parts) == 2:
                    self.goal_clear.add(parts[1])
                # else: ignore malformed goal fact
            # Ignore other goal types if any

        # Static facts are not directly used in the calculation logic,
        # but the grid structure implied by up/down/left/right is assumed
        # for Manhattan distance calculation based on tile names.
        # available-color facts are implicitly handled by assuming needed colors exist.
        # self.static = task.static # Not strictly needed for this heuristic logic

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        state_painted = {} # Map tile_str -> color_str
        state_robot_at = None # tile_str
        state_robot_has = None # color_str

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "painted":
                # State format: (painted tile_X_Y color_C)
                 if len(parts) == 3:
                    state_painted[parts[1]] = parts[2]
                 # else: ignore malformed state fact
            elif parts[0] == "robot-at":
                # State format: (robot-at robot1 tile_X_Y)
                # Assuming only one robot named robot1
                 if len(parts) == 3 and parts[1] == 'robot1':
                    state_robot_at = parts[2]
                 # else: ignore malformed state fact
            elif parts[0] == "robot-has":
                # State format: (robot-has robot1 color_C)
                # Assuming only one robot named robot1 and it has only one color
                 if len(parts) == 3 and parts[1] == 'robot1':
                    state_robot_has = parts[2]
                 # else: ignore malformed state fact
            # Ignore other state facts like 'clear', 'up', 'down', etc. for this heuristic

        # --- Heuristic Calculation ---
        h = 0

        # 3. Check for unsolvable states: If any tile required to be clear in the goal is currently painted
        for tile_str in self.goal_clear:
            if tile_str in state_painted:
                # Problem is likely unsolvable if a goal requires a tile to be clear but it's painted.
                # Returning a large number instead of inf might be slightly better for some search algorithms,
                # but inf is standard for unsolvable. Let's stick to inf.
                return float('inf')

        # 4. Identify unpainted goal tiles and 5. Identify needed colors
        unpainted_goals = set() # Stores (tile_str, color_str)
        needed_colors = set() # Stores color_str

        for tile_str, goal_color_str in self.goal_painted.items():
            if tile_str not in state_painted or state_painted[tile_str] != goal_color_str:
                unpainted_goals.add((tile_str, goal_color_str))
                needed_colors.add(goal_color_str)

        # 7. Add the number of unpainted goal tiles (cost for paint actions)
        h += len(unpainted_goals)

        # 8. Calculate color change cost
        colors_to_get = needed_colors.copy()
        if state_robot_has in colors_to_get:
            colors_to_get.remove(state_robot_has)
        h += len(colors_to_get) # Add 1 for each color change needed

        # 9. Calculate movement cost
        if unpainted_goals and state_robot_at: # Need robot location to calculate movement
            # Helper to parse tile coordinates
            def parse_tile_coords(tile_str):
                 # Example: 'tile_1_5' -> (1, 5)
                 parts = tile_str.split('_')
                 # Expecting format like 'tile_row_col'
                 if len(parts) == 3 and parts[0] == 'tile':
                     try:
                         # Convert row and column parts to integers
                         return int(parts[1]), int(parts[2])
                     except ValueError:
                         # Handle cases where row/col are not integers
                         return None, None
                 return None, None # Handle unexpected format

            robot_row, robot_col = parse_tile_coords(state_robot_at)

            if robot_row is None or robot_col is None:
                # If robot location cannot be parsed, we cannot estimate movement cost.
                # This might indicate a malformed state or problem definition.
                # Print a warning and skip movement cost calculation.
                print(f"Error: Could not parse robot location tile string: {state_robot_at}. Skipping movement cost.")
                min_dist_to_adj = float('inf') # Ensures this part doesn't add to h
            else:
                min_dist_to_adj = float('inf')

                for tile_str, _ in unpainted_goals:
                    tile_row, tile_col = parse_tile_coords(tile_str)

                    if tile_row is None or tile_col is None:
                         print(f"Error: Could not parse goal tile string: {tile_str}. Skipping movement cost for this tile.")
                         continue # Skip this goal tile

                    dist = abs(robot_row - tile_row) + abs(robot_col - tile_col)
                    # Robot needs to be adjacent (Manhattan distance 1).
                    # Minimum moves to get from distance D to distance 1 is max(0, D - 1).
                    dist_to_adj = max(0, dist - 1)
                    min_dist_to_adj = min(min_dist_to_adj, dist_to_adj)

                if min_dist_to_adj != float('inf'):
                     h += min_dist_to_adj

        # 10. Add cost for robot blocking a clear goal tile
        # Check if robot is on a tile that should be clear AND that tile is not painted
        # (the painted check is redundant because we return inf earlier if a clear goal tile is painted)
        if state_robot_at in self.goal_clear: # and state_robot_at not in state_painted:
             h += 1 # Robot needs to move off this tile

        # 11. Return the final calculated value of h
        return h
