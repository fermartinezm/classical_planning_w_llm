# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle invalid fact format defensively
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
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Need a way to parse tile names
def parse_tile_name(tile_name):
    """Parses 'tile_row_col' string into (row, col) integers."""
    if not isinstance(tile_name, str):
        return None
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # PDDL rows/cols are 0-indexed or 1-indexed?
            # Example static: (up tile_1_1 tile_0_1) implies tile_1_1 is row 1, tile_0_1 is row 0.
            # This suggests 0-indexed rows and columns.
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Invalid format
    return None # Invalid format


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It considers the cost of painting each tile, moving the
    robot to be adjacent to the tile, and changing the robot's color when necessary.
    It sums the estimated cost for each unpainted goal tile and adds the cost
    for acquiring the necessary colors.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost between adjacent tiles is 1.
    - The grid structure allows calculating Manhattan distance between tiles.
    - Tiles painted with the wrong color cannot be repainted directly and are assumed
      not to be goal tiles that need fixing in valid problem instances.
    - The robot can always move to an adjacent tile if the destination tile itself is clear.
    - There is only one robot, named 'robot1'.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
      Stores this mapping (tile_name -> required_color).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Robot State:** Find the robot's current location (tile) and the color it is currently holding from the state facts. Parse the robot's tile name to get its grid coordinates (row, col).
    2. **Initialize Costs and Needed Colors:** Set the total heuristic cost for movement and painting to 0. Create an empty set to store the colors required for all unpainted goal tiles.
    3. **Iterate Through Goal Tiles:** For each tile and its required color specified in the task's goal conditions:
       a. **Check if Goal is Met:** Check if the current state contains the fact `(painted tile_name required_color)`. If it does, this goal is already satisfied for this tile, so continue to the next goal tile.
       b. **Check Tile Status:** If the goal is not met, check the status of the tile in the current state:
          i. **Tile is Clear:** If the state contains `(clear tile_name)`:
             - This tile needs to be painted. Add the `required_color` to the set of `needed_colors`.
             - Add 1 to the `total_movement_and_paint_cost` (for the paint action itself).
             - Calculate the Manhattan distance between the robot's current coordinates and the tile's coordinates.
             - If the Manhattan distance is 1, the robot is already adjacent to the tile, so the movement cost to get adjacent is 0.
             - If the Manhattan distance is greater than 1, the minimum number of moves required to get from the robot's current location to *any* tile adjacent to the target tile is the Manhattan distance minus 1. Add this value to the `total_movement_and_paint_cost`.
          ii. **Robot is On the Tile:** If the state contains `(robot-at robot1 tile_name)` (and the tile is not clear, which is implied if robot is on it according to domain rules):
             - This tile needs to be painted, but the robot is blocking it. Add the `required_color` to the set of `needed_colors`.
             - The robot must first move off the tile (cost 1). This makes the tile clear.
             - Then, the robot must paint the tile (cost 1). After moving off, the robot is adjacent to the tile it just left, so no additional movement cost is needed to become adjacent for the paint action.
             - Add a total of 2 to the `total_movement_and_paint_cost` for this tile (1 for move-off + 1 for paint).
          iii. **Tile is Wrongly Painted:** If the tile is neither clear nor occupied by the robot (meaning it's painted with the wrong color), assume this is an unfixable state in valid problems and ignore this tile for the heuristic calculation.
    4. **Calculate Color Change Cost:** Determine the set of colors in `needed_colors` that the robot does *not* currently have (`needed_colors` minus the robot's current color). The cost to acquire these colors is the size of this set (each requires one `change_color` action). Add this `color_change_cost` to the total heuristic.
    5. **Return Total Heuristic:** The final heuristic value is the sum of `total_movement_and_paint_cost` and `color_change_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal paintings.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Store goal locations and required colors for each tile.
        # Mapping: tile_name -> required_color
        self.goal_paintings = {}
        for goal in self.goals:
            # Example goal: '(painted tile_1_2 black)'
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Ensure the fact has the expected number of parts
                if len(parts) == 3:
                    tile_name, color = parts[1], parts[2]
                    self.goal_paintings[tile_name] = color
                # else: ignore malformed goal fact

        # Static facts are not explicitly needed for this heuristic's calculation logic,
        # as grid structure is inferred from tile names and available colors are assumed
        # to be sufficient based on goal requirements.
        # static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Robot State
        robot_location = None
        robot_color = None
        # Assuming only one robot named 'robot1'
        robot_name = 'robot1'

        # Convert state frozenset to a set for potentially faster lookups if state is large,
        # although frozenset lookups are generally efficient (O(1) on average).
        # Let's stick to frozenset as it's the input type.
        # state_set = set(state) # Optional optimization

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3 and parts[1] == robot_name:
                robot_location = parts[2]
            elif parts and parts[0] == "robot-has" and len(parts) == 3 and parts[1] == robot_name:
                robot_color = parts[2]

        # If robot state cannot be determined, something is wrong with the state representation
        # or domain assumptions. Return infinity or a large value.
        if robot_location is None or robot_color is None:
             # This state is likely invalid or unreachable in a standard problem
             return float('inf')

        robot_coords = parse_tile_name(robot_location)
        if robot_coords is None:
             # Invalid tile name format for robot location
             return float('inf')
        robot_row, robot_col = robot_coords

        # 2. Initialize costs and needed colors
        total_movement_and_paint_cost = 0
        needed_colors = set()

        # 3. Iterate Through Goal Tiles
        for tile_name, required_color in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            if f"(painted {tile_name} {required_color})" in state:
                continue # Goal satisfied for this tile

            # Check the status of the tile
            tile_is_clear = f"(clear {tile_name})" in state
            robot_is_on_tile = f"(robot-at {robot_name} {tile_name})" in state

            if tile_is_clear:
                # Tile is clear, needs painting
                needed_colors.add(required_color)
                cost_for_tile = 1 # Paint action

                # Calculate movement cost
                tile_coords = parse_tile_name(tile_name)
                if tile_coords is None:
                    # Invalid tile name format for goal tile
                    return float('inf')
                tile_row, tile_col = tile_coords

                manhattan_dist = abs(robot_row - tile_row) + abs(robot_col - tile_col)

                # Movement cost to get adjacent is Manhattan distance - 1
                # If already adjacent (dist 1), cost is 0. If same tile (dist 0), this case is handled below.
                # If dist > 1, cost is dist - 1.
                movement_cost = max(0, manhattan_dist - 1)

                cost_for_tile += movement_cost
                total_movement_and_paint_cost += cost_for_tile

            elif robot_is_on_tile:
                 # Tile is not clear because robot is on it
                 needed_colors.add(required_color)
                 # Cost = 1 (move away) + 1 (paint)
                 # After moving away, robot is adjacent, so no extra movement cost
                 cost_for_tile = 2
                 total_movement_and_paint_cost += cost_for_tile

            # Else: Tile is not clear and robot is not on it (wrongly painted). Ignore.

        # 4. Calculate Color Change Cost
        # Count how many needed colors the robot doesn't currently have
        colors_to_acquire = needed_colors - {robot_color}
        color_change_cost = len(colors_to_acquire)

        # 5. Return Total Heuristic
        total_heuristic = total_movement_and_paint_cost + color_change_cost

        return total_heuristic
