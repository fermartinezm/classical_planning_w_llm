# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

# Helper function for Manhattan distance
def manhattan_distance(tile1_str, tile2_str):
    """
    Calculates Manhattan distance between two tiles based on their names.
    Assumes tile names are in the format 'tile_row_col'.
    """
    try:
        # Parse row and column from tile names
        # Example: 'tile_0_1' -> row=0, col=1
        _, row1_str, col1_str = tile1_str.split('_')
        _, row2_str, col2_str = tile2_str.split('_')
        row1, col1 = int(row1_str), int(col1_str)
        row2, col2 = int(row2_str), int(col2_str)
        return abs(row1 - row2) + abs(col1 - col2)
    except (ValueError, IndexError):
        # Handle unexpected tile name format gracefully
        # This might indicate an issue with the problem definition or state
        # Return a large value to discourage paths involving malformed tiles
        return float('inf')


# Assuming Heuristic base class is imported as specified in the problem description
# from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their desired colors. It sums, for each unpainted goal tile, the estimated
    cost to get a robot with the correct color to that tile and paint it.

    # Assumptions:
    - Tiles are arranged in a grid, and movement cost between adjacent tiles is 1.
      Manhattan distance is used as an estimate for movement cost.
    - Robots can switch colors if the desired color is available. Switching color costs 1 action.
    - A tile must be clear before it can be painted. Reachable states are assumed
      to have unpainted goal tiles that are clear. If an unpainted goal tile is not clear,
      the state is considered a dead end (heuristic returns infinity).
    - The cost of painting a tile is 1 action.
    - The heuristic sums the minimum cost for each unpainted goal tile independently,
      ignoring potential synergies (e.g., one robot painting multiple nearby tiles)
      or conflicts (e.g., multiple robots needing the same tile).

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
    - Extracts the set of available colors from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted ?t ?c)`. Store the required
       color `?c` for each goal tile `?t`.
    2. Extract the set of available colors from the static facts `(available-color ?c)`.
    3. In the current state, identify which of the goal tiles are *not* yet painted
       with the correct color. These are the tiles that still need work.
    4. For each tile `t` that needs to be painted with color `c`:
       a. Check if `(clear t)` is true in the current state. If not, return infinity
          (dead end).
       b. Check if color `c` is in the set of available colors. If not, return infinity
          (impossible goal).
    5. In the current state, find the location `(robot-at ?r ?loc)` and color `(robot-has ?r ?color)`
       for each robot `?r`.
    6. Initialize the total heuristic cost to 0.
    7. For each tile `t` that needs to be painted with color `c` (as determined in step 3, and passed checks in step 4):
       a. Initialize `min_robot_cost_for_tile = infinity`.
       b. For each robot `r` found in step 5:
          i. Get its current location `loc` and color `color`.
          ii. Calculate the movement cost: `move_cost = manhattan_distance(loc, t)`.
          iii. Calculate the color switching cost: `switch_cost = 1` if `color != c` else `0`.
          iv. The painting cost is always 1.
          v. The total cost for this robot to paint this tile is `robot_cost = move_cost + switch_cost + 1`.
          vi. Update `min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)`.
       c. If no robots were found or `min_robot_cost_for_tile` is still infinity, return infinity (impossible).
       d. Add `min_robot_cost_for_tile` to the total heuristic cost.
    8. Return the total heuristic cost. If the state is a goal state, this process
       will result in a cost of 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that do not change during plan execution.
        static_facts = task.static

        # Store goal colors for each tile.
        # Map: tile_name -> required_color
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_colors[tile] = color
                # else: Warning could be added here for malformed goals

        # Store available colors from static facts
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "available-color" and len(parts) == 2:
                self.available_colors.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the state is a goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Track robot locations and colors
        robot_locations = {} # Map: robot_name -> tile_name
        robot_colors = {}    # Map: robot_name -> color_name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_cost = 0  # Initialize action cost counter.
        infinity = float('inf') # Use a variable for clarity

        # Identify tiles that need painting according to the goal
        tiles_to_paint = [] # List of (tile, required_color)
        for goal_tile, required_color in self.goal_colors.items():
            # Check if the tile is already painted correctly
            is_painted_correctly = f"(painted {goal_tile} {required_color})" in state

            if not is_painted_correctly:
                # This tile needs painting.
                # Check if it's clear - required precondition for paint
                is_clear = f"(clear {goal_tile})" in state
                if not is_clear:
                    # Tile needs painting but is not clear. Likely a dead end in this domain.
                    return infinity

                # Check if the required color is even available in the domain
                if required_color not in self.available_colors:
                     # This goal is impossible to achieve
                     return infinity

                tiles_to_paint.append((goal_tile, required_color))

        # If no tiles need painting, but it's not a goal state, this shouldn't happen
        # if the goal only consists of painted facts. The initial goal check handles it.

        # Calculate cost for each tile that needs painting
        for tile, required_color in tiles_to_paint:
            min_robot_cost_for_tile = infinity
            found_any_robot = False # Track if there's at least one robot

            # Find the minimum cost among all robots to paint this tile
            for robot in robot_locations: # Iterate through robots found in state
                found_any_robot = True
                robot_loc = robot_locations[robot]
                robot_color = robot_colors.get(robot) # Get color, None if not found (shouldn't happen in valid states)

                if robot_color is None:
                    # Robot exists but doesn't have a color? Problematic state.
                    continue # Skip this robot

                move_cost = manhattan_distance(robot_loc, tile)
                if move_cost == infinity:
                    # Cannot calculate distance, maybe malformed tile name?
                    continue # Skip this robot for this tile

                # Cost to switch color: 1 if robot has wrong color, 0 otherwise
                switch_cost = 1 if robot_color != required_color else 0
                paint_cost = 1

                robot_cost = move_cost + switch_cost + paint_cost
                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)

            # If there are no robots at all, the goal is impossible.
            if not found_any_robot:
                 return infinity

            # If min_robot_cost_for_tile is still inf, it means robots exist but
            # perhaps color info was missing or distance calculation failed for all.
            # If it's inf here, it implies no robot could potentially paint it.
            if min_robot_cost_for_tile == infinity:
                 return infinity

            total_cost += min_robot_cost_for_tile

        return total_cost
