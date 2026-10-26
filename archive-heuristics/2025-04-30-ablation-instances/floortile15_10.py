from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to parse tile coordinates from 'tile_r_c' format
def parse_tile_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        # Assuming format is always tile_row_col
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (ValueError, IndexError):
        # Handle unexpected tile name format - return None or raise error
        # Returning None and handling it in __call__
        return None

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles
    specified in the goal that are currently unpainted. It sums the estimated
    cost for each unpainted goal tile independently, considering the robot's
    current location and color.

    # Assumptions
    - Tiles cannot be repainted once painted. If a tile is painted with the
      wrong color according to the goal, the state is likely a dead end
      (this heuristic does not explicitly penalize such states, focusing
      only on tiles that are clear and need painting).
    - The robot always holds exactly one color.
    - Tile names follow the format 'tile_row_col' allowing coordinate parsing.
    - Movement cost between adjacent tiles is 1.
    - Manhattan distance is used as a proxy for movement cost on the grid,
      ignoring potential obstacles (other painted tiles) for simplicity
      in this non-admissible heuristic.
    - The robot's name is 'robot1'.

    # Heuristic Initialization
    - Extracts the set of goal facts specifying which tiles need to be painted
      and with which color. Stores this as a dictionary mapping tile names
      to required colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location tile name and the color name it is holding by iterating through the state facts.
    2. Parse the robot's location tile name to get its (row, col) coordinates (Rr, Rc). If parsing fails or robot location/color is not found, return a large penalty.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each tile specified in the goal that needs to be painted
       with a specific color (let's call the tile T_str and the required color C_goal_str).
    5. For the current goal tile T_str:
       - Check if T_str is already painted with color C_goal_str in the current state. If yes,
         this goal is satisfied for this tile; add 0 to the total cost and proceed
         to the next goal tile.
       - Check if T_str is currently clear. If yes, this tile needs to be painted.
         Calculate the estimated cost to paint this tile:
         a. Parse the tile name T_str to get its (row, col) coordinates (Tr, Tc). If parsing fails, add a penalty for this tile and continue.
         b. Calculate the Manhattan distance `d` between the robot's current
            location (Rr, Rc) and tile T (Tr, Tc): `d = abs(Rr - Tr) + abs(Rc - Tc)`.
         c. Calculate the estimated movement cost for the robot to reach a tile
            adjacent to T. This cost is 1 if the robot is currently at T (`d == 0`),
            0 if the robot is already adjacent to T (`d == 1`), and `d - 1` if
            the robot is further away (`d > 1`). This can be calculated as
            `d - 1 if d >= 1 else 1`.
         d. Calculate the color change cost: 1 if the robot's current color
            (robot_color_str) is not equal to the required color (C_goal_str),
            0 otherwise.
         e. The paint action itself costs 1.
         f. The estimated cost for this specific tile T is the sum of the movement
            cost (c), the color change cost (d), and the paint action cost (e).
         g. Add this estimated cost for tile T to the total heuristic cost.
       - If T_str is painted with a color other than C_goal_str, this state is
         likely a dead end. This heuristic does not add a penalty in this case.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Extract goal painted tiles: {tile_name: color_name}
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot's current location and color
        robot_location_str = None
        robot_color_str = None
        # Assuming robot name is 'robot1' based on example state and domain
        robot_name = 'robot1'

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3 and parts[1] == robot_name:
                robot_location_str = parts[2]
            elif parts and parts[0] == "robot-has" and len(parts) == 3 and parts[1] == robot_name:
                 robot_color_str = parts[2]

        # Safety check: robot must be located and have a color
        if robot_location_str is None or robot_color_str is None:
             # This state is likely invalid or a dead end. Assign a high cost.
             return 1000000 # A large penalty

        # Parse robot coordinates
        robot_coords = parse_tile_coords(robot_location_str)
        if robot_coords is None:
             # Penalty for unparseable robot location
             return 1000000

        Rr, Rc = robot_coords

        total_cost = 0  # Initialize action cost counter.

        # Iterate through goal tiles that need to be painted
        for tile_str, goal_color_str in self.goal_painted_tiles.items():
            # Check if the tile is already painted correctly
            if f"(painted {tile_str} {goal_color_str})" in state:
                continue # Goal for this tile is met

            # Check if the tile is clear (needs painting)
            if f"(clear {tile_str})" in state:
                # This tile needs to be painted with goal_color_str

                # Parse tile coordinates
                tile_coords = parse_tile_coords(tile_str)
                if tile_coords is None:
                    # Cannot calculate distance, add a penalty for this unparseable tile
                    total_cost += 1000
                    continue

                Tr, Tc = tile_coords

                # 1. Calculate movement cost to reach an adjacent tile
                d = abs(Rr - Tr) + abs(Rc - Tc) # Manhattan distance to the tile itself

                # Cost to reach an adjacent tile:
                # If d=0 (robot at tile), needs 1 move away.
                # If d=1 (robot adjacent), needs 0 moves.
                # If d>1, needs d-1 moves.
                move_cost = d - 1 if d >= 1 else 1

                # 2. Calculate color change cost
                color_cost = 1 if robot_color_str != goal_color_str else 0

                # 3. Paint action cost
                paint_cost = 1

                # Total cost for this tile
                total_cost += move_cost + color_cost + paint_cost

            # If the tile is painted with a wrong color, we ignore it based on assumption.
            # The goal won't be met via this tile in this state.

        return total_cost
