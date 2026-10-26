from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         return [] # Return empty list for malformed facts

    # Remove parentheses and split by whitespace
    return fact_str[1:-1].split()


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles
    that are not currently painted correctly according to the goal. It sums
    the number of tiles needing painting, the estimated number of color changes,
    and the estimated movement cost to reach the vicinity of the nearest
    unpainted tile.

    # Assumptions
    - The goal specifies the desired color for a subset of tiles using the
      `(painted tile color)` predicate. Any tile not specified in the goal
      is assumed to not require painting (or its state doesn't matter).
      We assume tiles painted with the wrong color cannot be fixed (based on
      action definitions), so we only count tiles that are supposed to be
      painted with a specific color but are not currently in that state.
    - Movement cost is estimated using Manhattan distance on the grid coordinates
      derived from tile names (assuming 'tile_r_c' format), ignoring the 'clear'
      predicate constraint for simplicity (non-admissible).
    - There is only one robot, named 'robot1'.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the desired color for each tile
      that needs to be painted. Stores this in a dictionary mapping tile names
      to goal colors (`self.goal_colors`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Robot State:** Find the robot's current location (tile name)
        and the color it is currently holding from the state facts. Parse the
        robot's tile name to get its grid coordinates (row, column).
    2.  **Identify Unpainted Goal Tiles:** Iterate through the `self.goal_colors`
        dictionary. For each tile and its required goal color, check if the
        fact `(painted tile goal_color)` exists in the current state. Collect
        all tiles for which this fact is missing into a list (`tiles_to_paint`).
    3.  **Count Paint Actions:** The number of tiles in `tiles_to_paint` is a
        direct estimate of the minimum number of `paint` actions required. Add
        this count to the total heuristic cost.
    4.  **Identify Needed Colors:** Collect the set of distinct goal colors
        required for the tiles in `tiles_to_paint`.
    5.  **Estimate Color Change Actions:**
        - If `tiles_to_paint` is empty, no color changes are needed (cost is 0).
        - If `tiles_to_paint` is not empty:
            - If the robot's current color is one of the `needed_colors`, the
              estimated cost for color changes is the number of distinct
              `needed_colors` minus 1 (as the first needed color is already held).
            - If the robot's current color is not one of the `needed_colors`, the
              estimated cost is the number of distinct `needed_colors` (as the
              robot must first change to one of the needed colors, and then
              potentially switch between the others).
        Add this estimated color change cost to the total heuristic cost.
    6.  **Estimate Movement Actions:**
        - If `tiles_to_paint` is empty, no movement is needed (cost is 0).
        - Otherwise, calculate the Manhattan distance from the robot's current
          coordinates to the coordinates of each tile in `tiles_to_paint`.
          Find the minimum of these distances (`min_dist_to_any_tile`).
        - The robot needs to move to a tile *adjacent* to the target tile
          to perform a paint action.
        - If `min_dist_to_any_tile` is 0 (robot is currently *at* the target
          tile), it needs 1 move to get to an adjacent tile.
        - If `min_dist_to_any_tile` is greater than 0, the minimum moves
          required to reach an adjacent tile is `min_dist_to_any_tile - 1`.
        Add this estimated movement cost to the total heuristic cost.
    7.  **Return Total Cost:** The sum calculated in steps 3, 5, and 6 is the
        heuristic value for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        super().__init__(task) # Call base class constructor

        # Extract goal colors for tiles that need to be painted
        # self.goals is a frozenset of goal facts
        self.goal_colors = {}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "painted":
                # Fact is expected to be (painted tile_name color)
                if len(parts) == 3:
                    tile_name = parts[1]
                    color = parts[2]
                    self.goal_colors[tile_name] = color


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of facts).

        # 1. Identify Robot State
        robot_location = None
        current_color = None
        robot_name = "robot1" # Assuming robot is named robot1

        # Iterate through state facts to find robot location and color
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            if parts[0] == "robot-at" and len(parts) == 3 and parts[1] == robot_name:
                robot_location = parts[2] # e.g., 'tile_0_4'
            elif parts[0] == "robot-has" and len(parts) == 3 and parts[1] == robot_name:
                current_color = parts[2] # e.g., 'white'

        # If robot location is not found, something is wrong with the state representation
        if robot_location is None:
             return float('inf') # Should not happen in valid states

        # Parse robot coordinates (assuming tile_r_c format)
        r_robot, c_robot = -1, -1 # Use invalid default values
        try:
            loc_parts = robot_location.split('_') # e.g., ['tile', '0', '4']
            if len(loc_parts) == 3 and loc_parts[0] == 'tile':
                r_robot = int(loc_parts[1])
                c_robot = int(loc_parts[2])
            else:
                 return float('inf') # Cannot compute heuristic for unexpected format
        except (ValueError, IndexError):
             return float('inf') # Cannot compute heuristic on parsing error


        # 2. & 3. Identify tiles needing painting and count them
        tiles_to_paint = []
        needed_colors = set()

        # Iterate through the tiles that *should* be painted according to the goal
        for tile, goal_color in self.goal_colors.items():
            # Check if the tile is NOT painted correctly in the current state
            # The goal fact (painted tile goal_color) must be present in the state
            if (f"(painted {tile} {goal_color})") not in state:
                 # This tile needs attention (either unpainted or wrong color)
                 tiles_to_paint.append(tile)
                 needed_colors.add(goal_color)

        # If all goal tiles are painted correctly, the heuristic is 0
        if not tiles_to_paint:
            return 0

        # Base cost: 1 action per tile needing painting
        total_cost = len(tiles_to_paint)

        # 4. & 5. Calculate color change cost
        color_cost = 0
        if needed_colors:
            if current_color in needed_colors:
                # Robot has one of the needed colors, needs changes for the rest
                color_cost = len(needed_colors) - 1
            else:
                # Robot has a color not needed, needs to change to one of the needed colors
                color_cost = len(needed_colors)
        total_cost += color_cost

        # 6. Calculate movement cost to reach vicinity of nearest unpainted tile
        movement_cost = 0
        min_dist_to_any_tile = float('inf')

        for tile in tiles_to_paint:
            # Parse tile name to get coordinates (r_T, c_T)
            r_T, c_T = -1, -1 # Use invalid default values
            try:
                tile_parts = tile.split('_') # e.g., ['tile', '1', '5']
                if len(tile_parts) == 3 and tile_parts[0] == 'tile':
                    r_T = int(tile_parts[1])
                    c_T = int(tile_parts[2])
                else:
                     # Skip this tile for distance calc if name format is unexpected
                     continue
            except (ValueError, IndexError):
                 # Skip this tile for distance calc on parsing error
                 continue

            # Calculate Manhattan distance from robot to this unpainted tile
            dist_to_tile = abs(r_robot - r_T) + abs(c_robot - c_T)
            min_dist_to_any_tile = min(min_dist_to_any_tile, dist_to_tile)

        # Estimate moves to reach a tile adjacent to the closest unpainted tile
        # If min_dist_to_any_tile is float('inf'), it means no unpainted tiles had parsable names.
        # This case should ideally not happen if tiles_to_paint is not empty.
        if min_dist_to_any_tile == float('inf'):
             movement_cost = 0 # Default or error handling
        elif min_dist_to_any_tile == 0:
             movement_cost = 1 # Need 1 move to get adjacent
        else: # min_dist_to_any_tile > 0
             movement_cost = min_dist_to_any_tile - 1 # Need dist-1 moves to get adjacent

        total_cost += movement_cost

        # 7. Return Total Cost
        return total_cost
