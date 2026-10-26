# Add necessary imports at the top
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Helper functions (can be defined outside the class or inside if preferred, outside is cleaner)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty facts or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    # Basic check: number of parts must match number of args unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Detailed check using fnmatch
    # Use min length to avoid index errors if parts is shorter than args (due to malformed fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args[:len(parts)]))


def get_coords(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    if not isinstance(tile_name, str):
        return None
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Handle potential non-tile objects passed in
    return None # Indicate failure to parse

def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two (row, col) tuples."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance if coordinates are missing
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# The Heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It considers the number of tiles needing
    painting, the colors that robots need to acquire, and an estimate of the
    movement cost for robots to reach the vicinity of the tiles. It assigns a
    large penalty if any goal tile is painted with the wrong color or if tiles
    need painting but no robots exist, as these states are likely unsolvable
    in this domain.

    # Assumptions
    - Tiles are arranged in a grid structure, and tile names follow the format 'tile_r_c'
      where r and c are row and column numbers.
    - Movement between adjacent tiles (up, down, left, right) costs 1 action.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - A tile is either 'clear' or 'painted' with exactly one color.
    - If a goal tile is not 'clear' and not painted with the goal color, the state
      is considered blocked and likely unsolvable, incurring a large penalty.
    - If tiles need painting but no robots are present, the state is unsolvable,
      incurring a large penalty.
    - The heuristic calculation for movement sums the minimum distance from any
      robot to each unpainted goal tile (minus 1 to account for needing to be
      adjacent, not on the tile itself). This is a non-admissible estimate that
      doesn't account for robot coordination or efficiency.

    # Heuristic Initialization
    - Extracts the set of goal facts specifying which tiles need to be painted
      and with which color.
    - Parses all tile names found in the problem definition (initial state, goals,
      static facts) to build a mapping from tile name string to (row, col) coordinates.
      This mapping is used for calculating Manhattan distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
       - The current location of each robot.
       - The current color held by each robot.
       - The painting status of each tile (clear, or painted with a specific color).
    2. Identify the set of goal tiles that are not currently painted with their
       required goal color.
    3. If this set is empty, the goal is reached, and the heuristic is 0.
    4. Check if any of the unpainted goal tiles are *not* clear (i.e., they are
       painted with a different color or have an invalid status). If such a tile
       exists, return a very large penalty value, as this state is likely unsolvable.
    5. If all unpainted goal tiles are clear, check if there are any robots. If
       tiles need painting but no robots exist, return a very large penalty.
    6. If all unpainted goal tiles are clear and robots exist, calculate the
       heuristic components:
       a.  **Paint Cost:** Add 1 for each unpainted goal tile (representing the
           paint action needed).
       b.  **Color Cost:** Identify the set of colors required by the unpainted
           goal tiles. For each required color, if no robot currently holds that
           color, add 1 to the cost (representing the need for at least one robot
           to acquire that color).
       c.  **Movement Cost:** For each unpainted goal tile, find the minimum
           Manhattan distance from any robot's current position to that tile.
           Sum `max(0, distance - 1)` over all unpainted goal tiles. This estimates
           the movement needed to get robots adjacent to the target tiles.
    7. The total heuristic value is the sum of the paint cost, color cost, and
       movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract goal painted facts
        self.goal_painted = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                if len(args) == 2: # Ensure correct number of arguments
                    tile, color = args
                    self.goal_painted.add((tile, color))

        # Build tile coordinates map
        self.tile_coords = {}
        all_tile_names = set()

        # Collect all tile names from initial state, goals, and static facts
        # Iterate through all facts and their parts
        for fact in self.initial_state | self.goals | self.static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tile_names.add(part)

        # Parse coordinates for each tile name
        for tile_name in all_tile_names:
            coords = get_coords(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse current state
        current_robot_pos = {} # robot -> tile
        current_robot_colors = {} # robot -> color
        current_painted_status = {} # tile -> color or 'clear'

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                current_robot_pos[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                current_robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted_status[tile] = color
            elif predicate == "clear" and len(parts) == 2:
                tile = parts[1]
                current_painted_status[tile] = 'clear'


        # 2. Identify unpainted goal tiles
        unpainted_goal_tiles = set() # set of (tile, color)
        for tile, color in self.goal_painted:
            if current_painted_status.get(tile) != color:
                 unpainted_goal_tiles.add((tile, color))

        # 3. If goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 4. Check for blocked tiles
        for tile, color in unpainted_goal_tiles:
            # If the tile is not clear, it must be painted with the wrong color
            # or its status is not explicitly listed (invalid state).
            # In a valid state, an unpainted goal tile must be clear.
            if current_painted_status.get(tile) != 'clear':
                return 1000000 # Large penalty

        # 5. If all unpainted goal tiles are clear, check if robots exist
        if not current_robot_pos:
             # Tiles need painting but no robots are available
             return 1000000 # Large penalty

        # 6. All unpainted goal tiles are clear and robots exist. Calculate components.
        clear_unpainted_goal_tiles = unpainted_goal_tiles # Rename for clarity
        total_cost = 0

        # a. Paint Cost
        total_cost += len(clear_unpainted_goal_tiles)

        # b. Color Cost
        required_colors = {color for tile, color in clear_unpainted_goal_tiles}
        robot_colors = {current_robot_colors.get(robot) for robot in current_robot_pos}
        robot_colors.discard(None) # Remove None if any robot has no color listed

        colors_to_acquire = required_colors - robot_colors
        total_cost += len(colors_to_acquire)

        # c. Movement Cost
        movement_cost = 0
        target_tiles = {tile for tile, color in clear_unpainted_goal_tiles}

        # We already checked that target_tiles is not empty (step 3) and robots exist (step 5)
        for target_tile in target_tiles:
            target_coords = self.tile_coords.get(target_tile)
            if target_coords is None:
                 # This tile name was collected but couldn't be parsed. Should not happen.
                 continue

            min_dist = float('inf')

            for robot, robot_pos in current_robot_pos.items():
                robot_coords = self.tile_coords.get(robot_pos)
                if robot_coords is None:
                    # Robot is at a location that couldn't be parsed. Should not happen.
                    continue

                dist = manhattan_distance(robot_coords, target_coords)
                min_dist = min(min_dist, dist)

            # Add cost to get adjacent (distance - 1)
            if min_dist != float('inf'): # Ensure a robot location was valid
                movement_cost += max(0, min_dist - 1)


        total_cost += movement_cost

        return total_cost
