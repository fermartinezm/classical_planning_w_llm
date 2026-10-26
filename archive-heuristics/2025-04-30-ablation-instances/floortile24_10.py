from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats, maybe log a warning
        # For this problem, we assume valid PDDL fact strings
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with the correct colors. It considers the number of tiles that need painting,
    the color changes required, and the movement needed to reach the vicinity
    of the unpainted tiles.

    # Assumptions
    - Each paint action costs 1.
    - Each color change action costs 1.
    - Movement cost is approximated by the maximum Manhattan distance from the
      robot's current location to a tile adjacent to any unpainted goal tile,
      ignoring obstacles (like other painted tiles).
    - The robot always has a color.
    - All required colors are available.

    # Heuristic Initialization
    - Extracts the goal painting requirements (which tile needs which color)
      from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Robot State: Determine the robot's current tile location and
       the color it is currently holding. This is done by scanning the state
       facts for predicates like `(robot-at ...)` and `(robot-has ...)`.
    2. Identify Unpainted Goal Tiles: Compare the current state's painted facts
       with the goal painting requirements stored during initialization. Collect
       all goal tiles that are not yet painted with their required color.
    3. Base Cost (Paint Actions): The number of unpainted goal tiles is a lower
       bound on the number of paint actions required (each unpainted tile needs
       at least one paint action). Add this count to the total heuristic cost.
       If there are no unpainted goal tiles, the heuristic is 0, as the goal is reached.
    4. Color Change Cost: Determine the set of distinct colors required for the
       unpainted goal tiles.
       - If the robot's current color is one of the needed colors, it can start
         painting tiles of that color immediately. It will need to change color
         to paint tiles requiring other colors. The minimum number of color
         changes required is the number of distinct needed colors minus 1.
       - If the robot's current color is not one of the needed colors, it must
         first change color to one of the needed colors (cost 1), and then potentially
         change colors for the remaining distinct needed colors. The minimum number
         of color changes required is the number of distinct needed colors.
       Add this calculated value to the total heuristic cost.
    5. Movement Cost: The robot needs to move to a tile adjacent to each unpainted
       goal tile to paint it. For each unpainted goal tile, calculate the minimum
       Manhattan distance from the robot's current tile to any tile adjacent
       to the goal tile. The minimum distance from robot coordinates (rr, cr) to
       a tile adjacent to goal coordinates (rg, cg) is max(0, abs(rr - rg) + abs(cr - cg) - 1).
       The movement cost component of the heuristic is the maximum of these
       minimum distances over all unpainted goal tiles. This represents the
       distance to the "hardest to reach" unpainted tile's vicinity. Add this
       maximum value to the total heuristic cost.
    6. Total Heuristic: The sum of the base cost (paint actions), color change
       cost, and movement cost is the estimated heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painting requirements.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and colors for each tile that needs painting.
        # Assuming goal facts are only of the form (painted tile color)
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                # Fact is (painted tile color)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color
                # else: Handle unexpected goal fact format if necessary

    def _parse_tile_coords(self, tile_name):
        """Parses a tile name like 'tile_r_c' into (row, col) integers."""
        try:
            parts = tile_name.split('_')
            # Expecting format like 'tile_1_5' -> ['tile', '1', '5']
            if len(parts) == 3 and parts[0] == 'tile':
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            else:
                # Handle unexpected tile name format
                # print(f"Warning: Unexpected tile name format: {tile_name}")
                return None # Or raise an error
        except (ValueError, IndexError) as e:
            # Handle cases where parts are not integers or list indexing fails
            # print(f"Warning: Could not parse tile coordinates for {tile_name}: {e}")
            return None

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Robot State
        robot_tile = None
        robot_color = None
        # Assuming there's only one robot and it's always at a tile and has a color
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    # Fact is (robot-at robot_name tile_name)
                    robot_tile = parts[2]
                elif parts[0] == "robot-has" and len(parts) == 3:
                     # Fact is (robot-has robot_name color_name)
                    robot_color = parts[2]
            # Optimization: Stop searching once both are found
            if robot_tile and robot_color:
                break

        # If robot state cannot be determined, heuristic is infinite or large value (problematic)
        # Assuming valid states always contain robot-at and robot-has
        if not robot_tile or not robot_color:
             # This state might be unreachable or invalid according to domain rules
             # Returning infinity indicates this state is likely not on a valid path
             return float('inf')

        robot_coords = self._parse_tile_coords(robot_tile)
        if robot_coords is None:
             # Handle parsing error - should not happen with valid tile names
             return float('inf') # Indicate an invalid state

        # 2. Identify Unpainted Goal Tiles
        unpainted_goal_tiles = [] # List of (tile_name, goal_color) tuples
        current_painted_facts = set(fact for fact in state if match(fact, "painted", "*", "*"))

        for goal_tile, goal_color in self.goal_paintings.items():
            required_fact = f"(painted {goal_tile} {goal_color})"
            if required_fact not in current_painted_facts:
                 unpainted_goal_tiles.append((goal_tile, goal_color))

        # If no unpainted goal tiles, the goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 3. Base Cost (Paint Actions)
        num_unpainted = len(unpainted_goal_tiles)

        # 4. Color Change Cost
        needed_colors = set(color for tile, color in unpainted_goal_tiles)
        color_change_cost = 0
        if robot_color not in needed_colors:
            # Must change color at least once to get a needed color
            color_change_cost = len(needed_colors)
        else:
            # Already have a needed color, only need to change for other needed colors
            color_change_cost = len(needed_colors) - 1
        # Ensure cost is not negative (happens if len(needed_colors) is 1 and robot has it)
        color_change_cost = max(0, color_change_cost)


        # 5. Movement Cost
        robot_r, robot_c = robot_coords
        distances_to_neighbors = []
        for goal_tile, _ in unpainted_goal_tiles:
            goal_coords = self._parse_tile_coords(goal_tile)
            if goal_coords is None:
                # Handle parsing error - should not happen
                return float('inf') # Indicate an invalid state

            goal_r, goal_c = goal_coords

            # Manhattan distance from robot to goal tile
            dist_to_goal = abs(robot_r - goal_r) + abs(robot_c - goal_c)

            # Minimum Manhattan distance from robot to any tile adjacent to goal tile
            # An adjacent tile is 1 step away from the goal tile.
            # The minimum distance from robot to an adjacent tile is max(0, dist_to_goal - 1).
            min_dist_to_adjacent = max(0, dist_to_goal - 1)
            distances_to_neighbors.append(min_dist_to_adjacent)

        # The movement cost is the maximum distance to reach the vicinity of any unpainted tile
        max_movement_distance = max(distances_to_neighbors)


        # 6. Total Heuristic
        total_cost = num_unpainted + color_change_cost + max_movement_distance

        return total_cost
