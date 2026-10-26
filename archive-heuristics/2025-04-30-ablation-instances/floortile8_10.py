from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid facts


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper functions
def get_coords(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integer coordinates."""
    parts = tile_name.split('_')
    # Expecting format 'tile_row_col'
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # Row and column indices are 0-based integers
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None
    else:
        return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates the Manhattan distance between two tiles."""
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)
    if coords1 is None or coords2 is None:
        return float('inf') # Indicate invalid tile names

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

def position_and_paint_cost(robot_loc_name, target_tile_name):
    """
    Estimates the minimum actions (movement + paint) required to get the robot
    into a position adjacent to the target tile and paint it, assuming the robot
    already possesses the correct color.

    - If the robot is currently on the target tile (distance 0):
      It must move off the tile (1 move) to an adjacent tile, and then paint from there (1 paint action). Total = 2 actions.
    - If the robot is currently adjacent to the target tile (distance 1):
      It is already in a position to paint. Needs 1 paint action. Total = 1 action.
    - If the robot is further away from the target tile (distance d > 1):
      It needs at least `d - 1` moves to reach a tile adjacent to the target tile. Then it needs 1 paint action. Total = `(d - 1) + 1 = d` actions.

    This function calculates the minimum number of actions assuming optimal movement
    to an adjacent tile followed by the paint action.
    Returns float('inf') if tile names are invalid.
    """
    dist = manhattan_distance(robot_loc_name, target_tile_name)
    if dist == float('inf'):
        return float('inf')

    if dist == 0: # Robot is on the tile to be painted
        return 2
    elif dist == 1: # Robot is adjacent
        return 1
    else: # Robot is further away (dist > 1)
        return dist


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their specified colors. It sums the estimated costs for color changes and
    the cost for positioning the robot and painting each unpainted goal tile.

    # Assumptions
    - Each paint action costs 1.
    - Each color change action costs 1.
    - Movement between adjacent tiles costs 1 (Manhattan distance applies).
    - The robot always holds exactly one color.
    - Tiles, once painted, cannot be unpainted or repainted. The goal must specify
      the final color for tiles.
    - The grid structure is defined by up/down/left/right predicates, allowing
      Manhattan distance calculation based on tile names like 'tile_row_col'.
    - The 'clear' predicate means the robot is not on that tile.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Static facts (like adjacency or available colors) are not explicitly stored
      in dedicated structures beyond the base `self.static`, as tile coordinates
      and adjacency are derived from tile names and Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles: Iterate through the goal conditions (`self.goals`). For each goal fact `(painted tile_t color_c)`, check if this exact fact is present in the current state (`node.state`). If not, add the tuple `(tile_t, color_c)` to a list of unpainted goal tasks.

    2. Find Robot's State: Extract the robot's current location (tile name) and the color it is currently holding from the current state facts.

    3. Estimate Color Change Cost:
       - Collect the set of unique colors required for the unpainted goal tiles identified in Step 1.
       - If there are no unpainted goal tiles, the color change cost is 0.
       - If there are unpainted goal tiles:
         - Find the color the robot currently holds.
         - If the robot's current color is one of the needed colors, the estimated color change cost is `|needed_colors| - 1`.
         - If the robot's current color is *not* one of the needed colors, the estimated color change cost is `|needed_colors|`.
       - Ensure the calculated color change cost is not negative.

    4. Estimate Movement and Paint Cost:
       - For each unpainted goal tile `t` identified in Step 1, estimate the minimum number of actions required to get the robot into a position adjacent to `t` and perform the paint action. This is calculated using the `position_and_paint_cost(robot_location, t)` helper function. This function accounts for whether the robot is currently on the tile, adjacent to the tile, or further away, using Manhattan distance as a basis for movement cost.
       - Sum this estimated cost for all unpainted goal tasks. If any individual cost is infinite (due to invalid tile names), the total cost is infinite.

    5. Calculate Total Heuristic: The total heuristic value is the sum of the estimated color change cost (Step 3) and the total estimated movement and paint cost (Step 4).

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        super().__init__(task)
        # Store goal conditions for easy access
        self.goals = task.goals

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state facts

        # 1. Identify Unpainted Goal Tiles
        unpainted_goal_tasks = [] # List of (tile_name, color_name) tuples
        goal_painted_facts = set()
        for goal_fact in self.goals:
            # Only consider 'painted' facts in the goal
            if match(goal_fact, "painted", "*", "*"):
                 goal_painted_facts.add(goal_fact)

        # If there are no painted facts in the goal, the goal is trivial (or malformed for this domain)
        if not goal_painted_facts:
             return 0 # Goal is already met or impossible to paint

        current_painted_facts = set()
        robot_location = None
        robot_color = None

        # Extract current state information efficiently
        for fact in state:
            if match(fact, "painted", "*", "*"):
                current_painted_facts.add(fact)
            elif match(fact, "robot-at", "*", "*"):
                # Assuming only one robot
                parts = get_parts(fact)
                if len(parts) == 3: # Expecting (robot-at robot_name tile_name)
                    robot_location = parts[2] # e.g., 'tile_0_4'
            elif match(fact, "robot-has", "*", "*"):
                 # Assuming only one robot and it always has one color
                parts = get_parts(fact)
                if len(parts) == 3: # Expecting (robot-has robot_name color_name)
                    robot_color = parts[2] # e.g., 'white'

        # Find goal tiles that are not painted correctly in the current state
        # A tile needs painting if the goal requires (painted t c) but the state
        # does *not* contain (painted t c). We assume no unpainting action exists,
        # so if a tile is painted with the wrong color, the problem is unsolvable.
        # The heuristic assumes solvable problems and thus only counts tiles
        # that are not yet painted according to the goal.
        for goal_fact in goal_painted_facts:
             if goal_fact not in current_painted_facts:
                 # This tile needs to be painted according to the goal, but isn't yet.
                 parts = get_parts(goal_fact)
                 if len(parts) == 3: # Expecting (painted tile_name color_name)
                    _, tile_name, color_name = parts
                    unpainted_goal_tasks.append((tile_name, color_name))
                 else:
                     # Handle unexpected goal fact format
                     return float('inf') # Indicate potential problem

        # If all goal tiles are painted correctly, heuristic is 0.
        if not unpainted_goal_tasks:
            return 0

        # 2. Find Robot's State (Done in step 1 loop)
        # Check if robot state was found (should always be true in valid states)
        if robot_location is None or robot_color is None:
             # This indicates an invalid state representation
             return float('inf') # Indicate unsolvable or invalid state

        # 3. Estimate Color Change Cost
        needed_colors = {color for tile, color in unpainted_goal_tasks}
        num_color_changes = 0
        # If there are needed colors (which is true if unpainted_goal_tasks is not empty)
        if robot_color not in needed_colors:
            # Robot has a color it doesn't need for any remaining task, must change
            num_color_changes = len(needed_colors)
        else:
            # Robot has a color it needs, saves one initial change
            num_color_changes = len(needed_colors) - 1
        # Ensure color changes is not negative (e.g., if needed_colors was empty, handled above)
        num_color_changes = max(0, num_color_changes)


        # 4. Estimate Movement and Paint Cost
        total_movement_and_paint_cost = 0
        for tile_name, color_name in unpainted_goal_tasks:
            cost_for_tile = position_and_paint_cost(robot_location, tile_name)
            if cost_for_tile == float('inf'): # Handle case where tile name parsing failed
                 return float('inf')
            total_movement_and_paint_cost += cost_for_tile

        # 5. Calculate Total Heuristic
        total_cost = num_color_changes + total_movement_and_paint_cost

        return total_cost
