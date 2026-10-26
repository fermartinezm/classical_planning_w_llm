from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

# Helper function to parse fact strings
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by spaces
    # Handle potential extra spaces
    return fact[1:-1].split()

# Helper function to match fact parts with arguments (including wildcards)
def match(fact, *args):
    """
    Checks if a fact string matches a pattern of arguments.
    Supports fnmatch wildcards in args.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    estimated costs for each unsatisfied goal tile. The estimated cost for
    a single tile includes:
    1. The paint action itself (cost 1).
    2. The cost to acquire the correct color if no robot currently has it (cost 1, simplified).
    3. The minimum movement cost for any robot to reach a tile adjacent to the target tile.

    Assumptions:
    - The problem instances represent a grid of tiles named 'tile_R_C' where R and C are integers.
    - The adjacency predicates (up, down, left, right) define connections consistent with this grid structure.
    - There is no action to unpaint a tile. If a goal tile is painted with the wrong color, the state is considered a dead end (heuristic returns infinity).
    - Solvable problems do not require unpainting.
    - Robots always hold a color.
    - The heuristic does not account for potential conflicts between robots (e.g., needing the same tile or color simultaneously) or the 'clear' precondition for movement/painting beyond the initial check for wrong colors.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic pre-processes the task information.
    It identifies all tile objects from the task definition and parses their grid coordinates
    (row and column) from their names (assuming 'tile_R_C' format). This mapping from
    tile name to (row, col) coordinate is stored for efficient distance calculations later.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic function (`__call__`) takes a state node and computes its heuristic value:
    1. Parse the current locations and held colors for all robots from the state facts.
    2. Identify which tiles are currently painted and with which color.
    3. Iterate through the goal conditions. For each goal `(painted T C)`:
       a. Check if the tile `T` is already painted in the current state.
       b. If `T` is painted with a color `C'` different from the goal color `C`, the state is considered unsolvable, and the heuristic returns `infinity`.
       c. If `T` is not painted, or is painted with the correct color `C` (meaning the goal is already satisfied for this tile), note whether the goal is unsatisfied (i.e., `(painted T C)` is a goal but not in the state).
    4. If there are no unsatisfied goal tiles, the state is a goal state, and the heuristic returns 0.
    5. Initialize the total heuristic value `h` to 0.
    6. Add the number of unsatisfied goal tiles to `h`. This accounts for the minimum number of paint actions required.
    7. Determine the set of colors required by the unsatisfied goal tiles.
    8. Determine the set of colors currently held by the robots.
    9. Identify the colors that are needed but not currently held by any robot. Add the count of these colors to `h`. This estimates the minimum number of `change_color` actions needed to make the required colors available.
    10. For each unsatisfied goal tile `T`:
        a. Find the minimum Manhattan distance from any robot's current location to the target tile `T`. Let this be `D`.
        b. Calculate the minimum number of moves required for a robot to reach *any* tile adjacent to `T`. This is `D - 1` if `D >= 1`, and `1` if `D == 0`.
        c. Add this minimum movement cost to `h`. This estimates the movement effort needed to get a robot into a position to paint the tile.
    11. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task) # Call the base class constructor
        self.goals = task.goals
        self.tile_coords = {}

        # Extract tile objects and their coordinates from task facts
        # Assuming tile names are like 'tile_R_C'
        for fact_str in task.facts:
            parts = get_parts(fact_str)
            # Check if the fact represents a tile object declaration
            # Fact format is typically (type object_name) or (type object_name - parent_type)
            if len(parts) >= 2 and parts[0] == 'tile':
                 tile_name = parts[1]
                 if tile_name.startswith('tile_'):
                    try:
                        # Split 'tile_R_C' into 'tile', 'R', 'C'
                        _, row_str, col_str = tile_name.split('_')
                        self.tile_coords[tile_name] = (int(row_str), int(col_str))
                    except ValueError:
                        # Handle unexpected tile name format if necessary
                        # print(f"Warning: Unexpected tile name format: {tile_name}")
                        pass # Ignore objects not matching tile_R_C format

        # Optional: Check if tile_coords is empty - might indicate issue with parsing or domain
        # if not self.tile_coords:
        #      print("Warning: No tile coordinates parsed. Heuristic might be incorrect.")


    def __call__(self, node):
        state = node.state

        # Parse robot locations and colors from the current state
        robot_locations = {} # Map robot name -> tile name
        robot_colors = {}    # Map robot name -> color name
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robot_locations[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # Identify painted tiles in the current state
        painted_in_state = {} # Map tile name -> color name
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                painted_in_state[tile] = color

        # Identify unsatisfied goal tiles and check for wrong colors
        unsatisfied_goals = [] # List of (tile_name, color_name) tuples
        
        for goal in self.goals:
            # We only care about 'painted' goals for this heuristic
            if match(goal, "painted", "*", "*"):
                _, target_tile, target_color = get_parts(goal)

                if target_tile not in painted_in_state:
                    # The tile is not painted, but needs to be
                    unsatisfied_goals.append((target_tile, target_color))
                elif painted_in_state[target_tile] != target_color:
                    # The tile is painted with the wrong color - this state is likely a dead end
                    # Return infinity as the heuristic value
                    return math.inf

        # If all goal 'painted' conditions are met, the heuristic is 0
        if not unsatisfied_goals:
             return 0

        h = 0

        # 1. Cost component: Paint actions
        # Each unsatisfied tile needs one paint action
        h += len(unsatisfied_goals)

        # 2. Cost component: Color changes
        # Identify colors needed for unsatisfied goals
        needed_colors = {color for (tile, color) in unsatisfied_goals}
        # Identify colors currently held by robots
        held_colors = set(robot_colors.values())
        # Colors that need to be acquired by some robot
        colors_to_acquire = needed_colors - held_colors
        # Add the number of colors that need to be acquired
        h += len(colors_to_acquire)

        # 3. Cost component: Movement
        # For each unsatisfied goal tile, estimate the minimum movement cost
        # for *any* robot to get adjacent to it.
        for (target_tile, target_color) in unsatisfied_goals:
            target_coord = self.tile_coords.get(target_tile)
            if target_coord is None:
                 # This tile was not found in the parsed tile coordinates.
                 # This might indicate an issue with the problem definition or parsing.
                 # Treat this as an unsolvable state from here.
                 # print(f"Error: Coordinates not found for tile {target_tile}")
                 return math.inf

            min_moves_to_adj_for_this_tile = math.inf

            # Find the robot that is closest (in Manhattan distance) to the target tile
            for robot, robot_tile in robot_locations.items():
                robot_coord = self.tile_coords.get(robot_tile)
                if robot_coord is None:
                     # This robot's location tile was not found in parsed coordinates.
                     # print(f"Error: Coordinates not found for robot tile {robot_tile}")
                     continue # Skip this robot, maybe others are fine

                # Calculate Manhattan distance between robot and target tile
                D = abs(robot_coord[0] - target_coord[0]) + abs(robot_coord[1] - target_coord[1])

                # Calculate minimum moves needed to reach a tile adjacent to the target tile
                # If D=0 (robot is on the target tile), it needs 1 move to get to an adjacent tile.
                # If D>=1 (robot is 1 or more steps away), it needs D-1 moves to reach an adjacent tile.
                moves_to_adj = (D - 1) if D > 0 else 1

                min_moves_to_adj_for_this_tile = min(min_moves_to_adj_for_this_tile, moves_to_adj)

            # If min_moves_to_adj_for_this_tile is still infinity, it means no robots were found
            # or none had valid locations. This shouldn't happen in a valid problem instance.
            if min_moves_to_adj_for_this_tile == math.inf:
                 # print("Error: Could not calculate movement cost for a tile (no robots or invalid locations).")
                 return math.inf

            h += min_moves_to_adj_for_this_tile

        return h
