import re
import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before attempting slicing
    if not isinstance(fact, str):
        # This might happen if the state contains non-string representations,
        # though typically states are sets of strings in this framework.
        # Handle or raise an error as appropriate for the planner's state representation.
        # Assuming facts are always strings like '(predicate arg1 arg2)'
        raise TypeError(f"Expected string fact, but got {type(fact)}: {fact}")
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but useful for clarity)
# def match(fact, *args):
#     """Check if a PDDL fact matches a given pattern."""
#     parts = get_parts(fact)
#     # Check if the number of parts matches the number of args, unless args contains wildcards
#     if len(parts) != len(args) and '*' not in args:
#          return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all goal tiles correctly.
    It sums the estimated cost for each individual unpainted goal tile.
    The estimated cost for a single unpainted tile includes:
    1. The paint action itself (cost 1).
    2. The minimum cost for any robot to reach a position adjacent to the tile
       from which it can be painted, and simultaneously have the required color.
       This minimum cost is calculated as the Manhattan distance from the robot's
       current location to the required painting location, plus 1 if the robot
       needs to change color to match the target color for that tile.

    # Assumptions
    - Tiles are arranged in a grid, and tile names like 'tile_x_y' indicate coordinates (row_index, col_index).
    - Movement actions correspond to grid movements (up, down, left, right) between adjacent tiles.
    - The cost of any action (move, change_color, paint) is 1.
    - If a tile is painted with the wrong color in the current state, the problem is considered unsolvable under the domain rules (as there's no unpaint action and paint requires the tile to be clear).
    - The heuristic calculates movement distance using Manhattan distance on the grid, ignoring the 'clear' predicate for movement paths within the heuristic calculation itself for efficiency. The 'clear' predicate is only considered to identify unsolvable states (a wrongly painted tile is not clear and cannot be repainted).
    - Robots start with a color (the `change_color` precondition requires `(robot-has ?r ?c)`).

    # Heuristic Initialization
    - Parses the goal conditions from the task to identify the target color for each tile that needs painting.
    - Identifies all tile objects mentioned in the initial state or static facts and maps their names ('tile_x_y') to grid coordinates (row_index, col_index).
    - Builds a map indicating, for each target tile, which adjacent tiles a robot must be on to paint it, based on the static 'up', 'down', 'left', 'right' predicates. For example, if `(up tile_1_1 tile_0_1)` is a static fact, it means `tile_1_1` is above `tile_0_1`, and a robot at `tile_0_1` can paint `tile_1_1`. Thus, `tile_0_1` is a 'paintable_from' location for `tile_1_1`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current location and held color for each robot, and the painted status (and color) of each tile.
    2. Identify all goal tiles that are not currently painted with the correct color.
    3. Check if any goal tile is currently painted with the *wrong* color. If a tile needs color C_goal but is painted with C_current != C_goal, the state is considered unsolvable under the domain rules (as paint requires the tile to be clear, and a painted tile is not clear). In this case, a very large heuristic value (infinity) is returned.
    4. If there are no wrongly painted goal tiles, iterate through the remaining unpainted goal tiles (those that are clear but need painting according to the goal).
    5. For each unpainted goal tile (target_tile, target_color):
       a. Initialize the minimum combined cost (movement + color change) for any robot to paint this tile to infinity.
       b. Iterate through each robot:
          i. Determine the cost for this robot to get the correct color (1 if its current color is different from target_color, 0 otherwise).
          ii. Find the minimum Manhattan distance from the robot's current location to *any* tile from which the target_tile can be painted (using the precomputed 'paintable_from' map).
          iii. The combined cost for this robot to be ready to paint this tile is the minimum movement distance plus the color change cost.
          iv. Update the minimum combined cost for this tile if the current robot's combined cost is lower.
       c. If the minimum combined cost for this tile is still infinity (meaning no robot can reach a paintable location), the state is likely unsolvable, return infinity.
       d. Add 1 to the total heuristic (for the paint action itself).
       e. Add the minimum combined robot cost (movement + color) found in step 5b to the total heuristic.
    6. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile coordinates,
        and paintable-from relationships from the task definition.
        """
        self.goals = task.goals
        static_facts = task.static
        # Include initial state facts to ensure we find all objects (especially tiles)
        all_facts_and_init = task.initial_state | static_facts

        self.goal_tile_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tile_colors[tile] = color

        self.tile_coords = {}
        tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
        
        # Find all objects mentioned in the initial state or static facts
        all_objects = set()
        for fact in all_facts_and_init:
             # Add all arguments of predicates as potential objects
            all_objects.update(get_parts(fact)[1:])

        # Parse coordinates for all identified tile objects
        for obj in all_objects:
            match = tile_pattern.match(obj)
            if match:
                # PDDL tile names are typically tile_row_col
                row, col = int(match.group(1)), int(match.group(2))
                self.tile_coords[obj] = (row, col)

        self.paintable_from = {}
        # Build map: target_tile -> list of tiles robot can be on to paint it
        # (direction target_tile robot_location_tile) means robot at robot_location_tile
        # can paint target_tile using the corresponding paint action.
        # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1.
        # Robot at tile_0_1 can paint tile_1_1 using paint_up.
        # So, tile_0_1 is a paintable_from location for tile_1_1.
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                target_tile, robot_location_tile = parts[1], parts[2]
                if target_tile not in self.paintable_from:
                    self.paintable_from[target_tile] = []
                self.paintable_from[target_tile].append(robot_location_tile)

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles using precomputed coordinates."""
        # Ensure both tiles exist in our coordinate map
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
             # This indicates an issue with parsing or problem definition
             # Return infinity as these tiles cannot be reached/used
             return float('inf')

        x1, y1 = self.tile_coords[tile1_name]
        x2, y2 = self.tile_coords[tile2_name]
        return abs(x1 - x2) + abs(y1 - y2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}
        # clear_tiles = set() # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            # elif parts[0] == "clear":
            #     clear_tiles.add(parts[1])

        unpainted_goals = set()
        for tile, goal_color in self.goal_tile_colors.items():
            if tile in painted_tiles:
                # Check for wrongly painted tiles (unsolvable)
                if painted_tiles[tile] != goal_color:
                    # This state is likely unsolvable as there's no unpaint action
                    # and paint requires the tile to be clear.
                    return float('inf')
                # If painted correctly, it's satisfied, do nothing
            else:
                # Tile needs to be painted
                unpainted_goals.add(tile)

        if not unpainted_goals:
            return 0 # Goal reached

        total_h = 0
        for tile in unpainted_goals:
            target_color = self.goal_tile_colors[tile]

            min_robot_task_cost = float('inf')

            # Find the minimum cost for any robot to get into position and have the color
            for robot, R_loc in robot_locations.items():
                # Cost to get the correct color
                # Assumes robots always have a color initially based on domain precondition
                color_cost = 1 if robot_colors.get(robot) != target_color else 0

                # Minimum movement cost for this robot to a paintable location for 'tile'
                min_move_cost = float('inf')
                paint_from_tiles = self.paintable_from.get(tile, [])

                # If there are no tiles from which this target tile can be painted,
                # this tile is unreachable/unpaintable. This shouldn't happen in valid grids.
                # If it does, the problem might be unsolvable.
                if not paint_from_tiles:
                     return float('inf') # Unreachable tile

                for paint_from_tile in paint_from_tiles:
                    # Calculate distance from robot's current location to the potential painting location
                    dist = self.manhattan_distance(R_loc, paint_from_tile)
                    min_move_cost = min(min_move_cost, dist)

                # Total cost for this robot to be ready to paint this tile
                # (movement cost + color change cost)
                robot_task_cost = min_move_cost + color_cost
                min_robot_task_cost = min(min_robot_task_cost, robot_task_cost)

            # If min_robot_task_cost is still inf, it means no robot can reach
            # a paintable spot for this tile (e.g., if paintable_from was empty).
            # This state is likely unsolvable.
            if min_robot_task_cost == float('inf'):
                return float('inf')

            # Add cost for the paint action itself (1) + the minimum cost for the best robot to get ready
            total_h += 1 + min_robot_task_cost

        return total_h
