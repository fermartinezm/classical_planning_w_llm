# Import necessary modules
import re
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In the actual environment, this import will be from the framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a minimal dummy class if the real one isn't found
    class Heuristic:
        def __init__(self, task):
            # Dummy implementation
            self.goals = task.goals
            self.static = task.static
            pass
        def __call__(self, node):
            # Dummy implementation
            return 0 # Or raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Handle potential extra spaces or malformed facts gracefully
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are currently
    unpainted or painted with the wrong color. It sums the estimated cost for
    each such tile, considering the minimum effort required by any robot.
    The estimated cost for a single tile includes the cost for a robot to
    acquire the correct color, move to a tile adjacent to the target tile,
    and perform the paint action. Manhattan distance is used to estimate movement cost.

    # Assumptions
    - Tiles are arranged in a grid, inferred from 'up', 'down', 'left', 'right' predicates.
    - Manhattan distance is used as a lower bound for movement cost, ignoring obstacles ('clear' tiles) for movement paths.
    - Robots always hold a color initially. Changing color costs 1 action.
    - Painting a tile costs 1 action and requires the robot to be on an adjacent tile with the correct color, and the target tile to be clear.
    - Once a tile is painted, it cannot be repainted or unpainted. If a tile is painted with the wrong color, the problem is unsolvable (heuristic returns infinity).
    - If a tile needing paint is not clear and no robot is on it, it's considered a dead end (heuristic returns infinity).
    - The heuristic sums costs for each unpainted/wrongly painted goal tile independently, which might overestimate but aims to guide greedy search.

    # Heuristic Initialization
    - Parses the static facts to build a mapping from tile names (e.g., 'tile_r_c') to grid coordinates (r, c).
    - Stores the goal state, specifically the required color for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Initialize Total Cost:** Set the total heuristic cost to 0.
    2. **Extract State Information:**
       - Identify the current location and color held by each robot.
       - Identify which tiles are currently painted and with what color.
    3. **Identify Unpainted Goal Tiles and Check for Dead Ends:**
       - Iterate through the goal conditions to find all tiles that need to be painted with a specific color.
       - For each goal tile:
         - Check if it is currently painted.
         - If it is painted with a color *different* from the goal color, the state is unsolvable. Return infinity immediately.
         - If it is not painted with the goal color (meaning it's either clear or painted with the wrong color, which is handled), add it to a list of tiles needing paint.
    4. **Handle Goal State:** If the list of tiles needing paint is empty, the current state is a goal state. Return 0.
    5. **Estimate Cost for Each Unpainted Goal Tile:**
       - For each tile `T` that needs to be painted with color `C`:
         - **Check if Paintable:** Verify if the tile `T` is currently `clear`. If it is not `clear`, check if any robot is currently located *at* tile `T`.
         - If `T` is not `clear` and no robot is located at `T`, then `T` must be painted with the wrong color (already checked) or blocked in an unsolvable way. Return infinity.
         - **Determine Applicable Robots:** If a robot `R_on_T` is located at tile `T`, only `R_on_T` can paint it next (by moving off first). If tile `T` is `clear`, any robot can potentially paint it.
         - **Calculate Minimum Robot Cost:** Initialize the minimum cost for painting tile `T` to infinity. Iterate through the applicable robots determined in the previous step:
           - Get robot `R`'s current location (`LocR`) and color (`ColorR`).
           - Calculate the cost for `R` to change color to `C`: 1 if `ColorR` is not `C`, otherwise 0.
           - Calculate the cost for `R` to move and paint tile `T`:
             - If `LocR` is the same as `T`: The robot must first move off `T` to an adjacent tile (cost 1), then paint `T` from that adjacent tile (cost 1). Total move+paint cost = 2.
             - If `LocR` is different from `T`: The robot must move from `LocR` to a tile `X` adjacent to `T` (minimum Manhattan distance `max(0, distance(LocR, T) - 1)`), then paint `T` from `X` (cost 1). Total move+paint cost = `max(0, distance(LocR, T) - 1) + 1`.
           - The total cost for robot `R` to paint tile `T` is `color_change_cost + move_paint_cost`.
           - Update the minimum cost for tile `T` with the minimum cost found among the applicable robots.
         - **Check Solvability for Tile:** If the minimum cost for tile `T` is still infinity (e.g., no applicable robots exist), return infinity.
         - **Add to Total:** Add the minimum cost for tile `T` to the `total_heuristic_cost`.
    6. **Return Total Cost:** Return the `total_heuristic_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and grid structure."""
        self.goals = task.goals
        static_facts = task.static

        # Map tile names to (row, col) coordinates
        self.tile_coords = {}
        # Map (row, col) coordinates to tile names
        self.coords_tile = {}

        # Extract tile names and infer grid structure
        tile_names = set()

        # Collect all objects that are tiles by looking at predicates involving tiles
        # Assumes tile names start with 'tile_'
        for fact in static_facts:
            parts = get_parts(fact)
            # Consider predicates that involve tiles as arguments
            if parts and parts[0] in ['up', 'down', 'left', 'right', 'clear', 'painted', 'robot-at']:
                 for part in parts[1:]:
                     if part.startswith('tile_'):
                         tile_names.add(part)

        # Parse tile names like 'tile_r_c' to get coordinates
        for tile_name in tile_names:
            match_rc = re.match(r'tile_(\d+)_(\d+)', tile_name)
            if match_rc:
                r, c = int(match_rc.group(1)), int(match_rc.group(2))
                self.tile_coords[tile_name] = (r, c)
                self.coords_tile[(r, c)] = tile_name
            # Tiles not matching 'tile_r_c' pattern will not have coordinates,
            # and distance calculations involving them will return infinity.

        # Store goal requirements: {tile_name: color}
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles by name."""
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
            # This indicates an object is not a recognized tile in the grid.
            return float('inf')
        r1, c1 = self.tile_coords[tile1_name]
        r2, c2 = self.tile_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, location = get_parts(fact)[1:]
                if robot not in robot_info: robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1:]
                if robot not in robot_info: robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # Identify tiles that still need to be painted with the correct color
        unpainted_goal_tiles = {} # {tile_name: goal_color}
        painted_facts = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "painted", "*", "*")}

        for goal_tile, goal_color in self.goal_tiles.items():
            if goal_tile not in painted_facts or painted_facts[goal_tile] != goal_color:
                 # Check for dead ends: tile painted with wrong color
                 if goal_tile in painted_facts and painted_facts[goal_tile] != goal_color:
                     return float('inf')
                 # Tile needs painting
                 unpainted_goal_tiles[goal_tile] = goal_color

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        total_heuristic_cost = 0

        # For each unpainted goal tile, estimate the minimum cost to paint it
        for tile_to_paint, required_color in unpainted_goal_tiles.items():
            min_tile_cost = float('inf')

            # Check if the tile is clear and if a robot is on it
            tile_is_clear = f"(clear {tile_to_paint})" in state
            robot_on_tile_name = None
            for robot, info in robot_info.items():
                if info.get('location') == tile_to_paint:
                    robot_on_tile_name = robot
                    break

            # Determine which robots can potentially paint this tile next
            robots_to_consider = []
            if robot_on_tile_name:
                 # Only the robot currently on the tile can make it clear and paint it next.
                 robots_to_consider.append(robot_on_tile_name)
            elif tile_is_clear:
                 # The tile is clear, any robot can potentially paint it.
                 robots_to_consider = list(robot_info.keys())
            # If not tile_is_clear and no robot_on_tile_name, it's a dead end (caught below)

            if not robots_to_consider:
                 # No robots available to paint this tile, and it needs painting. Unsolvable.
                 # This also covers the case where not tile_is_clear and no robot_on_tile_name.
                 return float('inf')

            for robot in robots_to_consider:
                 info = robot_info.get(robot) # Use .get for safety
                 if not info or 'location' not in info or 'color' not in info:
                     continue # Should not happen if robots_to_consider comes from robot_info.keys()

                 robot_location = info['location']
                 robot_color = info['color']

                 # Cost to change color if needed
                 color_change_cost = 1 if robot_color != required_color else 0

                 # Cost to move and paint
                 move_paint_cost = float('inf')

                 if robot_location == tile_to_paint:
                     # Robot is on the tile. Must move off (1) then paint from adjacent (1).
                     move_paint_cost = 1 + 1
                 elif tile_is_clear:
                     # Robot is not on the tile, and the tile is clear.
                     # Robot must move from its location to a tile adjacent to tile_to_paint (distance) + paint (1).
                     # Min moves to adjacent X is max(0, distance(LocR, T) - 1).
                     dist_to_tile = self.manhattan_distance(robot_location, tile_to_paint)
                     if dist_to_tile != float('inf'): # Ensure distance is calculable
                         move_cost = max(0, dist_to_tile - 1)
                         paint_cost = 1
                         move_paint_cost = move_cost + paint_cost
                     # else: move_paint_cost remains infinity

                 # Total cost for this robot to paint this tile
                 current_robot_tile_cost = color_change_cost + move_paint_cost

                 min_tile_cost = min(min_tile_cost, current_robot_tile_cost)

            # Add the minimum cost for this tile to the total
            if min_tile_cost == float('inf'):
                 # This tile cannot be painted by any considered robot. Unsolvable.
                 return float('inf')

            total_heuristic_cost += min_tile_cost

        return total_heuristic_cost
