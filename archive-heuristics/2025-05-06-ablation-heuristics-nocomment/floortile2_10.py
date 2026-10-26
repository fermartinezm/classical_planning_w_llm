import re
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) tuple of integers."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen for valid tile names

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated costs
    for each goal tile that is not yet painted correctly. The estimated cost
    for a single unpainted goal tile includes:
    1. A cost for the paint action (1).
    2. A cost for changing color, if the required color is not held by any robot
       that needs to paint a tile of that color (1 per color needed).
    3. A cost for robot movement, estimated by the minimum Manhattan distance
       from any robot's current location to any tile adjacent to the goal tile.
       This ignores the 'clear' predicate constraint on movement.

    Assumptions:
    - Tile names follow the format 'tile_R_C' where R and C are integers representing
      row and column indices.
    - The grid is rectangular and connected according to 'up', 'down', 'left', 'right'
      predicates, consistent with the 'tile_R_C' naming convention (R increases downwards,
      C increases rightwards).
    - Tiles painted with the wrong color in the initial state do not occur in solvable
      problems (as there is no unpaint action). The heuristic only considers goal tiles
      that are not yet painted with the correct color.
    - The 'clear' predicate constraint on movement is relaxed, using Manhattan distance
      as a lower bound for movement cost.
    - Multiple robots can work in parallel and their tasks (painting different tiles)
      are independent for heuristic calculation purposes (costs are summed).
    - 'available-color' predicate is assumed to be true for all colors at all locations
      where a robot might need to change color (typically where the robot is).

    Heuristic Initialization:
    - Parses the goal state to identify all tiles that need to be painted and their
      required colors, storing them in `self.goal_paintings`.
    - Parses static facts and initial state facts to determine the grid dimensions
      (max_row, max_col) based on tile names, which is used to check for valid
      adjacent tile coordinates.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h = 0`.
    2. Initialize a set `colors_needed_by_unpainted_tiles` to track colors required
       by goal tiles that are not yet painted correctly.
    3. Initialize dictionaries `robot_locations` and `robot_colors` to store the
       current location and held color for each robot from the current state.
    4. Iterate through the facts in the current state:
       - If the fact is `(robot-at R T)`, store `robot_locations[R] = T`.
       - If the fact is `(robot-has R C)`, store `robot_colors[R] = C`.
       - If the fact is `(painted T C)`, store this information (implicitly, by
         checking against the goal later).
    5. Initialize a list `unpainted_goal_tiles` to store (tile, color) pairs for
       goal tiles not yet painted correctly.
    6. Iterate through the `self.goal_paintings` dictionary (tile -> required_color).
       For each goal tile `T` and its required color `C`:
       - Check if the fact `(painted T C)` exists in the current state.
       - If `(painted T C)` is NOT in the current state:
         - Add 1 to `h` (cost for the paint action).
         - Add `C` to the `colors_needed_by_unpainted_tiles` set.
         - Add `(T, C)` to the `unpainted_goal_tiles` list.
    7. Calculate the cost for changing colors:
       - Create a set `colors_held` containing colors currently held by robots.
       - For each color `C` in `colors_needed_by_unpainted_tiles`:
         - If `C` is NOT in `colors_held`, add 1 to `h` (cost for one robot to change to color C).
       This counts the minimum number of color changes needed across all robots to cover all required colors.
    8. Calculate the movement cost:
       - For each `(goal_tile, goal_color)` pair in `unpainted_goal_tiles`:
         - Parse `goal_tile` to get its coordinates `(Tr, Tc)`.
         - Initialize `min_dist_to_adjacent = infinity`.
         - Define potential adjacent coordinates: `(Tr-1, Tc)`, `(Tr+1, Tc)`, `(Tr, Tc-1)`, `(Tr, Tc+1)`.
         - For each potential adjacent coordinate `(Ar, Ac)`:
           - Check if `(Ar, Ac)` is a valid tile coordinate within the grid bounds
             (0 <= Ar <= max_row and 0 <= Ac <= max_col).
           - If valid:
             - For each robot `R` at `R_loc` (with coordinates `(Rr, Rc)`):
               - Calculate Manhattan distance: `dist = abs(Rr - Ar) + abs(Rc - Ac)`.
               - Update `min_dist_to_adjacent = min(min_dist_to_adjacent, dist)`.
         - If `min_dist_to_adjacent` is still infinity (e.g., tile has no valid neighbors, unlikely in grid), handle appropriately (e.g., return infinity or a large number, though grid problems usually have neighbors). Otherwise, add `min_dist_to_adjacent` to `h`.
    9. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Also check initial state for tiles

        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Determine grid dimensions from tile names in static facts and initial state
        # Assuming tile names are 'tile_R_C' and R, C are 0-indexed
        max_row = 0
        max_col = 0
        all_facts = static_facts | initial_state # Consider all facts initially present
        for fact in all_facts:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith('tile_'):
                    coords = parse_tile_name(part)
                    if coords:
                        max_row = max(max_row, coords[0])
                        max_col = max(max_col, coords[1])
        self.max_row = max_row
        self.max_col = max_col

    def __call__(self, node):
        state = node.state

        h = 0
        colors_needed_by_unpainted_tiles = set()
        robot_locations = {}
        robot_colors = {}

        # Parse current state for robot info
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        unpainted_goal_tiles = []

        # Identify unpainted goal tiles and needed colors
        for goal_tile, goal_color in self.goal_paintings.items():
            # Check if the goal fact is NOT in the current state
            if f'(painted {goal_tile} {goal_color})' not in state:
                unpainted_goal_tiles.append((goal_tile, goal_color))
                h += 1 # Cost for paint action
                colors_needed_by_unpainted_tiles.add(goal_color)

        # If all goal tiles are painted correctly, h is 0 already.
        if not unpainted_goal_tiles:
            return 0

        # Calculate change_color cost
        # Count how many needed colors are not held by any robot
        colors_held = set(robot_colors.values())
        for color in colors_needed_by_unpainted_tiles:
            if color not in colors_held:
                h += 1 # Cost for one robot to change to this color

        # Calculate movement cost for each unpainted goal tile
        for goal_tile, _ in unpainted_goal_tiles:
            goal_r, goal_c = parse_tile_name(goal_tile)

            min_dist_to_adjacent = float('inf')

            # Potential adjacent coordinates (up, down, left, right)
            adjacent_coords = [
                (goal_r - 1, goal_c), # Up
                (goal_r + 1, goal_c), # Down
                (goal_r, goal_c - 1), # Left
                (goal_r, goal_c + 1)  # Right
            ]

            # Check if there are any robots before calculating distance
            if not robot_locations:
                 # Should not happen in valid problems, but handle defensively
                 return 1000000 # Unsolvable

            for adj_r, adj_c in adjacent_coords:
                # Check if adjacent coordinate is within grid bounds
                if 0 <= adj_r <= self.max_row and 0 <= adj_c <= self.max_col:
                    # This adjacent tile exists in the grid
                    # Find minimum distance from any robot to this adjacent tile
                    for robot, robot_tile in robot_locations.items():
                        robot_r, robot_c = parse_tile_name(robot_tile)
                        dist = abs(robot_r - adj_r) + abs(robot_c - adj_c)
                        min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            # Add the minimum distance required to reach an adjacent tile
            # If min_dist_to_adjacent is still inf, it means the goal tile has no valid neighbors
            # within the determined grid bounds. This is unlikely for typical grid problems
            # but could happen for a single tile problem or a tile on the edge/corner
            # if the bounds calculation is slightly off or the grid is sparse.
            # If it happens, the tile might be unreachable by moving to an adjacent cell.
            # Returning a large number indicates a likely unsolvable state from here.
            if min_dist_to_adjacent == float('inf'):
                 return 1000000 # A large finite value

            h += min_dist_to_adjacent

        return h
