from fnmatch import fnmatch
# Assuming heuristic_base is available in the heuristics directory
from heuristics.heuristic_base import Heuristic
import math # Used for float('inf')

# Helper functions (can be outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Extract (row, col) from a tile name like 'tile_R_C'."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            # PDDL uses tile_R_C where R is row, C is column
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a valid tile name format (R or C not integer)
    return None

def get_tile_name(row, col):
    """Construct a tile name 'tile_R_C' from (row, col)."""
    # Note: This constructs the PDDL name, not necessarily a valid tile in the problem
    return f"tile_{row}_{col}"

def manhattan_distance(coord1, coord2):
    """Calculate Manhattan distance between two (row, col) coordinates."""
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

# The heuristic class
class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on:
    1. Number of tiles that need painting.
    2. Number of tiles that need painting but are currently blocked (e.g., by a robot).
    3. Minimum color changes required across all robots to cover needed colors.
    4. Estimated movement cost: sum of minimum Manhattan distances from any robot
       to a valid painting position for each tile needing paint.

    Assumes tile names are in the format 'tile_R_C' where R is row and C is column.
    Assumes row indices increase upwards based on the 'up'/'down' predicate structure
    in the example PDDL, so a mapping to standard grid coordinates (row increasing downwards)
    is used for distance calculations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes tile coordinates (both PDDL and grid) and goal painted tiles.
        """
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Extract all tile objects and their PDDL coordinates (R, C)
        self.pddl_tile_coords = {}
        self.all_tiles = set()

        # Tiles can be found in initial state predicates or static facts (adjacency)
        # Collect all potential object names from initial state and static facts
        all_objects_in_facts = set()
        for fact in task.initial_state | task.static:
             all_objects_in_facts.update(get_parts(fact)[1:]) # Add all parameters as potential objects

        # Filter for objects that look like tiles and parse PDDL coordinates
        for obj_name in all_objects_in_facts:
            coords = parse_tile_name(obj_name)
            if coords is not None:
                self.pddl_tile_coords[obj_name] = coords
                self.all_tiles.add(obj_name)

        # Determine MaxRow to map PDDL row index (increasing upwards)
        # to standard grid row index (increasing downwards)
        self.MaxRow = 0
        if self.pddl_tile_coords:
             self.MaxRow = max(r for r, c in self.pddl_tile_coords.values())

        # Create mapping to standard grid coordinates (row increases downwards)
        self.grid_tile_coords = {
            tile_name: (self.MaxRow - r, c)
            for tile_name, (r, c) in self.pddl_tile_coords.items()
        }

        # Store goal painted tiles
        self.goal_painted_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                # Ensure the goal tile exists in the problem
                if tile in self.all_tiles:
                    self.goal_painted_tiles[tile] = color
                # else: Goal refers to non-existent tile.
                # This heuristic will effectively ignore this goal.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check for unsolvable states (tile painted with wrong color)
        current_painted_tiles = {}
        current_clear_tiles = set()
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "painted" and len(parts) == 3:
                current_painted_tiles[parts[1]] = parts[2]
            elif parts[0] == "clear" and len(parts) == 2:
                current_clear_tiles.add(parts[1])
            elif parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]

        # Check if any tile is painted with the wrong color according to the goal
        for tile, color_painted in current_painted_tiles.items():
            if tile in self.goal_painted_tiles and self.goal_painted_tiles[tile] != color_painted:
                return float('inf') # Tile painted with wrong color, likely unsolvable

        # Identify pending tasks (tiles that need to be painted correctly)
        pending_tasks = set()
        for tile, goal_color in self.goal_painted_tiles.items():
            # Only consider goals for tiles that actually exist in the problem
            if tile in self.all_tiles:
                if tile not in current_painted_tiles or current_painted_tiles[tile] != goal_color:
                    pending_tasks.add((tile, goal_color))

        # If no pending tasks, goal is reached
        if not pending_tasks:
            return 0

        # --- Heuristic Component 1: Paint Actions ---
        # Each pending task requires at least one paint action
        h_paint = len(pending_tasks)

        # --- Heuristic Component 2: Clear Actions ---
        # Count tiles that need painting but are not clear (e.g., robot is on it)
        # These tiles need a robot to move off first.
        unclear_pending_tiles = {tile for tile, color in pending_tasks if tile not in current_clear_tiles}
        h_clear = len(unclear_pending_tiles)

        # --- Heuristic Component 3: Color Change Actions ---
        # Count colors needed by pending tasks
        color_demand = {}
        for tile, color in pending_tasks:
            color_demand[color] = color_demand.get(color, 0) + 1

        # Count robots having each color
        robot_supply = {}
        for robot, color in robot_colors.items():
            robot_supply[color] = robot_supply.get(color, 0) + 1

        # Minimum color changes needed to satisfy demand (simplified)
        # This counts how many "painting slots" of a certain color are needed
        # beyond the number of robots currently holding that color.
        h_color = 0
        for color, demand in color_demand.items():
             supply = robot_supply.get(color, 0)
             h_color += max(0, demand - supply)


        # --- Heuristic Component 4: Movement Actions ---
        # Estimate movement cost: sum of minimum Manhattan distances from any robot
        # to a valid painting position for each pending tile.
        h_movement = 0
        for tile_X, color_Y in pending_tasks:
            # Get PDDL coordinates of the tile to be painted
            pddl_r, pddl_c = self.pddl_tile_coords[tile_X]

            # Determine potential robot locations (PDDL names) to paint tile_X
            # Robot at tile_R'_C' can paint tile_R_C if (up tile_R_C tile_R'_C') or (down tile_R_C tile_R'_C')
            # (up tile_R_C tile_R'_C') means tile_R_C is above tile_R'_C'. PDDL row R < R'. So R' = R+1.
            # (down tile_R_C tile_R'_C') means tile_R_C is below tile_R'_C'. PDDL row R > R'. So R' = R-1.
            # Potential robot locations are tile_{R+1}_C and tile_{R-1}_C (in PDDL names)
            paint_pos_pddl_names = [
                get_tile_name(pddl_r + 1, pddl_c), # Robot below, paints up
                get_tile_name(pddl_r - 1, pddl_c)  # Robot above, paints down
            ]

            # Filter for actual existing tiles and get their grid coordinates
            paint_pos_grid_coords = []
            for pddl_name in paint_pos_pddl_names:
                if pddl_name in self.all_tiles:
                    paint_pos_grid_coords.append(self.grid_tile_coords[pddl_name])

            # If no valid painting positions found for a goal tile, something is wrong
            # with the problem definition or grid structure.
            if not paint_pos_grid_coords:
                 # This tile cannot be painted vertically according to static facts.
                 # If it's a goal tile, the problem is likely unsolvable.
                 # Return infinity.
                 return float('inf')


            min_dist_for_task = float('inf')
            # If there are no robots, movement cost is infinite (cannot paint)
            if not robot_locations:
                 return float('inf')

            for robot_name, robot_loc_tile in robot_locations.items():
                # Ensure robot is at a known tile location
                if robot_loc_tile not in self.grid_tile_coords:
                     # Robot is at an unknown location? Should not happen in valid state.
                     # Treat this robot as unable to contribute to movement for now.
                     continue

                robot_grid_coord = self.grid_tile_coords[robot_loc_tile]

                for pp_grid_coord in paint_pos_grid_coords:
                    dist = manhattan_distance(robot_grid_coord, pp_grid_coord)
                    min_dist_for_task = min(min_dist_for_task, dist)

            # If min_dist_for_task is still inf, it means no *available* robot could reach any painting position.
            # This might happen if robots are at unknown locations or grid is disconnected.
            # Treat as unsolvable or very high cost.
            if min_dist_for_task == float('inf'):
                 return float('inf') # Cannot reach painting position

            h_movement += min_dist_for_task

        # --- Total Heuristic ---
        total_h = h_paint + h_clear + h_color + h_movement

        return total_h
