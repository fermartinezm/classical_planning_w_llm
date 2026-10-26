import re
from collections import deque
from fnmatch import fnmatch
import math # For infinity

# Assume heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# In the actual planning environment, the import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions (adapted from examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Custom helper function for floortile
def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_row_col' into a (row, col) tuple."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen in valid instances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all required tiles with the correct colors.
    It sums the estimated cost for each unpainted goal tile independently.
    The estimated cost for a single tile includes:
    1. Cost to move a robot off the tile if it's currently occupied.
    2. Minimum cost for any robot to reach a tile adjacent to the target tile (from which it can paint).
    3. Cost for that robot to change color if it doesn't have the required color.
    4. Cost of the paint action itself.

    # Assumptions
    - Tiles are arranged in a grid, and movement/painting adjacency follows the grid structure defined by `up`, `down`, `left`, `right` predicates.
    - Tile names follow the format `tile_row_col`.
    - Tiles needing painting in the goal are initially clear or occupied by a robot. If occupied, the robot must move off.
    - If a tile is painted with the wrong color, the problem is considered unsolvable (no unpaint action).
    - The heuristic ignores potential conflicts like multiple robots needing the same tile or blocking each other, and assumes color changes are always possible if the color is available (which is guaranteed for goal colors).
    - Distance is calculated as shortest path on the grid graph defined by adjacency predicates.

    # Heuristic Initialization
    - Parses all tile names to get their grid coordinates.
    - Builds an adjacency graph of tiles based on `up`, `down`, `left`, `right` static facts.
    - Computes all-pairs shortest paths on the grid graph using BFS.
    - Identifies for each tile, the set of tiles from which it can be painted (based on `up`, `down`, `left`, `right` static facts).
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal facts of the form `(painted ?tile ?color)`.
    2. Filter these goal facts to find the set of unpainted goal tiles `U = {(T, C) | (painted T C) is a goal and (painted T C) is not in the current state}`.
    3. If `U` is empty, the goal is reached, return 0.
    4. Check if any tile `T` in `U` is currently painted with a *different* color `C'`. If so, the state is likely unsolvable, return a large value (infinity).
    5. Initialize `total_heuristic_cost = 0`.
    6. Identify the current location of each robot and the color each robot holds from the state.
    7. Identify which tiles are currently occupied by robots.
    8. For each unpainted goal tile `(T, C)` in `U`:
        a. Initialize `cost_for_this_tile = 0`.
        b. If tile `T` is currently occupied by a robot, add 1 to `cost_for_this_tile` (cost to move the robot off).
        c. Calculate the minimum cost for *any* robot to paint tile `T` with color `C`:
            i. Initialize `min_robot_painting_cost = math.inf`.
            ii. Find the set of tiles `Adj_T` from which tile `T` can be painted (precomputed in `__init__`). If `Adj_T` is empty, this tile cannot be painted, return infinity for the total heuristic.
            iii. For each robot `R` at location `Loc_R` holding color `Color_R`:
                - Calculate the minimum distance from `Loc_R` to any tile `X` in `Adj_T` using the precomputed shortest paths. Let this be `min_dist_R_to_Adj_T`. If `Loc_R` or any `X` in `Adj_T` is not in the distance matrix (e.g., isolated tile), this path is impossible.
                - If `Loc_R` not in self.dist_matrix:
                    # Robot is at an unknown location, cannot calculate distance
                    continue # Try next robot

                current_robot_min_dist_to_adj = math.inf
                for adj_tile in paint_from_tiles:
                    # Ensure the adjacent tile is a known tile and reachable from current_loc
                    if adj_tile in self.dist_matrix.get(current_loc, {}):
                         dist = self.dist_matrix[current_loc][adj_tile]
                         current_robot_min_dist_to_adj = min(current_robot_min_dist_to_adj, dist)

                if current_robot_min_dist_to_adj == math.inf:
                    # Robot cannot reach any paintable-from tile for this goal tile
                    continue # Try next robot

                # Calculate the cost for robot R to paint T
                cost_for_robot = current_robot_min_dist_to_adj # Cost of move actions
                # Cost to change color if needed
                # Use .get for robot_colors as a robot might not have a color initially
                if robot_colors.get(robot) != goal_color:
                    cost_for_robot += 1 # Cost of change_color action
                cost_for_robot += 1 # Cost of paint action

                min_robot_painting_cost = min(min_robot_painting_cost, cost_for_robot)

            iv. If `min_robot_painting_cost` is still infinity after checking all robots, return infinity for the total heuristic (no robot can paint this tile).
        d. Add `cost_for_this_tile + min_robot_painting_cost` to `total_heuristic_cost`.
    9. Return `total_heuristic_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Include initial state to find all objects

        self.tile_coords = {}
        self.adj_list = {}
        self.paintable_from = {}
        all_tiles = set()

        # 1. Collect all tile objects and parse coordinates
        # Iterate through initial state, goals, and static facts to find arguments that look like tiles
        tile_pattern = re.compile(r'tile_\d+_\d+')
        for fact in initial_state | self.goals | static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if tile_pattern.match(part):
                     all_tiles.add(part)

        for tile in all_tiles:
             coords = parse_tile_name(tile)
             if coords:
                 self.tile_coords[tile] = coords
                 self.adj_list[tile] = set()
                 self.paintable_from[tile] = set()

        # 2. Build adjacency list and paintable_from mapping from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                dir_pred, tile_y, tile_x = parts # (dir tile_y tile_x) means y is dir from x
                # Ensure both tiles are recognized before adding adjacency
                if tile_x in self.adj_list and tile_y in self.adj_list:
                    # Adjacency is bidirectional for movement
                    self.adj_list[tile_x].add(tile_y)
                    self.adj_list[tile_y].add(tile_x)
                    # Painting tile_y from tile_x is possible if (dir tile_y tile_x)
                    self.paintable_from[tile_y].add(tile_x)

        # 3. Compute all-pairs shortest paths using BFS
        self.dist_matrix = {}
        for start_tile in self.tile_coords:
            self.dist_matrix[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            while q:
                current_tile, dist = q.popleft()
                self.dist_matrix[start_tile][current_tile] = dist

                # Ensure current_tile has entries in adj_list before iterating
                if current_tile in self.adj_list:
                    for neighbor in self.adj_list.get(current_tile, set()): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles
        unpainted_goals = set()
        wrongly_painted = False
        for goal in self.goals:
            g_parts = get_parts(goal)
            if g_parts and g_parts[0] == 'painted' and len(g_parts) == 3:
                goal_tile, goal_color = g_parts[1], g_parts[2]
                is_painted_correctly = False
                is_painted_wrongly = False

                # Check current state for the status of the goal tile
                for fact in state:
                    f_parts = get_parts(fact)
                    if f_parts and f_parts[0] == 'painted' and len(f_parts) == 3 and f_parts[1] == goal_tile:
                        if f_parts[2] == goal_color:
                            is_painted_correctly = True
                        else:
                            is_painted_wrongly = True
                            break # Found wrong color, no need to check other facts for this tile

                if is_painted_wrongly:
                    wrongly_painted = True
                    break # Found a wrongly painted tile, return infinity
                if not is_painted_correctly:
                    unpainted_goals.add((goal_tile, goal_color))

        # 2. Check if goal is reached or state is unsolvable due to wrong color
        if not unpainted_goals:
            return 0 # Goal reached
        if wrongly_painted:
            return math.inf # Unsolvable state

        # 3. Initialize cost and get robot info
        total_heuristic_cost = 0
        robot_locs = {}
        robot_colors = {}
        occupied_tiles = set()

        for fact in state:
            f_parts = get_parts(fact)
            if f_parts and f_parts[0] == 'robot-at' and len(f_parts) == 3:
                robot, tile = f_parts[1], f_parts[2]
                robot_locs[robot] = tile
                occupied_tiles.add(tile)
            elif f_parts and f_parts[0] == 'robot-has' and len(f_parts) == 3:
                robot, color = f_parts[1], f_parts[2]
                robot_colors[robot] = color

        # 4. Calculate cost for each unpainted goal tile
        for goal_tile, goal_color in unpainted_goals:
            cost_for_this_tile = 0

            # Cost to move robot off if tile is occupied
            if goal_tile in occupied_tiles:
                 # Add cost for a robot to move off this tile
                 cost_for_this_tile += 1 # Cost of move action

            # Calculate minimum cost for any robot to paint this tile
            min_robot_painting_cost = math.inf

            # Get paintable-from tiles for the goal tile
            paint_from_tiles = self.paintable_from.get(goal_tile, set())
            if not paint_from_tiles:
                 # This tile cannot be painted from anywhere based on static facts
                 # This should ideally not happen in valid problems where the tile is reachable and paintable
                 # Treat as unsolvable or unreachable goal part
                 return math.inf

            # Ensure goal_tile is a known tile in the grid
            if goal_tile not in self.tile_coords:
                 return math.inf # Goal tile is somehow not in the grid graph

            for robot, current_loc in robot_locs.items():
                cost_for_robot = 0

                # Cost to change color if needed
                # Use .get for robot_colors as a robot might not have a color initially
                if robot_colors.get(robot) != goal_color:
                    cost_for_robot += 1 # Cost of change_color action

                # Minimum distance from robot's current location to any paintable-from tile
                min_dist_to_adj = math.inf

                # Ensure robot's current location is a known tile in the grid
                if current_loc in self.dist_matrix:
                    for adj_tile in paint_from_tiles:
                        # Ensure the adjacent tile is a known tile and reachable from current_loc
                        if adj_tile in self.dist_matrix.get(current_loc, {}):
                             dist = self.dist_matrix[current_loc][adj_tile]
                             min_dist_to_adj = min(min_dist_to_adj, dist)

                if min_dist_to_adj == math.inf:
                    # Robot cannot reach any paintable-from tile for this goal tile
                    continue # Try next robot

                cost_for_robot += min_dist_to_adj # Cost of move actions
                cost_for_robot += 1 # Cost of paint action

                min_robot_painting_cost = min(min_robot_painting_cost, cost_for_robot)

            if min_robot_painting_cost == math.inf:
                 # No robot can paint this tile
                 return math.inf

            total_heuristic_cost += cost_for_this_tile + min_robot_painting_cost

        return total_heuristic_cost
