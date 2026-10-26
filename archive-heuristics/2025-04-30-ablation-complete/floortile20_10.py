from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the required number of actions to reach the goal
    by summing an estimated cost for each unpainted goal tile. The estimated
    cost for a single tile includes the paint action itself, the cost to get
    a robot with the correct color, and the minimum movement cost for any
    robot to reach a location from which the tile can be painted.

    # Assumptions
    - The domain represents a grid world where tiles are connected by up/down/left/right predicates.
    - Tile names often follow the format 'tile_R_C' where R is the row and C is the column, allowing grid parsing (though the heuristic relies on explicit adjacency facts).
    - Movement between adjacent tiles costs 1 action.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - A tile must be clear to be painted. If a goal tile is painted with the wrong color, the state is considered unreachable.
    - The heuristic calculates movement costs based on static grid connectivity, ignoring dynamic obstacles like other robots or non-clear tiles (except for wrongly painted goal tiles).
    - Assumes all required goal colors are available in the domain.

    # Heuristic Initialization
    The constructor precomputes the grid structure and distances:
    1. Identifies all tile objects present in the static facts and goals.
    2. Builds an undirected graph representing tile connectivity based on up/down/left/right facts.
    3. Computes all-pairs shortest paths (distances) between tiles using BFS.
    4. Stores the goal conditions, mapping each goal tile to its required color.
    5. Stores the mapping from each tile to the set of locations from which it can be painted, based on static adjacency facts.
    6. Checks if all required goal colors are available in the domain's static facts. If not, the goal is unreachable.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if any required goal color is not available in the domain. If so, return infinity.
    2. Identify all goal tiles that are not yet painted correctly. Let this set be `U`.
    3. For each tile `T` in `U`, check if it is currently painted with a color different from the goal color. If so, the state is unreachable, return infinity.
    4. If `U` is empty, the goal is reached, return 0.
    5. Initialize the total heuristic value `h` to 0.
    6. Get the current location and color for each robot from the current state.
    7. For each tile `T` in the set of unpainted goal tiles `U`:
        a. Determine the required color `C` for `T` from the goal conditions.
        b. Find the set of possible paint locations `Paint_locs_for_T` from which `T` can be painted (based on static adjacency facts precomputed in init).
        c. Calculate the minimum cost for *any* robot to reach *any* of the paint locations in `Paint_locs_for_T` while having the required color `C`. This minimum cost is calculated as:
           `min_cost_for_tile = min_{robot r} ( (1 if r's current color != C else 0) + shortest_distance(r's location, nearest_paint_location_for_T) )`.
           If a robot does not have a color or cannot reach any paint location for `T`, it is not considered for the minimum. If no robot can satisfy the condition, this minimum cost remains infinity.
        d. If `min_cost_for_tile` is infinity (meaning no robot can reach any paint location for this tile with the required color), the state is unreachable, return infinity.
        e. Add `1` (for the paint action) plus `min_cost_for_tile` to the total heuristic value `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, distances,
        goal information, and paint locations from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify all tile objects
        all_tiles = set()
        # Find all terms that look like tiles in static facts and goal facts.
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tiles.add(part)
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith("tile_"):
                     all_tiles.add(part)

        # Map tile names to coordinates (optional for BFS, but useful for understanding)
        self.name_to_coords = {}
        self.coords_to_name = {}
        for tile_name in all_tiles:
            try:
                # Assuming format tile_R_C
                _, r_str, c_str = tile_name.split('_')
                r, c = int(r_str), int(c_str)
                self.name_to_coords[tile_name] = (r, c)
                self.coords_to_name[(r, c)] = tile_name
            except ValueError:
                # Ignore tiles not in tile_R_C format for coordinate mapping
                pass

        # 2. Build adjacency list for movement graph (undirected)
        self.adj = {tile: [] for tile in all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1:] # Fact is (direction t1 t2)
                # t1 is [direction] from t2. Movement is possible between t1 and t2.
                if t1 in self.adj and t2 in self.adj: # Ensure tiles are recognized
                    self.adj[t1].append(t2)
                    self.adj[t2].append(t1)

        # 3. Compute all-pairs shortest paths (BFS from each tile)
        self.dist = {}
        for start_node in all_tiles:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr_node, d = q.popleft()
                self.dist[start_node][curr_node] = d
                # Ensure curr_node is in adj before accessing it
                if curr_node in self.adj:
                    for neighbor in self.adj[curr_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # 4. Store goal tiles and their required colors
        self.goal_tiles_info = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1:]
                self.goal_tiles_info[tile] = color

        # 5. Store paint locations mapping
        self.paint_locs_map = {tile: set() for tile in all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                painted_tile, robot_at_tile = parts[1:]
                # Fact is (direction painted_tile robot_at_tile)
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # To paint tile_1_1 using paint_up, robot must be at tile_0_1
                # So, tile_0_1 is a paint location for tile_1_1
                if painted_tile in self.paint_locs_map and robot_at_tile in all_tiles: # Ensure both are recognized tiles
                    self.paint_locs_map[painted_tile].add(robot_at_tile)

        # 6. Store available colors and check if goal colors are available
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}
        self.goal_colors_available = True
        for required_color in self.goal_tiles_info.values():
            if required_color not in self.available_colors:
                self.goal_colors_available = False
                break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if any required goal color is not available
        if not self.goal_colors_available:
             return float('inf')

        # 2. Identify unpainted goal tiles and check for wrongly painted tiles
        unpainted_goal_tiles = set()
        for goal_tile, required_color in self.goal_tiles_info.items():
            is_painted_correctly = False
            is_painted_wrongly = False

            # Check if the tile is painted
            for fact in state:
                if match(fact, "painted", goal_tile, "*"):
                    painted_color = get_parts(fact)[2]
                    if painted_color == required_color:
                        is_painted_correctly = True
                    else:
                        is_painted_wrongly = True
                    break # Found painted status

            if is_painted_wrongly:
                # 3. Check for unreachable states (wrong color painted)
                return float('inf')

            if not is_painted_correctly:
                 unpainted_goal_tiles.add(goal_tile)

        # 4. If U is empty, goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 5. Initialize heuristic
        h = 0

        # 6. Get robot info (location and color)
        robot_info = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, location = get_parts(fact)[1:]
                # Find robot's color from the state
                robot_color = None
                for color_fact in state:
                    if match(color_fact, "robot-has", robot, "*"):
                        robot_color = get_parts(color_fact)[2]
                        break
                # If robot_color is None, it cannot paint, but we still track its location
                robot_info[robot] = (location, robot_color)

        # 7. Calculate cost for each unpainted tile
        for tile in unpainted_goal_tiles:
            required_color = self.goal_tiles_info[tile]

            paint_locations = self.paint_locs_map.get(tile, set())

            # If a goal tile cannot be painted from anywhere, it's unreachable
            if not paint_locations:
                 return float('inf')

            min_cost_for_tile = float('inf')

            # 7.c. Calculate min cost for any robot to reach a paint location with the required color
            for robot, (r_loc, r_color) in robot_info.items():
                # A robot without a color cannot paint
                if r_color is None:
                    continue

                color_cost = 1 if r_color != required_color else 0

                min_move_cost_for_robot_to_paint_tile = float('inf')
                for paint_loc in paint_locations:
                    # Check if paint_loc is a recognized tile and reachable from robot's current location
                    if r_loc in self.dist and paint_loc in self.dist[r_loc]:
                         min_move_cost_for_robot_to_paint_tile = min(min_move_cost_for_robot_to_paint_tile, self.dist[r_loc][paint_loc])

                # If the robot can reach at least one paint location
                if min_move_cost_for_robot_to_paint_tile != float('inf'):
                    min_cost_for_robot = color_cost + min_move_cost_for_robot_to_paint_tile
                    min_cost_for_tile = min(min_cost_for_tile, min_cost_for_robot)

            # 7.d. If no robot can reach any paint location for this tile with the required color
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            # 7.e. Add cost for this tile (paint + min_robot_cost)
            h += 1 + min_cost_for_tile

        # 8. Return total heuristic
        return h
