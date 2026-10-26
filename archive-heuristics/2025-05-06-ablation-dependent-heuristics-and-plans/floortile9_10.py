from fnmatch import fnmatch
from collections import defaultdict, deque

# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


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
    # Ensure we don't go out of bounds if parts is shorter than args
    if len(parts) < len(args):
        return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted and clear. For each such tile, it calculates the
    minimum cost for any robot to reach an adjacent painting location, change color
    if necessary, and paint the tile. The total heuristic is the sum of these
    minimum costs over all unpainted goal tiles.

    # Assumptions
    - Goal tiles are initially either clear or painted with the correct color.
    - Tiles painted with the wrong color do not occur for goal tiles in the initial state.
    - Robots always hold a color (the 'free-color' predicate is not used in actions).
    - The grid structure defined by up/down/left/right predicates is consistent.
    - Movement is only possible to clear tiles.
    - Painting is only possible from an adjacent tile on the correct side, and the target tile must be clear.

    # Heuristic Initialization
    - Parses goal facts to create a mapping from goal tile to required color.
    - Parses static facts to build the grid graph (adjacency list) and a mapping
      from each tile to the set of adjacent tiles from which it can be painted.
    - Identifies available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are currently unpainted and clear in the state.
       These are the tiles that contribute to the heuristic value.
    2. For each robot, determine its current location and the color it holds.
    3. For each unpainted, clear goal tile T that needs color C:
       a. Initialize the minimum cost for this tile, `min_cost_T`, to infinity.
       b. Identify all possible adjacent tiles (`PaintLoc`) from which tile T can be painted, based on the static grid structure.
       c. For each such `PaintLoc`:
          i. For each robot R at location Loc_R with color Color_R:
              - Calculate the shortest path distance `d` from Loc_R to `PaintLoc` using BFS. The BFS considers only tiles that are currently clear as traversable destinations, *including* the `PaintLoc` itself. The starting tile Loc_R is the source.
              - If `PaintLoc` is not reachable via clear tiles (e.g., `PaintLoc` is not clear, or no path exists through clear tiles), the distance `d` will be infinity.
              - If `d` is infinity, this robot cannot reach this `PaintLoc` for this tile in the current state via clear tiles; skip this combination.
              - Calculate the color change cost `c`: 0 if Color_R is C, 1 if Color_R is different from C. (We assume required color C is always available if it appears in a goal).
              - The total cost for robot R to paint tile T from `PaintLoc` is `d + c + 1` (move + change_color + paint).
              - Update `min_cost_T = min(min_cost_T, d + c + 1)`.
       d. If after checking all robots and all valid `PaintLoc`s for tile T, `min_cost_T` is still infinity, the problem is likely unsolvable from this state; return infinity for the total heuristic.
       e. Add `min_cost_T` to the total heuristic value.
    4. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task)

        # Store goal locations and required colors for each tile.
        # {tile_name: color_name, ...}
        self.goal_colors = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_colors[tile] = color

        # Build the grid graph (adjacency list) from static facts.
        # {tile_name: [neighbor_tile_name, ...], ...}
        self.grid_graph = defaultdict(list)
        # Build the adjacent paint locations mapping.
        # {painted_tile: [robot_location_tile, ...], ...}
        self.adj_paint_locations = defaultdict(list)

        # If (dir tile1 tile2) is true, robot at tile2 can paint tile1 using paint_dir.
        # So, tile2 is an adjacent paint location for tile1.
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                direction, tile1, tile2 = parts
                # Add edges to the grid graph (bidirectional)
                self.grid_graph[tile1].append(tile2)
                self.grid_graph[tile2].append(tile1)

                # Add to adjacent paint locations mapping
                # If (dir tile1 tile2) is true, robot at tile2 can paint tile1 using paint_dir.
                # So, tile2 is an adjacent paint location for tile1.
                self.adj_paint_locations[tile1].append(tile2)

        # Available colors are implicitly available for change_color action.
        # We need to check if the required color is available.
        self.available_colors = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "available-color", "*")
        }


    def bfs_distance(self, start_tile, target_tile, state):
        """
        Calculates the shortest path distance from start_tile to target_tile
        considering only 'clear' tiles as traversable destinations.
        The start_tile itself does not need to be clear.
        The target_tile must be clear to be reached.
        """
        # The target tile must be clear to be a valid destination for movement.
        if f"(clear {target_tile})" not in state:
             return float('inf')

        if start_tile == target_tile:
            return 0

        # Tiles that are clear in the current state (excluding the start tile if it's not clear)
        # The BFS logic handles the start tile correctly by adding it initially.
        # Subsequent moves require the neighbor to be clear.
        clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        q = deque([(start_tile, 0)])
        visited = {start_tile}

        while q:
            current_tile, dist = q.popleft()

            # Neighbors are defined by the grid graph
            for neighbor in self.grid_graph.get(current_tile, []):
                # A tile is traversable if it is clear
                if neighbor in clear_tiles and neighbor not in visited:
                    if neighbor == target_tile:
                        return dist + 1 # Found target
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))

        # Target not reachable via clear tiles
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        # Using the task's goal_reached method is more robust
        if self.task.goal_reached(state):
             return 0

        total_heuristic = 0

        # Identify current robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}, ...}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif match(fact, "robot-has", "*", "*"):
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # Identify unpainted goal tiles that are clear
        unpainted_clear_goal_tiles = {} # {tile_name: required_color, ...}
        # state is a frozenset of strings like '(predicate arg1 arg2)'

        for goal_tile, required_color in self.goal_colors.items():
            # Check if the tile is already painted with the correct color
            is_painted_correctly = f"(painted {goal_tile} {required_color})" in state

            # If not painted correctly, check if it's clear and needs painting
            if not is_painted_correctly:
                 is_clear = f"(clear {goal_tile})" in state
                 if is_clear:
                     unpainted_clear_goal_tiles[goal_tile] = required_color
                 # If it's not clear and not painted correctly, it's either painted
                 # wrong or occupied. The heuristic ignores these for simplicity,
                 # assuming they don't block goal achievement or are handled implicitly
                 # by other parts of the search/domain.

        # If no unpainted clear goal tiles, heuristic is 0 (as goal_reached check handles the rest)
        if not unpainted_clear_goal_tiles:
             return 0

        for goal_tile, required_color in unpainted_clear_goal_tiles.items():
            min_cost_for_tile = float('inf')

            # Find adjacent locations from which this tile can be painted
            possible_paint_locations = self.adj_paint_locations.get(goal_tile, [])

            for paint_loc in possible_paint_locations:
                 # The BFS checks if paint_loc is clear, so no need to filter here.

                 for robot_name, info in robot_info.items():
                    robot_location = info['location']
                    robot_color = info['color']

                    # Calculate distance from robot's current location to the paint location
                    # BFS considers only clear tiles as traversable destinations, and the target must be clear.
                    distance = self.bfs_distance(robot_location, paint_loc, state)

                    if distance == float('inf'):
                        continue # Robot cannot reach this paint location via clear tiles

                    # Calculate color change cost
                    color_cost = 0
                    if robot_color != required_color:
                        # Need to change color. Assume change_color action is always possible
                        # if the target color is available (checked in __init__).
                        # The cost is 1 action.
                        # We should check if the required color is available at all.
                        if required_color not in self.available_colors:
                             # This goal tile requires a color that is not available.
                             # The problem is unsolvable.
                             return float('inf')
                        color_cost = 1 # Cost to change color


                    # Total cost for this robot to paint this tile from this location
                    cost = distance + color_cost + 1 # move + change_color + paint

                    min_cost_for_tile = min(min_cost_for_tile, cost)

            if min_cost_for_tile == float('inf'):
                # This goal tile is unreachable or unpaintable from the current state
                # by any robot via clear tiles.
                # Return infinity as the problem is likely unsolvable from here.
                return float('inf')

            total_heuristic += min_cost_for_tile

        return total_heuristic
