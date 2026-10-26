from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
         return False
    if len(parts) > len(args) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_tile, end_tile, grid_adj):
    """
    Performs Breadth-First Search to find the shortest path distance
    between two tiles in the grid.
    """
    if start_tile == end_tile:
        return 0
    queue = deque([(start_tile, 0)])
    visited = {start_tile}
    while queue:
        current_tile, dist = queue.popleft()
        if current_tile in grid_adj:
            for neighbor in grid_adj[current_tile]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    if neighbor == end_tile: # Found target, return distance
                        return dist + 1
    return float('inf') # Not reachable

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated cost for each unpainted goal tile
    independently, plus a global cost for acquiring necessary colors. The cost for
    a single tile includes making the tile clear (if necessary), moving a robot
    to an adjacent tile from which it can paint, and the paint action itself.

    # Assumptions
    - If a goal tile is not painted with the correct color and is not 'clear', it is assumed
      that a robot is currently occupying that tile, and moving the robot off will make it clear.
      (Problems with goal tiles wrongly painted initially are considered unsolvable).
    - Any robot can paint any tile if it is in the correct adjacent position and has the correct color.
    - The cost of acquiring a needed color is 1 for each unique color required by an unpainted
      goal tile that is not currently held by any robot, provided that color is available. This cost is added once globally.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates represents valid movement paths for robots.
    - The problem is solvable (i.e., required colors are available, grid is connected, no wrongly painted goal tiles initially).

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which colors.
    - Extracts the set of available colors from static facts.
    - Builds the grid adjacency list (`grid_adj`) from 'up', 'down', 'left', 'right' static facts to enable BFS for distance calculation.
    - Builds the paint adjacency list (`paint_adj`) from 'up', 'down', 'left', 'right' static facts to identify which tiles a robot must be on to paint a target tile.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize `total_cost = 0`.
    2. Identify all goal facts of the form `(painted tile color)`.
    3. Identify the set of `unpainted_goal_tiles`, which are goal facts `(painted t c)` that are not present in the current state.
    4. Cache robot locations: `robot_locations = {r: loc for (r, loc) in ... if match(fact, "robot-at", r, loc)}`.
    5. If no robots exist but there are unpainted goal tiles, return `float('inf')`.
    6. For each `(t, c)` in `unpainted_goal_tiles`:
        a. Check if `(clear t)` is in the state.
        b. If `(clear t)` is NOT in the state:
            i. Check if any robot is currently located at tile `t`. Find robots at tile: `robots_at_tile = defaultdict(list)`. Populate this by iterating state facts matching `(robot-at * *)`.
            ii. If no robot is at tile `t`, this implies the tile is painted with a color other than `c` (assuming valid initial states). Since there is no action to unpaint, the problem is unsolvable from this state. Return `float('inf')`.
            iii. If a robot IS at tile `t`, add 1 to `total_cost`. This represents the cost of moving the robot off tile `t` to make it clear and paintable.
    7. Identify the set of unique colors `needed_colors` required by the `unpainted_goal_tiles`.
    8. Identify the set of colors `held_colors` currently held by any robot in the state.
    9. Calculate the `color_acquisition_cost`: Count how many colors in `needed_colors` are *not* in `held_colors`. For each such color, check if it is in the set of `available_colors` (precomputed in `__init__`). If a needed color is not held and not available, the problem is unsolvable. Otherwise, add 1 to the cost for each needed but not held color. Add this total `color_acquisition_cost` to `total_cost`.
    10. For each `(t, c)` in `unpainted_goal_tiles`:
        a. Find the set of tiles `paint_adj_tiles` from which tile `t` can be painted (using the precomputed `paint_adj` map).
        b. Calculate the minimum movement cost for any robot to reach any tile in `paint_adj_tiles`. This is done by performing BFS from each robot's current location to every tile in `paint_adj_tiles` and taking the minimum distance found. Let this be `min_dist_to_paint_adj`.
        c. If `min_dist_to_paint_adj` is `float('inf')`, it means the tile is unreachable. Return `float('inf')`.
        d. Add `min_dist_to_paint_adj` to `total_cost`.
        e. Add 1 to `total_cost` for the paint action itself.
    11. Return the final `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal tiles and their required colors
        self.goal_tiles = {} # {tile_name: color_name}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Store available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                self.available_colors.add(parts[1])


        # Build grid adjacency for movement (BFS)
        self.grid_adj = defaultdict(set)
        # Build paint adjacency (tile_to_paint -> tiles_robot_can_be_at)
        self.paint_adj = defaultdict(set)

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                # Grid adjacency is symmetric for movement
                self.grid_adj[t1].add(t2)
                self.grid_adj[t2].add(t1)

                # Paint adjacency: robot at t2 can paint t1
                self.paint_adj[t1].add(t2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        total_cost = 0
        unpainted_goal_tiles = [] # List of (tile, color) tuples
        needed_colors = set()

        # Cache robot locations and robots at tiles
        robot_locations = {}
        robots_at_tile = defaultdict(list) # Map tile -> list of robots at that tile
        for fact in state:
             if match(fact, "robot-at", "*", "*"):
                 parts = get_parts(fact)
                 robot, location = parts[1], parts[2]
                 robot_locations[robot] = location
                 robots_at_tile[location].append(robot)

        # If no robots exist but there are goal tiles, it's unsolvable unless all goals are met
        if not robot_locations and self.goal_tiles:
             all_goals_met = True
             for tile, color in self.goal_tiles.items():
                 if f"(painted {tile} {color})" not in state:
                     all_goals_met = False
                     break
             if not all_goals_met:
                 return float('inf') # Unsolvable without robots

        # Identify unpainted goal tiles and handle non-clear ones
        for tile, color in self.goal_tiles.items():
            # Check if the goal (painted tile color) is NOT in the state
            goal_fact = f"(painted {tile} {color})"
            if goal_fact not in state:
                unpainted_goal_tiles.append((tile, color))
                needed_colors.add(color)

                # Check if the tile is clear.
                is_clear = f"(clear {tile})" in state
                if not is_clear:
                    # If not clear and goal not met, check if a robot is on it.
                    is_robot_at_t = len(robots_at_tile[tile]) > 0

                    if not is_robot_at_t:
                        # Not clear, goal not met, and no robot is at the tile.
                        # This implies the tile is painted with the wrong color. Unsolvable.
                        return float('inf')
                    else:
                        # Not clear, goal not met, and a robot IS at the tile.
                        # Add cost to move robot off.
                        total_cost += 1 # Cost to move robot off the tile

        # If no unpainted goal tiles, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # Identify colors held by robots
        held_colors = set()
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                color = parts[2] # Only need the color
                held_colors.add(color)

        # Calculate color acquisition cost
        # Count colors needed that no robot currently has.
        colors_to_acquire_count = 0
        for color in needed_colors:
            if color not in held_colors:
                # If a needed color is not held, it must be available to be acquired.
                if color not in self.available_colors:
                    # Needed color is not available. Unsolvable.
                    return float('inf')
                colors_to_acquire_count += 1 # Cost to acquire this color

        total_cost += colors_to_acquire_count

        # Calculate movement and paint cost for each unpainted tile
        for tile, color in unpainted_goal_tiles:
            min_dist_to_paint_adj = float('inf')

            # Find the set of tiles from which this tile can be painted
            paint_adj_tiles = self.paint_adj.get(tile, set())

            if not paint_adj_tiles:
                 # This tile cannot be painted from anywhere based on static facts.
                 # This implies an unsolvable problem or a malformed domain/instance.
                 return float('inf')

            # Find min distance from any robot to any paint-adjacent tile
            # We already checked if robot_locations is empty above.
            for robot_loc in robot_locations.values():
                 for adj_tile in paint_adj_tiles:
                     dist = bfs(robot_loc, adj_tile, self.grid_adj)
                     min_dist_to_paint_adj = min(min_dist_to_paint_adj, dist)

            # If any paint-adjacent tile is unreachable from all robots, problem is unsolvable
            if min_dist_to_paint_adj == float('inf'):
                 return float('inf')

            # Add movement cost and paint action cost
            total_cost += min_dist_to_paint_adj + 1 # +1 for the paint action

        return total_cost
