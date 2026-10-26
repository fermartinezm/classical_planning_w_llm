from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper to parse a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    Estimates the cost to reach the goal state by summing up the estimated
    costs for each unpainted goal tile that is currently clear. The estimated
    cost for a single unpainted goal tile includes the cost of the paint action
    itself, the cost for a robot to change color if necessary, and the minimum
    movement cost for any robot to reach a tile adjacent to the target tile.
    If a goal tile is painted with the wrong color, the heuristic returns
    infinity, indicating an unsolvable state. This heuristic is not admissible
    as it sums costs for individual tiles independently, potentially
    double-counting robot effort or movement. However, it aims to be informative
    by considering the required color and proximity of robots to unpainted tiles.

    Assumptions:
    - Tile names follow the format 'tile_row_col' where row and col are integers (used for Manhattan fallback, but BFS is primary).
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      forms a connected graph.
    - All colors required by the goal are available colors.
    - Tiles are initially 'clear' unless specified as 'painted' in the initial state.
    - Once a tile is painted, it remains painted with that color.
    - States passed to the heuristic are valid (e.g., robot-at and robot-has facts exist for each robot).
    - If a tile is not explicitly 'clear' or 'painted' in the state, it is considered not paintable.

    Heuristic Initialization:
    - Parses static facts to identify all tiles present in the domain/problem.
    - Builds a directed graph representing possible robot moves based on
      'up', 'down', 'left', 'right' predicates (e.g., `(up y x)` means move x -> y).
    - Computes all-pairs shortest path distances between tiles using BFS on the directed move graph.
    - Builds an undirected graph representing tile adjacency for painting
      (e.g., `(up y x)` means x is adjacent to y). A robot at x can paint y.
    - Stores the goal state facts for quick lookup, specifically the required
      color for each goal tile.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h` to 0.
    2. Identify the state of all goal tiles: which are painted correctly,
       painted incorrectly, or clear.
       Store currently painted tiles and their colors, and currently clear tiles.
    3. Identify the set of goal tiles that are not painted correctly.
       For each goal `(painted tile color)`:
       - If `tile` is painted with `color` in the state, this goal is met.
       - If `tile` is painted with a *different* color in the state, the state is unsolvable. Return `float('inf')`.
       - If `tile` is *clear* in the state, it is an unpainted goal tile that needs painting. Add it to `unpainted_goals = {tile: color, ...}`.
       - If `tile` is neither painted nor clear, it's considered not paintable in this state. Ignore it or handle as unsolvable if it's a goal tile. We assume it must be clear to be paintable.
    4. If `unpainted_goals` is empty (and no goal tile was painted incorrectly), the state is a goal state, return `h = 0`.
    5. Extract the current location and color for each robot from the state.
       Store this in a dictionary `robot_info = {robot: {'loc': tile, 'color': color}, ...}`.
    6. For each `(tile, color)` pair in `unpainted_goals.items()`:
       a. This tile needs to be painted with `color`. This requires at least 1 paint action.
       b. A robot must perform this action. Find the minimum cost for *any* robot
          to be ready to paint this tile.
       c. Initialize `min_robot_cost_for_this_tile = float('inf')`.
       d. Find the set of tiles adjacent to the target tile `tile` using the
          precomputed undirected adjacency graph: `adjacent_tiles = list(self.undirected_adj.get(tile, []))`.
          If `adjacent_tiles` is empty, this goal tile cannot be painted. Return `float('inf')`.
       e. For each robot `r` and its info (`loc_r`, `color_r`) in `robot_info`:
          i. Calculate the minimum movement cost for robot `r` from its current location `loc_r`
             to any tile in `adjacent_tiles`. This is `min_{adj_tile in adjacent_tiles} self.distances[loc_r][adj_tile]`.
             Let this be `moves`.
          ii. If `moves` is infinity, this robot cannot reach any adjacent tile. Skip this robot for this tile.
          iii. Calculate the color change cost for robot `r`. If `color_r` is not
               the required `color`, the robot needs to change color, costing 1 action.
               Let this be `color_change = 1 if color_r != color else 0`.
          iv. The total cost for robot `r` to *get ready* to paint this tile is
              `moves + color_change`.
          v. Update `min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, moves + color_change)`.
       f. If `min_robot_cost_for_this_tile` is still infinity after checking all robots,
          this tile is unreachable by any robot. The state is likely unsolvable.
          Return `float('inf')`.
       g. The estimated cost to paint this specific tile is `min_robot_cost_for_this_tile + 1` (for the paint action itself).
       h. Add this estimated cost for the tile to the total heuristic: `h += (min_robot_cost_for_this_tile + 1)`.
    7. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.all_tiles = set()
        # Collect all tiles from static facts involving tiles
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) > 1 and parts[0] in ['up', 'down', 'left', 'right', 'clear', 'painted', 'robot-at']:
                 # These predicates involve tiles. Collect all tile arguments.
                 for part in parts[1:]:
                     # Basic check if it looks like a tile name
                     if isinstance(part, str) and part.startswith('tile_'):
                         self.all_tiles.add(part)

        # 1. Build directed graph for robot movement
        self.move_adj = {tile: [] for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                # PDDL: (predicate target source) e.g., (up tile_1_1 tile_0_1)
                # Move action is from source to target: move_up ?r ?x ?y means move from x to y
                source_tile = parts[2]
                target_tile = parts[1]
                if source_tile in self.move_adj: # Ensure tile exists in our collected set
                    self.move_adj[source_tile].append(target_tile)
                # else: source_tile was not collected, maybe an issue? Assume all relevant tiles are collected.


        # 2. Precompute all-pairs shortest paths using BFS on the directed move graph
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()
                self.distances[start_node][current_node] = dist

                if current_node in self.move_adj: # Ensure current_node has neighbors in move graph
                    for neighbor in self.move_adj[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

        # Handle unreachable tiles (set distance to infinity)
        for start_node in self.all_tiles:
             for end_node in self.all_tiles:
                 if end_node not in self.distances[start_node]:
                     self.distances[start_node][end_node] = float('inf')


        # 3. Build undirected graph for finding adjacent tiles for painting
        self.undirected_adj = {tile: set() for tile in self.all_tiles}
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                  tile1, tile2 = parts[1], parts[2]
                  if tile1 in self.undirected_adj: self.undirected_adj[tile1].add(tile2)
                  if tile2 in self.undirected_adj: self.undirected_adj[tile2].add(tile1)


        # 4. Store goal painted facts
        self.goal_painted_facts = {
            get_parts(goal)[1]: get_parts(goal)[2] # {tile: color}
            for goal in self.goals
            if match(goal, "painted", "*", "*")
        }

    def __call__(self, node):
        state = node.state
        h = 0

        # 1. Identify the state of goal tiles
        unpainted_goals = {} # {tile: color} for goal tiles that are currently clear
        current_painted_tiles = {} # {tile: color} for tiles currently painted
        current_clear_tiles = set()

        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1:3]
                current_painted_tiles[tile] = color
            elif match(fact, "clear", "*"):
                tile = get_parts(fact)[1]
                current_clear_tiles.add(tile)

        for tile, goal_color in self.goal_painted_facts.items():
            if tile in current_painted_tiles:
                # Tile is already painted
                current_color = current_painted_tiles[tile]
                if current_color != goal_color:
                    # Painted with wrong color - unsolvable
                    return float('inf')
                # Else: Painted with correct color - goal achieved for this tile, ignore.
            elif tile in current_clear_tiles:
                # Tile is clear, needs painting
                unpainted_goals[tile] = goal_color
            # Else: Tile is neither painted nor clear. Assume not paintable. Ignore.


        # 2. If goal reached, return 0
        if not unpainted_goals:
            return 0

        # 3. Extract robot info
        robot_info = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, loc = get_parts(fact)[1:3]
                if robot not in robot_info: robot_info[robot] = {}
                robot_info[robot]['loc'] = loc
            elif match(fact, "robot-has", "*", "*"):
                 robot, color = get_parts(fact)[1:3]
                 if robot not in robot_info: robot_info[robot] = {}
                 robot_info[robot]['color'] = color

        # Basic check for valid robot info (should be guaranteed by planner state representation)
        if not all('loc' in info and 'color' in info for info in robot_info.values()):
             # Handle error or return infinity if state is malformed
             # print("Warning: Malformed robot state in heuristic")
             return float('inf') # Indicate unsolvable or invalid state


        # 4. Calculate heuristic for each unpainted goal tile
        for tile, color in unpainted_goals.items():
            min_robot_cost_for_this_tile = float('inf')

            # Find tiles adjacent to the target tile for painting using precomputed undirected graph
            adjacent_tiles = list(self.undirected_adj.get(tile, []))

            # If a goal tile has no adjacent tiles defined, it's likely an error or unsolvable
            if not adjacent_tiles:
                 # print(f"Warning: Goal tile {tile} has no adjacent tiles defined.")
                 return float('inf') # Indicate unsolvable

            for robot, info in robot_info.items():
                loc_r = info['loc']
                color_r = info['color']

                # Calculate minimum moves to any adjacent tile
                min_moves = float('inf')
                # Ensure robot's current location is in the distance map (should be if collected)
                if loc_r in self.distances:
                    for adj_tile in adjacent_tiles:
                        if adj_tile in self.distances[loc_r]:
                             min_moves = min(min_moves, self.distances[loc_r][adj_tile])
                        # else: This adj_tile is unreachable from robot's current location.

                if min_moves == float('inf'):
                     # This robot cannot reach any adjacent tile. Skip this robot for this tile.
                     continue

                # Calculate color change cost
                color_change = 1 if color_r != color else 0

                # Total cost for this robot to paint this tile (moves + color_change + paint)
                # The moves cost is to get *to* the adjacent tile.
                # The paint action is 1 cost.
                robot_cost_for_this_tile = min_moves + color_change + 1

                min_robot_cost_for_this_tile = min(min_robot_cost_for_this_tile, robot_cost_for_this_tile)

            # If min_robot_cost_for_this_tile is still infinity, it means no robot can paint this tile.
            # This state is likely unsolvable. Return infinity.
            if min_robot_cost_for_this_tile == float('inf'):
                 return float('inf') # Indicate unsolvable

            h += min_robot_cost_for_this_tile

        return h
