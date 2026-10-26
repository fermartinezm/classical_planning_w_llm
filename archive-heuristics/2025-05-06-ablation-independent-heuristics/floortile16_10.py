# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available

import collections # For BFS queue
# import logging # Optional: for debugging

# Configure logging if needed
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def parse_fact(fact_string):
    """
    Helper function to parse a PDDL fact string.
    e.g., '(predicate arg1 arg2)' -> ('predicate', ['arg1', 'arg2'])
    """
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    if not parts:
        return None, [] # Handle empty string case, though unlikely for facts
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the required number of actions to reach a goal state
    by summing three main cost components for the unsatisfied goal conditions:
    1. The number of tiles that need to be painted (each requires a paint action).
    2. The number of colors required by unpainted goal tiles that no robot currently holds
       (assuming one color change action is sufficient per needed color not held).
    3. The minimum movement cost for a robot to get adjacent to each tile that needs painting.
       The movement cost for a tile is the minimum shortest path distance from any robot's
       current location to any tile adjacent to the target tile, calculated on the grid
       graph ignoring the 'clear' predicate for movement.

    This heuristic is designed for greedy best-first search and is not admissible.
    It aims to prioritize states where more goal tiles are painted, required colors
    are held, and robots are closer to the tiles they need to paint.

    Assumptions:
    - Tile names follow a structure that allows identifying them as tiles (e.g., 'tile_R_C').
    - The grid structure is fully defined by 'up', 'down', 'left', 'right' predicates
      connecting all relevant tiles.
    - Goal tiles that are not clear and not painted correctly represent dead ends.
    - The 'clear' precondition for movement is ignored when calculating distances
      in the grid graph relaxation.
    - Robots can instantly change color if the color is available (cost 1).
    - One robot changing color to a specific color can satisfy the need for that
      color for any number of unpainted tiles requiring that color.
    - All colors required by the goal are available colors. If a needed color is
      not available, the problem is unsolvable (dead end).

    Heuristic Initialization:
    - The constructor parses the static facts from the task description.
    - It builds an adjacency map (`self.adj_map`) representing the grid graph
      based on the 'up', 'down', 'left', and 'right' predicates.
    - It identifies and stores the set of all tile objects (`self.tiles`) involved
      in the adjacency relations.
    - It stores the goal facts (`self.goals`).
    - It identifies and stores the set of available colors (`self.available_colors`).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the heuristic value `h` to 0.
    2. Parse the current state (`node.state`) to extract:
        - The current position of each robot (`robot-at`).
        - The color currently held by each robot (`robot-has`).
        - The status of each tile (either `clear` or `painted` with a specific color).
    3. Identify the set `unpainted_goal_tiles`. This set contains tuples `(tile, color)`
       for each goal fact `'(painted tile color)'` that is not present in the current state.
    4. While identifying `unpainted_goal_tiles`, check for dead ends:
        - If a goal fact `'(painted T C)'` is not in the state, and the tile `T` is
          not `clear` in the current state:
            - Check if `T` is painted with a *different* color `C'`. If `'(painted T C')'`
              is in the state for `C' != C`, the goal is unreachable for this tile
              (as there's no unpaint/repaint action on non-clear tiles). Return `float('inf')`.
            - If `T` is not clear and not painted with *any* color (this state is unlikely
              in a valid execution but handled defensively), also return `float('inf')`.
    5. If the heuristic is already `float('inf')` due to a dead end detection, return it.
    6. If `unpainted_goal_tiles` is empty, it means all painting goals are satisfied. Return `h = 0`.
    7. Add the cost component for painting actions: `h += len(unpainted_goal_tiles)`.
    8. Determine the set of colors required by the `unpainted_goal_tiles`.
    9. Check if all required colors are available. If any required color is not in `self.available_colors`, return `float('inf')`.
    10. Determine the set of colors currently held by the robots.
    11. Identify the colors that are needed but not currently held by any robot. Add the count
        of these colors to `h`. This estimates the minimum color change actions needed.
    12. Add the cost component for movement: For each `(tile, color)` in `unpainted_goal_tiles`:
        - Find the set of tiles adjacent to `tile` using the precomputed adjacency map.
        - For each robot, compute the shortest path distance from its current position
          to all other tiles in the grid using BFS on the grid graph (ignoring 'clear').
        - Find the minimum distance from *any* robot's current position to *any* tile
          adjacent to the target `tile`.
        - Add this minimum distance to `h`.
    13. Return the final calculated value of `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.tiles = set()
        self.available_colors = set()
        # Adjacency map: tile_name -> list of adjacent_tile_names
        self.adj_map = collections.defaultdict(list)

        # Parse static facts to build grid graph, identify tiles and available colors
        for fact_string in self.static:
            pred, args = parse_fact(fact_string)
            if pred in ['up', 'down', 'left', 'right']:
                # Predicates are like (up tile_R1_C1 tile_R2_C2) meaning tile_R1_C1 is up from tile_R2_C2
                # This implies an edge from tile_R2_C2 to tile_R1_C1 (move up)
                # and an edge from tile_R1_C1 to tile_R2_C2 (move down)
                # The predicates define bidirectional connections for movement.
                tile_a, tile_b = args # e.g., tile_1_1, tile_0_1 for (up tile_1_1 tile_0_1)
                self.adj_map[tile_a].append(tile_b) # tile_a is adjacent to tile_b
                self.adj_map[tile_b].append(tile_a) # tile_b is adjacent to tile_a
                self.tiles.add(tile_a)
                self.tiles.add(tile_b)
            elif pred == 'available-color':
                 color = args[0]
                 self.available_colors.add(color)


        # Remove duplicates from adjacency lists
        for tile in self.adj_map:
            self.adj_map[tile] = list(set(self.adj_map[tile]))

        # logger.info(f"Initialized floortileHeuristic with {len(self.tiles)} tiles, {len(self.adj_map)} adjacency entries, and {len(self.available_colors)} available colors.")
        # logger.debug(f"Adj map: {self.adj_map}")
        # logger.debug(f"Available colors: {self.available_colors}")


    def get_adjacent_tiles(self, tile_name):
        """Get adjacent tiles from the precomputed map."""
        return self.adj_map.get(tile_name, [])

    def bfs(self, start_tile, all_tiles):
        """
        Computes shortest path distances from start_tile to all other tiles
        in the grid graph, ignoring 'clear' predicate.
        """
        distances = {tile: float('inf') for tile in all_tiles}

        # Ensure start_tile is a known tile in the grid
        if start_tile not in all_tiles:
             # If robot is on a tile not in the main grid, it cannot reach anything.
             # Distances remain infinity.
             # logger.warning(f"BFS start tile {start_tile} not found in known tiles.")
             return distances

        distances[start_tile] = 0
        queue = collections.deque([start_tile])

        while queue:
            current_tile = queue.popleft()
            current_dist = distances[current_tile]

            # Check if current_tile exists in adj_map keys before iterating
            if current_tile in self.adj_map:
                for neighbor in self.adj_map[current_tile]:
                    # Check if neighbor is in the set of all_tiles we care about
                    if neighbor in all_tiles and distances[neighbor] == float('inf'):
                         distances[neighbor] = current_dist + 1
                         queue.append(neighbor)
            # else: logger.debug(f"Tile {current_tile} not found in adj_map keys during BFS.")

        return distances


    def __call__(self, node):
        state = node.state
        h = 0

        # 1. Parse state
        robot_positions = {} # robot_name -> tile_name
        robot_colors = {}    # robot_name -> color_name
        tile_status = {}     # tile_name -> 'clear' or 'painted_color_name'

        # Build a set of state facts for faster lookup
        state_facts = set(state)

        for fact_string in state_facts:
            pred, args = parse_fact(fact_string)
            if pred == 'robot-at':
                robot, tile = args
                robot_positions[robot] = tile
            elif pred == 'robot-has':
                robot, color = args
                robot_colors[robot] = color
            elif pred == 'clear':
                tile = args[0]
                tile_status[tile] = 'clear'
            elif pred == 'painted':
                tile, color = args
                tile_status[tile] = f'painted_{color}'

        # 2. Identify unpainted goal tiles and check for dead ends
        unpainted_goal_tiles = set() # set of (tile_name, color_name) tuples

        for goal_fact_string in self.goals:
            # Goal facts are always '(painted T C)'
            _, goal_args = parse_fact(goal_fact_string)
            goal_tile, goal_color = goal_args

            if goal_fact_string not in state_facts:
                # This goal fact is not satisfied
                current_status = tile_status.get(goal_tile)

                if current_status != 'clear':
                    # Tile is not clear. Check if it's painted with the wrong color.
                    if current_status is not None and current_status.startswith('painted_'):
                         painted_color = current_status.split('_')[1]
                         if painted_color != goal_color:
                             # Painted with wrong color, likely dead end
                             # logger.debug(f"Dead end: {goal_tile} painted {painted_color}, needs {goal_color}")
                             return float('inf')
                         # If painted_color == goal_color, the goal fact should have been in state_facts.
                         # If it wasn't, the state is inconsistent, or the goal check is flawed.
                         # Assuming consistency, this branch implies painted_color != goal_color.
                    else:
                         # Tile is not clear, but not painted? (e.g. occupied by robot, or other unhandled status)
                         # In this domain, not clear usually means painted or occupied by robot.
                         # If occupied by robot, it needs to move. If painted wrong, it's a dead end.
                         # If it's a goal tile needing paint, and it's not clear and not painted correctly,
                         # it cannot be painted. Treat as dead end.
                         # logger.debug(f"Dead end: {goal_tile} not clear and not painted correctly.")
                         return float('inf')

                # If we reach here, the tile is clear but needs painting
                unpainted_goal_tiles.add((goal_tile, goal_color))

        # 3. If U is empty, goal is reached
        if not unpainted_goal_tiles:
            # logger.debug("All painting goals satisfied.")
            return 0

        # 4. Add cost components

        # Cost 1: Paint actions
        h += len(unpainted_goal_tiles)
        # logger.debug(f"Cost (Paint actions): {len(unpainted_goal_tiles)}")

        # Cost 2: Color changes
        colors_needed = {color for _, color in unpainted_goal_tiles}

        # Check if all needed colors are available
        if not colors_needed.issubset(self.available_colors):
             # logger.debug(f"Dead end: Needed color not available. Needed: {colors_needed}, Available: {self.available_colors}")
             return float('inf')

        colors_held = set(robot_colors.values())
        needed_colors_not_held = colors_needed - colors_held
        h += len(needed_colors_not_held) # Assumes one robot can change to cover a needed color
        # logger.debug(f"Cost (Color changes): {len(needed_colors_not_held)}")

        # Cost 3: Movement
        # Precompute distances from each robot's position to all tiles
        robot_distances = {} # robot_name -> {tile_name -> distance}
        for robot, pos in robot_positions.items():
             # Need distances from robot pos to all tiles in the grid
             robot_distances[robot] = self.bfs(pos, self.tiles)
             # logger.debug(f"BFS from {pos} for robot {robot} completed.")


        for goal_tile, goal_color in unpainted_goal_tiles:
            min_dist_to_adj_for_tile = float('inf')
            adjacent_tiles = self.get_adjacent_tiles(goal_tile)

            if not adjacent_tiles:
                 # A goal tile has no adjacent tiles? Problematic grid.
                 # Treat as unreachable for safety, though unlikely in valid problems.
                 # logger.error(f"Goal tile {goal_tile} has no adjacent tiles defined.")
                 return float('inf')

            for robot in robot_positions:
                robot_pos = robot_positions[robot]
                distances_from_robot = robot_distances.get(robot) # Use .get for safety

                if distances_from_robot is None or distances_from_robot.get(robot_pos, float('inf')) == float('inf'):
                     # Robot position is not in the known tile set or unreachable from itself (shouldn't happen)
                     # This robot cannot contribute to painting this tile.
                     continue

                min_dist_for_robot = float('inf')
                for adj_tile in adjacent_tiles:
                    # Ensure adj_tile is in the computed distances (i.e., reachable from robot_pos)
                    if adj_tile in distances_from_robot and distances_from_robot[adj_tile] != float('inf'):
                         min_dist_for_robot = min(min_dist_for_robot, distances_from_robot[adj_tile])

                # min_dist_for_robot is the minimum moves for this robot to reach ANY adjacent tile
                min_dist_to_adj_for_tile = min(min_dist_to_adj_for_tile, min_dist_for_robot)

            # Add the minimum movement cost for this specific goal tile
            if min_dist_to_adj_for_tile == float('inf'):
                 # This means no robot can reach any adjacent tile of the goal tile.
                 # This state is likely unsolvable under the relaxation.
                 # logger.debug(f"Dead end: No robot can reach adjacent to {goal_tile}.")
                 return float('inf') # Cannot reach adjacent tile

            h += min_dist_to_adj_for_tile
            # logger.debug(f"Cost (Movement for {goal_tile}): {min_dist_to_adj_for_tile}")


        # logger.debug(f"Total heuristic value: {h}")
        return h
