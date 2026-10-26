from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        three components:
        1. The number of tiles that need to be painted according to the goal
           but are currently clear.
        2. The number of distinct colors required by these unpainted goal tiles
           that are not currently held by any robot.
        3. The sum of minimum distances for each unpainted goal tile, where the
           minimum distance for a tile is the shortest path distance from any
           robot's current location to any tile adjacent to the target tile.

    Assumptions:
        - Tile names follow the format 'tile_row_col' allowing parsing of coordinates.
        - The grid structure is defined solely by 'up', 'down', 'left', 'right'
          predicates in the static facts, forming a connected graph.
        - Solvable instances do not require unpainting tiles; tiles painted with
          the wrong color are considered dead ends and result in a high heuristic value.
        - Robots always hold a color (the 'free-color' predicate is unused in actions).

    Heuristic Initialization:
        The constructor parses the static facts provided in the task.
        - It identifies all tile names and builds a mapping from tile name to
          (row, col) coordinates and vice versa, assuming the 'tile_row_col' format.
        - It constructs an adjacency list representation of the grid graph based
          on the 'up', 'down', 'left', and 'right' predicates.
        - It parses the goal facts to store the required color for each goal tile
          in a dictionary `goal_paintings`.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method computes the heuristic value for a given state node.
        1.  **Goal Check:** First, check if the current state is the goal state. If yes, return 0.
        2.  **Parse State:** Extract current information from the state: robot locations,
            colors held by robots, tiles that are painted (with their color), and
            tiles that are clear.
        3.  **Identify Unpainted and Wrongly Painted Tiles:** Iterate through the
            goal painting requirements (`self.goal_paintings`).
            - If a goal tile is currently painted with a different color, the state
              is likely unsolvable. Return a large heuristic value (e.g., 10000).
            - If a goal tile is currently clear, add it to a list of tiles that
              need painting (`unpainted_goal_tiles_info`).
            - Note: If a goal tile is neither painted correctly nor clear,
              it must be painted with the wrong color (handled above)
              or painted with the correct color (goal met for this tile).
        4.  **Calculate Base Cost (Paint Actions):** The minimum number of paint
            actions required is the number of tiles in `unpainted_goal_tiles_info`.
            Add this count to the total heuristic value `h`.
        5.  **Calculate Color Cost:** Determine the set of distinct colors required
            by the tiles in `unpainted_goal_tiles_info`. Determine the set of distinct
            colors currently held by robots. The number of colors that are needed
            but not held represents the minimum number of `change_color` actions
            required across all robots. Add this count to `h`.
        6.  **Calculate Movement Cost:**
            - If there are no tiles needing paint, the movement cost is 0.
            - Otherwise, perform a Breadth-First Search (BFS) starting simultaneously
              from all current robot locations to compute the shortest distance
              from any robot to every reachable tile.
            - For each tile that needs painting (`unpainted_goal_tiles_info`), find
              its neighbors using the pre-computed adjacency list.
            - Find the minimum distance from any robot's current location to any
              of these adjacent tiles using the distances computed by BFS.
            - If any tile needing paint is unreachable (no adjacent tile is reachable
              by any robot), the state is likely unsolvable. Return a large
              heuristic value (e.g., 10000).
            - Sum these minimum distances for all tiles needing paint. Add this
              total movement cost to `h`.
        7.  **Return Heuristic Value:** Return the calculated value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals

        self.tile_coords = {}
        self.coords_tile = {}
        self.adj = {}
        self.all_tile_names = set()

        # Build grid graph and parse tile coordinates
        for fact in task.static:
            parts = self.get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) == 3:
                    t1, t2 = parts[1], parts[2]
                    self.all_tile_names.add(t1)
                    self.all_tile_names.add(t2)
                    self.adj.setdefault(t1, []).append(t2)
                    self.adj.setdefault(t2, []).append(t1) # Grid is undirected

        for tile_name in self.all_tile_names:
            try:
                # Assuming tile names are like 'tile_row_col'
                _, row_str, col_str = tile_name.split('_')
                row, col = int(row_str), int(col_str)
                self.tile_coords[tile_name] = (row, col)
                self.coords_tile[(row, col)] = tile_name
            except ValueError:
                # Handle unexpected tile naming format if necessary
                # print(f"Warning: Could not parse tile coordinates from name: {tile_name}")
                pass

        # Store goal painting requirements
        self.goal_paintings = {}
        for goal in self.goals:
            parts = self.get_parts(goal)
            if parts and parts[0] == 'painted':
                 if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color

    def get_parts(self, fact):
        """Helper to split a PDDL fact string into predicate and arguments."""
        # Ensure fact is a string and has expected format
        if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
            return []
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Helper to check if a fact matches a predicate and arguments (with wildcards)."""
        parts = self.get_parts(fact)
        if not parts:
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def bfs_from_multiple_sources(self, sources):
        """
        Performs BFS starting from multiple source tiles to find shortest distances
        to all reachable tiles in the grid graph.
        """
        dist = {tile: float('inf') for tile in self.all_tile_names}
        q = deque()

        # Initialize queue with all source tiles
        for source_tile in sources:
            if source_tile in self.all_tile_names and dist[source_tile] == float('inf'):
                 dist[source_tile] = 0
                 q.append(source_tile)

        # Run BFS
        while q:
            curr_tile = q.popleft()
            # Ensure curr_tile is a valid tile in our graph representation
            if curr_tile in self.adj:
                for neighbor in self.adj[curr_tile]:
                    if neighbor in self.all_tile_names and dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[curr_tile] + 1
                        q.append(neighbor)
        return dist

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Parse state
        current_paintings = {}
        current_clear_tiles = set()
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = self.get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == 'painted':
                if len(parts) == 3:
                    current_paintings[parts[1]] = parts[2]
            elif predicate == 'clear':
                if len(parts) == 2:
                    current_clear_tiles.add(parts[1])
            elif predicate == 'robot-at':
                if len(parts) == 3:
                    robot_locations[parts[1]] = parts[2]
            elif predicate == 'robot-has':
                if len(parts) == 3:
                    robot_colors[parts[1]] = parts[2]

        # 2. Identify unpainted goal tiles and wrongly painted tiles
        unpainted_goal_tiles_info = {} # tile_name -> goal_color
        wrongly_painted_tiles = set() # tile_name

        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile in current_paintings:
                if current_paintings[goal_tile] != goal_color:
                    wrongly_painted_tiles.add(goal_tile)
            elif goal_tile in current_clear_tiles:
                 unpainted_goal_tiles_info[goal_tile] = goal_color
            # Note: If a goal tile is neither painted correctly nor clear,
            # it must be painted with the wrong color (handled above)
            # or painted with the correct color (goal met for this tile).

        # If any tile is painted with the wrong color, it's likely a dead end
        if wrongly_painted_tiles:
            return 10000 # Large value indicating likely unsolvable

        # 3. Calculate base cost (paint actions)
        h = len(unpainted_goal_tiles_info)

        # 4. Calculate color cost
        C_needed = set(unpainted_goal_tiles_info.values())
        C_held = set(robot_colors.values())
        colors_to_acquire = C_needed - C_held
        h += len(colors_to_acquire)

        # 5. Calculate movement cost
        if not unpainted_goal_tiles_info:
            movement_cost = 0
        else:
            # Compute distances from all robot locations
            robot_loc_list = list(robot_locations.values())
            # Handle case where no robots are present (shouldn't happen in valid problems)
            if not robot_loc_list:
                 return 10000 # Cannot paint without robots

            dist_from_robots = self.bfs_from_multiple_sources(robot_loc_list)

            total_movement_cost = 0
            for tile_to_paint, _ in unpainted_goal_tiles_info.items():
                min_dist_to_adjacent = float('inf')
                # Find neighbors of the tile that needs painting
                # Ensure tile_to_paint is in the graph (it should be if it's a goal tile)
                if tile_to_paint in self.adj:
                    for neighbor in self.adj[tile_to_paint]:
                        # Check if neighbor is a valid tile and reachable from robots
                        if neighbor in dist_from_robots and dist_from_robots[neighbor] != float('inf'):
                             min_dist_to_adjacent = min(min_dist_to_adjacent, dist_from_robots[neighbor])
                # else: tile_to_paint is not in adj, maybe it's an isolated tile? Unlikely in this domain.

                # If a tile needing paint has no reachable adjacent tile, it's unsolvable
                if min_dist_to_adjacent == float('inf'):
                     return 10000 # Unreachable tile

                total_movement_cost += min_dist_to_adjacent

            h += total_movement_cost

        return h
