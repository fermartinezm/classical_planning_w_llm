import math
from collections import deque

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        three components:
        1. The number of goal tiles that are not yet painted correctly
           (representing the required paint actions).
        2. The number of distinct colors required by unpainted goal tiles
           that are not currently held by any robot (representing the required
           color change actions).
        3. For each unpainted goal tile, the minimum distance from any robot's
           current location to any tile from which that goal tile can be painted
           (representing the required movement actions).

    Assumptions:
        - The input task represents a solvable problem instance.
        - Tile names follow the format 'tile_row_col'.
        - The grid structure defined by up/down/left/right predicates is consistent
          and forms a connected graph for movement.
        - Goal tiles are paintable from at least one adjacent tile as defined
          by the static predicates.
        - Available colors are defined in static facts.
        - Robots initially hold an available color.

    Heuristic Initialization:
        The constructor pre-processes the static information from the task:
        - It identifies all tiles in the domain by parsing facts.
        - It builds an undirected graph representing possible robot movements
          based on up/down/left/right predicates.
        - It builds a directed mapping indicating which tiles can be painted
          from which adjacent tiles based on up/down/left/right predicates.
        - It computes all-pairs shortest path distances between all tiles
          using Breadth-First Search (BFS) on the movement graph.
        - It extracts the target color for each goal tile from the goal facts.

    Step-By-Step Thinking for Computing Heuristic (__call__):
        1. Check if the current state is a goal state using task.goal_reached(). If yes, return 0.
        2. Initialize an empty list `unpainted_goal_tiles` and an empty set `needed_colors`.
        3. Iterate through the `goal_tiles` dictionary (precomputed in `__init__`). For each `(tile, goal_color)` pair:
           - Check if the fact `(painted tile goal_color)` exists in the current state.
           - If it does not exist, add `tile` to `unpainted_goal_tiles` and `goal_color` to `needed_colors`.
        4. If `unpainted_goal_tiles` is empty (meaning all goal tiles are painted correctly), the state must be a goal state (already handled in step 1), so this branch should not be reached if step 1 is correct.
        5. Extract the current location and color for each robot from the current state facts `(robot-at R L)` and `(robot-has R C)`. Store them in dictionaries `robot_locations` and `robot_colors`, and collect current colors in `current_robot_colors` set.
        6. Initialize the heuristic value `h` to 0.
        7. Add the number of unpainted goal tiles to `h`. This accounts for the paint actions needed.
        8. Identify colors that are needed for unpainted tiles but are not currently held by any robot (`colors_to_acquire = needed_colors - current_robot_colors`). Add the count of these colors to `h`. This accounts for the color change actions needed.
        9. For each `tile` in `unpainted_goal_tiles`:
           - Initialize `min_dist_to_adjacency` to infinity.
           - Get the list of tiles from which `tile` can be painted (`paint_from_tiles = self.paintable_from.get(tile, [])`).
           - If `paint_from_tiles` is empty, this goal tile cannot be painted, indicating an unsolvable state. Return `float('inf')`.
           - For each `paint_from_tile` in `paint_from_tiles`:
             - Ensure the `paint_from_tile` is a valid tile in our graph. If not, return `float('inf')`.
             - For each robot `R` and its current location `robot_loc` in `robot_locations`:
               - Find the precomputed distance `dist(robot_loc, paint_from_tile)`.
               - If the distance exists (i.e., the tile is reachable), update `min_dist_to_adjacency` with the minimum distance found so far.
           - If `min_dist_to_adjacency` is still infinity after checking all robots and all paintable-from tiles, it means the tile is unreachable by any robot from any valid painting position. Problem unsolvable. Return `float('inf')`.
           - Add `min_dist_to_adjacency` to `h`.
        10. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self.goal_tiles = {} # {tile_name: color_name}
        self.move_neighbors = {} # {tile_name: [neighbor_tile_name, ...]} (undirected)
        self.paintable_from = {} # {tile_name: [tile_name_from_where_it_can_be_painted, ...]} (directed)
        self.all_tiles = set()
        self.distances = {} # {(tile1, tile2): distance}

        self._process_static_info()
        self._compute_distances()

    def _parse_fact(self, fact_str):
        """Parses a fact string into a tuple (predicate, [args])."""
        # Remove surrounding parentheses and split by space
        parts = fact_str.strip('()').split()
        return parts[0], parts[1:]

    def _process_static_info(self):
        """Extracts grid structure and goal information from static facts and task goals."""
        # Collect all tiles mentioned in facts
        # We need all tiles to build the distance graph correctly
        all_fact_strings = set(self.task.facts) | set(self.task.initial_state) | set(self.task.goals) | set(self.task.static)
        for fact_str in all_fact_strings:
             pred, args = self._parse_fact(fact_str)
             for arg in args:
                 if arg.startswith('tile_'):
                     self.all_tiles.add(arg)

        # Initialize neighbor dictionaries
        for tile in self.all_tiles:
            self.move_neighbors[tile] = []
            self.paintable_from[tile] = []

        # Build neighbor graphs from static facts
        for fact_str in self.task.static:
            pred, args = self._parse_fact(fact_str)
            if pred == 'up': # (up y x) means y is up from x
                y, x = args
                if x in self.all_tiles and y in self.all_tiles:
                    self.move_neighbors[x].append(y) # Can move up from x to y
                    self.move_neighbors[y].append(x) # Can move down from y to x
                    self.paintable_from[y].append(x) # Can paint y from x
            elif pred == 'down': # (down y x) means y is down from x
                y, x = args
                if x in self.all_tiles and y in self.all_tiles:
                    self.move_neighbors[x].append(y) # Can move down from x to y
                    self.move_neighbors[y].append(x) # Can move up from y to x
                    self.paintable_from[y].append(x) # Can paint y from x
            elif pred == 'left': # (left y x) means y is left from x
                y, x = args
                if x in self.all_tiles and y in self.all_tiles:
                    self.move_neighbors[x].append(y) # Can move left from x to y
                    self.move_neighbors[y].append(x) # Can move right from y to x
                    self.paintable_from[y].append(x) # Can paint y from x
            elif pred == 'right': # (right y x) means y is right from x
                y, x = args
                if x in self.all_tiles and y in self.all_tiles:
                    self.move_neighbors[x].append(y) # Can move right from x to y
                    self.move_neighbors[y].append(x) # Can move left from y to x
                    self.paintable_from[y].append(x) # Can paint y from x

        # Extract goal tiles and colors
        # Goal is a frozenset of facts
        for goal_fact_str in self.task.goals:
            pred, args = self._parse_fact(goal_fact_str)
            if pred == 'painted':
                tile, color = args
                self.goal_tiles[tile] = color

    def _compute_distances(self):
        """Computes all-pairs shortest paths using BFS."""
        for start_node in self.all_tiles:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                for neighbor in self.move_neighbors.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An integer or float('inf') heuristic value.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        unpainted_goal_tiles = []
        needed_colors = set()

        # Identify unpainted/incorrectly painted goal tiles and needed colors
        for goal_tile, goal_color in self.goal_tiles.items():
            # If the goal fact is not in the state, this tile needs attention
            if f'(painted {goal_tile} {goal_color})' not in state:
                unpainted_goal_tiles.append(goal_tile)
                needed_colors.add(goal_color)

        # Extract robot locations and colors from the current state
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        current_robot_colors = set()

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'robot-at':
                robot, tile = args
                robot_locations[robot] = tile
            elif pred == 'robot-has':
                robot, color = args
                robot_colors[robot] = color
                current_robot_colors.add(color)

        h = 0

        # Component 1: Cost for paint actions (1 per unpainted goal tile)
        h += len(unpainted_goal_tiles)

        # Component 2: Cost for acquiring needed colors (1 per needed color not held)
        colors_to_acquire = needed_colors - current_robot_colors
        h += len(colors_to_acquire)

        # Component 3: Cost for movement (sum of min distances for each unpainted tile)
        for tile in unpainted_goal_tiles:
            min_dist_to_adjacency = float('inf')
            paint_from_tiles = self.paintable_from.get(tile, [])

            if not paint_from_tiles:
                 # This goal tile cannot be painted from anywhere. Problem unsolvable.
                 return float('inf')

            for paint_from_tile in paint_from_tiles:
                # Ensure the paint_from_tile is a valid tile in our graph
                if paint_from_tile not in self.all_tiles:
                     # This shouldn't happen if static facts are consistent, but as a safeguard
                     return float('inf')

                for robot in robot_locations:
                    robot_loc = robot_locations[robot]
                    # Check if the distance was computed (i.e., path exists)
                    if (robot_loc, paint_from_tile) in self.distances:
                         min_dist_to_adjacency = min(min_dist_to_adjacency, self.distances[(robot_loc, paint_from_tile)])
                    # else: This paint_from_tile is unreachable from this robot's location.
                    # This is handled by min() staying inf if no robot can reach any paint_from_tile.

            if min_dist_to_adjacency == float('inf'):
                 # This goal tile is unreachable by any robot from any valid painting position.
                 # Problem unsolvable.
                 return float('inf')

            h += min_dist_to_adjacency

        return h
