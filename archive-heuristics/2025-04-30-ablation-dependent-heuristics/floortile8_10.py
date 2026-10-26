from collections import deque

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        three components:
        1. The estimated cost of acquiring necessary colors that no robot
           currently possesses.
        2. The estimated cost of moving robots to positions adjacent to
           the tiles that need painting.
        3. The cost of performing the paint action for each tile that needs
           painting.

        It is designed for greedy best-first search and is not admissible,
        but aims to guide the search effectively by prioritizing states
        where more goal conditions are met, necessary colors are held by
        robots, and robots are closer to the tiles they need to paint.

    Assumptions:
        - The heuristic is used with a greedy best-first search and does
          not need to be admissible.
        - The grid structure is defined by the adjacency facts (`up`, `down`,
          `left`, `right`).
        - Robots always hold a color (no 'free-color' state).

    Heuristic Initialization:
        During initialization (`__init__`), the heuristic pre-calculates
        static information from the task definition:
        1. It parses the static adjacency facts (`up`, `down`, `left`, `right`)
           to build a graph representation of the tile grid, storing it
           as an adjacency list (`self.adj`). It also collects all unique
           tile names (`self.tile_names`).
        2. It computes the shortest path distance between every pair of
           tiles in the grid using Breadth-First Search (BFS). These
           distances are stored in `self.distances`. This allows for quick
           lookup of movement costs during heuristic computation.
        3. It parses the goal facts (`painted`) to identify which tiles
           need to be painted and with which color, storing this in
           `self.goal_paintings`.

    Step-By-Step Thinking for Computing Heuristic:
        The heuristic value for a given state (`__call__`) is computed as follows:
        1. Identify the current location and held color for each robot
           by examining the state facts (`robot-at`, `robot-has`).
        2. Determine which goal painting conditions are not yet met in
           the current state. These are the tiles that still need to be
           painted (`needed_paintings`). During this step, it also checks
           if any goal tile is painted with the wrong color (i.e., it's
           not clear and not painted with the goal color). If such a tile
           is found, the state is considered unsolvable, and the heuristic
           returns infinity. If all goal paintings are met, the heuristic is 0.
        3. Calculate the 'color change cost': Identify the set of colors
           required by the tiles in `needed_paintings`. Count how many
           of these required colors are *not* currently held by *any* robot.
           This count is a simple estimate of the minimum number of
           `change_color` actions needed across all robots to acquire the
           missing color types.
        4. Calculate the 'movement and paint cost': For each tile `T` that
           needs painting (from `needed_paintings`):
           a. Find the set of tiles `Adj_T` that are adjacent to `T` in the
              grid graph (these are the tiles a robot must be *at* to paint `T`).
           b. For each robot `R`, find the shortest distance from its current
              location `R_loc` to *any* tile in `Adj_T`. This is the minimum
              movement cost for robot `R` to get into a position to paint `T`.
           c. Find the minimum of these distances over all robots. This is the
              minimum movement cost for *any* robot to get into a position
              to paint `T`.
           d. Add this minimum movement cost plus 1 (for the paint action itself)
              to a running total (`movement_paint_cost`). If any tile is
              unreachable from all robots, the heuristic returns infinity.
        5. The total heuristic value is the sum of the 'color change cost'
           and the 'movement and paint cost'.
    """
    def __init__(self, task):
        self.task = task
        self.adj = {} # Adjacency list for tiles: tile -> set of adjacent tiles
        self.tile_names = set() # Set of all tile names
        self.distances = {} # Shortest path distances between tiles: (tile1, tile2) -> distance
        self.goal_paintings = {} # {tile: color} for goal paintings

        # 1. Build adjacency map and collect tile names from static facts
        # Adjacency facts define the grid structure and possible moves/paint locations.
        # (up y x) means y is up from x. Robot at x can move to y, robot at x can paint y.
        # So y is adjacent to x. The relationship is symmetric for distance.
        for fact_str in task.static:
            if fact_str.startswith('('):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    pred, arg1, arg2 = parts
                    if pred in {'up', 'down', 'left', 'right'}:
                        # The arguments are (neighbor_tile, current_tile) in terms of robot movement/painting
                        neighbor_tile, current_tile = arg1, arg2

                        self.tile_names.add(neighbor_tile)
                        self.tile_names.add(current_tile)

                        if current_tile not in self.adj:
                            self.adj[current_tile] = set()
                        if neighbor_tile not in self.adj:
                            self.adj[neighbor_tile] = set()

                        # Add symmetric adjacency for distance calculation
                        self.adj[current_tile].add(neighbor_tile)
                        self.adj[neighbor_tile].add(current_tile)

        # Ensure all tiles mentioned in goal are in tile_names, even if isolated
        for fact_str in task.goals:
             if fact_str.startswith('(painted '):
                 parts = fact_str.strip('()').split()
                 if len(parts) == 3:
                     _, tile, color = parts
                     self.tile_names.add(tile)
                     if tile not in self.adj: # Add isolated tiles to adj map
                         self.adj[tile] = set()


        # 2. Compute all-pairs shortest paths using BFS
        for start_tile in self.tile_names:
            self.distances.update(self._bfs(start_tile))

        # 3. Parse goal facts
        for fact_str in task.goals:
            if fact_str.startswith('(painted '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    _, tile, color = parts
                    self.goal_paintings[tile] = color

    def _bfs(self, start_tile):
        """
        Performs BFS starting from start_tile to find distances to all reachable tiles.
        Returns a dictionary mapping (start_tile, end_tile) tuple to distance.
        """
        q = deque([(start_tile, 0)])
        visited = {start_tile}
        distances = {(start_tile, start_tile): 0}

        # Handle case where start_tile is isolated (not in adj graph)
        if start_tile not in self.adj:
             return distances # Can only reach itself

        while q:
            current_tile, dist = q.popleft()

            # Neighbors are defined by the adjacency list
            for neighbor in self.adj.get(current_tile, set()): # Use .get for safety
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[(start_tile, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

        return distances


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        :param state: The current state (frozenset of fact strings).
        :return: The estimated number of actions to reach the goal, or infinity
                 if the state is detected as a dead end (e.g., goal tile painted
                 with the wrong color).
        """
        # 1. Get current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        current_facts = set(state) # Convert frozenset to set for faster lookups

        for fact_str in current_facts:
            if fact_str.startswith('('):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    pred, obj1, obj2 = parts
                    if pred == 'robot-at':
                        robot_locations[obj1] = obj2
                    elif pred == 'robot-has':
                        robot_colors[obj1] = obj2

        # 2. Identify unpainted goal tiles and check for dead ends
        needed_paintings = {} # {tile: color}
        for tile, color in self.goal_paintings.items():
            goal_painted_fact = f'(painted {tile} {color})'

            if goal_painted_fact not in current_facts:
                 # This tile is not painted with the goal color.
                 # Check if it's painted with *any* other color (i.e., not clear).
                 is_clear_fact = f'(clear {tile})'
                 if is_clear_fact not in current_facts:
                     # It's not clear, and not painted with the goal color.
                     # This means it's painted with the wrong color.
                     # This is a dead end in this domain as there's no unpaint action.
                     return float('inf')
                 else:
                     # It's clear, and needs to be painted with the goal color.
                     needed_paintings[tile] = color

        # 3. If goal is reached, heuristic is 0
        if not needed_paintings:
            return 0

        # 4. Calculate color change cost
        # Count distinct colors needed by unpainted tiles that no robot currently holds.
        needed_colors = set(needed_paintings.values())
        held_colors = set(robot_colors.values())
        missing_colors = needed_colors - held_colors
        color_change_cost = len(missing_colors)

        # 5. Calculate movement and paint cost
        movement_paint_cost = 0
        for tile_to_paint, required_color in needed_paintings.items():
            # Find adjacent tiles to the tile that needs painting.
            # A robot must be AT an adjacent tile to paint it.
            # The set of tiles a robot can be at to paint tile_to_paint
            # is the set of tiles adjacent to tile_to_paint in the grid graph.
            adjacent_tiles_to_paint = self.adj.get(tile_to_paint, set())

            if not adjacent_tiles_to_paint:
                 # Tile is isolated or not in adjacency graph - likely unreachable
                 # If a goal tile is unreachable, the state is likely unsolvable.
                 return float('inf') # Indicate unsolvable from here

            min_dist_to_adjacent = float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            for robot, robot_loc in robot_locations.items():
                min_dist_for_robot = float('inf')
                # Ensure robot_loc is a valid tile in our graph
                if robot_loc in self.tile_names:
                    for adj_tile in adjacent_tiles_to_paint:
                        # Distance from robot_loc to adj_tile
                        # Use .get to handle cases where robot_loc or adj_tile might be isolated
                        # or not reachable from each other (though BFS should cover connected components)
                        dist = self.distances.get((robot_loc, adj_tile), float('inf'))
                        min_dist_for_robot = min(min_dist_for_robot, dist)

                # Update the minimum distance considering all robots
                min_dist_to_adjacent = min(min_dist_to_adjacent, min_dist_for_robot)

            if min_dist_to_adjacent == float('inf'):
                 # No robot can reach any adjacent tile for this painting task
                 # This state is likely unsolvable
                 return float('inf') # Indicate unsolvable from here

            # Cost for this tile: min moves for best robot + 1 (paint action)
            movement_paint_cost += min_dist_to_adjacent + 1

        # 6. Total heuristic
        return color_change_cost + movement_paint_cost
