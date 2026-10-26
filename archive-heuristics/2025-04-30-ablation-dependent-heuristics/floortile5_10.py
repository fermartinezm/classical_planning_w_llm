import collections

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each unpainted goal tile. For each unpainted
        goal tile, it calculates the minimum cost for any robot to reach
        a tile adjacent to the goal tile, change color if necessary, and
        perform the paint action. The distance calculation considers the
        current state's clear tiles for movement traversability.

    Assumptions:
        - Tile names are strings, typically in 'tile_row_col' format.
        - Robot names are strings, typically starting with 'robot'.
        - Color names are strings.
        - The grid structure is defined by 'up', 'down', 'left', 'right'
          predicates in the static facts.
        - Goal states are defined by a set of '(painted tile color)' facts.
        - If a goal tile is painted with the wrong color, the state is
          considered unsolvable (heuristic returns infinity).
        - If a goal tile needs painting, it is assumed to be clear in the
          current state unless painted with the wrong color. (Ignores the case
          where a robot might be standing on an unpainted goal tile).
        - All available colors are listed in static facts or initial state.

    Heuristic Initialization:
        The constructor processes the static facts and goal facts to build
        necessary data structures:
        - Maps between tile names and integer IDs.
        - Adjacency list representing the grid graph based on 'up', 'down',
          'left', 'right' predicates.
        - A dictionary mapping goal tile names to their required colors.
        - Sets of robot names and available color names.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state to extract robot locations, robot held colors,
            currently painted tiles and their colors, and clear tiles.
        2.  Check if any goal tile is currently painted with a color different
            from the required goal color. If so, return infinity, as this state
            is likely a dead end in the standard domain.
        3.  Identify the set of goal tiles that are not yet painted with the
            correct color (i.e., they are either clear or not mentioned as painted,
            and are goal tiles).
        4.  If there are no unpainted goal tiles, the state is a goal state,
            return 0.
        5.  For each robot, compute the shortest distance from its current location
            to all other reachable tiles, considering only 'clear' tiles as valid
            destinations for move actions. This is done using a Breadth-First Search (BFS)
            starting from the robot's current tile, traversing only through clear tiles.
        6.  Initialize the total heuristic value `h` to 0.
        7.  For each unpainted goal tile `t` that needs color `c`:
            a.  Initialize `min_cost_for_tile` to infinity. This will store the minimum
                estimated cost for *any* robot to paint this specific tile.
            b.  Identify the tiles adjacent to `t` using the precomputed adjacency list.
            c.  For each robot `r`:
                i.  Get the robot's current location `r_tile` and color `r_color`.
                ii. Find the minimum distance from `r_tile` to *any* tile adjacent to `t`.
                    This distance is obtained from the state-dependent BFS results computed
                    in step 5. If an adjacent tile is unreachable by robot `r` (e.g., no
                    path through clear tiles), the distance will be infinity.
                iii. If an adjacent tile is reachable (`min_dist_to_adj` is not infinity):
                    - Calculate the color change cost: 1 if `r_color` is not `c`, 0 otherwise.
                    - Calculate the estimated cost for robot `r` to paint tile `t`:
                      `min_dist_to_adj + color_change_cost + 1` (the +1 is for the paint action itself).
                    - Update `min_cost_for_tile = min(min_cost_for_tile, estimated_cost_for_r)`.
            d.  If, after checking all robots, `min_cost_for_tile` is still infinity, it means
                no robot can reach an adjacent tile to paint `t`. This state is likely
                unsolvable, so return infinity.
            e.  Add `min_cost_for_tile` to the total heuristic value `h`.
        8.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information and goals.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.task = task
        self.goal_facts = task.goals
        self.static_facts = task.static

        # Data structures for precomputation
        self.tile_to_id = {}
        self.id_to_tile = {}
        self.adj_list = collections.defaultdict(list)
        self.robots = set()
        self.available_colors = set()
        self.goal_tiles = {} # tile_name -> color_name

        self._process_static_and_goal_facts()

    def _process_static_and_goal_facts(self):
        """
        Parses static facts and goal facts to populate internal data structures.
        Collects all potential objects first, then builds maps and adjacency.
        """
        all_objects = set()
        for fact_str in set(self.task.initial_state) | set(self.static_facts) | set(self.goal_facts):
             parts = self._parse_fact(fact_str)
             # All parts after the predicate are objects/arguments
             all_objects.update(parts[1:])

        tile_id_counter = 0
        all_tiles = set()

        # Classify objects based on predicate usage in domain or naming convention
        for obj in all_objects:
            if obj.startswith('tile_'):
                all_tiles.add(obj)
            elif obj.startswith('robot'):
                self.robots.add(obj)

        # Add colors found in initial state or static facts
        for fact_str in set(self.task.initial_state) | set(self.static_facts):
             parts = self._parse_fact(fact_str)
             predicate = parts[0]
             if predicate == 'robot-has' and len(parts) == 3:
                 # parts[1] is robot, parts[2] is color
                 self.available_colors.add(parts[2])
             elif predicate == 'available-color' and len(parts) == 2:
                 # parts[1] is color
                 self.available_colors.add(parts[1])


        for tile in sorted(list(all_tiles)): # Sort for consistent ID assignment
            self.tile_to_id[tile] = tile_id_counter
            self.id_to_tile[tile_id_counter] = tile
            tile_id_counter += 1

        # Build adjacency list from static facts
        for fact_str in self.static_facts:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                if tile1 in self.tile_to_id and tile2 in self.tile_to_id:
                    id1, id2 = self.tile_to_id[tile1], self.tile_to_id[tile2]
                    self.adj_list[id1].append(id2)
                    self.adj_list[id2].append(id1) # Grid is undirected for movement

        # Extract goal tiles and colors
        for fact_str in self.goal_facts:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

    def _parse_fact(self, fact_str):
        """
        Parses a PDDL fact string into a list of strings.
        e.g., '(robot-at robot1 tile_0_1)' -> ['robot-at', 'robot1', 'tile_0_1']
        Handles potential extra spaces.
        """
        # Remove surrounding brackets and split by space, filter empty strings
        return [part for part in fact_str[1:-1].split() if part]

    def _bfs(self, start_tile_id, clear_tile_ids):
        """
        Performs BFS from a start tile, only traversing through clear tiles.

        Args:
            start_tile_id: The integer ID of the starting tile.
            clear_tile_ids: A set of integer IDs of tiles that are currently clear.

        Returns:
            A dictionary mapping reachable tile IDs to their shortest distance
            from the start tile.
        """
        distances = {tile_id: float('inf') for tile_id in self.id_to_tile.keys()}
        queue = collections.deque([(start_tile_id, 0)])
        distances[start_tile_id] = 0

        while queue:
            current_id, d = queue.popleft()

            # A robot can move *from* any tile, but only *to* a clear tile.
            # The BFS calculates distance to reach a tile. So, we check if the
            # neighbor is clear before adding it to the queue.
            for neighbor_id in self.adj_list.get(current_id, []):
                if neighbor_id in clear_tile_ids and distances[neighbor_id] == float('inf'):
                    distances[neighbor_id] = d + 1
                    queue.append((neighbor_id, d + 1))

        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of fact strings representing the current state.

        Returns:
            An integer or float('inf') representing the estimated cost to goal.
        """
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {}    # robot_name -> color_name
        painted_tiles = {}   # tile_name -> color_name
        clear_tiles = set()  # tile_name set

        # Parse current state facts
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            elif predicate == 'clear' and len(parts) == 2:
                tile = parts[1]
                clear_tiles.add(tile)

        h = 0
        unpainted_goals = []

        # Check for wrongly painted goal tiles and identify unpainted ones
        for goal_tile, required_color in self.goal_tiles.items():
            if goal_tile in painted_tiles:
                if painted_tiles[goal_tile] != required_color:
                    # Goal tile painted with wrong color - likely unsolvable
                    return float('inf')
                # Else: correctly painted, goal satisfied for this tile
            else:
                # Goal tile not painted - needs painting
                unpainted_goals.append((goal_tile, required_color))

        if not unpainted_goals:
            # All goal tiles are correctly painted
            return 0

        # Compute state-dependent robot reachability distances
        robot_distances = {} # robot_name -> {tile_id -> distance}
        # Only include clear tiles that are actually in our tile map
        clear_tile_ids = {self.tile_to_id[t] for t in clear_tiles if t in self.tile_to_id}

        for robot in self.robots:
            if robot in robot_locations and robot_locations[robot] in self.tile_to_id:
                 start_tile_id = self.tile_to_id[robot_locations[robot]]
                 # A robot's current tile is traversable *from* for BFS, even if not clear.
                 # But it can only move *to* clear tiles. The BFS handles this by checking
                 # neighbor_id in clear_tile_ids.
                 # The start tile itself doesn't need to be clear to start BFS from it.
                 robot_distances[robot] = self._bfs(start_tile_id, clear_tile_ids)
            else:
                 # Robot state unknown or location not a valid tile, cannot contribute
                 robot_distances[robot] = {} # Empty distances

        # Calculate cost for each unpainted goal tile
        for goal_tile, required_color in unpainted_goals:
            min_cost_for_tile = float('inf')
            goal_tile_id = self.tile_to_id.get(goal_tile)

            if goal_tile_id is None:
                 # Goal tile not found in tile map - problem with parsing or domain
                 return float('inf')

            adjacent_tile_ids = self.adj_list.get(goal_tile_id, [])

            # Note: A tile must be clear to be painted. We assume unpainted goal tiles are clear.
            # If a robot is on an unpainted goal tile, it makes it not clear.
            # This heuristic doesn't explicitly handle the cost of moving a robot *off*
            # a goal tile first. It assumes the tile is paintable if not wrongly painted.
            # The distance calculation is to an *adjacent* tile, which must be clear to move onto.

            for robot in self.robots:
                if robot not in robot_locations or robot not in robot_colors:
                    continue # Robot state unknown

                r_tile_id = self.tile_to_id.get(robot_locations[robot])
                r_color = robot_colors[robot]

                if r_tile_id is None:
                    continue # Robot location not a valid tile

                # Find min distance from robot's current tile to any adjacent tile of the goal tile
                min_dist_to_adj = float('inf')
                robot_dist_map = robot_distances.get(robot, {})

                for adj_id in adjacent_tile_ids:
                    # Distance from robot's current tile to adj_id
                    # Check if adj_id is reachable by the robot
                    dist = robot_dist_map.get(adj_id, float('inf'))
                    min_dist_to_adj = min(min_dist_to_adj, dist)

                if min_dist_to_adj != float('inf'):
                    # Robot can reach an adjacent tile that is clear (because BFS only traversed clear tiles)
                    color_cost = 1 if r_color != required_color else 0
                    # Cost = distance to adjacent + color change (if needed) + paint action (1)
                    cost_r_paints_t = min_dist_to_adj + color_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_r_paints_t)

            if min_cost_for_tile == float('inf'):
                # No robot can reach an adjacent tile to paint this goal tile through clear path
                return float('inf')

            h += min_cost_for_tile

        return h
