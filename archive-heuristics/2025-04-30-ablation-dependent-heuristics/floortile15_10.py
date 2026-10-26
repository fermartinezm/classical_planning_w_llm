import collections
import math

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the minimum
    estimated costs for each goal tile that is not yet painted with the correct color.
    For each such tile, the minimum cost is calculated over all robots. The cost
    for a robot to paint a specific goal tile is estimated as the sum of:
    1. Cost to change the robot's color to the required color (0 or 1).
    2. Shortest path distance (number of move actions) for the robot to reach
       any tile adjacent to the goal tile, considering only clear tiles are traversable.
    3. Cost of the paint action (1).

    Assumptions:
    - Tile names are in the format 'tile_row_col' where row and col are integers.
    - The grid defined by 'up', 'down', 'left', 'right' facts is connected.
    - Solvable states do not have goal tiles painted with the wrong color.
    - Solvable states allow robots to eventually reach a clear tile adjacent
      to any unpainted goal tile. If no clear adjacent tile is currently reachable,
      the heuristic assumes the state is unsolvable for that tile and returns infinity.

    Heuristic Initialization:
    The constructor pre-processes the static information from the task:
    - Parses tile names to extract row and column coordinates, storing them
      in a dictionary `tile_coords`. (Note: `tile_coords` is not strictly used
      in the current heuristic calculation but is included based on common
      practice for grid domains and the checklist item about parsing objects).
    - Builds an adjacency map `adjacency_map` where keys are tile names and
      values are sets of adjacent tile names, based on 'up', 'down', 'left',
      and 'right' facts.
    - Extracts the goal requirements, storing them in a dictionary `goal_painted`
      mapping goal tile names to their required colors.
    - Identifies all robot names present in the initial state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state using `task.goal_reached(state)`. If yes, return 0.
    2. Extract the current locations (`robot-at`) and colors (`robot-has`) of all robots from the state facts.
    3. Extract the set of tiles that are currently clear (`clear`) from the state facts.
    4. Extract the set of tiles that are currently painted (`painted`) from the state facts.
    5. Identify the set of unpainted goal tiles. These are tiles `T` such that
       `(painted T C_goal)` is in the goal facts but not in the current state facts.
       If a goal tile `T` is found to be painted with a color `C_wrong` where `C_wrong`
       is not the required goal color `C_goal`, the state is considered unsolvable
       by this heuristic, and `float('inf')` is returned.
    6. Initialize the total heuristic value `h` to 0.
    7. For each unpainted goal tile `T` needing color `C_goal`:
        a. Find the set of tiles `Adj(T)` adjacent to `T` using the pre-computed `adjacency_map`.
        b. Filter `Adj(T)` to find `ClearAdj(T, state)`, the set of adjacent tiles that are currently clear.
        c. If `ClearAdj(T, state)` is empty, the tile cannot be painted directly in this state. Return `float('inf')` as the state is likely unsolvable or requires complex actions not captured by the simple heuristic model.
        d. Initialize `min_robot_cost` for this tile to `float('inf')`.
        e. For each robot `R`:
            i. Get the robot's current location `L_R` and color `C_R`. If robot state is incomplete, skip.
            ii. Calculate `cost_color`: 0 if `C_R == C_goal`, 1 otherwise (for `change_color`).
            iii. Calculate `dist`: the shortest path distance (number of moves) from `L_R` to any tile in `ClearAdj(T, state)` using only tiles in the current `clear_tiles` set as traversable nodes in a BFS graph. The BFS starts at `L_R` (distance 0), and moves are only allowed to adjacent tiles that are in `clear_tiles`.
            iv. If `dist` is `float('inf')` (no path exists), this robot cannot reach a clear adjacent tile. Continue to the next robot.
            v. Calculate `robot_cost = cost_color + dist + 1` (1 for the paint action).
            vi. Update `min_robot_cost = min(min_robot_cost, robot_cost)`.
        f. If `min_robot_cost` is still `float('inf')` after checking all robots, it means no robot can paint this tile in this state according to the heuristic model. Return `float('inf')`.
        g. Add `min_robot_cost` to the total heuristic `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static task information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.tile_coords = {}
        self.adjacency_map = collections.defaultdict(set)
        self.goal_painted = {}
        self.robots = set()
        self.colors = set()

        # Pre-process static facts
        for fact in task.static:
            fact_parts = fact.strip('()').split()
            if not fact_parts: continue # Skip empty facts

            predicate = fact_parts[0]

            if predicate in ('up', 'down', 'left', 'right') and len(fact_parts) == 3:
                # Adjacency facts are symmetric in grid domains
                tile1, tile2 = fact_parts[1], fact_parts[2]
                self.adjacency_map[tile1].add(tile2)
                self.adjacency_map[tile2].add(tile1)
                self._parse_tile_name(tile1)
                self._parse_tile_name(tile2)
            elif predicate == 'available-color' and len(fact_parts) == 2:
                self.colors.add(fact_parts[1])

        # Extract robots from initial state (they must be there)
        # A more robust way would be to parse objects from the PDDL problem file
        # but we can infer them from initial state facts like robot-at or robot-has
        for fact in task.initial_state:
             fact_parts = fact.strip('()').split()
             if not fact_parts: continue # Skip empty facts
             if fact_parts[0] in ('robot-at', 'robot-has') and len(fact_parts) >= 2:
                 # Assuming anything appearing as arg1 of robot-at/robot-has is a robot.
                 self.robots.add(fact_parts[1])


        # Extract goal painted tiles
        for goal_fact in task.goals:
            goal_parts = goal_fact.strip('()').split()
            if not goal_parts: continue # Skip empty facts
            if goal_parts[0] == 'painted' and len(goal_parts) == 3:
                tile, color = goal_parts[1], goal_parts[2]
                self.goal_painted[tile] = color
                self._parse_tile_name(tile) # Ensure goal tiles are in tile_coords

    def _parse_tile_name(self, tile_name):
        """Parses 'tile_row_col' name and stores coordinates."""
        if tile_name not in self.tile_coords and tile_name.startswith('tile_'):
            try:
                parts = tile_name.split('_')
                # Check if there are enough parts and the row/col are digits
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[tile_name] = (row, col)
                # else: print(f"Warning: Unexpected tile name format: {tile_name}")
            except (ValueError, IndexError):
                # print(f"Warning: Could not parse tile name: {tile_name}")
                pass # Ignore tiles that don't match the expected format

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An integer heuristic value or float('inf').
        """
        # Check if the state is a goal state
        if self.task.goal_reached(state):
            return 0

        # Extract dynamic state information
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles = {} # tile -> color

        for fact in state:
            fact_parts = fact.strip('()').split()
            if not fact_parts: continue # Skip empty facts

            predicate = fact_parts[0]
            if predicate == 'robot-at' and len(fact_parts) == 3:
                robot, tile = fact_parts[1], fact_parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has' and len(fact_parts) == 3:
                robot, color = fact_parts[1], fact_parts[2]
                robot_colors[robot] = color
            elif predicate == 'clear' and len(fact_parts) == 2:
                clear_tiles.add(fact_parts[1])
            elif predicate == 'painted' and len(fact_parts) == 3:
                tile, color = fact_parts[1], fact_parts[2]
                painted_tiles[tile] = color

        unpainted_goal_tiles = {} # tile -> required_color
        for goal_tile, required_color in self.goal_painted.items():
            current_color = painted_tiles.get(goal_tile)

            if current_color is None:
                 # Tile is not painted
                 unpainted_goal_tiles[goal_tile] = required_color
            elif current_color != required_color:
                 # Tile is painted with the wrong color - unsolvable based on domain actions
                 return float('inf')
            # else: tile is painted with the correct color, it's a satisfied goal

        h = 0
        for tile_to_paint, required_color in unpainted_goal_tiles.items():
            min_robot_cost = float('inf')

            # Find clear adjacent tiles for the current tile_to_paint
            adjacent_tiles = self.adjacency_map.get(tile_to_paint, set())
            # A tile must be clear to be painted onto, and the robot must move onto
            # an adjacent clear tile to paint.
            clear_adjacent_tiles = {adj_tile for adj_tile in adjacent_tiles if adj_tile in clear_tiles}

            # If no adjacent tile is clear, this tile cannot be painted directly.
            # This state is likely unsolvable under the heuristic's assumptions.
            # A more complex heuristic would consider moving robots off neighbors.
            # For this simple heuristic, we return inf.
            if not clear_adjacent_tiles:
                 return float('inf')

            for robot in self.robots:
                robot_loc = robot_locations.get(robot)
                robot_col = robot_colors.get(robot)

                # Ensure robot state is known
                if robot_loc is None or robot_col is None:
                    continue

                # Cost to get the correct color
                cost_color = 0 if robot_col == required_color else 1

                # Calculate shortest path distance from robot_loc to any clear_adjacent_tile
                # BFS searches for path on clear tiles, starting from robot_loc.
                dist = self._bfs_distance(robot_loc, clear_adjacent_tiles, clear_tiles)

                if dist != float('inf'):
                    # Total cost for this robot to paint this tile
                    robot_cost = cost_color + dist + 1 # +1 for the paint action
                    min_robot_cost = min(min_robot_cost, robot_cost)

            # If no robot can paint this tile
            if min_robot_cost == float('inf'):
                return float('inf')

            h += min_robot_cost

        return h

    def _bfs_distance(self, start_tile, target_tiles, clear_tiles):
        """
        Performs BFS to find the shortest distance (number of moves) from start_tile
        to any tile in target_tiles, using only tiles in clear_tiles as valid
        destinations for moves.

        Args:
            start_tile: The tile where the robot starts.
            target_tiles: A set of target tile names (these must be clear).
            clear_tiles: A set of tile names that are currently clear.

        Returns:
            The minimum distance (number of moves) or float('inf') if no target is reachable.
        """
        # The robot is at start_tile. It can move to an adjacent tile if that tile is clear.
        # The BFS finds the shortest path from start_tile to any target_tile,
        # where each step in the path (except the start) must land on a clear tile.

        queue = collections.deque([(start_tile, 0)])
        visited = {start_tile} # Mark the starting tile as visited

        while queue:
            current_tile, dist = queue.popleft()

            # If the current tile is one of the targets, we found a path.
            # The distance `dist` is the number of moves to reach this target tile.
            if current_tile in target_tiles:
                 return dist

            # Explore neighbors
            for neighbor in self.adjacency_map.get(current_tile, set()):
                # A move is possible to 'neighbor' if 'neighbor' is clear AND not yet visited
                if neighbor in clear_tiles and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If the queue is empty and no target was reached
        return float('inf')
