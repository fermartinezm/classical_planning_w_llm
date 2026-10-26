import collections

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum estimated
        cost for each unpainted goal tile. The minimum estimated cost for a
        tile is the minimum over all robots of the cost for that robot to
        get the correct color, move to a tile adjacent to the goal tile, and paint.

    Assumptions:
        - The grid structure defined by up/down/left/right predicates is static.
        - All colors required in the goal are available (defined by available-color).
        - Robots initially have a color (as per example instances).
        - A tile painted with the wrong color makes the state unsolvable in this domain
          (since there's no unpaint action and painting requires the tile to be clear).
        - The shortest path distance on the static grid is a reasonable estimate
          for movement cost, ignoring dynamic clear status of intermediate tiles
          or the target adjacent tile itself during distance calculation.

    Heuristic Initialization:
        - Parses static facts (up, down, left, right) to build the tile adjacency graph.
        - Computes all-pairs shortest paths between tiles using BFS.
        - Parses goal facts to identify target colors for goal tiles.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the set of goal tiles that are not painted with the correct color
           in the current state.
        2. For each such unpainted goal tile T with target color C_goal:
           a. Check if T is painted with any color other than C_goal in the current state.
              This is done by checking if T is painted, and if so, if the color is not C_goal.
              If T is painted with a wrong color, the state is unsolvable, return infinity.
           b. Find all tiles X that are adjacent to T based on the static grid structure.
           c. Calculate the minimum cost for any robot R to paint tile T:
              i. Initialize minimum cost for this tile to infinity.
              ii. For each robot R:
                  - Determine the robot's current location and color from the state.
                  - Calculate the cost for R to obtain color C_goal: 0 if R already has C_goal, 1 otherwise (assuming the robot has some color and C_goal is available).
                  - Calculate the minimum cost for R to move from its current location to any tile X adjacent to T. This is the minimum precomputed shortest path distance from R's location to any adjacent tile X.
                  - If a path exists from the robot's location to at least one adjacent tile:
                      - The estimated cost for R to paint T is (color cost) + (minimum move cost) + 1 (for the paint action).
                      - Update the minimum cost for tile T if this robot's cost is lower.
           d. If after checking all robots, the minimum cost for tile T is still infinity, the state is likely unsolvable (e.g., tile is unreachable), return infinity.
           e. Add the minimum estimated cost for tile T to the total heuristic value.
        3. The total heuristic value is the sum of the minimum estimated costs for each unpainted goal tile.
        4. If the state is a goal state (all goal tiles painted correctly), the heuristic is 0 because there are no unpainted goal tiles.
    """

    def __init__(self, task):
        self.task = task
        self.goal_tiles = {}  # tile -> target_color
        self.tile_neighbors = collections.defaultdict(set) # tile -> set of adjacent tiles
        self.tile_distances = {} # tile1 -> tile2 -> distance
        self.all_tiles = set()

        # Parse static facts to build grid graph
        # (up y x) means y is up from x. Adjacency is symmetric for movement.
        # (down y x) means y is down from x.
        # (left y x) means y is left from x.
        # (right y x) means y is right from x.
        # The predicates define the relationship FROM the second argument TO the first argument.
        # Example: (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1.
        # This implies tile_0_1 and tile_1_1 are adjacent.
        for fact in self.task.static:
            fact_parts = fact.strip("()").split()
            if len(fact_parts) == 3 and fact_parts[0] in ['up', 'down', 'left', 'right']:
                pred, tile1, tile2 = fact_parts
                # Add adjacency in both directions
                self.tile_neighbors[tile1].add(tile2)
                self.tile_neighbors[tile2].add(tile1)
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # Compute all-pairs shortest paths using BFS
        for start_tile in self.all_tiles:
            self.tile_distances[start_tile] = {}
            queue = collections.deque([(start_tile, 0)])
            visited = {start_tile}
            while queue:
                current_tile, dist = queue.popleft()
                self.tile_distances[start_tile][current_tile] = dist
                for neighbor in self.tile_neighbors.get(current_tile, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Parse goal facts
        for goal_fact in self.task.goals:
            goal_parts = goal_fact.strip("()").split()
            if len(goal_parts) == 3 and goal_parts[0] == 'painted':
                _, tile, color = goal_parts
                self.goal_tiles[tile] = color

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        h = 0
        robot_locations = {}
        robot_colors = {}
        painted_tiles_in_state = {} # tile -> color

        # Extract dynamic info from state
        for fact in state:
            fact_parts = fact.strip("()").split()
            if len(fact_parts) >= 2: # Handle predicates with different arities
                 pred = fact_parts[0]
                 if pred == 'robot-at' and len(fact_parts) == 3:
                     robot_locations[fact_parts[1]] = fact_parts[2]
                 elif pred == 'robot-has' and len(fact_parts) == 3:
                     robot_colors[fact_parts[1]] = fact_parts[2]
                 elif pred == 'painted' and len(fact_parts) == 3:
                     # Store the color the tile is painted with.
                     # Assuming a tile is painted with at most one color relevant to the goal.
                     painted_tiles_in_state[fact_parts[1]] = fact_parts[2]

        # Calculate heuristic for unpainted goal tiles
        for goal_tile, target_color in self.goal_tiles.items():
            # Check if the goal tile is already painted with the target color
            if goal_tile in painted_tiles_in_state and painted_tiles_in_state[goal_tile] == target_color:
                continue # This goal is already satisfied

            # Check if the goal tile is painted with the wrong color
            if goal_tile in painted_tiles_in_state and painted_tiles_in_state[goal_tile] != target_color:
                 # Tile is painted, but not with the target color. Unsolvable in this domain.
                 return float('inf')

            # Tile needs painting (and is currently not painted or painted wrong)
            # Since we checked for painted wrong above, if it's in painted_tiles_in_state
            # it must be painted correctly (which we skipped).
            # If it's not in painted_tiles_in_state, it needs painting.

            min_paint_cost_for_tile = float('inf')

            # Find adjacent tiles for the goal tile
            adjacent_tiles = self.tile_neighbors.get(goal_tile, set())

            # Consider each robot
            for robot_name, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_name)

                if robot_color is None:
                    # Robot has no color, cannot paint or change color.
                    continue

                # Cost to get the target color
                color_cost = 0 if robot_color == target_color else 1

                # Minimum move cost to an adjacent tile
                min_move_cost_to_adjacent = float('inf')
                if robot_loc in self.tile_distances: # Ensure robot location is a known tile
                    for adj_tile in adjacent_tiles:
                        if adj_tile in self.tile_distances[robot_loc]:
                             move_cost = self.tile_distances[robot_loc][adj_tile]
                             min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, move_cost)

                # If there is a path from the robot's location to at least one adjacent tile
                if min_move_cost_to_adjacent != float('inf'):
                    # Total estimated cost for this robot to paint this tile
                    # color_cost (0 or 1) + move_cost + paint_action (1)
                    cost_for_this_robot = color_cost + min_move_cost_to_adjacent + 1
                    min_paint_cost_for_tile = min(min_paint_cost_for_tile, cost_for_this_robot)

            # Add the minimum cost to paint this tile to the total heuristic
            # If min_paint_cost_for_tile is still infinity, it means no robot can reach
            # an adjacent tile or get the color. This state is likely unsolvable.
            if min_paint_cost_for_tile == float('inf'):
                 return float('inf') # Unsolvable state

            h += min_paint_cost_for_tile

        return h
