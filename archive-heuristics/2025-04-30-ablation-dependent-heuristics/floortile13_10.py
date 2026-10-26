import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string like '(predicate arg1 arg2)'"""
    # Remove outer parentheses
    content = fact_string.strip()
    if not content.startswith('(') or not content.endswith(')'):
         # Handle potential malformed fact strings, though unlikely with planner output
         # Or raise an error
         return None, [] # Or handle error appropriately
    content = content[1:-1].strip()
    if not content: return None, []

    # Split predicate and arguments
    parts = content.split()
    if not parts: return None, []

    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the number of actions required to reach the goal state.
        The heuristic sums the estimated cost for each goal tile that is
        not yet painted correctly. For each such tile, it calculates the
        minimum cost among all robots to paint that tile, considering
        color change, movement to a clear adjacent tile, and the paint action.
        It adds a large penalty if a goal tile is painted with the wrong color
        or if no robot can reach a clear adjacent tile to paint a clear goal tile,
        indicating a likely dead end.

    Assumptions:
        - The grid structure is defined by the up/down/left/right static predicates.
        - Tiles are either clear or painted.
        - There is no action to unpaint a tile.
        - Solvable instances do not require unpainting tiles.
        - The heuristic value is finite for solvable states.

    Heuristic Initialization:
        - Parses static facts to build the adjacency graph of tiles.
        - Computes all-pairs shortest paths (distances) between tiles using BFS
          on the grid graph.
        - Parses goal facts to identify which tiles need to be painted with which
          colors.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Parse the current state to identify:
           - The location of each robot.
           - The color held by each robot.
           - The current painting status of each tile (which color, or clear).
        3. Iterate through each tile specified in the goal as needing to be painted
           with a specific color (`tile_goal`, `color_goal`).
        4. For the current `tile_goal`:
           a. If `tile_goal` is already painted with `color_goal` in the current state,
              this goal requirement is met for this tile. Add 0 to `h` and continue
              to the next goal tile.
           b. If `tile_goal` is painted with any color *other than* `color_goal` in
              the current state, this tile cannot be repainted with the correct color
              due to the lack of an unpaint action. This state is likely a dead end
              for this goal. Add a large penalty (e.g., 1000) to `h` and continue
              to the next goal tile.
           c. If `tile_goal` is `clear` in the current state, it needs to be painted.
              Calculate the minimum cost for any robot to paint this tile:
              i. Initialize `min_cost_tile` to infinity.
              ii. Find all tiles adjacent to `tile_goal` using the precomputed graph.
              iii. For each robot:
                  - Get the robot's current location (`loc_r`) and color (`color_r`).
                  - Calculate the cost to get the correct color: 0 if `color_r` is
                    `color_goal`, 1 otherwise (for a `change_color` action).
                  - Calculate the minimum movement cost for this robot to reach *any*
                    adjacent tile (`adj_t`) of `tile_goal` that is currently `clear`.
                    This is the minimum precomputed distance from `loc_r` to any
                    `adj_t` where `(clear adj_t)` is true in the current state.
                    If no adjacent tile is clear and reachable, this robot cannot
                    paint the tile in the next step.
                  - If a reachable clear adjacent tile is found:
                      - The cost for this robot is `cost_color + min_move_cost + 1`
                        (where 1 is for the `paint` action).
                      - Update `min_cost_tile` with the minimum of its current value
                        and the cost for this robot.
              iv. If, after checking all robots, `min_cost_tile` is still infinity
                  (meaning no robot could reach a clear adjacent tile), this state
                  is likely a dead end for this goal. Add a large penalty (e.g., 1000)
                  to `h`.
              v. Otherwise, add `min_cost_tile` to `h`.
        5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self.adj = {} # tile -> set of adjacent tiles
        self.tiles = set()
        self.goal_painting = {} # tile -> color
        self.distances = {} # tile1 -> tile2 -> distance
        self.large_penalty = 1000 # Penalty for dead ends

        # 1. Parse static facts to build the adjacency graph and collect tiles
        for fact in self.task.static:
            predicate, args = parse_fact(fact)
            if predicate in ['up', 'down', 'left', 'right']:
                if len(args) == 2:
                    tile1, tile2 = args
                    self.adj.setdefault(tile1, set()).add(tile2)
                    self.adj.setdefault(tile2, set()).add(tile1)
                    self.tiles.add(tile1)
                    self.tiles.add(tile2)

        # 2. Compute all-pairs shortest paths (distances)
        for start_tile in self.tiles:
            self.distances[start_tile] = {}
            queue = collections.deque([(start_tile, 0)])
            visited = {start_tile: 0}

            while queue:
                current_tile, dist = queue.popleft()
                self.distances[start_tile][current_tile] = dist

                for neighbor in self.adj.get(current_tile, set()):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # 3. Parse goal facts
        for goal_fact in self.task.goals:
            predicate, args = parse_fact(goal_fact)
            if predicate == 'painted':
                if len(args) == 2:
                    tile, color = args
                    self.goal_painting[tile] = color

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # 2. Parse the current state
        robot_locations = {} # robot -> tile
        robot_colors = {} # robot -> color
        current_painting = {} # tile -> color
        current_clear_tiles = set()

        for fact in state:
            predicate, args = parse_fact(fact)
            if predicate == 'robot-at' and len(args) == 2:
                robot, tile = args
                robot_locations[robot] = tile
            elif predicate == 'robot-has' and len(args) == 2:
                robot, color = args
                robot_colors[robot] = color
            elif predicate == 'painted' and len(args) == 2:
                tile, color = args
                current_painting[tile] = color
            elif predicate == 'clear' and len(args) == 1:
                tile = args[0]
                current_clear_tiles.add(tile)

        # 1. Initialize heuristic
        h = 0

        # 3. Iterate through each goal tile
        for tile_goal, color_goal in self.goal_painting.items():
            # Check current status of the goal tile
            is_painted_correctly = tile_goal in current_painting and current_painting[tile_goal] == color_goal
            is_painted_wrong = tile_goal in current_painting and tile_goal not in current_clear_tiles and not is_painted_correctly # If not clear and not painted correctly, it must be painted wrong

            if is_painted_correctly:
                continue # Goal satisfied for this tile
            elif is_painted_wrong:
                h += self.large_penalty # Dead end for this tile
                continue
            # If not painted correctly and not painted wrong, it must be clear and needs painting

            min_cost_tile = float('inf')
            adj_tiles = self.adj.get(tile_goal, set())

            if not adj_tiles:
                 # Should not happen in valid grid problems, but handle defensively
                 h += self.large_penalty
                 continue

            can_paint_tile = False # Flag to check if any robot can paint this tile

            # iii. For each robot
            for robot, loc_r in robot_locations.items():
                color_r = robot_colors.get(robot)
                if color_r is None:
                    # Robot doesn't have a color? Assuming robot always has a color.
                    continue

                cost_color = 0 if color_r == color_goal else 1

                min_move_cost = float('inf')
                reachable_adjacent_found = False

                # Find minimum moves for this robot to a *clear* adjacent tile
                for adj_t in adj_tiles:
                    if adj_t in current_clear_tiles:
                         # Check if path exists in precomputed distances
                         if loc_r in self.distances and adj_t in self.distances[loc_r]:
                            dist = self.distances[loc_r][adj_t]
                            min_move_cost = min(min_move_cost, dist)
                            reachable_adjacent_found = True

                # If robot can reach a clear adjacent tile
                if reachable_adjacent_found:
                    cost_r = cost_color + min_move_cost + 1 # 1 for paint action
                    min_cost_tile = min(min_cost_tile, cost_r)
                    can_paint_tile = True # At least one robot can potentially paint it

            # iv. If no robot can paint this tile
            if not can_paint_tile:
                 h += self.large_penalty
            else:
                 # v. Add minimum cost for this tile
                 h += min_cost_tile

        # 5. Return total heuristic value
        return h
