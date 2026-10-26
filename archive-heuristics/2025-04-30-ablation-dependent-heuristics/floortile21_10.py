from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)'.

    @param fact_str: The PDDL fact string.
    @return: A tuple containing the predicate name and a list of arguments,
             or (None, []) if the string is not a valid fact format.
    """
    # Remove leading/trailing parentheses and split by spaces
    # Handle empty strings or malformed facts gracefully
    fact_str = fact_str.strip()
    if not fact_str or fact_str[0] != '(' or fact_str[-1] != ')':
        # Not a valid fact string format we expect
        return None, []

    content = fact_str[1:-1].strip()
    if not content:
        return None, []

    parts = content.split()
    if not parts:
        return None, []

    predicate = parts[0]
    args = parts[1:]
    return predicate, args


class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to satisfy unsatisfied goal conditions.
    For each tile that needs to be painted according to the goal but isn't,
    it calculates the minimum estimated cost for any robot to paint that tile.
    This minimum cost is the sum of:
    1. Cost to change the robot's color if it doesn't have the target color (1 if needed, 0 otherwise).
    2. Minimum cost (number of moves) for the robot to reach any tile adjacent to the target tile.
       This move cost calculation *relaxes* the requirement that the destination tile must be clear.
    3. Cost of the paint action (1).
    The total heuristic value is the sum of these minimum costs over all tiles that need painting.
    If any goal tile is painted with the wrong color, or if a clear goal tile has no adjacent tiles reachable by any robot,
    the heuristic returns infinity, indicating a likely dead-end state.

    Assumptions:
    - The problem represents a grid where tiles are connected by up/down/left/right predicates.
    - Goal facts only involve (painted tile color).
    - Robots start with a color and can change to any available color.
    - The set of all tiles, robots, and colors can be inferred from the initial state and static facts.
    - If a goal tile is painted with the wrong color, the state is unsolvable (no unpaint action).
    - All goal colors are available colors.
    - The heuristic calculation for move cost relaxes the 'clear' precondition for the destination tile.

    Heuristic Initialization:
    1. Parse static facts and initial state facts to identify all tiles, robots, and colors.
    2. Parse static facts to build the grid graph (adjacency list) based on up/down/left/right predicates.
    3. Parse goal facts to identify the target color for each goal tile.
    4. Pre-compute all-pairs shortest paths (distances) between tiles on the grid using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Extract current robot locations, robot colors, clear tiles, and painted tiles from the state.
    3. Identify tiles that need painting according to the goal but are not yet painted correctly.
    4. For each tile identified in step 3:
        a. Check if the tile is painted with a color different from the goal color. If yes, return infinity (unsolvable state).
        b. Check if the tile is clear. If not clear (and not painted wrong), this indicates a state inconsistency based on domain predicates; return infinity.
        c. If the tile is clear, find all tiles adjacent to it based on the pre-computed grid graph.
        d. If there are no adjacent tiles, return infinity (cannot paint this tile).
        e. Calculate the minimum cost for any robot to paint this tile:
            i. For each robot:
                - Find the robot's current location and color.
                - Calculate the cost to change color: 1 if the robot's current color is not the target color, 0 otherwise.
                - Calculate the minimum move cost: Find the minimum pre-computed distance from the robot's current location to any of the adjacent tiles (relaxing the clear precondition for the destination).
                - The estimated cost for this robot to paint this specific tile is (cost to change color) + (minimum move cost) + 1 (for the paint action).
            ii. The minimum cost for the tile is the minimum of the estimated costs calculated for each robot.
        f. Add the minimum cost for this tile to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static information.

        @param task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.goal_tiles = {} # {tile: target_color}
        self.tile_names = set()
        self.robot_names = set()
        self.color_names = set()
        self.grid_adj = {} # Adjacency list for the grid {tile: [neighbor_tile, ...]}
        self.distances = {} # Pre-computed distances between tiles {(tile1, tile2): distance}

        # 1. Infer objects (tiles, robots, colors) from initial state, static facts, and goal facts
        all_facts = set(task.initial_state) | set(task.static) | set(task.goals)
        for fact_str in all_facts:
            predicate, args = parse_fact(fact_str)
            if not predicate: continue

            if predicate in {'up', 'down', 'left', 'right'}:
                 if len(args) == 2:
                     self.tile_names.update(args)
            elif predicate == 'clear':
                 if len(args) == 1:
                     self.tile_names.add(args[0])
            elif predicate == 'painted':
                 if len(args) == 2:
                     self.tile_names.add(args[0])
                     self.color_names.add(args[1])
            elif predicate == 'robot-at':
                 if len(args) == 2:
                     self.robot_names.add(args[0])
                     self.tile_names.add(args[1])
            elif predicate == 'robot-has':
                 if len(args) == 2:
                     self.robot_names.add(args[0])
                     self.color_names.add(args[1])
            elif predicate == 'available-color':
                 if len(args) == 1:
                     self.color_names.add(args[0])

        # Initialize adjacency list for all found tiles
        for tile in self.tile_names:
            self.grid_adj[tile] = []

        # 2. Build grid graph from static facts
        for fact_str in task.static:
            predicate, args = parse_fact(fact_str)
            if not predicate: continue
            if predicate in {'up', 'down', 'left', 'right'}:
                if len(args) == 2:
                    tile1, tile2 = args
                    # Add bidirectional edges
                    if tile1 in self.grid_adj and tile2 in self.grid_adj: # Ensure both tiles were identified
                         self.grid_adj[tile1].append(tile2)
                         self.grid_adj[tile2].append(tile1)
                    # else: print(f"Warning: Grid fact involves unknown tile: {fact_str}") # Optional debug

        # 3. Parse goal facts
        for goal_fact_str in task.goals:
            predicate, args = parse_fact(goal_fact_str)
            if predicate == 'painted' and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

        # 4. Pre-compute all-pairs shortest paths (BFS from each tile)
        for start_tile in self.tile_names:
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0

            while q:
                current_tile, dist = q.popleft()

                # Get neighbors, handling potential missing keys if tile_names wasn't perfect
                neighbors = self.grid_adj.get(current_tile, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_tile, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal, or float('inf') if likely unsolvable.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        tiles_to_paint = {} # {tile: target_color}
        current_painted = {} # {tile: color}
        current_clear = set()
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}

        # Extract relevant information from the current state
        state_set = set(state) # Convert to set for O(1) average lookup

        for fact_str in state_set:
            predicate, args = parse_fact(fact_str)
            if not predicate: continue
            if predicate == 'painted' and len(args) == 2:
                current_painted[args[0]] = args[1]
            elif predicate == 'clear' and len(args) == 1:
                current_clear.add(args[0])
            elif predicate == 'robot-at' and len(args) == 2:
                robot_locations[args[0]] = args[1]
            elif predicate == 'robot-has' and len(args) == 2:
                robot_colors[args[0]] = args[1]

        # Identify tiles that need painting and check for dead ends
        for tile, target_color in self.goal_tiles.items():
            is_painted_correctly = f'(painted {tile} {target_color})' in state_set

            if is_painted_correctly:
                continue # Goal satisfied for this tile

            # Check if painted with wrong color
            is_painted_wrong = False
            if tile in current_painted and current_painted[tile] != target_color:
                 is_painted_wrong = True

            if is_painted_wrong:
                # Goal tile painted with the wrong color - likely unsolvable
                return float('inf')

            # If not painted correctly and not painted wrong, it must be clear to be paintable
            is_clear = f'(clear {tile})' in state_set

            if not is_clear:
                 # Goal tile is not painted correctly, not painted wrong, and not clear.
                 # This state is inconsistent with the domain predicates for a goal tile.
                 # A tile is either clear or painted.
                 # If it's a goal tile and not painted correctly, it must be clear to be paintable.
                 # If it's not clear, it cannot be painted.
                 # This implies a dead end.
                 return float('inf')

            # Tile is clear and needs painting
            tiles_to_paint[tile] = target_color


        # If all goal tiles are already painted correctly (should be caught by task.goal_reached, but safe check)
        if not tiles_to_paint:
             return 0

        total_h = 0

        # Calculate cost for each tile that needs painting
        for tile_to_paint, target_color in tiles_to_paint.items():
            # Find adjacent tiles
            adjacent_tiles = self.grid_adj.get(tile_to_paint, [])

            if not adjacent_tiles:
                # Cannot paint this tile - no adjacent tile to paint from
                # This tile is a goal tile but has no neighbors in the grid.
                return float('inf')

            min_cost_for_tile = float('inf')

            # Calculate minimum cost over all robots
            for robot in self.robot_names:
                robot_loc = robot_locations.get(robot)
                robot_color = robot_colors.get(robot)

                if robot_loc is None or robot_color is None:
                    # Robot state is inconsistent (e.g., robot exists but not at a location or has no color)
                    # Should not happen in valid states generated by the planner.
                    continue # Skip this robot

                # Cost to change color
                # Assumes target_color is available, which is a reasonable assumption
                # if it's a goal color.
                cost_change_color = 1 if robot_color != target_color else 0

                # Minimum move cost to any adjacent tile (relaxing the clear precondition for destination)
                min_move_cost_to_adjacent = float('inf')
                for adj_tile in adjacent_tiles:
                    # Distance lookup
                    move_cost = self.distances.get((robot_loc, adj_tile), float('inf'))
                    min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, move_cost)

                if min_move_cost_to_adjacent == float('inf'):
                    # This robot cannot reach any adjacent tile for this goal tile (disconnected part of grid?)
                    continue # Try next robot

                # Total estimated cost for this robot to paint this specific tile
                total_cost_for_robot = cost_change_color + min_move_cost_to_adjacent + 1 # +1 for paint action

                min_cost_for_tile = min(min_cost_for_tile, total_cost_for_robot)

            if min_cost_for_tile == float('inf'):
                 # No robot can paint this tile (either cannot reach any adjacent tile,
                 # or cannot get the right color - though color change is always possible if available)
                 # The only reason min_cost_for_tile would still be inf here is if no robot
                 # could reach *any* adjacent_tile.
                 return float('inf')

            total_h += min_cost_for_tile

        return total_h
