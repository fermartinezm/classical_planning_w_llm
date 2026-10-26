import collections

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum estimated cost for each unsatisfied goal tile to be painted
        with the required color. The cost for a single tile is estimated
        as the minimum cost for any robot to acquire the correct color,
        move to a tile adjacent to the target tile, and perform the paint action.
        Movement costs are estimated using precomputed shortest path distances
        on the grid graph, ignoring the dynamic 'clear' predicate for intermediate
        movement tiles (a relaxation). A large penalty is added if an unsatisfied
        goal tile is not clear, as it cannot be painted.

    Assumptions:
        - The grid defined by 'up', 'down', 'left', 'right' predicates is connected
          or relevant parts are connected. If a goal tile is unreachable from all
          robot starting positions in the static grid, a large penalty is applied.
        - Goal facts only involve the 'painted' predicate.
        - Robots always have exactly one color.
        - The set of objects (tiles, robots, colors) can be inferred from the
          initial state, goal, and static facts.

    Heuristic Initialization:
        The constructor analyzes the task definition (initial state, goal, static facts)
        to precompute necessary information:
        1. Identify all tiles, robots, and colors from the initial state, goal, and static facts.
        2. Build an adjacency map representing the grid connectivity based on 'up', 'down',
           'left', and 'right' static facts. This map stores for each tile, the set of
           adjacent tiles.
        3. Compute all-pairs shortest path distances between all tiles on this grid graph
           using Breadth-First Search (BFS). These distances represent the minimum number
           of move actions required to get from one tile to another, ignoring the 'clear'
           precondition for intermediate tiles.
        4. Store the goal facts that require tiles to be painted with specific colors.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the current location and color of each robot from the state facts.
        2. Identify the set of goal facts '(painted tile color)' that are not satisfied
           in the current state.
        3. If all goal facts are satisfied, the heuristic value is 0.
        4. Initialize the total heuristic value to 0.
        5. For each unsatisfied goal fact '(painted T C)':
           a. Check if '(clear T)' is in the current state.
           b. If '(clear T)' is NOT in the state:
              - Tile T cannot be painted. This state is likely a dead end for this goal.
                Add a large penalty to the total heuristic value for this tile.
           c. If '(clear T)' IS in the state:
              - Initialize the minimum cost to paint tile T with color C by any robot to infinity.
              - For each robot R:
                 i. Determine the cost for robot R to acquire color C. This is 1 if R currently
                    has a different color, and 0 if R already has color C.
                 ii. Determine the minimum cost for robot R to move from its current location
                     LocR to *any* tile AdjT that is adjacent to T. This is found by looking
                     up the precomputed shortest distance between LocR and each AdjT in the
                     adjacency map of T, and taking the minimum distance.
                 iii. The estimated cost for robot R to paint tile T is the sum of the color
                      acquisition cost, the minimum movement cost to an adjacent tile, and 1
                      for the paint action itself.
                 iv. Update the minimum cost to paint tile T with color C by any robot with
                      the minimum of the current minimum and the cost for robot R.
              - Add the minimum cost found for tile T to the total heuristic value. If no
                robot can reach any adjacent tile based on the static grid, add a large
                penalty value instead.
        6. Return the total heuristic value.
    """

    def __init__(self, task):
        # Parse objects
        self.robots = set()
        self.tiles = set()
        self.colors = set()

        # Helper to parse fact strings
        def parse_fact(fact_str):
            # Remove parentheses and split by space
            parts = fact_str.strip("()").split()
            return parts[0], parts[1:]

        # Collect objects from initial state, goal, and static facts
        # Iterate through all facts to find objects
        all_facts_strs = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_str in all_facts_strs:
            pred, args = parse_fact(fact_str)
            if pred == 'robot-at':
                # Expects 2 arguments: robot, tile
                if len(args) == 2:
                    self.robots.add(args[0])
                    self.tiles.add(args[1])
            elif pred == 'robot-has':
                 # Expects 2 arguments: robot, color
                 if len(args) == 2:
                    self.robots.add(args[0])
                    self.colors.add(args[1])
            elif pred in ['up', 'down', 'left', 'right']:
                 # Expects 2 arguments: tile, tile
                 if len(args) == 2:
                    self.tiles.add(args[0])
                    self.tiles.add(args[1])
            elif pred == 'clear':
                 # Expects 1 argument: tile
                 if len(args) == 1:
                    self.tiles.add(args[0])
            elif pred == 'painted':
                 # Expects 2 arguments: tile, color
                 if len(args) == 2:
                    self.tiles.add(args[0])
                    self.colors.add(args[1])
            elif pred == 'available-color':
                 # Expects 1 argument: color
                 if len(args) == 1:
                    self.colors.add(args[0])
            # Ignore 'free-color' as it's unused in actions or goals/initial state examples

        # Build adjacency map
        self.adj_map = collections.defaultdict(set)
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred in ['up', 'down', 'left', 'right']:
                # Expects 2 arguments: tile, tile
                if len(args) == 2:
                    tile1, tile2 = args[0], args[1]
                    # (up y x) means y is up from x, so x and y are adjacent
                    self.adj_map[tile1].add(tile2)
                    self.adj_map[tile2].add(tile1) # Grid is undirected for movement

        # Compute all-pairs shortest paths (distances) using BFS
        self.distances = {}
        for start_tile in self.tiles:
            q = collections.deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[(start_tile, start_tile)] = 0

            while q:
                current_tile, dist = q.popleft()

                # Ensure current_tile is in adj_map (it should be if it's in self.tiles)
                for neighbor in self.adj_map.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_tile, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Store goal painted tiles
        self.goal_painted_tiles = {}
        # Assuming goal is a conjunction of painted facts
        # The task.goals is a frozenset of goal fact strings
        for goal_fact_str in task.goals:
             # Check if it's a painted fact and has expected arguments
             if goal_fact_str.startswith('(painted '):
                 pred, args = parse_fact(goal_fact_str)
                 if len(args) == 2:
                    tile, color = args[0], args[1]
                    self.goal_painted_tiles[tile] = color
             # Handle potential other goal types if necessary, but domain suggests only painted

    def __call__(self, state):
        # Check if goal is reached
        # Goal is reached if all required painted facts are in the state
        if all(f'(painted {tile} {color})' in state for tile, color in self.goal_painted_tiles.items()):
            return 0

        # Get current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        # Iterate through state facts to find robot info
        for fact_str in state:
            if fact_str.startswith('(robot-at '):
                 parts = fact_str.strip("()").split()
                 # Ensure fact has expected number of parts
                 if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif fact_str.startswith('(robot-has '):
                 parts = fact_str.strip("()").split()
                 # Ensure fact has expected number of parts
                 if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        h = 0
        large_penalty = 1000000 # Penalty for potentially unreachable or unpaintable tiles

        # Iterate through unsatisfied goal tiles
        for tile, goal_color in self.goal_painted_tiles.items():
            # Check if the goal fact for this tile is NOT in the current state
            if f'(painted {tile} {goal_color})' not in state:
                # This tile needs painting. Check if it's clear.
                if f'(clear {tile})' not in state:
                    # The tile is not clear, cannot be painted. Likely a dead end.
                    h += large_penalty
                else:
                    # The tile is clear and needs painting. Calculate minimum cost.
                    min_robot_cost_for_tile = float('inf')

                    # Find the best robot for this tile
                    for robot in self.robots:
                        current_location = robot_locations.get(robot)
                        current_color = robot_colors.get(robot)

                        # A robot must have a location and a color to be useful
                        if current_location is None or current_color is None:
                             continue # Skip this robot if its state is incomplete or malformed

                        # Cost to get the right color
                        color_cost = 1 if current_color != goal_color else 0

                        # Cost to move to an adjacent tile
                        min_move_cost = float('inf')
                        adjacent_tiles = self.adj_map.get(tile, [])

                        if not adjacent_tiles:
                             # This tile has no adjacent tiles in the grid graph.
                             # This is an unusual grid structure, likely making the tile unreachable
                             # for painting (which requires being adjacent).
                             # Treat as unreachable for this robot.
                             pass # min_move_cost remains infinity
                        else:
                            for adj_tile in adjacent_tiles:
                                # Use precomputed distance from robot's current location
                                # to the adjacent tile. Ignore dynamic 'clear' for intermediate moves.
                                dist = self.distances.get((current_location, adj_tile), float('inf'))
                                min_move_cost = min(min_move_cost, dist)

                        # Total cost for this robot for this tile
                        # Only consider if movement to an adjacent tile is possible (grid is connected)
                        if min_move_cost != float('inf'):
                            # Cost = color change + movement + paint action
                            robot_cost = color_cost + min_move_cost + 1
                            min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)

                    # Add the minimum cost found for this tile to the total heuristic
                    # If min_robot_cost_for_tile is still infinity, it means no robot
                    # can reach any adjacent tile based on the static grid.
                    # This tile is likely unreachable for painting. Add a large penalty.
                    if min_robot_cost_for_tile == float('inf'):
                         h += large_penalty
                    else:
                        h += min_robot_cost_for_tile

        return h
