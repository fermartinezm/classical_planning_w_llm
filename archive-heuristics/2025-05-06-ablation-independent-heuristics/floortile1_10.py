import collections
from heuristics.heuristic_base import Heuristic
from task import Task

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of tiles that still need to be painted to match the goal (minimum paint actions).
    2. The sum, over each tile that needs painting, of the minimum movement cost for any robot
       to reach a tile adjacent to it.
    3. The number of distinct colors required by unsatisfied goal tiles that are not currently
       held by any robot (minimum color acquisition actions globally).

    Assumptions:
    - The grid structure defined by up/down/left/right predicates is connected for all relevant tiles.
    - Tiles are named consistently (e.g., starting with 'tile_').
    - The only way to satisfy a (painted tile color) goal is via a paint action.
    - Tiles painted with the wrong color represent a state from which the goal is unreachable
      or requires actions not modeled (like unpainting). The heuristic returns infinity in this case.
    - All robots are initially located on a tile and have a color.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the grid structure and the shortest path
    distances between all pairs of tiles using BFS. This information is static and
    reused for every state evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine robot locations, robot colors, and which tiles are painted
       and with what color.
    2. Parse the goal state to determine which tiles need to be painted and with what color.
    3. Identify the set of unsatisfied goal tiles (tiles that need painting but are not painted
       correctly).
    4. Check if any tile is painted with a color different from its goal color. If so, return
       infinity, as this state is likely a dead end.
    5. If there are no unsatisfied goal tiles, the goal is reached, return 0.
    6. Get the set of all robots. If there are no robots but there are unsatisfied goals, return infinity.
    7. Check reachability: For each unsatisfied goal tile, verify that at least one robot can reach
       a tile adjacent to it. If not, return infinity.
    8. Initialize the heuristic value `h` to 0.
    9. Add the number of unsatisfied goal tiles to `h`. This accounts for the minimum number
       of paint actions required (1 per tile).
    10. Calculate the total minimum movement cost for the unsatisfied tiles:
        For each unsatisfied goal tile `tile_X`:
          Find the set of tiles adjacent to `tile_X` using the precomputed grid graph.
          Find the minimum distance from *any* robot's current location to *any* tile
          adjacent to `tile_X`, using the precomputed distance map.
          Add this minimum distance to a running total for movement cost.
        Add this total movement cost to `h`.
    11. Calculate the minimum color acquisition cost:
        Identify the set of colors required by the unsatisfied goal tiles.
        Identify the set of colors currently held by robots.
        Count the number of colors in the required set that are not in the current set.
        Add this count to `h`.
    12. Return the final value of `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.all_tiles = set()
        self.adj = collections.defaultdict(set)
        self.dist_map = {}

        # Predicates that involve tiles and define grid structure
        tile_relations = {'up', 'down', 'left', 'right'}
        tile_predicates = tile_relations | {'robot-at', 'clear', 'painted'}

        # Collect all facts from initial state and static info
        all_facts = set(task.initial_state) | set(task.static)

        # Also add goal facts
        all_facts.update(task.goals)

        # 1. Identify all tiles and build adjacency list
        potential_tiles = set()
        for fact_str in all_facts:
            parts = self._parse_fact(fact_str)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate in tile_predicates:
                 # Add all arguments as potential tiles
                 for arg in args:
                     # Heuristic knowledge: tiles likely start with 'tile_'
                     if arg.startswith('tile_'):
                         potential_tiles.add(arg)

            # Build adjacency list from grid relations
            if predicate in tile_relations:
                if len(args) == 2:
                    tile1, tile2 = args[0], args[1]
                    # Only add adjacency if both are potential tiles
                    if tile1 in potential_tiles and tile2 in potential_tiles:
                         self.adj[tile1].add(tile2)
                         self.adj[tile2].add(tile1) # Undirected graph for distance

        # Filter potential_tiles to only include those that appeared in grid relations or goals/init
        # This ensures we only have tiles that are part of the connected grid or directly mentioned in init/goal.
        connected_tiles = set(self.adj.keys()) | set(t for neighbors in self.adj.values() for t in neighbors)
        
        goal_tiles_set = set()
        self.goal_tiles_map = {}
        for goal_str in self.goals:
            parts = self._parse_fact(goal_str)
            if parts[0] == 'painted':
                if len(parts) > 2: # Ensure fact has tile and color
                    tile = parts[1]
                    color = parts[2]
                    if tile.startswith('tile_'): # Apply heuristic knowledge here too
                        self.goal_tiles_map[tile] = color
                        goal_tiles_set.add(tile)

        init_tiles_set = set()
        for fact_str in task.initial_state:
             parts = self._parse_fact(fact_str)
             if parts[0] == 'robot-at':
                 if len(parts) > 2:
                     tile = parts[2]
                     if tile.startswith('tile_'):
                         init_tiles_set.add(tile)
             elif parts[0] in {'clear', 'painted'}:
                 if len(parts) > 1:
                     tile = parts[1]
                     if tile.startswith('tile_'):
                         init_tiles_set.add(tile)


        # Relevant tiles are those in the connected grid or explicitly in init/goal
        self.all_tiles = connected_tiles | goal_tiles_set | init_tiles_set

        # Ensure adjacency only contains relevant tiles
        self.adj = {t: {n for n in neighbors if n in self.all_tiles} for t, neighbors in self.adj.items() if t in self.all_tiles}


        # 2. Compute all-pairs shortest paths using BFS from each relevant tile
        self.dist_map = {} # Reset dist_map
        for start_tile in self.all_tiles:
            self.dist_map[start_tile] = self._bfs(start_tile)


    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into a tuple."""
        # Remove outer parentheses and split by spaces
        content = fact_string[1:-1]
        return tuple(content.split())

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # Should not happen if all_tiles is populated correctly
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Get neighbors from the adjacency list
            neighbors = self.adj.get(current_node, set())

            for neighbor in neighbors:
                if neighbor in self.all_tiles and neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def get_adjacent_tiles(self, tile):
        """Returns the set of tiles adjacent to the given tile."""
        return self.adj.get(tile, set())

    def dist(self, tile1, tile2):
        """Returns the shortest distance between two tiles."""
        if tile1 in self.dist_map and tile2 in self.dist_map[tile1]:
            return self.dist_map[tile1][tile2]
        return float('inf') # Tiles are disconnected or not in the map


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Parse state
        robot_locs = {}
        robot_colors = {}
        painted_tiles = {}
        robots = set() # Keep track of all robot names

        for fact_str in state:
            parts = self._parse_fact(fact_str)
            if not parts: continue # Skip empty facts

            if parts[0] == 'robot-at':
                if len(parts) > 2:
                    robot, tile = parts[1], parts[2]
                    robot_locs[robot] = tile
                    robots.add(robot)
            elif parts[0] == 'robot-has':
                if len(parts) > 2:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                    robots.add(robot)
            elif parts[0] == 'painted':
                if len(parts) > 2:
                    tile, color = parts[1], parts[2]
                    painted_tiles[tile] = color
            # 'clear' facts are implicitly handled: a tile is either clear or painted.
            # If it's not in painted_tiles and is a goal tile, it's clear and needs painting.


        # 2. Identify unsatisfied goals and check for wrongly painted tiles
        unsat_goals = {} # tile -> goal_color
        wrongly_painted = False

        for goal_tile, goal_color in self.goal_tiles_map.items():
            if goal_tile in painted_tiles:
                current_color = painted_tiles[goal_tile]
                if current_color != goal_color:
                    wrongly_painted = True
                    break # Found a wrongly painted tile, state is likely dead end
            else:
                # Tile is not painted, or is clear. It needs painting.
                unsat_goals[goal_tile] = goal_color

        if wrongly_painted:
            return float('inf') # State is likely a dead end

        # 3. If goal is reached
        if not unsat_goals:
            return 0

        # 4. Get robots and handle case with no robots but unsatisfied goals
        # Robots are collected during state parsing
        if not robots:
             return float('inf')

        # 5. Check reachability for all unsatisfied goal tiles
        for tile_X, color_Y in unsat_goals.items():
            reachable_by_any_robot = False
            adjacent_tiles = self.get_adjacent_tiles(tile_X)

            if not adjacent_tiles:
                 # Cannot paint if no adjacent tiles. This goal is impossible.
                 return float('inf')

            for robot in robots:
                R_loc = robot_locs.get(robot)
                if R_loc is None:
                    # Robot location unknown or robot not on grid? Assume impossible for this robot.
                    continue

                for tile_Adj in adjacent_tiles:
                     # Check if distance is finite
                     if self.dist(R_loc, tile_Adj) != float('inf'):
                         reachable_by_any_robot = True
                         break # Found one robot that can reach adjacent
                if reachable_by_any_robot:
                     break # This tile is reachable by at least one robot

            if not reachable_by_any_robot:
                 # No robot can reach an adjacent tile for this unsat goal tile.
                 return float('inf')

        # If we reach here, all unsat goal tiles are reachable by at least one robot.

        # 6. Calculate heuristic components
        h = 0

        # Component 1: Minimum paint actions
        h += len(unsat_goals)

        # Component 2: Minimum movement cost for unsatisfied tiles
        total_min_moves_for_tiles = 0
        for tile_X, color_Y in unsat_goals.items():
            min_move_cost_for_tile_X = float('inf') # Min dist for *any* robot to adjacent
            adjacent_tiles = self.get_adjacent_tiles(tile_X) # Already checked not empty

            for robot in robots:
                R_loc = robot_locs.get(robot)
                if R_loc is None: continue # Should be handled by reachability check, but defensive

                min_move_cost_to_adjacent = float('inf')
                for tile_Adj in adjacent_tiles:
                     move_cost = self.dist(R_loc, tile_Adj)
                     min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, move_cost)

                min_move_cost_for_tile_X = min(min_move_cost_for_tile_X, min_move_cost_to_adjacent)

            # min_move_cost_for_tile_X should be finite here due to reachability check
            total_min_moves_for_tiles += min_move_cost_for_tile_X

        h += total_min_moves_for_tiles

        # Component 3: Minimum color acquisition cost
        # Count colors needed for unsat goals that no robot currently has.
        # Each such color needs at least one change_color action globally.
        needed_colors = set(unsat_goals.values())
        current_colors = set(robot_colors.values())
        colors_to_acquire_globally = needed_colors - current_colors

        h += len(colors_to_acquire_globally)

        return h

