from heuristics.heuristic_base import Heuristic
from collections import deque
import math

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing, for each clear goal tile,
    the minimum cost for any robot to reach an adjacent tile with the required
    color and paint it. The cost includes movement (estimated by BFS on
    traversable tiles), color change (if needed), and the paint action.
    It returns infinity if any goal tile is painted with the wrong color or
    if a clear goal tile is unreachable by any robot.

    Assumptions:
    - Tile names are in the format 'tile_R_C' where R and C are integers,
      or other formats as long as adjacency predicates define the grid structure.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - The problem instances are solvable unless a goal tile is painted with the wrong color
      or becomes unreachable.
    - Movement is restricted to 'clear' tiles. A robot's current tile is not 'clear'
      but is the starting point for BFS.

    Heuristic Initialization:
    - Parses static facts ('up', 'down', 'left', 'right', 'available-color').
    - Builds a static tile graph (adjacency list) based on adjacency predicates.
    - Stores goal facts and available colors.
    - Collects all unique tile names mentioned in static facts and goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state facts.
    2. Initialize heuristic value `h = 0`.
    3. Identify unsatisfied goal facts and check for dead ends:
       - Create a dictionary `painted_tiles` mapping tiles to their current color from the state.
       - Create a dictionary `clear_goal_tiles_to_paint` mapping clear goal tiles to their required color.
       - Iterate through `self.goals`. For each goal fact `(painted tile_Y color_Y)`:
         - If `tile_Y` is in `painted_tiles`:
           - If `painted_tiles[tile_Y] != color_Y`, return `math.inf` (dead end).
           - Otherwise, the goal for this tile is satisfied.
         - If `tile_Y` is not in `painted_tiles` (implies it's clear):
           - Add `tile_Y` to `clear_goal_tiles_to_paint` with its required `color_Y`.
    4. If `clear_goal_tiles_to_paint` is empty, return `h = 0` (all goal tiles are painted correctly).
    5. Identify robot locations and colors: Create a dictionary `robot_info` mapping robot names to a tuple `(location, color)`.
    6. Compute BFS distances from each robot location:
       - Define a helper function `_bfs(start_tile, state)` that performs BFS starting from `start_tile`, only traversing through tiles that are `clear` in the current `state`. It returns a dictionary of distances.
       - Create a dictionary `robot_distances` mapping robot names to the distance dictionary returned by `_bfs` for that robot's location.
    7. Calculate the total heuristic based on clear goal tiles:
       - For each `goal_tile` and its `required_color` in `clear_goal_tiles_to_paint`:
         - Initialize `min_cost_for_tile = math.inf`.
         - Get the neighbors of `goal_tile` from the static `self.tile_graph`.
         - For each robot `r` and its `(loc, current_color)` in `robot_info`:
           - Get the distance dictionary `distances_from_robot` for robot `r`.
           - Initialize `min_dist_to_neighbor = math.inf`.
           - For each `neighbor_tile` of `goal_tile`:
             - If `neighbor_tile` is reachable from `loc` (i.e., `distances_from_robot.get(neighbor_tile, math.inf)` is not `math.inf`), update `min_dist_to_neighbor`.
           - If `min_dist_to_neighbor` is finite:
             - Calculate the cost for robot `r` to paint `goal_tile`: `min_dist_to_neighbor` (moves) + `(1 if current_color != required_color else 0)` (color change) + `1` (paint action).
             - Update `min_cost_for_tile = min(min_cost_for_tile, cost_to_paint)`.
         - If `min_cost_for_tile` is still `math.inf`, return `math.inf` (clear goal tile is unreachable).
         - Add `min_cost_for_tile` to `h`.
    9. Return `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # Build static tile graph
        self.tile_graph = {} # Adjacency list: tile -> list of adjacent tiles
        self.all_tiles = set()
        self.available_colors = set()

        for fact in self.static_facts:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate in ['up', 'down', 'left', 'right']:
                tile1 = parts[1]
                tile2 = parts[2]
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)
                if tile1 not in self.tile_graph:
                    self.tile_graph[tile1] = []
                if tile2 not in self.tile_graph:
                    self.tile_graph[tile2] = []
                # Add bidirectional edges
                self.tile_graph[tile1].append(tile2)
                self.tile_graph[tile2].append(tile1)
            elif predicate == 'available-color':
                self.available_colors.add(parts[1])

        # Ensure all tiles from goals are in the graph even if they have no static neighbors listed
        # (though this shouldn't happen in valid PDDL)
        for goal in self.goals:
             parts = goal.strip('()').split()
             if parts[0] == 'painted':
                 tile = parts[1]
                 self.all_tiles.add(tile)
                 if tile not in self.tile_graph:
                     self.tile_graph[tile] = []


    def _bfs(self, start_tile, state):
        """
        Performs BFS from start_tile, only traversing through 'clear' tiles.
        Returns a dictionary of distances from start_tile to reachable tiles.
        """
        distances = {tile: math.inf for tile in self.all_tiles}
        q = deque([(start_tile, 0)])
        distances[start_tile] = 0
        visited = {start_tile}

        # Helper to check if a tile is clear in the current state
        is_clear = lambda tile: f'(clear {tile})' in state

        while q:
            current_tile, dist = q.popleft()

            if current_tile not in self.tile_graph:
                 continue # Should not happen

            for neighbor in self.tile_graph.get(current_tile, []): # Use .get for safety
                # Can move to neighbor if it is clear AND not visited
                if is_clear(neighbor) and neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    q.append((neighbor, dist + 1))

        return distances


    def __call__(self, node):
        state = node.state
        h_value = 0

        # 1. Identify clear goal tiles and check for dead ends
        clear_goal_tiles_to_paint = {} # tile -> required_color
        painted_tiles = {} # tile -> color

        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'painted':
                painted_tiles[parts[1]] = parts[2]

        for goal in self.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'painted':
                goal_tile = parts[1]
                goal_color = parts[2]

                if goal_tile in painted_tiles:
                    # Tile is painted
                    current_color = painted_tiles[goal_tile]
                    if current_color != goal_color:
                        # Painted with wrong color - dead end
                        return math.inf
                    # Else: Painted with correct color - goal satisfied for this tile
                else:
                    # Tile is not painted - must be clear (assuming domain consistency)
                    # This tile needs to be painted
                    clear_goal_tiles_to_paint[goal_tile] = goal_color

        # If all goal tiles are already painted correctly, heuristic is 0
        if not clear_goal_tiles_to_paint:
            return 0

        # 2. Identify robot locations and colors
        robot_info = {} # robot -> (location, color)
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                location = parts[2]
                if robot not in robot_info:
                    robot_info[robot] = [None, None]
                robot_info[robot][0] = location
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot not in robot_info:
                    robot_info[robot] = [None, None]
                robot_info[robot][1] = color

        # 3. Compute BFS distances from each robot location considering 'clear' tiles
        robot_distances = {} # robot -> {tile: distance}
        for robot, (loc, color) in robot_info.items():
             if loc: # Ensure robot location is known
                robot_distances[robot] = self._bfs(loc, state) # Pass state to BFS


        # 4. Calculate heuristic based on clear goal tiles
        for goal_tile, required_color in clear_goal_tiles_to_paint.items():
            min_cost_for_tile = math.inf

            # Find neighbors of the goal tile from the static graph
            goal_tile_neighbors = self.tile_graph.get(goal_tile, [])

            # Cost for this tile is the minimum over all robots
            for robot, (loc, current_color) in robot_info.items():
                if loc is None: continue # Should not happen in valid state

                distances_from_robot = robot_distances.get(robot)
                if distances_from_robot is None: continue # Should not happen

                min_dist_to_neighbor = math.inf

                # Find minimum distance from robot's location to any neighbor of the goal tile
                for neighbor_tile in goal_tile_neighbors:
                    # Check if the neighbor is reachable from the robot's location
                    # using the BFS distances computed on clear tiles.
                    # The BFS finds distance to 'clear' tiles. A robot paints from an adjacent tile.
                    # The adjacent tile must be reachable (i.e., distance is finite).
                    dist_to_neighbor = distances_from_robot.get(neighbor_tile, math.inf)
                    if dist_to_neighbor != math.inf:
                         min_dist_to_neighbor = min(min_dist_to_neighbor, dist_to_neighbor)


                # If a path exists to at least one neighbor
                if min_dist_to_neighbor != math.inf:
                    # Cost = moves + color_change + paint
                    # Moves = min_dist_to_neighbor
                    # Color change = 1 if robot doesn't have the required color, else 0
                    # Paint action = 1
                    cost_to_paint = min_dist_to_neighbor + (1 if current_color != required_color else 0) + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_to_paint)

            # If no robot can reach a clear neighbor of this goal tile, it's a dead end
            if min_cost_for_tile == math.inf:
                 return math.inf

            h_value += min_cost_for_tile

        return h_value
