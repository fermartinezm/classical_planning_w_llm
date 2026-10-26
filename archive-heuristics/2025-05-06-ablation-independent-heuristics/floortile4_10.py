from heuristics.heuristic_base import Heuristic
from task import Task
import math
import re
from collections import deque

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing, for each unpainted
        goal tile, the minimum estimated cost for any robot to paint that tile.
        The estimated cost for a robot to paint a tile includes:
        1. Minimum movement cost from the robot's current location to any tile
           adjacent to the target tile.
        2. Cost to change the robot's color if it doesn't have the required color (1 action).
        3. Cost of the paint action (1 action).
        If a goal tile is painted with the wrong color, the heuristic returns infinity.

    Assumptions:
        - Tile names follow the format 'tile_R_C' where R and C are integers.
        - Movement between adjacent tiles (up, down, left, right) is bidirectional.
        - All colors required by the goal are available (available-color predicate is true).
        - Robots initially have some color (robot-has predicate is true).
        - If a goal tile is painted with a color different from the goal color,
          it is a dead end (no unpaint action).
        - Tiles not specified in the goal as painted can be ignored regarding
          their painted/clear status for goal satisfaction. However, painting
          a non-goal tile might block movement, which this heuristic doesn't
          explicitly check for.

    Heuristic Initialization:
        - Parses static facts to build a graph representing the tile grid
          based on adjacency predicates (up, down, left, right).
        - Extracts tile names and maps them to (row, col) coordinates.
        - Computes all-pairs shortest paths between tiles using BFS.
        - Stores goal facts (painted tiles and required colors).
        - Stores available colors.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize heuristic value `h = 0`.
        2. Parse the current state to find robot locations and their current colors.
        3. Identify the set of goal facts that are not satisfied in the current state. These are the 'pending' goal facts, specifically `(painted T C)` where `T` is not painted with `C`.
        4. If there are no pending goal facts, the state is a goal state, return `h = 0`.
        5. For each pending goal fact `(painted T C)`:
            a. Check if tile `T` is currently painted with a *different* color `C_wrong`. This is done by checking if any fact `(painted T C_wrong)` exists in the state where `C_wrong != C`.
            b. If `T` is painted with a wrong color, the state is considered a dead end. Return `float('inf')`.
            c. If `T` is not painted with the wrong color (it must be clear or painted correctly, but we already filtered for unpainted goal tiles, so it must be clear):
                i. Check if the required color `C` is available. If not, return `float('inf')`.
                ii. Calculate the minimum cost for *any* robot `R` to paint tile `T` with color `C`.
                iii. Initialize `min_robot_cost_for_T = float('inf')`.
                iv. For each robot `R` with current location `L_R` and color `C_R`:
                    - Find the set of tiles adjacent to `T` using the pre-calculated graph.
                    - Calculate the minimum distance from `L_R` to any tile in the adjacent set using the pre-calculated shortest paths. Let this be `min_dist_R_to_adj_T`. If no adjacent tile is reachable (e.g., due to a disconnected graph, though unlikely in this domain), this distance might be infinity.
                    - Calculate the color change cost for robot `R`: `color_cost_R = 1` if `C_R != C`, otherwise `0`. (Assumes required color `C` is available, which was checked in step 5.c.i).
                    - The estimated cost for robot `R` to paint `T` is `cost_R_paint_T = min_dist_R_to_adj_T + color_cost_R + 1` (move cost + color change cost + paint action cost).
                    - Update `min_robot_cost_for_T = min(min_robot_cost_for_T, cost_R_paint_T)`.
                v. If `min_robot_cost_for_T` is still `float('inf')` (e.g., no robot can reach an adjacent tile), the state might be a dead end. Return `float('inf')`.
                vi. Add `min_robot_cost_for_tile` to the total heuristic `h`.
        6. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # Preprocessing: Build tile graph and compute distances
        self.tile_coords = {} # Map tile name to (row, col)
        self.tile_names = [] # List of all tile names
        self.tile_graph = {} # Adjacency list: tile -> [adjacent_tiles]

        # Collect all potential tile names from static facts and goals
        potential_tiles = set()
        for fact_str in self.static_facts:
            parts = self._parse_fact(fact_str)
            if len(parts) >= 2:
                 for arg in parts[1:]:
                     # Check if argument looks like a tile name
                     if arg.startswith('tile_'):
                         potential_tiles.add(arg)
        for goal_fact_str in self.goals:
             parts = self._parse_fact(goal_fact_str)
             if len(parts) >= 2:
                 for arg in parts[1:]:
                     if arg.startswith('tile_'):
                         potential_tiles.add(arg)


        # Initialize graph and coords for all potential tiles
        for tile_name in potential_tiles:
             if tile_name not in self.tile_coords:
                 try:
                     # Assuming tile names are like 'tile_R_C'
                     parts_name = tile_name.split('_')
                     row = int(parts_name[1])
                     col = int(parts_name[2])
                     self.tile_coords[tile_name] = (row, col)
                     self.tile_names.append(tile_name)
                     self.tile_graph[tile_name] = [] # Initialize adjacency list
                 except (IndexError, ValueError):
                      # Ignore objects that look like tiles but aren't formatted correctly
                      pass


        # Add edges to the graph from adjacency facts
        for fact_str in self.static_facts:
             parts = self._parse_fact(fact_str)
             if len(parts) == 3 and parts[0] in {'up', 'down', 'left', 'right'}:
                 tile1, tile2 = parts[1], parts[2]
                 # Ensure both are valid tiles found during initial scan
                 if tile1 in self.tile_graph and tile2 in self.tile_graph:
                     # Add bidirectional edges
                     if tile2 not in self.tile_graph[tile1]:
                         self.tile_graph[tile1].append(tile2)
                     if tile1 not in self.tile_graph[tile2]:
                         self.tile_graph[tile2].append(tile1)

        # Compute all-pairs shortest paths using BFS
        self.distances = {} # distances[tile1][tile2] = distance
        for start_tile in self.tile_names:
            self.distances[start_tile] = self._bfs(start_tile)

        # Store goal tiles and their required colors
        self.goal_tiles_info = {} # {tile_name: color}
        for goal_fact_str in self.goals:
            parts = self._parse_fact(goal_fact_str)
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles_info[tile] = color

        # Store available colors (from static facts)
        self.available_colors = set()
        for fact_str in self.static_facts:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'available-color' and len(parts) == 2:
                self.available_colors.add(parts[1])


    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string into a list of strings."""
        # Remove parentheses and split by spaces
        # Use regex to handle potential multiple spaces or different formats robustly
        return re.findall(r'\S+', fact_str[1:-1])

    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all other nodes."""
        distances = {node: float('inf') for node in self.tile_names}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph before accessing neighbors
            if current_node in self.tile_graph:
                for neighbor in self.tile_graph[current_node]:
                    # Ensure neighbor is a valid tile
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def _get_adjacent_tiles(self, tile_name):
        """Returns a list of tiles adjacent to the given tile."""
        return self.tile_graph.get(tile_name, []) # Return empty list if tile not in graph

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic for the floortile domain.
        """
        state = node.state
        h = 0

        # Get current robot info
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color}}
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {'location': None, 'color': None} # Initialize if first time seeing robot
                robot_info[robot]['location'] = location
            elif parts[0] == 'robot-has' and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot not in robot_info:
                     robot_info[robot] = {'location': None, 'color': None} # Initialize if first time seeing robot
                 robot_info[robot]['color'] = color

        # Check each goal tile
        pending_goal_tiles = []
        for goal_tile, required_color in self.goal_tiles_info.items():
            # Check if the goal fact (painted goal_tile required_color) is true
            goal_satisfied = f'(painted {goal_tile} {required_color})' in state

            if not goal_satisfied:
                # Check if the tile is painted with the wrong color
                is_painted_wrong = False
                for fact_str in state:
                    parts = self._parse_fact(fact_str)
                    if parts[0] == 'painted' and len(parts) == 3:
                        painted_tile, painted_color = parts[1], parts[2]
                        if painted_tile == goal_tile and painted_color != required_color:
                            is_painted_wrong = True
                            break

                if is_painted_wrong:
                    # Dead end state
                    return float('inf')
                else:
                    # Tile is not painted correctly and not painted wrong -> must be clear
                    # Add to pending list to calculate its cost
                    pending_goal_tiles.append((goal_tile, required_color))

        # If all goal tiles are satisfied, h is 0
        if not pending_goal_tiles:
             return 0

        # Calculate cost for each pending goal tile
        for goal_tile, required_color in pending_goal_tiles:
            # Check if the required color is available. If not, cannot paint.
            if required_color not in self.available_colors:
                 # Cannot obtain the required color. Dead end for this tile.
                 return float('inf')

            min_robot_cost_for_tile = float('inf')

            adjacent_tiles = self._get_adjacent_tiles(goal_tile)
            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles? Should not happen in valid grid problems.
                 # Treat as unreachable.
                 return float('inf')

            # Ensure goal_tile is a valid tile in our graph/distances
            if goal_tile not in self.tile_graph:
                 # This goal tile was not found in static adjacency facts or other facts.
                 # Should not happen in valid problems.
                 return float('inf')

            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                # Robot must have a known location that is a valid tile in our graph
                if robot_location is None or robot_location not in self.distances:
                    continue # This robot cannot reach the tile from here

                # Find min distance from robot's location to any adjacent tile
                min_dist_robot_to_adj = float('inf')
                # robot_location is guaranteed to be in self.distances keys here
                for adj_tile in adjacent_tiles:
                    # Ensure adj_tile is a valid tile and reachable from robot_location
                    if adj_tile in self.distances[robot_location]:
                         dist = self.distances[robot_location][adj_tile]
                         min_dist_robot_to_adj = min(min_dist_robot_to_adj, dist)

                # If robot cannot reach any adjacent tile
                if min_dist_robot_to_adj == float('inf'):
                    continue # This robot cannot paint this tile

                # Calculate color change cost
                color_cost = 0
                # Check if robot_color is known and different from required_color
                if robot_color is not None and robot_color != required_color:
                    # Need to change color. We already checked required_color is available.
                    color_cost = 1
                # If robot_color is None, this is unexpected based on domain.
                # Assume it needs a color change if its color isn't the required one.
                # The PDDL implies robot-has is always true for some color.

                # Total estimated cost for this robot to paint this tile
                cost_this_robot = min_dist_robot_to_adj + color_cost + 1 # move + color_change + paint

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_this_robot)

            # If no robot can paint this tile (all costs were inf)
            if min_robot_cost_for_tile == float('inf'):
                 return float('inf') # Dead end

            h += min_robot_cost_for_tile

        return h
