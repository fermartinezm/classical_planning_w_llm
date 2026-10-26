import re
from collections import deque, defaultdict

from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator are available in the environment
# from task import Operator, Task

class floortileHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the floortile domain. It estimates the cost
    to reach the goal state by summing up several components:
    1. The number of tiles that need to be painted.
    2. The number of colors required for unpainted goal tiles that no robot
       currently possesses.
    3. The number of unpainted goal tiles that are currently occupied by a robot
       (requiring the robot to move off).
    4. The sum, over all unpainted goal tiles, of the minimum distance from any
       robot to any tile adjacent to the goal tile (a valid paint position).

    Assumptions:
    - The problem instance is solvable (unless detected otherwise by the heuristic).
    - Tile names follow the format 'tile_r_c' where r and c are integers representing
      row and column indices.
    - The grid connectivity is defined by 'up', 'down', 'left', 'right' predicates
      and forms a connected graph (or disconnected components).
    - Robots can change color if the target color is available.
    - Robots always have some color initially (no 'free-color' state needs explicit handling).
    - Tiles painted with the wrong color cannot be repainted or cleared (unsolvable state).

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes static information from the task:
    - Builds an adjacency list representation of the tile grid graph based on
      'up', 'down', 'left', 'right' predicates.
    - Parses tile names to extract row and column coordinates, storing a mapping.
    - Stores the set of goal facts.
    - Stores the set of available colors.
    - Checks if all colors required by the goal are available; if not, marks the
      problem as unsolvable.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the problem was determined to be unsolvable during initialization
       (e.g., required colors are not available). If so, return infinity.
    2. Identify the set of goal facts '(painted tile color)' that are not true
       in the current state. If this set is empty, the goal is reached, return 0.
    3. Initialize the heuristic value `h` to 0.
    4. Add the number of unpainted goal tiles to `h`. This accounts for the minimum
       number of paint actions required.
    5. Identify the set of colors needed for the unpainted goal tiles.
    6. Identify the set of colors currently held by robots.
    7. Add the number of colors needed for unpainted goal tiles that are not
       currently held by any robot to `h`. This accounts for the minimum number
       of color change actions required across all robots.
    8. Identify tiles currently occupied by robots.
    9. Identify unpainted goal tiles that are currently occupied by a robot. Add
       the count of these tiles to `h`. This accounts for the minimum number of
       move actions required to clear these tiles so they can be painted.
    10. Calculate the movement cost component:
        a. For each robot, perform a Breadth-First Search (BFS) starting from its
           current location to find the shortest distance to all other reachable tiles.
           Store these distances.
        b. Initialize a total minimum distance `total_min_dist_to_paint_pos_for_all_tiles` to 0.
        c. For each unpainted goal tile `t` requiring color `c`:
           i. Find the set of tiles adjacent to `t` (these are the valid paint positions for `t`).
           ii. If `t` has no adjacent tiles in the graph, the problem is unsolvable; return infinity.
           iii. Find the minimum distance from *any* robot's current location to *any* of the paint positions for `t`. This requires checking the distances computed in step 10a for each robot and each paint position.
           iv. If no robot can reach any paint position for `t` (minimum distance is infinity), the problem is unsolvable; return infinity.
           v. Add this minimum distance (from any robot to a paint position for `t`) to `total_min_dist_to_paint_pos_for_all_tiles`.
        d. Add `total_min_dist_to_paint_pos_for_all_tiles` to `h`.
    11. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.adj = defaultdict(list)
        self.tile_coords = {} # Not strictly used for distance, but good for debugging/structure
        self.available_colors = set()
        self.unsolvable = False

        # Parse static facts
        for fact_str in self.static:
            # Example: '(up tile_1_1 tile_0_1)'
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty lines or malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate in ['up', 'down', 'left', 'right']:
                # These define the grid connectivity
                if len(args) == 2:
                    tile1, tile2 = args
                    # Add symmetric edges
                    self.adj[tile2].append(tile1) # tile1 is adjacent to tile2 (e.g., tile_1_1 is up from tile_0_1, so tile_1_1 is adjacent to tile_0_1)
                    self.adj[tile1].append(tile2) # Connectivity is symmetric

                    # Parse tile coordinates if not already known
                    for tile_name in [tile1, tile2]:
                        if tile_name not in self.tile_coords:
                            match = re.match(r'tile_(\d+)_(\d+)', tile_name)
                            if match:
                                r, c = int(match.group(1)), int(match.group(2))
                                self.tile_coords[tile_name] = (r, c)
                            # else: tile name format unexpected, ignore for now
            elif predicate == 'available-color':
                if len(args) == 1:
                    self.available_colors.add(args[0])
                # else: available-color format unexpected, ignore for now

        # Ensure adjacency list has unique neighbors for robustness
        for tile in self.adj:
             self.adj[tile] = list(set(self.adj[tile]))

        # Check if all goal colors are available
        goal_colors = set()
        for goal_fact in self.goals:
             parts = goal_fact.strip('()').split()
             if parts and parts[0] == 'painted':
                 # Goal is (painted tile color)
                 if len(parts) == 3:
                     goal_colors.add(parts[2])
                 # else: unexpected goal format, ignore for now

        if not goal_colors.issubset(self.available_colors):
             self.unsolvable = True
             # In a real planner, you might log this:
             # import logging
             # logging.warning("Floortile heuristic: Goal requires unavailable colors. Problem is unsolvable.")


    def _bfs(self, start_node, graph):
        """Performs BFS from start_node on the graph."""
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph adjacency list
            # It should exist if it's a valid tile name from static facts or state
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        if self.unsolvable:
            return float('inf')

        state = node.state

        # Extract dynamic state information
        robot_locs = {} # robot_name -> tile_name
        robot_colors = {} # robot_name -> color
        tiles_occupied_by_robots = set() # tile_name
        painted_in_state = set() # (painted tile color) facts

        for fact_str in state:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty lines or malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'robot-at' and len(args) == 2:
                robot_name, tile_name = args
                robot_locs[robot_name] = tile_name
                tiles_occupied_by_robots.add(tile_name)
            elif predicate == 'robot-has' and len(args) == 2:
                 robot_name, color = args
                 robot_colors[robot_name] = color
            elif predicate == 'painted' and len(args) == 2:
                 painted_in_state.add(fact_str) # Store the full fact string

        # 1. Identify unsatisfied goal tiles
        unsat_goals = set() # set of (tile_name, color) tuples
        for goal_fact in self.goals:
            if goal_fact not in painted_in_state:
                 # Extract tile and color from goal fact string
                 parts = goal_fact.strip('()').split()
                 if parts and parts[0] == 'painted' and len(parts) == 3:
                     tile_name, color = parts[1], parts[2]
                     unsat_goals.add((tile_name, color))
                 # else: unexpected goal format, ignore or handle

        # If goal is reached
        if not unsat_goals:
            return 0

        h = 0

        # Component 1: Paint actions
        h += len(unsat_goals)

        # Component 2: Color changes
        colors_needed_for_unsat = {c for (t, c) in unsat_goals}
        colors_robots_have = set(robot_colors.values())
        colors_to_acquire = colors_needed_for_unsat - colors_robots_have
        h += len(colors_to_acquire)

        # Component 3: Clearing occupied tiles
        unpainted_goal_tiles_occupied = {t for (t, c) in unsat_goals if t in tiles_occupied_by_robots}
        h += len(unpainted_goal_tiles_occupied)

        # Component 4: Movement cost
        dist_from_robot = {} # robot_name -> {tile_name -> distance}
        for robot_name, start_tile in robot_locs.items():
             # Only run BFS if the start tile exists in our graph (should always be the case for valid states)
             if start_tile in self.adj:
                dist_from_robot[robot_name] = self._bfs(start_tile, self.adj)
             else:
                 # Robot is at a tile not in the grid graph? Unlikely in valid problems.
                 # Treat as unreachable from this robot.
                 dist_from_robot[robot_name] = {}


        total_min_dist_to_paint_pos_for_all_tiles = 0
        for (t_name, c) in unsat_goals:
            min_dist_any_robot_to_paint_pos_for_t = float('inf')
            paint_positions_for_t = self.adj.get(t_name, []) # Tiles adjacent to t_name

            if not paint_positions_for_t:
                # Tile has no adjacent tiles, cannot be painted
                # import logging
                # logging.warning(f"Floortile heuristic: Goal tile {t_name} has no adjacent tiles. Problem likely unsolvable.")
                return float('inf')

            for paint_pos_name in paint_positions_for_t:
                for robot_name in robot_locs:
                    # Check if the robot's distances were computed (i.e., robot was at a valid tile)
                    if robot_name in dist_from_robot and paint_pos_name in dist_from_robot[robot_name]:
                        dist_r_to_paint_pos = dist_from_robot[robot_name][paint_pos_name]
                        min_dist_any_robot_to_paint_pos_for_t = min(min_dist_any_robot_to_paint_pos_for_t, dist_r_to_paint_pos)

            if min_dist_any_robot_to_paint_pos_for_t == float('inf'):
                # No robot can reach any paint position for this tile
                # import logging
                # logging.warning(f"Floortile heuristic: Goal tile {t_name} unreachable by any robot. Problem likely unsolvable.")
                return float('inf')

            total_min_dist_to_paint_pos_for_all_tiles += min_dist_any_robot_to_paint_pos_for_t

        h += total_min_dist_to_paint_pos_for_all_tiles

        return h
