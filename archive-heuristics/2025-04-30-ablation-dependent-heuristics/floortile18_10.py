import re
from collections import deque

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing three components:
    1. The number of tiles that need to be painted according to the goal but are not yet painted correctly.
    2. The number of distinct colors required by the unpainted goal tiles that are not currently held by any robot.
    3. The sum, over all unpainted goal tiles, of the minimum distance from any robot to any tile adjacent to that goal tile.

    Assumptions:
    - The input task represents a solvable problem instance.
    - Tile names follow the format 'tile_row_col'.
    - The adjacency facts (up, down, left, right) define a connected graph among tiles relevant to the problem.
    - Tiles that are goal tiles but not yet painted correctly are assumed to be clear (as repainting is not possible).

    Heuristic Initialization:
    During initialization, the heuristic identifies all tile objects by their naming convention. It then precomputes the graph structure of the tiles based on the static adjacency facts (up, down, left, right). It computes the shortest path distance between every pair of tiles using Breadth-First Search (BFS). This distance information is stored for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal facts related to painting tiles, e.g., `(painted tile_X color_Y)`.
    2. In the current state, determine which of these goal painting facts are not satisfied. Collect the set of tiles that need to be painted (`unpainted_goal_tiles_info`) and the set of colors required for these tiles (`needed_colors`).
    3. Identify the current location of each robot and the color each robot is holding. Collect the set of colors currently held by robots (`held_colors`).
    4. Calculate the first component of the heuristic: The number of unpainted goal tiles, `len(unpainted_goal_tiles_info)`. This represents a lower bound on the number of paint actions required.
    5. Calculate the second component: The number of colors in `needed_colors` that are not in `held_colors`. `len(needed_colors - held_colors)`. This represents a lower bound on the number of `change_color` actions required across all robots.
    6. Calculate the third component: For each tile `T` in `unpainted_goal_tiles_info`:
       a. Find all tiles `Adj_T` that are adjacent to `T` based on the precomputed graph.
       b. For each robot `R` at location `Loc_R`, find the minimum distance from `Loc_R` to any tile in `Adj_T` using the precomputed distances.
       c. Find the minimum of these distances over all robots `R`. This is the minimum distance any robot needs to travel to get adjacent to tile `T`.
       d. Sum these minimum distances over all tiles `T` in `unpainted_goal_tiles_info`. This sum represents an estimate of the total movement cost required.
    7. The total heuristic value is the sum of the three components calculated in steps 4, 5, and 6.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all objects from initial state, goals, and static facts
        all_objects = set()
        # Iterate through initial state, goals, and static facts
        for fact_set in [task.initial_state, task.goals, task.static]:
            for fact in fact_set:
                # Split fact string into parts, ignoring predicate and brackets
                parts = fact.strip('()').split()
                # Objects are arguments after the predicate
                for part in parts[1:]:
                    # Remove type information if present (like 'robot1 - robot')
                    obj_name = part.split(' - ')[0]
                    all_objects.add(obj_name)

        # Identify tiles using naming convention
        self.all_tiles = {obj for obj in all_objects if re.match(r'tile_\d+_\d+', obj)}

        # Build adjacency list from static facts
        self.adj_list = {tile: [] for tile in self.all_tiles}

        for fact in self.static:
            if fact.startswith('(up '):
                parts = fact.strip('()').split()
                y, x = parts[1], parts[2]
                if x in self.adj_list and y in self.adj_list: # Ensure both are tiles we know
                    self.adj_list[x].append(y)
                    self.adj_list[y].append(x)
            elif fact.startswith('(down '):
                parts = fact.strip('()').split()
                y, x = parts[1], parts[2]
                if x in self.adj_list and y in self.adj_list:
                    self.adj_list[x].append(y)
                    self.adj_list[y].append(x)
            elif fact.startswith('(left '):
                parts = fact.strip('()').split()
                y, x = parts[1], parts[2]
                if x in self.adj_list and y in self.adj_list:
                    self.adj_list[x].append(y)
                    self.adj_list[y].append(x)
            elif fact.startswith('(right '):
                parts = fact.strip('()').split()
                y, x = parts[1], parts[2]
                if x in self.adj_list and y in self.adj_list:
                    self.adj_list[x].append(y)
                    self.adj_list[y].append(x)

        # Remove duplicates from adjacency lists
        for tile in self.adj_list:
             self.adj_list[tile] = list(set(self.adj_list[tile]))

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all other nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # Should not happen if start_node comes from self.all_tiles
             return distances # Return inf for all if start is invalid

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in adj_list before iterating (should be if in all_tiles)
            if current_node in self.adj_list:
                for neighbor in self.adj_list[current_node]:
                    if neighbor in distances and distances[neighbor] == float('inf'): # Ensure neighbor is a known tile
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def get_adjacent_tiles(self, tile):
         """Helper to get adjacent tiles from the precomputed list."""
         return self.adj_list.get(tile, [])


    def get_distance(self, tile1, tile2):
         """Helper to get distance from precomputed table."""
         # Return infinity if tiles are not in the graph or unreachable
         return self.distances.get(tile1, {}).get(tile2, float('inf'))


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of facts representing the current state.
        @return: An integer heuristic value. Returns a large integer (1000000)
                 if the state is likely unsolvable (e.g., goal tile unreachable).
        """
        unpainted_goal_tiles_info = set() # Stores (tile, color) for unpainted goal tiles
        needed_colors = set()
        held_colors = set()
        robot_locations = {} # Stores {robot: tile}

        # Parse state facts
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact.strip('()').split()
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif fact.startswith('(robot-has '):
                parts = fact.strip('()').split()
                robot, color = parts[1], parts[2]
                held_colors.add(color)

        # Identify unpainted goal tiles and needed colors
        goal_painted_facts = set()
        for goal_fact in self.goals:
             if goal_fact.startswith('(painted '):
                  goal_painted_facts.add(goal_fact)

        for goal_fact in goal_painted_facts:
             if goal_fact not in state:
                  # This goal fact is not satisfied
                  parts = goal_fact.strip('()').split()
                  tile, color = parts[1], parts[2]
                  unpainted_goal_tiles_info.add((tile, color))
                  needed_colors.add(color)

        # If all painted goals are satisfied, heuristic is 0
        if not unpainted_goal_tiles_info:
            return 0

        # Heuristic Component 1: Number of unpainted goal tiles
        h = len(unpainted_goal_tiles_info)

        # Heuristic Component 2: Number of needed colors not held by any robot
        colors_to_acquire = needed_colors - held_colors
        h += len(colors_to_acquire)

        # Heuristic Component 3: Movement cost
        total_movement_cost = 0
        large_unreachable_cost = 1000000 # Use a large integer for unreachable states

        for tile, color in unpainted_goal_tiles_info:
            adjacent_tiles = self.get_adjacent_tiles(tile)
            if not adjacent_tiles:
                # If a goal tile has no adjacent tiles, it cannot be painted.
                # This implies unsolvability. Return a large value.
                return large_unreachable_cost

            min_dist_to_adj_for_tile = float('inf')

            for robot, robot_loc in robot_locations.items():
                min_dist_for_robot = float('inf')
                # Ensure robot_loc is a valid tile in our graph
                if robot_loc in self.all_tiles:
                    for adj_tile in adjacent_tiles:
                        dist = self.get_distance(robot_loc, adj_tile)
                        min_dist_for_robot = min(min_dist_for_robot, dist)

                min_dist_to_adj_for_tile = min(min_dist_to_adj_for_tile, min_dist_for_robot)

            if min_dist_to_adj_for_tile == float('inf'):
                 # No robot can reach any adjacent tile for this goal tile.
                 # Problem likely unsolvable or this tile is unreachable.
                 # Return a large value.
                 return large_unreachable_cost
            else:
                total_movement_cost += min_dist_to_adj_for_tile

        # Check if total_movement_cost is still infinity (e.g., if unpainted_goal_tiles_info was not empty
        # but all tiles were unreachable - though the loop above should catch this).
        # This check is mostly redundant due to the check inside the loop, but kept for safety.
        if total_movement_cost == float('inf'):
             return large_unreachable_cost


        h += total_movement_cost

        return h
