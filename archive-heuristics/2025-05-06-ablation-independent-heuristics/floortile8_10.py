from collections import deque
import math

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum costs
        for each unpainted goal tile. The minimum cost for a goal tile is
        calculated over all robots, considering the cost to change color
        and the shortest path movement cost through clear tiles to a
        location from which the tile can be painted, plus the paint action cost.

    Assumptions:
        - The grid structure is defined by static up/down/left/right predicates.
        - Goal colors are available (available-color is static).
        - Tiles are named in a consistent format (e.g., tile_row_col), although
          the heuristic primarily relies on the adjacency graph from static facts.
        - A tile painted with the wrong color represents a dead end.

    Heuristic Initialization:
        - Parses goal facts to store the required color for each goal tile.
        - Parses static up/down/left/right facts to build:
            - An adjacency map representing the grid connectivity for movement.
            - A map from goal tiles to the set of tiles from which they can be painted.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Identify unpainted goal tiles and check for dead ends (goal tile painted with wrong color or unpaintable). If a dead end is detected, return infinity.
        3. Identify the current location and color of each robot.
        4. Identify the set of tiles that are currently clear.
        5. For each robot, perform a Breadth-First Search (BFS) starting from the robot's current location. The BFS explores the grid graph considering only paths that move *to* clear tiles. This computes the shortest distance from the robot's location to every reachable clear tile.
        6. Initialize the total heuristic value to 0.
        7. For each unpainted goal tile:
            a. Determine the required color.
            b. Find the set of potential painting locations (tiles adjacent to the goal tile in the appropriate direction, precomputed in initialization).
            c. Calculate the minimum cost for any robot to paint this specific goal tile:
                i. For each robot:
                    - Calculate the color change cost (0 if robot has the required color, 1 otherwise).
                    - Calculate the minimum movement cost: Find the minimum distance from the robot's current location (using the BFS results for this robot) to any of the potential painting locations *that are currently clear*. If no clear painting location is reachable, the movement cost is infinity.
                    - The total cost for this robot for this goal tile is color_cost + movement_cost + 1 (for the paint action).
                ii. The minimum cost for the goal tile is the minimum of the total costs across all robots.
            d. If the minimum cost for the goal tile is infinity, the goal is unreachable by any robot to a suitable clear paint location; return infinity.
            e. Add the minimum cost for the goal tile to the total heuristic value.
        8. Return the total heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static

        # Parse goal facts to get required colors
        self.goal_tiles_info = {} # tile -> required_color
        for goal_fact in self.goals:
            if goal_fact.startswith('(painted '):
                parts = goal_fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_tiles_info[tile] = color

        # Build adjacency map and paint locations from static facts
        self.adj_map = {} # tile -> list of adjacent tiles for movement
        self.paint_locations = {} # goal_tile -> set of tiles from which it can be painted

        for static_fact in self.static_facts:
            parts = static_fact[1:-1].split()
            predicate = parts[0]
            if predicate in ['up', 'down', 'left', 'right']:
                tile_a = parts[1]
                tile_b = parts[2] # tile_b is adjacent to tile_a

                # Build adjacency map (bidirectional for movement)
                self.adj_map.setdefault(tile_a, []).append(tile_b)
                self.adj_map.setdefault(tile_b, []).append(tile_a)

                # Build paint locations map
                # (dir tile_A tile_B) means tile_A is DIR from tile_B.
                # Robot at tile_B can paint tile_A using paint_DIR action.
                # If tile_A is a goal tile, tile_B is a paint location for it.
                if tile_a in self.goal_tiles_info:
                     self.paint_locations.setdefault(tile_a, set()).add(tile_b)

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify unpainted goal tiles and check for dead ends
        unpainted_goal_tiles = set()
        current_painted_status = {} # tile -> color or 'clear'

        # Populate current_painted_status for all tiles that are painted or clear
        for fact in state:
            if fact.startswith('(painted '):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                current_painted_status[tile] = color
            elif fact.startswith('(clear '):
                 parts = fact[1:-1].split()
                 tile = parts[1]
                 current_painted_status[tile] = 'clear' # Use 'clear' sentinel

        for goal_tile, required_color in self.goal_tiles_info.items():
            current_status = current_painted_status.get(goal_tile)

            if current_status == required_color:
                # Already painted correctly
                pass
            elif current_status is not None and current_status != 'clear' and current_status != required_color:
                 # Painted with the wrong color - dead end
                 return math.inf
            elif current_status == 'clear':
                # Needs painting and is clear (paintable)
                unpainted_goal_tiles.add(goal_tile)
            # else: current_status is None (not painted, not clear)
            # If a goal tile is not painted correctly and not clear, it cannot be painted.
            # This implies a dead end for this goal tile.
            elif goal_tile in self.goal_tiles_info: # Check if it's actually a goal tile we care about
                 # It's a goal tile that needs painting, but it's not clear. Dead end.
                 return math.inf


        if not unpainted_goal_tiles:
             # This case should be covered by the initial goal check, but good safety
             return 0

        # 3. Identify robot info
        robot_info = {} # robot_name -> {'loc': tile_name, 'color': color_name}
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact[1:-1].split()
                robot = parts[1]
                loc = parts[2]
                robot_info.setdefault(robot, {})['loc'] = loc
            elif fact.startswith('(robot-has '):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_info.setdefault(robot, {})['color'] = color

        # 4. Create set of clear tiles
        clear_tiles = {fact[1:-1].split()[1] for fact in state if fact.startswith('(clear ')}

        # 5. Perform BFS from each robot location
        robot_distances = {} # robot_name -> {tile: distance}
        # Collect all tiles potentially in the grid graph
        all_grid_tiles = set(self.adj_map.keys())
        for neighbors in self.adj_map.values():
             all_grid_tiles.update(neighbors)
        # Add any goal tiles that might not be connected in the static graph (unlikely but safe)
        all_grid_tiles.update(self.goal_tiles_info.keys())


        for robot_name, info in robot_info.items():
            start_loc = info['loc']
            distances = {tile: math.inf for tile in all_grid_tiles}

            # Only start BFS if the robot's location is a known tile in the grid
            if start_loc in distances:
                distances[start_loc] = 0
                queue = deque([start_loc])

                while queue:
                    curr_tile = queue.popleft()
                    curr_dist = distances[curr_tile]

                    # Explore neighbors
                    for neighbor_tile in self.adj_map.get(curr_tile, []):
                        # Can only move *to* a clear tile
                        if neighbor_tile in clear_tiles and distances[neighbor_tile] == math.inf:
                            distances[neighbor_tile] = curr_dist + 1
                            queue.append(neighbor_tile)
            robot_distances[robot_name] = distances # Store distances for this robot

        # 6. Initialize total heuristic value
        h = 0

        # 7. Calculate cost for each unpainted goal tile
        for goal_tile in unpainted_goal_tiles:
            required_color = self.goal_tiles_info[goal_tile]
            min_cost_for_goal_tile = math.inf

            # Find min cost over all robots
            for robot_name, info in robot_info.items():
                # robot_loc = info['loc'] # Not needed directly here, use distances
                robot_color = info['color']

                # Cost to get color
                color_cost = 0 if robot_color == required_color else 1

                # Cost to move to a paint location
                min_move_cost_for_robot = math.inf
                potential_paint_locs = self.paint_locations.get(goal_tile, set())

                for paint_loc in potential_paint_locs:
                    # Robot must move to a paint_loc that is clear
                    if paint_loc in clear_tiles:
                        # Get distance from the BFS result for this robot
                        dist = robot_distances[robot_name].get(paint_loc, math.inf)
                        min_move_cost_for_robot = min(min_move_cost_for_robot, dist)

                # Total cost for this robot for this goal tile
                if min_move_cost_for_robot != math.inf:
                    cost_for_robot = color_cost + min_move_cost_for_robot + 1 # +1 for paint action
                    min_cost_for_goal_tile = min(min_cost_for_goal_tile, cost_for_robot)

            # If no robot can reach a clear paint location for this goal tile
            if min_cost_for_goal_tile == math.inf:
                # This goal tile is unreachable by any robot to a suitable clear paint location.
                # This state is likely a dead end.
                return math.inf

            h += min_cost_for_goal_tile

        # 8. Return total heuristic value
        return h
