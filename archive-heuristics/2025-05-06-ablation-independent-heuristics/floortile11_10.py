# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Task
import re
import collections # For deque for BFS
import math # For math.inf

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum costs
    for each unsatisfied goal tile. The minimum cost for a single goal tile
    (painted T C) is the minimum over all robots of the cost to get the
    correct color C plus the shortest path distance for the robot to reach
    a clear tile adjacent to T, plus 1 for the paint action. Movement is
    restricted to currently clear tiles. If a tile is painted with the wrong
    color, the heuristic is infinity.

    Assumptions:
    - Tile names are in the format 'tile_X_Y' where X is the row and Y is the column.
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - 'available-color' predicates define all colors that can be obtained.
    - The state representation is a frozenset of strings.
    - The Task object provides initial_state, goals, and static facts.
    - All goal colors are present in 'available-color' facts.

    Heuristic Initialization:
    - Parses static facts to build the tile adjacency graph.
    - Parses goal facts to store the target color for each goal tile.
    - Parses available colors.
    - Parses all facts to identify all tile names in the problem.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value h = 0.
    2. Identify robot locations and current colors from the current state.
    3. Identify the set of clear tiles from the current state.
    4. For each goal fact '(painted T C)' in the task's goals:
        a. Check if '(painted T C)' is already true in the current state. If yes, continue to the next goal fact.
        b. Check if T is painted with any *other* color C' in the current state. If yes, the state is a dead end for this goal, return math.inf.
        c. If the goal is not satisfied and not painted wrong, the tile T must be clear according to the domain definition (a tile is either clear or painted).
        d. Tile T needs color C. Calculate the minimum cost to paint T:
            i. Find all tiles X adjacent to T using the precomputed adjacency graph.
            ii. Filter these adjacent tiles to keep only those that are currently clear. Let this set be Adj_clear(T).
            iii. If Adj_clear(T) is empty, no robot can paint T from an adjacent clear tile. This goal is unreachable. Return math.inf.
            iv. Calculate min_cost_for_tile = math.inf.
            v. For each robot R:
                - Get R's current location L and color C_R.
                - Calculate color_cost = 0 if C_R == C else 1 (cost of change_color action).
                - Calculate movement_cost = shortest path distance from L to any tile in Adj_clear(T) using BFS. The BFS traverses the graph of all tiles but only allows moving *to* tiles in the current clear_tiles_set. The starting tile L does not need to be clear.
                - If movement_cost is finite:
                    - robot_cost = color_cost + movement_cost + 1 (cost of paint action).
                    - min_cost_for_tile = min(min_cost_for_tile, robot_cost).
            vi. If min_cost_for_tile is still math.inf, it means no robot can reach a suitable adjacent clear tile. This goal is unreachable. Return math.inf.
            vii. Add min_cost_for_tile to h.
    5. Return h.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static
        self.all_facts_set = task.facts # Set of all possible fact strings

        # Preprocess static information
        self.tile_neighbors = self._parse_tile_neighbors(self.static_facts)
        self.goal_tiles = self._parse_goal_tiles(self.goals)
        self.available_colors = self._parse_available_colors(self.static_facts)
        self.all_tiles = self._parse_all_tiles(self.all_facts_set) # Get all tile names

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_str[1:-1].split()
        if not parts: # Handle empty string case if necessary
             return None, []
        return parts[0], parts[1:]

    def _parse_tile_neighbors(self, static_facts):
        """Builds adjacency list for tiles based on static facts."""
        neighbors = {}
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred in ['up', 'down', 'left', 'right']:
                # Predicate is (dir tile1 tile2), meaning tile1 is dir of tile2.
                # So tile1 and tile2 are neighbors.
                tile1, tile2 = args
                neighbors.setdefault(tile1, []).append(tile2)
                neighbors.setdefault(tile2, []).append(tile1)
        return neighbors

    def _parse_goal_tiles(self, goals):
        """Extracts target color for each goal tile."""
        goal_tiles = {} # Map tile_name -> color
        for goal_str in goals:
            pred, args = self._parse_fact(goal_str)
            if pred == 'painted':
                tile, color = args
                goal_tiles[tile] = color
        return goal_tiles

    def _parse_available_colors(self, static_facts):
        """Extracts available colors."""
        available_colors = set()
        for fact_str in static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'available-color':
                available_colors.add(args[0])
        return available_colors

    def _parse_all_tiles(self, all_facts_set):
        """Extracts all tile names from all possible facts."""
        all_tiles = set()
        # Iterate through all possible facts defined in the domain/problem
        for fact_str in all_facts_set:
             # Simple regex to find tile_X_Y patterns
             matches = re.findall(r'tile_\d+_\d+', fact_str)
             all_tiles.update(matches)
        return list(all_tiles) # Return as list


    def _get_adjacent_tiles(self, tile_name):
        """Returns list of tiles adjacent to tile_name."""
        return self.tile_neighbors.get(tile_name, [])

    def _bfs_from_robot_location(self, start_tile, target_tiles_set, clear_tiles_set):
        """
        Finds the shortest path distance from start_tile (robot location)
        to any tile in target_tiles_set.
        Movement from any tile U to its neighbor V requires V to be clear.
        The start_tile itself does not need to be clear to start the BFS from it.
        """
        q = collections.deque([(start_tile, 0)])
        visited = {start_tile} # Keep track of tiles added to queue

        # If the robot is already on a target adjacent clear tile
        if start_tile in target_tiles_set:
             return 0

        while q:
            current_tile, dist = q.popleft()

            # Explore neighbors
            for neighbor in self._get_adjacent_tiles(current_tile):
                # Can only move to a neighbor if it is clear and not visited
                if neighbor in clear_tiles_set and neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = dist + 1
                    # Check if the neighbor is one of the target tiles
                    if neighbor in target_tiles_set:
                        return new_dist # Found shortest path to a target
                    q.append((neighbor, new_dist))

        # No path found
        return math.inf


    def __call__(self, node):
        state = node.state
        h = 0

        # 2. Identify robot locations and current colors
        robot_info = {} # Map robot_name -> {'location': tile_name, 'color': color_name}
        robots = set() # Collect all robot names present in the state
        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'robot-at':
                robot, tile = args
                robots.add(robot)
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = tile
            elif pred == 'robot-has':
                robot, color = args
                robots.add(robot)
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # 3. Identify the set of clear tiles
        clear_tiles_set = {self._parse_fact(fact_str)[1][0] for fact_str in state if self._parse_fact(fact_str)[0] == 'clear'}

        # 4. For each goal fact '(painted T C)'
        for goal_str in self.goals:
            pred, args = self._parse_fact(goal_str)
            if pred != 'painted':
                continue # Ignore non-painted goals if any

            target_tile, required_color = args

            # a. Check if already satisfied
            if goal_str in state:
                continue

            # b. Check if painted with wrong color
            is_painted_wrong = False
            for fact_str in state:
                p, a = self._parse_fact(fact_str)
                if p == 'painted' and a[0] == target_tile and a[1] != required_color:
                    is_painted_wrong = True
                    break
            if is_painted_wrong:
                # Dead end
                return math.inf

            # c. If the goal is not satisfied and not painted wrong, the tile T must be clear.
            # The paint action requires the target tile to be clear.
            # If it's not clear here, something is wrong with state representation or domain understanding.
            # Assuming it must be clear if not painted correctly or wrong.
            # Check explicitly just in case, although it should theoretically be true.
            if f'(clear {target_tile})' not in state:
                 # This state is likely invalid or represents an unreachable situation
                 # based on standard STRIPS assumptions for this domain.
                 # Treat as unreachable.
                 return math.inf


            # d. Tile T needs color C. Calculate the minimum cost to paint T:
            min_cost_for_tile = math.inf

            # Find clear tiles adjacent to the target tile
            adjacent_to_target = self._get_adjacent_tiles(target_tile)
            target_adjacent_clear_tiles = {
                adj_tile for adj_tile in adjacent_to_target
                if adj_tile in clear_tiles_set
            }

            # If no clear tile adjacent to the target, this goal is unreachable by painting
            if not target_adjacent_clear_tiles:
                 return math.inf

            # Consider each robot
            for robot_name in robots:
                info = robot_info.get(robot_name)
                # Ensure robot info is available (should be if robot is in state)
                if not info or 'location' not in info or 'color' not in info:
                     # This robot's state is incomplete, skip or treat as unable to help
                     continue

                robot_location = info['location']
                robot_color = info['color']

                # Calculate color cost
                color_cost = 0
                if robot_color != required_color:
                    # Cost is 1 action (change_color) if the required color is available.
                    # We assume required_color is always available if it's a goal color.
                    color_cost = 1

                # Calculate movement cost using BFS
                movement_cost = self._bfs_from_robot_location(
                    robot_location, target_adjacent_clear_tiles, clear_tiles_set
                )

                if movement_cost != math.inf:
                    # Total cost for this robot to paint this tile
                    robot_cost = color_cost + movement_cost + 1 # +1 for the paint action
                    min_cost_for_tile = min(min_cost_for_tile, robot_cost)

            # If after checking all robots, no one can paint this tile
            if min_cost_for_tile == math.inf:
                 # This goal is unreachable from this state
                 return math.inf

            h += min_cost_for_tile

        return h
