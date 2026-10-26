import math
from collections import deque
from heuristics.heuristic_base import Heuristic
# Assuming task and node classes are available in the environment
# from task import Task, Node # Not needed for the heuristic code itself

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing up costs for each
    unpainted goal tile. The cost for a single goal tile includes:
    1. A base cost of 1 for the paint action.
    2. The minimum grid distance for any robot to reach any *clear* tile
       adjacent to the goal tile. This distance is calculated on the full
       grid graph, but only clear adjacent tiles in the current state are
       considered as potential destinations for the final move before painting.
    3. An additional cost of 1 if the required color for this tile is not
       currently held by any robot (summed over unique needed colors).
    The heuristic also returns a large value if a goal tile is painted with the
    wrong color in the current state, as this is assumed to be a dead end.
    It also returns a large value if a goal tile is unreachable via a clear
    adjacent tile by any robot.

    Assumptions:
    - Tile names are in the format 'tile_X_Y' where X and Y are integers
      (although the heuristic doesn't strictly rely on parsing X, Y, it assumes
       the grid structure is fully defined by up/down/left/right facts).
    - The grid defined by up/down/left/right facts is connected.
    - Goal tiles are initially either clear or painted with the correct color.
      (The heuristic checks for wrong-colored goal tiles in subsequent states).
    - All colors required by the goal are available (available-color predicate).
    - The 'clear' precondition for the paint action itself (that the tile being
      painted must be clear) is relaxed/ignored, assuming robots will move off
      goal tiles if necessary.

    Heuristic Initialization:
    1. Parses static facts to build the grid graph (adjacency list) based on
       up/down/left/right predicates.
    2. Identifies all unique tile objects mentioned in static or goal facts.
    3. Computes all-pairs shortest paths between all tiles using BFS on the
       grid graph. Stores these distances.
    4. Parses goal facts to store the target color for each goal tile.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state facts.
    2. Extract the current location and color held by each robot from the state.
    3. Check if any goal tile is currently painted with a color different from
       its target color in the goal. If so, return a large value (dead end).
    4. Identify all goal tiles that are not yet painted with their target color
       in the current state.
    5. If there are no unpainted goal tiles, the goal is reached, return 0.
    6. Initialize the heuristic value H to 0.
    7. Keep track of colors needed for the unpainted goal tiles.
    8. For each unpainted goal tile (T, C) needing color C:
        a. Add 1 to H (representing the paint action).
        b. Add C to the set of needed colors.
        c. Find all tiles adjacent to T using the precomputed grid graph.
        d. Identify which of these adjacent tiles are currently 'clear' in the state.
           Calculate the minimum distance from any robot's current location
           to any of these *clear* adjacent tiles, using the precomputed
           shortest paths.
        e. If no clear adjacent tile is reachable by any robot on the grid graph
           (e.g., no robots, or all adjacent tiles are not clear or unreachable),
           return a large value.
        f. Add this minimum distance to H.
    9. Count how many colors in the set of needed colors are not currently held
       by any robot. Add this count to H (representing color change actions).
    10. Return H.
    """

    DEAD_END_VALUE = 1000000 # Use a large finite value for dead ends

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_facts = task.goals
        self.static_facts = task.static

        # --- Heuristic Initialization ---

        # 1. Build grid graph and identify tiles
        self.adj = {}
        self.all_tiles = set()
        for fact_str in self.static_facts:
            if fact_str.startswith('(up ') or fact_str.startswith('(down ') or \
               fact_str.startswith('(left ') or fact_str.startswith('(right '):
                parts = fact_str.strip('()').split()
                # Ensure fact has expected number of parts
                if len(parts) == 3:
                    t1 = parts[1]
                    t2 = parts[2]
                    self.all_tiles.add(t1)
                    self.all_tiles.add(t2)
                    if t1 not in self.adj:
                        self.adj[t1] = []
                    if t2 not in self.adj:
                        self.adj[t2] = []
                    # Add bidirectional edges
                    self.adj[t1].append(t2)
                    self.adj[t2].append(t1)

        # Ensure all tiles mentioned in goal facts are included, even if isolated
        # (unlikely in floortile, but robust)
        for goal_fact_str in self.goal_facts:
             if goal_fact_str.startswith('(painted '):
                 parts = goal_fact_str.strip('()').split()
                 if len(parts) == 3:
                     tile = parts[1]
                     self.all_tiles.add(tile)
                     if tile not in self.adj:
                         self.adj[tile] = [] # Add isolated tiles

        # 2. Compute all-pairs shortest paths
        self.tile_distances = {}
        for start_tile in self.all_tiles:
            self.tile_distances[start_tile] = self._bfs(start_tile)

        # 3. Store goal facts (tile and target color)
        self.goal_tile_colors = {}
        for goal_fact_str in self.goal_facts:
            if goal_fact_str.startswith('(painted '):
                parts = goal_fact_str.strip('()').split()
                if len(parts) == 3:
                    tile = parts[1]
                    color = parts[2]
                    self.goal_tile_colors[tile] = color

    def _bfs(self, start_tile):
        """Performs BFS from start_tile to find distances to all other tiles."""
        distances = {tile: math.inf for tile in self.all_tiles}
        if start_tile not in distances:
             # Should not happen if all_tiles is populated correctly
             return distances

        distances[start_tile] = 0
        queue = deque([start_tile])

        while queue:
            curr_tile = queue.popleft()
            # Use get(curr_tile, []) to handle tiles that might be in all_tiles
            # but have no adjacencies (isolated).
            for neighbor_tile in self.adj.get(curr_tile, []):
                if distances[neighbor_tile] == math.inf:
                    distances[neighbor_tile] = distances[curr_tile] + 1
                    queue.append(neighbor_tile)
        return distances

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Get current state facts (already have 'state')

        # 2. Extract robot locations and colors, and clear tiles
        robot_locations = {}
        robot_colors = {}
        clear_tiles_in_state = set()

        for fact_str in state:
            if fact_str.startswith('(robot-at '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    robot = parts[1]
                    tile = parts[2]
                    robot_locations[robot] = tile
            elif fact_str.startswith('(robot-has '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    robot = parts[1]
                    color = parts[2]
                    robot_colors[robot] = color
            elif fact_str.startswith('(clear '):
                 parts = fact_str.strip('()').split()
                 if len(parts) == 2:
                     tile = parts[1]
                     clear_tiles_in_state.add(tile)


        # 3. Check for dead ends (goal tile painted wrong color)
        for fact_str in state:
            if fact_str.startswith('(painted '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    painted_tile = parts[1]
                    painted_color = parts[2]
                    # Check if this tile is a goal tile and the color is wrong
                    if painted_tile in self.goal_tile_colors and \
                       self.goal_tile_colors[painted_tile] != painted_color:
                        # Goal tile painted with the wrong color - likely unsolvable
                        return self.DEAD_END_VALUE

        # 4. Identify unpainted goal tiles
        unpainted_goals = [] # List of (tile, target_color)
        for goal_tile, target_color in self.goal_tile_colors.items():
            if '(painted {} {})'.format(goal_tile, target_color) not in state:
                unpainted_goals.append((goal_tile, target_color))

        # 5. If no unpainted goal tiles, goal is reached
        if not unpainted_goals:
            return 0

        # 6. Initialize heuristic value
        h_value = 0
        needed_colors_set = set()

        # 8. For each unpainted goal tile
        for goal_tile, target_color in unpainted_goals:
            # a. Add 1 for paint action
            h_value += 1
            # b. Add color to needed set
            needed_colors_set.add(target_color)

            # c. Find adjacent tiles
            adjacent_tiles = self.adj.get(goal_tile, [])

            # d. Calculate min distance from any robot to any *clear* adjacent tile
            min_dist_to_clear_adjacent = math.inf
            # Check if there are any robots
            if not robot_locations:
                 return self.DEAD_END_VALUE # No robots to paint

            for robot, robot_loc in robot_locations.items():
                # Ensure robot_loc is a valid tile in our distance map
                if robot_loc not in self.tile_distances:
                     # Robot is at an unknown location? Should not happen in valid states.
                     continue # Skip this robot if location is unknown

                for adj_tile in adjacent_tiles:
                    # Check if adjacent tile is clear in the current state
                    if adj_tile in clear_tiles_in_state:
                        # Ensure adj_tile is a valid tile in our distance map
                        if adj_tile in self.tile_distances[robot_loc]:
                             dist = self.tile_distances[robot_loc][adj_tile]
                             min_dist_to_clear_adjacent = min(min_dist_to_clear_adjacent, dist)

            # e. If no *clear* adjacent tile is reachable by any robot
            if min_dist_to_clear_adjacent == math.inf:
                 # This goal tile is unreachable because no adjacent tile is clear and reachable.
                 # Could be a temporary block or a dead end (e.g., adjacent tile painted).
                 # Returning DEAD_END_VALUE seems appropriate for a greedy search.
                 return self.DEAD_END_VALUE

            # f. Add min distance to H
            h_value += min_dist_to_clear_adjacent

        # 9. Count color changes needed
        color_change_cost = 0
        for color in needed_colors_set:
            # Check if any robot currently has this color
            has_color = False
            for robot, robot_color in robot_colors.items():
                if robot_color == color:
                    has_color = True
                    break
            if not has_color:
                color_change_cost += 1

        h_value += color_change_cost

        # 10. Return H
        return h_value
