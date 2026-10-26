import collections
import re
import logging

from heuristics.heuristic_base import Heuristic

# Assuming Task and Operator classes are available from task.py
# from task import Task, Operator

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each individual goal fact that is not yet satisfied.
    For a goal fact (painted tile_t color_c) that is not satisfied and where
    tile_t is currently clear or occupied by a robot, the estimated cost is
    calculated as:
    1 (to move a robot off tile_t if one is there)
    + minimum cost for any robot to reach a tile adjacent to tile_t from which
      it can be painted, with the robot having color_c
    + 1 (for the paint action).
    The minimum cost for a robot to reach a painting-adjacent tile with the
    correct color is the minimum over all robots of (shortest path distance
    from robot's current location to a painting-adjacent tile of tile_t +
    1 if the robot needs to change color).
    If a goal tile is painted with the wrong color in the current state, the
    heuristic returns infinity, as there is no action to unpaint or repaint.

    Assumptions:
    - Tiles are named in the format 'tile_row_col' where row and col are integers.
    - The grid defined by up/down/left/right predicates is connected.
    - Goal facts only require painting tiles that are initially clear or already
      painted with the correct color. If a tile is painted with the wrong color
      in the state, it's a dead end.
    - Robots always have some color. Changing color costs 1 action.
    - All actions have cost 1.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes the grid structure and
    shortest path distances between all pairs of tiles.
    1. Parses all 'tile_row_col' objects from static facts to identify tiles
       and their coordinates (though coordinates are not strictly used in
       the current heuristic logic, they are stored).
    2. Builds a directed graph where nodes are tiles and edges represent
       possible robot movements based on 'up', 'down', 'left', 'right'
       static facts. An edge exists from tile A to tile B if a move action
       allows a robot to go from A to B.
    3. Computes all-pairs shortest paths on this movement graph using BFS
       starting from each tile. These distances are stored for quick lookup
       during heuristic computation.
    4. Precomputes, for each tile, the set of tiles from which it can be painted.
       A tile Y can be painted from tile X if a static fact like (up Y X) exists.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h = 0`.
    2. Get the current state and the goal facts from the task.
    3. Extract the current location and held color for each robot from the state.
    4. Iterate through each goal fact `(painted tile_t color_c)`.
    5. For the current goal fact `(painted tile_t color_c)`:
        a. Check if the fact `(painted tile_t color_c)` is already true in the current state. If yes, this goal is satisfied for this tile, continue to the next goal fact.
        b. Check the current state of `tile_t`.
        c. If `tile_t` is painted with a color `c'` where `c' != color_c`:
            - This is a dead end. Return `float('inf')`.
        d. If `tile_t` is occupied by a robot `robot_r`:
            - Add 1 to the cost for this tile (for moving `robot_r` off).
        e. If `tile_t` is neither clear, nor painted (correct or wrong), nor occupied:
            - This indicates an unexpected state. Treat as a dead end and return `float('inf')`.
        f. If `tile_t` is clear (or was occupied and the cost to move off was added):
            i. Initialize `min_robot_ready_cost = float('inf')`.
            ii. Get the list of tiles `painting_adj_tiles` from which `tile_t` can be painted (precomputed).
            iii. If `painting_adj_tiles` is empty, return `float('inf')` (tile cannot be painted).
            iv. For each robot `robot_r` and its current location `r_loc`:
                - Calculate `color_cost = 1` if `robot_r` does not have `color_c`, else `0`.
                - Initialize `min_move_cost_r = float('inf')`.
                - For each tile `adj_t` in `painting_adj_tiles`:
                    - Look up the precomputed shortest path distance `dist(r_loc, adj_t)`.
                    - Update `min_move_cost_r = min(min_move_cost_r, dist(r_loc, adj_t))`.
                - If `min_move_cost_r` is not `float('inf')`, update `min_robot_ready_cost = min(min_robot_ready_cost, min_move_cost_r + color_cost)`.
            v. If `min_robot_ready_cost` is still `float('inf')` (no robot can reach a painting-adjacent tile with the required color), return `float('inf')`.
            vi. Add `min_robot_ready_cost + 1` (for the paint action) to the total heuristic value `h`.
    6. Return the total sum `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # Precompute grid structure and distances
        self._tile_coords = {} # Map tile name to (row, col)
        self._coords_tile = {} # Map (row, col) to tile name
        self._movement_adjacency_list = collections.defaultdict(list) # Map tile name to list of tiles reachable by one move
        self._distances = {} # Map (tile1, tile2) to shortest path distance for movement
        self._painting_adjacency_map = collections.defaultdict(list) # Map tile name to list of tiles from which it can be painted

        self._parse_static_facts()
        self._compute_distances()
        self._compute_painting_adjacency()


    def _parse_static_facts(self):
        """
        Parses static facts to build the grid structure and movement adjacency list.
        Identifies tiles and their coordinates.
        Builds movement adjacency list based on movement predicates (up, down, left, right).
        """
        tiles = set()
        # Extract all tile names first to build coordinate maps
        tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
        for fact_str in self.static_facts:
            parts = fact_str[1:-1].split()
            for part in parts:
                if part.startswith('tile_'):
                    tiles.add(part)

        # Map tile names to coordinates and vice versa
        for tile_name in tiles:
            match = tile_pattern.match(tile_name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                self._tile_coords[tile_name] = (row, col)
                self._coords_tile[(row, col)] = tile_name

        # Build movement adjacency list based on movement predicates
        # (up ?y ?x), (down ?y ?x), (left ?y ?x), (right ?y ?x)
        # mean robot can move from x to y. Add edge x -> y.
        for fact_str in self.static_facts:
            parts = fact_str[1:-1].split()
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                pred, tile_y, tile_x = parts
                if tile_x in self._tile_coords and tile_y in self._tile_coords:
                     # Add edge from tile_x to tile_y in the movement graph
                    self._movement_adjacency_list[tile_x].append(tile_y)

    def _compute_distances(self):
        """
        Computes shortest path distances between all pairs of tiles using BFS
        on the movement graph. Stores results in self._distances.
        """
        for start_tile in self._tile_coords:
            self._distances[start_tile] = {}
            queue = collections.deque([(start_tile, 0)])
            visited = {start_tile}

            while queue:
                current_tile, dist = queue.popleft()
                self._distances[start_tile][current_tile] = dist

                # Check if current_tile is in adjacency list (it might not be if it's an isolated tile, though unlikely in grid)
                if current_tile in self._movement_adjacency_list:
                    for neighbor_tile in self._movement_adjacency_list[current_tile]:
                        if neighbor_tile not in visited:
                            visited.add(neighbor_tile)
                            queue.append((neighbor_tile, dist + 1))

    def _compute_painting_adjacency(self):
        """
        Computes, for each tile, the set of tiles from which it can be painted.
        A tile Y can be painted from tile X if a static fact like (up Y X) exists.
        Stores results in self._painting_adjacency_map.
        """
        # (up ?y ?x), (down ?y ?x), (left ?y ?x), (right ?y ?x)
        # mean robot at x can paint y. Add x to _painting_adjacency_map[y].
        for fact_str in self.static_facts:
            parts = fact_str[1:-1].split()
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                pred, tile_y, tile_x = parts
                if tile_x in self._tile_coords and tile_y in self._tile_coords:
                     # tile_y can be painted from tile_x
                    self._painting_adjacency_map[tile_y].append(tile_x)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state
        h_value = 0

        # Extract robot locations and colors from the current state
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact_str in state:
            parts = fact_str[1:-1].split()
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot_name, tile_name = parts[1], parts[2]
                if robot_name not in robot_info:
                    robot_info[robot_name] = {}
                robot_info[robot_name]['location'] = tile_name
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot_name, color_name = parts[1], parts[2]
                if robot_name not in robot_info:
                    robot_info[robot_name] = {}
                robot_info[robot_name]['color'] = color_name

        # Identify unpainted goal tiles that are clear or occupied
        unpainted_paintable_goals = []
        dead_end = False

        for goal_fact_str in self.goals:
            # Goal fact is '(painted tile_t color_c)'
            parts = goal_fact_str[1:-1].split()
            if parts[0] == 'painted' and len(parts) == 3:
                tile_t, color_c = parts[1], parts[2]

                # Check if goal is already satisfied
                if goal_fact_str in state:
                    continue # Goal tile is already painted correctly

                # Check state of tile_t
                is_clear = f'(clear {tile_t})' in state
                is_occupied = any(info.get('location') == tile_t for info in robot_info.values())
                is_painted_wrong = False
                for fact_str in state:
                     p_parts = fact_str[1:-1].split()
                     if p_parts[0] == 'painted' and len(p_parts) == 3 and p_parts[1] == tile_t and p_parts[2] != color_c:
                         is_painted_wrong = True
                         break

                if is_painted_wrong:
                    # Tile is painted the wrong color - dead end
                    dead_end = True
                    break
                elif is_clear or is_occupied:
                    # Tile needs painting and is clear or occupied (can be made clear)
                    unpainted_paintable_goals.append((tile_t, color_c, is_occupied))
                # else: The tile is neither clear, nor painted (correct or wrong), nor occupied.
                # This state shouldn't happen based on domain predicates, but if it does,
                # it's likely an unpaintable state or a state representation issue.
                # Treat as dead end for safety.
                elif not is_clear and not is_occupied and not is_painted_wrong:
                     logging.warning(f"Tile {tile_t} is not clear, not occupied, and not painted with goal color {color_c}, but also not painted with any other color in state. Treating as dead end.")
                     dead_end = True
                     break


        if dead_end:
            return float('inf')

        # Calculate cost for each unpainted clear/occupied goal tile
        for tile_t, color_c, is_occupied in unpainted_paintable_goals:
            tile_cost = 0
            if is_occupied:
                tile_cost += 1 # Cost to move robot off the tile

            min_robot_ready_cost = float('inf')
            painting_adjacent_tiles = self._painting_adjacency_map.get(tile_t, [])

            if not painting_adjacent_tiles:
                 # Should not happen in a grid where tiles have neighbors, but handle defensively
                 logging.warning(f"Tile {tile_t} has no painting-adjacent tiles. Treating as dead end.")
                 return float('inf')

            for robot_name, info in robot_info.items():
                r_loc = info.get('location')
                r_color = info.get('color')

                if r_loc is None or r_color is None:
                    # Robot info incomplete - should not happen
                    continue

                color_cost = 1 if r_color != color_c else 0

                min_move_cost_r = float('inf')
                if r_loc in self._distances:
                    for adj_t in painting_adjacent_tiles:
                        if adj_t in self._distances[r_loc]:
                             move_cost = self._distances[r_loc][adj_t]
                             min_move_cost_r = min(min_move_cost_r, move_cost)

                if min_move_cost_r != float('inf'):
                    min_robot_ready_cost = min(min_robot_ready_cost, min_move_cost_r + color_cost)

            if min_robot_ready_cost == float('inf'):
                 # No robot can reach a painting-adjacent tile with the required color
                 # This could happen if grid is disconnected or no robot can get the color (e.g. color not available)
                 # The domain guarantees available-color, so likely grid issue or no path.
                 # Treat as dead end.
                 logging.warning(f"No robot can reach a painting-adjacent tile of {tile_t} with color {color_c}. Treating as dead end.")
                 return float('inf')


            # Total cost for this tile: move_off_cost + min_robot_ready_cost + paint_action_cost
            h_value += tile_cost + min_robot_ready_cost + 1

        return h_value
