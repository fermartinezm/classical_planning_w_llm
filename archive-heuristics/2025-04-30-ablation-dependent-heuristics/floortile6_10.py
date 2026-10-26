from collections import deque
import math # Used for float('inf')

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum estimated cost
    for each unpainted goal tile independently. The estimated cost for a single
    unpainted goal tile is the sum of:
    1. The cost of the paint action (1).
    2. The minimum cost for any robot to reach a clear tile adjacent to the
       target tile with the required color. This cost is the minimum of:
       a) The shortest path distance from a robot that already has the color
          to a clear adjacent tile.
       b) 1 (for color change) + the shortest path distance from any robot
          to a clear adjacent tile (if the color is available).

    Assumptions:
    - The problem instance is solvable (unless the heuristic returns a large penalty).
    - Tile names are in the format 'tile_r_c' where r and c are integers.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      forms a connected graph (or disconnected components are handled by BFS).
    - Tiles painted with the wrong color in the initial state (i.e., not clear
      and not painted with the goal color) make the problem unsolvable.
    - The 'clear' precondition for moving and painting means robots cannot
      move onto or paint occupied/painted tiles. The heuristic considers
      only currently clear adjacent tiles as potential painting spots.
      It assumes robots can move through other clear tiles according to grid distances.

    Heuristic Initialization:
    1. Parses all tile names from the task's initial state and static facts
       to extract their (row, col) coordinates. Stores this mapping (`tile_coords`).
    2. Creates a reverse mapping from coordinates to tile names (`coords_to_tile_name`).
    3. Builds an adjacency list representation of the tile grid graph based
       on the 'up', 'down', 'left', 'right' static predicates.
    4. Computes all-pairs shortest paths (APSP) on the tile grid graph using
       BFS starting from each tile. Stores these distances.
    5. Stores the goal facts and static facts for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all goal facts of the form '(painted tile color)'.
    2. Filter these goal facts to find the set of 'unpainted goal tiles' -
       those where the goal fact '(painted tile color)' is not present in the
       current state.
    3. If there are no unpainted goal tiles, the heuristic is 0 (goal state).
    4. Extract the current position and color of each robot from the state.
    5. Initialize the total heuristic value to 0.
    6. Define a large penalty value for states deemed unsolvable or highly undesirable.
    7. For each unpainted goal tile (tile, target_color):
        a. Check if the tile is currently clear in the state. If not, it must be
           painted. If it's painted with a color different from the target color,
           the state is likely unsolvable. Return the large penalty.
        b. Find all tiles adjacent to the target tile based on the precomputed
           adjacency list (using coordinates).
        c. Filter the adjacent tiles to find those that are currently 'clear'
           in the state. These are the only tiles a robot can move onto to paint
           the target tile.
        d. If there are no clear adjacent tiles, the target tile cannot be painted
           from this state. Return the large penalty.
        e. Calculate the minimum cost for any robot to reach one of the clear
           adjacent tiles with the required target color. This involves two options:
           i. Minimum distance from any robot that *already* has the target color
              to any clear adjacent tile.
           ii. Minimum distance from *any* robot to any clear adjacent tile, plus 1
               for the color change action (if the target color is available).
           The minimum of these two options is the cost to get a robot with the
           right color adjacent.
        f. If the target color is needed but not available (checked against static facts)
           and no robot already has the color, the tile is unpaintable. Return
           the large penalty.
        g. If the minimum cost to get a robot with the right color adjacent is
           still infinity (e.g., disconnected graph, although unlikely), return
           the large penalty.
        h. Add the calculated minimum cost (movement + color change if needed)
           plus 1 (for the paint action) to the total heuristic value.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.large_penalty = 1000000 # Use a large number for unreachable states

        # 1. Identify all tile objects and parse coordinates
        self.tile_coords = {}
        self.coords_to_tile_name = {}
        all_objects = set()
        # Collect objects from initial state and static facts
        # A more robust way would be to parse the domain file objects section,
        # but for this problem, extracting from facts is sufficient based on examples.
        for fact in task.initial_state | task.static:
             # Simple parsing: find words starting with 'tile_'
             parts = fact.replace('(', ' ').replace(')', ' ').split()
             for part in parts:
                 if part.startswith('tile_'):
                     all_objects.add(part)

        for obj in all_objects:
            if obj.startswith('tile_'):
                try:
                    coords = self._parse_tile_name(obj)
                    self.tile_coords[obj] = coords
                    self.coords_to_tile_name[coords] = obj
                except ValueError:
                    # Handle cases where object is not a tile_r_c format if necessary
                    # Assuming all tile objects follow the format for valid problems
                    pass

        # 2. Build adjacency list
        self.adj = {coords: [] for coords in self.tile_coords.values()}
        for fact in self.static:
            if fact.startswith('(up ') or fact.startswith('(down ') or \
               fact.startswith('(left ') or fact.startswith('(right '):
                _, args = self._parse_fact(fact)
                t1, t2 = args # e.g., (up t1 t2) means t1 is up from t2
                if t1 in self.tile_coords and t2 in self.tile_coords:
                    coords1 = self.tile_coords[t1]
                    coords2 = self.tile_coords[t2]
                    # Add edge from t2's coords to t1's coords
                    self.adj[coords2].append(coords1)

        # 3. Compute all-pairs shortest paths (APSP)
        self.distances = {}
        all_tile_coords_list = list(self.tile_coords.values())
        for start_coords in all_tile_coords_list:
            self.distances[start_coords] = self._bfs(start_coords, self.adj, all_tile_coords_list)

    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string into predicate and arguments."""
        # Remove leading '(' and trailing ')' and space
        fact_str = fact_str[1:-1].strip()
        parts = fact_str.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _parse_tile_name(self, tile_name):
        """Parses a tile name like 'tile_r_c' into (r, c) tuple."""
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                r = int(parts[1])
                c = int(parts[2])
                return (r, c)
            except ValueError:
                raise ValueError(f"Invalid tile coordinate format in name: {tile_name}")
        else:
             raise ValueError(f"Invalid tile name format: {tile_name}")


    def _bfs(self, start_coords, adj_list, all_coords):
        """Performs BFS to find distances from start_coords to all other coords."""
        distances = {coords: float('inf') for coords in all_coords}
        distances[start_coords] = 0
        queue = deque([start_coords])

        while queue:
            current_coords = queue.popleft()
            current_dist = distances[current_coords]

            if current_coords in adj_list:
                for neighbor_coords in adj_list[current_coords]:
                    if distances[neighbor_coords] == float('inf'):
                        distances[neighbor_coords] = current_dist + 1
                        queue.append(neighbor_coords)
        return distances

    def _get_distance(self, tile_name1, tile_name2):
        """Looks up the precomputed distance between two tiles."""
        coords1 = self.tile_coords.get(tile_name1)
        coords2 = self.tile_coords.get(tile_name2)
        if coords1 is None or coords2 is None:
             # Should not happen if tiles are correctly parsed
             return float('inf')
        # Handle cases where start_coords or end_coords might not be in the distances map
        # (e.g., disconnected graph, although unlikely for grid problems)
        if coords1 not in self.distances or coords2 not in self.distances[coords1]:
             return float('inf')
        return self.distances[coords1][coords2]

    def _get_adjacent_tiles(self, tile_name):
        """Returns a list of tile names adjacent to the given tile."""
        coords = self.tile_coords.get(tile_name)
        if coords is None or coords not in self.adj:
            return [] # Tile not found or has no neighbors
        adjacent_coords = self.adj[coords]
        # Map adjacent coordinates back to tile names using the reverse mapping
        adjacent_tile_names = [self.coords_to_tile_name[c] for c in adjacent_coords if c in self.coords_to_tile_name]
        return adjacent_tile_names


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of facts representing the current state.
        @return: The estimated number of actions to reach the goal.
        """
        unpainted_goals = []
        for goal_fact in self.goals:
            if goal_fact.startswith('(painted '):
                if goal_fact not in state:
                    _, args = self._parse_fact(goal_fact)
                    tile = args[0]
                    color = args[1]
                    unpainted_goals.append((tile, color))

        if not unpainted_goals:
            return 0 # Goal reached

        robot_pos = {}
        robot_color = {}
        for fact in state:
            if fact.startswith('(robot-at '):
                _, args = self._parse_fact(fact)
                robot, pos = args
                robot_pos[robot] = pos
            elif fact.startswith('(robot-has '):
                _, args = self._parse_fact(fact)
                robot, color = args
                robot_color[robot] = color

        total_h = 0

        for tile, target_color in unpainted_goals:
            # Check if tile is paintable (must be clear or painted with target color)
            # If it's not clear, it must be painted. If painted, check if it's the target color.
            is_clear = f'(clear {tile})' in state
            is_painted_target_color = f'(painted {tile} {target_color})' in state
            is_painted_wrong_color = False
            if not is_clear and not is_painted_target_color:
                 # Check if it's painted with *any* color other than the target
                 for fact in state:
                     if fact.startswith(f'(painted {tile} '):
                         _, args = self._parse_fact(fact)
                         painted_color = args[1]
                         if painted_color != target_color:
                             is_painted_wrong_color = True
                             break # Found a painted fact for this tile with wrong color

            if is_painted_wrong_color:
                 # Tile is painted wrong. This state is likely unsolvable.
                 return self.large_penalty

            # If it's not clear and not painted wrong, it must be painted with the target color,
            # but we already filtered those out in unpainted_goals.
            # So, if it's not clear and not painted wrong, it's an invalid state representation
            # or a tile that is blocked by a robot but not clear? The domain says clear means not painted.
            # Let's assume if it's not clear and not painted with the target color, it's painted wrong.
            # The check `f'(clear {tile})' not in state` implies it's not clear.
            # If it's not clear, it must be painted. If it's not painted with the target color, it's wrong.
            # The check `is_painted_wrong_color` handles this.

            # Find adjacent tiles that are clear (potential painting spots)
            adjacent_tiles = self._get_adjacent_tiles(tile)
            reachable_adj_tiles = [adj_t for adj_t in adjacent_tiles if f'(clear {adj_t})' in state]

            if not reachable_adj_tiles:
                # Cannot reach any clear adjacent tile to paint this tile.
                # This tile is currently unpaintable.
                return self.large_penalty

            min_cost_tile = float('inf')

            # Option 1: Use a robot that already has the target color
            min_cost_existing_color = float('inf')
            for r, r_pos in robot_pos.items():
                if robot_color.get(r) == target_color: # Use .get() in case robot_has fact is missing (though domain implies it's always there)
                    min_dist_from_r = float('inf')
                    for adj_t in reachable_adj_tiles:
                        dist = self._get_distance(r_pos, adj_t)
                        min_dist_from_r = min(min_dist_from_r, dist)
                    min_cost_existing_color = min(min_cost_existing_color, min_dist_from_r)

            # Option 2: Use any robot and change color
            min_cost_change_color = float('inf')
            if f'(available-color {target_color})' in self.static:
                for r, r_pos in robot_pos.items():
                    min_dist_from_r = float('inf')
                    for adj_t in reachable_adj_tiles:
                        dist = self._get_distance(r_pos, adj_t)
                        min_dist_from_r = min(min_dist_from_r, dist)
                    if min_dist_from_r != float('inf'): # Check if robot can reach any reachable_adj_tile
                         min_cost_change_color = min(min_cost_change_color, 1 + min_dist_from_r)
            else:
                 # Target color is needed but not available. Problem unsolvable.
                 return self.large_penalty


            # Minimum cost to get a robot with the right color adjacent to the tile
            min_cost_to_get_adjacent_with_color = min(min_cost_existing_color, min_cost_change_color)

            if min_cost_to_get_adjacent_with_color == float('inf'):
                 # This should only happen if reachable_adj_tiles was empty (checked above),
                 # or if the graph is disconnected, or color is unavailable (checked above).
                 # If it happens, it means no robot can reach any reachable_adj_tile with the required color.
                 return self.large_penalty

            # Total cost for this tile = cost_to_get_adjacent_with_color + cost_to_paint
            total_h += min_cost_to_get_adjacent_with_color + 1

        return total_h
