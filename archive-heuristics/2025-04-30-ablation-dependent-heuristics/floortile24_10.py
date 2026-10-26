import re
import math # Used for float('inf')

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the estimated cost
    for each individual unpainted goal tile. For each unpainted goal tile (t, c),
    the estimated cost is 1 (for the paint action) plus the minimum cost for any robot
    to get into a position (adjacent to t) with the correct color (c). The cost for a
    robot to get into position with the correct color is estimated as the Manhattan
    distance from the robot's current location to the closest tile adjacent to t,
    plus 1 if the robot does not currently have color c (representing a color change action).
    This heuristic is non-admissible as it sums costs for individual tiles, potentially
    overcounting movement and color changes if a single robot handles multiple tiles.
    It also includes a check for unsolvable states where a goal tile is painted with
    the wrong color, returning infinity in such cases.

    Assumptions:
    - Tile names are in the format 'tile_r_c' where r and c are integers representing
      row and column. This allows calculating Manhattan distance.
    - The grid defined by the adjacency predicates (up, down, left, right) corresponds
      to these row/column coordinates, specifically:
        - (up y x) means y is tile_{r-1}_c if x is tile_r_c.
        - (down y x) means y is tile_{r+1}_c if x is tile_r_c.
        - (left y x) means y is tile_r_{c-1} if x is tile_r_c.
        - (right y x) means y is tile_r_{c+1} if x is tile_r_c.
    - The problem instances are generally solvable if no goal tile is painted with the wrong color initially.
    - The heuristic should be efficiently computable, favoring Manhattan distance over BFS for pathfinding in the heuristic calculation.

    Heuristic Initialization:
    The constructor processes the static facts and goal facts provided in the Task object.
    1. It extracts all tile names and parses their row/column coordinates to build
       a mapping from tile name string to (row, col) tuple (`tile_name_to_coords`).
       It also builds the reverse mapping (`coords_to_tile_name`).
    2. It builds an adjacency list (`adj_list`) representing the grid graph based on
       the 'up', 'down', 'left', 'right' static predicates. This is used to find
       tiles adjacent to a target tile.
    3. It extracts the set of goal painted facts `(tile, color)` from the Task's goals.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for the given state.
    1. Parse the current state to extract:
       - Robot locations (`robot_locations`: {robot_name: tile_name})
       - Robot current colors (`robot_colors`: {robot_name: color_name})
       - Currently painted tiles (`current_painted`: {(tile_name, color_name)})
       - Currently clear tiles (`current_clear`: {tile_name})
    2. Identify the set of unpainted goal tiles (`unpainted_goal_tiles`): These are the `(tile, color)` pairs from the precomputed `self.goal_painted_facts` that are not present as `(painted tile color)` facts in the current state.
    3. If `unpainted_goal_tiles` is empty, the state is a goal state, return 0.
    4. Check for unsolvable states: Iterate through the `current_painted` facts. If any painted fact `(t, c')` exists where `t` is a tile that needs to be painted with a *different* color `c` in the goal (`(t, c)` is in `self.goal_painted_facts` and `c' != c`), return `float('inf')` as this state is likely a dead end (no unpaint action exists).
    5. Initialize the heuristic value `h = 0`.
    6. Store robot info from the state for easy access: `robot_info = {r: {'loc': loc, 'color': color}}`.
    7. Iterate through each `(t, c)` in `unpainted_goal_tiles`:
       a. This tile `t` needs to be painted with color `c`. This requires at least 1 paint action.
       b. A robot needs to be adjacent to `t` and have color `c`.
       c. Find the minimum cost for *any* robot to achieve this state (being adjacent to `t` with color `c`).
          - Get the coordinates of tile `t` (`t_coords`). If not found, return `float('inf')`.
          - Find the coordinates of all tiles adjacent to `t` using `self.adj_list` and `self.tile_name_to_coords`. If no adjacent tiles found, return `float('inf')`.
          - Initialize `min_cost_to_enable_paint_for_t = float('inf')`.
          - For each robot `r` in `robot_info`:
            - Get the robot's current location `loc_r` and color `color_r`.
            - Get the coordinates of `loc_r` (`loc_r_coords`). If not found, skip this robot (or return inf if unexpected).
            - Calculate the minimum Manhattan distance from `loc_r_coords` to any of the adjacent coordinates of `t`. Let this be `min_dist_to_adj_t`.
            - If `min_dist_to_adj_t` is `float('inf')` (robot cannot reach any adjacent tile based on coordinate map), skip this robot for this tile.
            - Calculate the color change cost for robot `r` to get color `c`: `color_change_cost = 1 if color_r != c else 0`.
            - The cost for robot `r` to be ready to paint tile `t` is `min_dist_to_adj_t + color_change_cost`.
            - Update `min_cost_to_enable_paint_for_t = min(min_cost_to_enable_paint_for_t, min_dist_to_adj_t + color_change_cost)`.
       d. If `min_cost_to_enable_paint_for_t` is still `float('inf')` after checking all robots, it means no robot can reach a position to paint this tile. Return `float('inf')` (state is likely unsolvable).
       e. Add the cost for this unpainted tile to the total heuristic: `h += (1 + min_cost_to_enable_paint_for_t)`. The `1` is for the paint action itself.
    8. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static and goal information.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.goal_painted_facts = self._parse_goal_painted_facts(task.goals)
        # Collect all facts from static, initial, and goal to find all tile names
        all_facts = task.static | task.initial_state | task.goals
        self.tile_name_to_coords, self.coords_to_tile_name = self._build_tile_coords(all_facts)
        self.adj_list = self._build_adjacency_list(task.static)
        # self.all_tile_names = set(self.tile_name_to_coords.keys()) # Not strictly needed

    def _parse_goal_painted_facts(self, goals):
        """
        Parses goal facts to extract the set of (tile, color) pairs that need to be painted.
        """
        painted_goals = set()
        for fact_str in goals:
            # Example: '(painted tile_1_1 white)'
            match = re.match(r"\(painted (\S+) (\S+)\)", fact_str)
            if match:
                tile, color = match.groups()
                painted_goals.add((tile, color))
        return painted_goals

    def _build_tile_coords(self, facts):
        """
        Extracts tile names from facts and builds coordinate mappings.
        Assumes tile names are in 'tile_r_c' format.
        """
        tile_name_to_coords = {}
        coords_to_tile_name = {}
        tile_pattern = re.compile(r"tile_(\d+)_(\d+)")

        all_tile_names = set()
        # Extract tile names from various predicates across all facts
        for fact_str in facts:
             # Find all words in the fact string by splitting and cleaning
             for word in fact_str.replace('(', '').replace(')', '').split():
                 match = tile_pattern.match(word)
                 if match:
                     all_tile_names.add(word)

        for tile_name in all_tile_names:
            match = tile_pattern.match(tile_name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                tile_name_to_coords[tile_name] = (r, c)
                coords_to_tile_name[(r, c)] = tile_name

        return tile_name_to_coords, coords_to_tile_name

    def _build_adjacency_list(self, static_facts):
        """
        Builds an adjacency list for the tile grid from static facts.
        """
        # Initialize with all known tiles
        adj_list = {tile: [] for tile in self.tile_name_to_coords}
        adj_patterns = {
            'up': r"\(up (\S+) (\S+)\)",
            'down': r"\(down (\S+) (\S+)\)",
            'left': r"\(left (\S+) (\S+)\)",
            'right': r"\(right (\S+) (\S+)\)",
        }

        for fact_str in static_facts:
            for direction, pattern in adj_patterns.items():
                match = re.match(pattern, fact_str)
                if match:
                    tile1, tile2 = match.groups()
                    # Add bidirectional edges only if both tiles are in our map
                    if tile1 in adj_list and tile2 in adj_list:
                         adj_list[tile1].append(tile2)
                         adj_list[tile2].append(tile1) # Assuming adjacency is symmetric

        # Remove duplicates from adjacency lists
        for tile in adj_list:
             adj_list[tile] = list(set(adj_list[tile]))

        return adj_list

    def _get_coords(self, tile_name):
        """Helper to get coordinates, returning None if tile_name is not in map."""
        return self.tile_name_to_coords.get(tile_name)

    def _manhattan_distance(self, coords1, coords2):
        """Calculates Manhattan distance between two coordinate tuples (r, c)."""
        # This function should only be called with valid coordinates
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of strings representing the current state facts.

        Returns:
            An integer or float('inf') representing the estimated cost to reach the goal.
        """
        # 1. Parse current state
        robot_locations = {}
        robot_colors = {}
        current_painted = set()
        current_clear = set() # Not directly used in this heuristic logic, but good practice to parse

        for fact_str in state:
            match_robot_at = re.match(r"\(robot-at (\S+) (\S+)\)", fact_str)
            if match_robot_at:
                robot, tile = match_robot_at.groups()
                robot_locations[robot] = tile
                continue

            match_robot_has = re.match(r"\(robot-has (\S+) (\S+)\)", fact_str)
            if match_robot_has:
                robot, color = match_robot_has.groups()
                robot_colors[robot] = color
                continue

            match_painted = re.match(r"\(painted (\S+) (\S+)\)", fact_str)
            if match_painted:
                tile, color = match_painted.groups()
                current_painted.add((tile, color))
                continue

            match_clear = re.match(r"\(clear (\S+)\)", fact_str)
            if match_clear:
                tile = match_clear.group(1)
                current_clear.add(tile)
                continue

        # 2. Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        for (goal_tile, goal_color) in self.goal_painted_facts:
            if (goal_tile, goal_color) not in current_painted:
                 unpainted_goal_tiles.add((goal_tile, goal_color))

        # 3. Goal check
        if not unpainted_goal_tiles:
            return 0

        # 4. Check for unsolvable states (wrong color painted on a goal tile)
        # We only need to check current_painted facts that are on goal tiles
        goal_tiles_set = {t for (t, c) in self.goal_painted_facts}
        goal_tile_to_color = {t: c for (t, c) in self.goal_painted_facts} # Map for quick lookup

        for (painted_tile, painted_color) in current_painted:
             if painted_tile in goal_tiles_set:
                 required_color = goal_tile_to_color[painted_tile]
                 if painted_color != required_color:
                     # Goal tile is painted with the wrong color
                     return float('inf')

        # 5. Initialize heuristic
        h = 0

        # 6. Store robot info for easy access
        # Ensure we only process robots found in the current state
        robot_info = {}
        for r in robot_locations:
             loc = robot_locations[r]
             color = robot_colors.get(r) # Robot might not have a color if init state is malformed, though domain implies they do.
             if loc is not None and color is not None: # Check for safety
                 robot_info[r] = {'loc': loc, 'color': color}

        # If no robots are present but goal requires painting, it's unsolvable
        if not robot_info and unpainted_goal_tiles:
             return float('inf')


        # 7. Calculate cost for each unpainted goal tile
        for (t, c) in unpainted_goal_tiles:
            # Cost for this tile is 1 (paint action) + min_cost_to_enable_paint
            min_cost_to_enable_paint_for_t = float('inf')

            t_coords = self._get_coords(t)
            if t_coords is None:
                 # Goal tile not in our coordinate map - indicates a problem with input
                 return float('inf')

            # Find coordinates of adjacent tiles that could be paint locations
            adjacent_tiles = self.adj_list.get(t, [])
            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles - indicates a problem with input grid
                 return float('inf')

            adjacent_coords = []
            for adj_t in adjacent_tiles:
                 adj_coords = self._get_coords(adj_t)
                 if adj_coords is not None: # Should be in map if adj_t was in adj_list
                      adjacent_coords.append(adj_coords)

            # If for some reason no adjacent tile coordinates were found (e.g. adjacent tile not in map)
            if not adjacent_coords:
                 return float('inf')


            for r, info in robot_info.items():
                loc_r = info['loc']
                color_r = info['color']

                loc_r_coords = self._get_coords(loc_r)
                if loc_r_coords is None:
                    # Robot location not in our coordinate map - indicates a problem
                    # This robot cannot contribute to painting this tile
                    continue # Skip this robot

                min_dist_to_adj_t = float('inf')
                for adj_coords in adjacent_coords:
                    dist = self._manhattan_distance(loc_r_coords, adj_coords)
                    min_dist_to_adj_t = min(min_dist_to_adj_t, dist)

                # If min_dist_to_adj_t is still inf, this robot cannot reach any adjacent tile.
                # This shouldn't happen in a connected grid unless adjacent tiles are missing from map.
                if min_dist_to_adj_t == float('inf'):
                    continue # This robot cannot paint this tile

                # Cost for this robot to be ready to paint this tile
                color_change_cost = 1 if color_r != c else 0
                cost_r = min_dist_to_adj_t + color_change_cost
                min_cost_to_enable_paint_for_t = min(min_cost_to_enable_paint_for_t, cost_r)

            # If no robot can reach an adjacent tile with the correct color consideration,
            # min_cost_to_enable_paint_for_t remains inf
            if min_cost_to_enable_paint_for_t == float('inf'):
                 # This unpainted goal tile is unreachable/unpaintable by any robot
                 return float('inf') # State is likely unsolvable

            h += (1 + min_cost_to_enable_paint_for_t) # 1 for paint action

        return h
