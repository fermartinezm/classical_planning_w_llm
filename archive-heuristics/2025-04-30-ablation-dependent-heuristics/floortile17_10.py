import math

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal state by summing the minimum costs
    to paint each unpainted clear goal tile.
    """

    def __init__(self, task):
        """
        Heuristic Initialization:
        Initializes the heuristic by precomputing static information from the task.
        This includes identifying robots, tiles, and colors, building the grid
        adjacency list from static facts, and storing goal facts.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goal_facts = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Collect objects (robots, tiles, colors) from initial state, static facts, and goal facts
        all_objects = set()
        for fact in self.initial_state | self.static_facts | self.goal_facts:
             pred, args = self._parse_fact(fact)
             all_objects.update(args)

        self.robots = [obj for obj in all_objects if obj.startswith('robot')]
        self.all_tiles = [obj for obj in all_objects if obj.startswith('tile_')]
        # Assuming any object not starting with 'robot' or 'tile_' is a color
        self.colors = [obj for obj in all_objects if obj not in self.robots and obj not in self.all_tiles]

        self.adj_list = self._build_adjacency_list(self.static_facts)

        # Optional: Basic check for grid connectivity coverage
        connected_tiles = set(self.adj_list.keys()) | set(tile for adj_list in self.adj_list.values() for tile in adj_list)
        if not set(self.all_tiles).issubset(connected_tiles):
             # This warning indicates potential issues if the heuristic relies on all tiles being in the grid graph
             # print("Warning: Some tiles found in facts are not in the adjacency graph.")
             pass # Suppress warning for submission unless debugging


    def _parse_fact(self, fact):
        """Helper to parse a fact string like '(predicate arg1 arg2)'."""
        # Removes surrounding brackets and splits by space
        parts = fact[1:-1].split()
        if not parts: # Handle empty fact string case
             return None, []
        return parts[0], parts[1:]

    def _parse_tile_name(self, tile_name):
        """Helper to parse tile name 'tile_row_col' into (row, col)."""
        try:
            parts = tile_name.split('_')
            # Expecting at least 'tile', row, col parts
            if len(parts) < 3 or parts[0] != 'tile':
                 raise ValueError("Invalid tile name format")
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except (ValueError, IndexError) as e:
            # print(f"Error parsing tile name {tile_name}: {e}")
            return None, None # Indicate parsing failure


    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles based on their names."""
        r1, c1 = self._parse_tile_name(tile1_name)
        r2, c2 = self._parse_tile_name(tile2_name)
        if r1 is None or r2 is None:
             return 1000000 # Cannot calculate distance for invalid tile names
        return abs(r1 - r2) + abs(c1 - c2)

    def _build_adjacency_list(self, static_facts):
        """Builds adjacency list from static up/down/left/right facts."""
        adj = {}
        for fact in static_facts:
            pred, args = self._parse_fact(fact)
            if pred in ('up', 'down', 'left', 'right') and len(args) == 2:
                y, x = args # y is adjacent to x
                adj.setdefault(x, []).append(y)
                adj.setdefault(y, []).append(x) # Grid is bidirectional
        return adj

    def _get_adjacent_tiles(self, tile_name):
        """Returns list of tiles adjacent to tile_name based on precomputed adjacency."""
        return self.adj_list.get(tile_name, [])

    def _min_moves_to_be_adjacent(self, loc_r, t):
        """
        Calculates minimum Manhattan moves from loc_r to reach any tile adjacent to t.
        If loc_r is already adjacent to t, returns 0.
        """
        # Check if loc_r is a valid tile name before proceeding
        if self._parse_tile_name(loc_r) == (None, None):
             return 1000000 # Robot location is not a valid tile

        adjacent_tiles_to_t = self._get_adjacent_tiles(t)
        if not adjacent_tiles_to_t:
            # Target tile has no adjacent tiles defined in static facts
            return 1000000

        # Check if robot is already at a tile adjacent to t
        # This means t is in the adjacency list of loc_r
        if t in self._get_adjacent_tiles(loc_r):
             return 0 # Robot is already adjacent to the target tile

        # If not already adjacent, find the minimum moves to reach an adjacent tile
        min_d = 1000000
        for adj_t in adjacent_tiles_to_t:
            min_d = min(min_d, self._manhattan_distance(loc_r, adj_t))

        return min_d


    def __call__(self, state):
        """
        Step-By-Step Thinking for Computing Heuristic:
        1. Identify goal facts that require tiles to be painted.
        2. Check if any goal tile is currently painted with the wrong color. If so,
           the state is likely unsolvable in this domain, return a large value (infinity).
        3. Identify the set of goal tiles that are not yet painted correctly
           AND are currently clear. These are the tiles that still need painting.
        4. If this set is empty, the goal is reached, return 0.
        5. For each unpainted clear goal tile (tile, color):
           a. Determine the minimum cost for *any* robot to paint this specific tile
              with the required color.
           b. The cost for a robot involves:
              - Cost to change color if the robot doesn't have the required color (1 action).
              - Cost to move from the robot's current location to a tile adjacent
                to the target tile (Manhattan distance). If the robot is already
                adjacent, this cost is 0.
              - Cost of the paint action itself (1 action).
           c. The minimum cost for the tile is the minimum of these costs over all robots.
        6. The total heuristic value is the sum of these minimum costs for each
           unpainted clear goal tile. This relaxation assumes robots can work
           independently on different tiles and doesn't account for potential
           conflicts or optimal robot assignment.

        Assumptions:
        - Tile names follow the format 'tile_row_col' allowing coordinate extraction.
        - The grid structure defined by up/down/left/right facts is consistent.
        - All colors required by the goal are available colors in the domain.
        - A tile painted with the wrong color cannot be repainted.
        - Robot locations and colors are always specified in the state if the robot exists.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer heuristic value estimating the remaining cost.
        """
        # Check for dead ends (wrongly painted tiles)
        for goal_fact in self.goal_facts:
            pred, args = self._parse_fact(goal_fact)
            if pred == 'painted' and len(args) == 2:
                goal_tile, goal_color = args
                # Check if the tile is painted with a different color
                for state_fact in state:
                    state_pred, state_args = self._parse_fact(state_fact)
                    if state_pred == 'painted' and len(state_args) == 2:
                        state_tile, state_color = state_args
                        if state_tile == goal_tile and state_color != goal_color:
                            # Tile is painted with the wrong color - unsolvable state
                            return 1000000 # Return a large value for infinity

        # Identify unpainted clear goal tiles
        unpainted_clear_goal_tiles = [] # List of (tile, color) tuples
        for goal_fact in self.goal_facts:
            pred, args = self._parse_fact(goal_fact)
            if pred == 'painted' and len(args) == 2:
                goal_tile, goal_color = args
                # Check if the goal fact is NOT in the state
                if goal_fact not in state:
                    # Check if the tile is clear in the state
                    if f'(clear {goal_tile})' in state:
                         unpainted_clear_goal_tiles.append((goal_tile, goal_color))
                    # If it's not clear and not painted with the goal color, it must be painted wrong (handled above)

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_clear_goal_tiles:
            return 0

        # Get robot locations and colors from the current state
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for robot_name in self.robots:
            robot_info[robot_name] = {'location': None, 'color': None}
            for fact in state:
                pred, args = self._parse_fact(fact)
                if pred == 'robot-at' and len(args) == 2 and args[0] == robot_name:
                    robot_info[robot_name]['location'] = args[1]
                elif pred == 'robot-has' and len(args) == 2 and args[0] == robot_name:
                    robot_info[robot_name]['color'] = args[1]

        # Calculate heuristic: sum of minimum costs for each unpainted clear goal tile
        total_heuristic = 0
        for goal_tile, goal_color in unpainted_clear_goal_tiles:
            min_cost_to_paint_tile = 1000000 # Initialize with large value

            for robot_name, info in robot_info.items():
                robot_loc = info['location']
                robot_color = info['color']

                # Cannot calculate cost if robot location is unknown (shouldn't happen in valid states)
                if robot_loc is None:
                    continue

                # Cost to get the correct color
                color_cost = 0
                if robot_color != goal_color:
                    # Assumes goal_color is always available if needed
                    color_cost = 1 # Cost of change_color action

                # Cost to move to an adjacent tile
                move_cost = self._min_moves_to_be_adjacent(robot_loc, goal_tile)

                # Total cost for this robot to paint this tile (color change + movement)
                # We add the paint action cost (1) at the end for each tile
                cost_for_robot = color_cost + move_cost

                min_cost_to_paint_tile = min(min_cost_to_paint_tile, cost_for_robot)

            # Add the minimum cost to get a robot ready + the paint action cost (1)
            if min_cost_to_paint_tile >= 1000000: # Check against large number sentinel
                 # If no robot can reach an adjacent tile or get the color, this tile is unreachable
                 return 1000000 # Problem likely unsolvable

            total_heuristic += min_cost_to_paint_tile + 1

        return total_heuristic
