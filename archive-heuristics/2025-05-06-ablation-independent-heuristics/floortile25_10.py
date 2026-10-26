# Need to import Heuristic from heuristics.heuristic_base
# Need to import collections for deque and defaultdict
from heuristics.heuristic_base import Heuristic
import collections

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
        Estimates the cost to reach the goal state by summing the estimated
        costs for each goal tile that is not yet painted correctly. For each
        such tile, it calculates the minimum cost among all robots to reach
        an adjacent tile with the required color and perform the paint action.
        The cost for a single tile/robot/adjacent_tile combination is estimated
        as: (cost to change robot color if needed) + (grid distance from robot
        to adjacent tile) + 1 (for the paint action). The minimum of this value
        over all robots and all adjacent tiles is taken as the cost for that
        goal tile.

    Assumptions:
        - The grid structure (adjacency) is fully defined by the static
          predicates (up, down, left, right).
        - All colors required by the goal are available (available-color is true).
        - If a tile is painted with a color different from the goal color for
          that tile, the state is a dead end (no unpaint action exists).
        - The heuristic calculates movement cost based on the static grid
          distance (BFS), ignoring the 'clear' precondition for intermediate
          tiles on the path. This is a relaxation.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic preprocesses the task
        information:
        - It builds an adjacency list representation of the tile grid from
          the static 'up', 'down', 'left', 'right' facts.
        - It extracts the goal requirements, storing a mapping from goal tile
          names to their required colors.
        - It identifies all robot names and available colors from the initial
          state and static facts.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Initialize the total heuristic value `h` to 0.
        2.  Parse the current state to determine the location and color of each robot.
        3.  Parse the current state to determine the status of each tile (clear or painted with which color).
        4.  Iterate through each tile specified in the goal and its required color.
        5.  For the current goal tile and required color:
            a.  Check if the tile is already painted with the required color in the current state. If yes, this goal fact is satisfied; continue to the next goal tile.
            b.  Check if the tile is painted with a *different* color in the current state. If yes, the state is a dead end; return `float('inf')`.
            c.  (If neither of the above, the tile must be clear and needs painting). Calculate the minimum cost to paint this tile. Initialize `min_cost_for_this_tile` to `float('inf')`.
            d.  Find all tiles adjacent to the current goal tile using the preprocessed grid information.
            e.  For each robot:
                i.  Get the robot's current location and color from the state information.
                ii. Calculate the cost to acquire the required color: 0 if the robot already has it, 1 if it needs to change color (assuming the required color is available).
                iii. Calculate the minimum cost to move the robot from its current location to *any* of the adjacent tiles found in step 5d. This is done using BFS on the preprocessed static grid. Initialize `min_move_cost` to `float('inf')`.
                iv. For each adjacent tile: calculate the BFS distance from the robot's current location to the adjacent tile. Update `min_move_cost` with the minimum distance found.
                v.  If a path exists (`min_move_cost` is not `float('inf')`), calculate the total cost for this robot to paint the tile: `color_cost + min_move_cost + 1` (the +1 is for the paint action itself).
                vi. Update `min_cost_for_this_tile` with the minimum cost found among all robots considered so far for this goal tile.
            f.  If, after considering all robots, `min_cost_for_this_tile` is still `float('inf')`, it means no robot can reach an adjacent tile to paint the goal tile; return `float('inf')`.
            g.  Add `min_cost_for_this_tile` to the total heuristic value `h`.
        6.  After iterating through all goal tiles, return the total heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_tiles = {} # Map tile -> required_color
        self.grid_adj = collections.defaultdict(list) # Map tile -> list of adjacent tiles
        self.all_tiles = set()
        self.all_robots = set()
        self.available_colors = set()

        # Preprocess static facts and goal
        for fact_string in task.static:
            pred, args = self.parse_fact(fact_string)
            if pred in ['up', 'down', 'left', 'right'] and len(args) == 2:
                # Args are [tile1, tile2] meaning tile1 is adjacent to tile2
                # Add bidirectional edges
                t1, t2 = args
                self.grid_adj[t1].append(t2)
                self.grid_adj[t2].append(t1)
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)
            elif pred == 'available-color' and len(args) == 1:
                self.available_colors.add(args[0])

        # Preprocess goal facts
        for fact_string in task.goals:
            pred, args = self.parse_fact(fact_string)
            if pred == 'painted' and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color
                self.all_tiles.add(tile) # Ensure goal tiles are in all_tiles

        # Identify all robots and tiles from initial state facts
        for fact_string in task.initial_state:
             pred, args = self.parse_fact(fact_string)
             if pred == 'robot-at' and len(args) == 2:
                 robot, tile = args
                 self.all_robots.add(robot)
                 self.all_tiles.add(tile)
             elif pred == 'robot-has' and len(args) == 2:
                 robot, color = args
                 self.all_robots.add(robot)
             elif pred == 'clear' and len(args) == 1:
                 tile = args[0]
                 self.all_tiles.add(tile)
             elif pred == 'painted' and len(args) == 2:
                 tile, color = args
                 self.all_tiles.add(tile)


        # Ensure all tiles mentioned in goal or initial state are in the grid_adj keys,
        # even if they have no neighbors in static (unlikely but safe)
        for tile in self.all_tiles:
             if tile not in self.grid_adj:
                 self.grid_adj[tile] = [] # Initialize empty list if no neighbors found in static


    def parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Remove parentheses and split by space
        parts = fact_string.strip()[1:-1].split() # Use strip() just in case
        if not parts:
            return None, [] # Handle empty fact string case defensively
        return parts[0], parts[1:] # predicate, args

    def bfs_distance(self, start_node, end_node):
        """Calculates shortest path distance between two tiles using BFS on the grid."""
        if start_node == end_node:
            return 0
        # Check if nodes exist in the graph
        # If start_node or end_node are not in the grid_adj keys, it means they
        # were not connected to anything via up/down/left/right facts.
        # This implies they are unreachable from/to the main grid.
        if start_node not in self.grid_adj or end_node not in self.grid_adj:
             return float('inf')

        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            current_node, dist = queue.popleft()
            # Check if current_node has neighbors in the graph
            if current_node in self.grid_adj: # This check is redundant due to the initial check, but harmless
                for neighbor in self.grid_adj[current_node]:
                    if neighbor == end_node:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf') # Not reachable

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state node.
        """
        state = node.state
        h = 0

        # Extract current robot info
        robot_info = {} # Map robot -> {'location': tile, 'color': color}
        for robot_name in self.all_robots:
             robot_info[robot_name] = {'location': None, 'color': None} # Initialize

        for fact_string in state:
            pred, args = self.parse_fact(fact_string)
            if pred == 'robot-at' and len(args) == 2:
                robot, tile = args
                if robot in robot_info:
                    robot_info[robot]['location'] = tile
            elif pred == 'robot-has' and len(args) == 2:
                robot, color = args
                if robot in robot_info:
                    robot_info[robot]['color'] = color

        # Extract current tile info
        tile_info = {} # Map tile -> 'clear' or 'painted_color'
        # Initialize all known tiles as potentially clear if not painted
        for tile in self.all_tiles:
             tile_info[tile] = 'clear' # Default assumption if not explicitly painted

        for fact_string in state:
            pred, args = self.parse_fact(fact_string)
            if pred == 'painted' and len(args) == 2:
                tile, color = args
                tile_info[tile] = f'painted_{color}'
            # 'clear' facts are implicitly handled by the default initialization
            # and overridden by 'painted' facts if present.

        # Calculate heuristic for each goal tile
        for goal_tile, required_color in self.goal_tiles.items():
            current_tile_state = tile_info.get(goal_tile, 'clear') # Get state, default to clear if tile not mentioned

            # Check if goal is already satisfied for this tile
            if current_tile_state == f'painted_{required_color}':
                continue # This goal tile is satisfied

            # Check for dead end (painted wrong color)
            if current_tile_state.startswith('painted_') and current_tile_state != f'painted_{required_color}':
                 # Tile is painted with the wrong color, and there's no unpaint action.
                 # This state cannot reach the goal.
                 return float('inf')

            # Tile needs painting (it must be clear)
            # Find the minimum cost to paint this tile with the required_color
            min_cost_for_this_tile = float('inf')

            # Find adjacent tiles for the goal_tile
            adjacent_tiles = self.grid_adj.get(goal_tile, [])

            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles in the grid? Should not happen in valid problems.
                 # Treat as unreachable for safety.
                 return float('inf')

            # If there are no robots at all, this goal tile is unreachable
            if not self.all_robots:
                 return float('inf')

            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot info incomplete? Should not happen in valid states.
                    # Or robot is not on the grid? BFS will handle this.
                    continue # Skip this robot if its location/color is unknown

                # Cost to get the required color
                color_cost = 0
                if robot_color != required_color:
                    # Need to change color. Requires the required_color to be available.
                    # We assume available_color facts are in static and checked in init.
                    # If the required color is not available, this goal is unreachable.
                    # However, the problem guarantees solvable states have finite heuristic.
                    # So we assume required colors are available if they are goal colors.
                    color_cost = 1 # Cost of change_color action

                # Cost to move to an adjacent tile
                min_move_cost = float('inf')
                for adj_tile in adjacent_tiles:
                    dist = self.bfs_distance(robot_location, adj_tile)
                    min_move_cost = min(min_move_cost, dist)

                # Total cost for this robot to paint this tile
                if min_move_cost != float('inf'):
                    robot_paint_cost = color_cost + min_move_cost + 1 # +1 for the paint action
                    min_cost_for_this_tile = min(min_cost_for_this_tile, robot_paint_cost)

            # If no robot can paint this tile (cannot reach adjacent or no robots exist),
            # the goal is unreachable.
            if min_cost_for_this_tile == float('inf'):
                return float('inf')

            h += min_cost_for_this_tile

        return h
