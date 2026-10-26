# Required imports
import re
import math # For float('inf')

# Assume the Task class definition from <code-file-task> is available in the environment
# from task import Task # Not needed in the final output, but for context

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up
    several components for each goal tile that is not yet painted with the correct color:
    1. A base cost of 1 for the paint action itself.
    2. A cost of 1 for each distinct color required by unpainted goal tiles that are
       not currently held by any robot (representing color change actions).
    3. A movement and clearing cost:
       - If the goal tile is currently occupied by a robot, the cost is 1 (to move the robot off).
         This single move also places the robot on an adjacent tile, satisfying the adjacency
         precondition for painting.
       - If the goal tile is clear, the cost is the minimum Manhattan distance from any robot
         to any tile adjacent to the goal tile. This estimates the moves needed to get a
         robot into a painting position.

    Assumptions:
    - Tile names follow the pattern 'tile_R_C' where R and C are integers,
      allowing coordinate extraction for Manhattan distance calculation.
    - The grid structure implied by 'up', 'down', 'left', 'right' predicates
      corresponds to a standard grid where Manhattan distance is a reasonable
      approximation of shortest path moves.
    - Goal tiles are initially either clear or painted with the correct color.
      (The heuristic assumes unpainted goal tiles are either clear or occupied by a robot).
    - The problem is solvable (e.g., there are robots, available colors, etc., and the grid is connected).

    Heuristic Initialization:
    The constructor parses the static facts from the task definition.
    - It identifies all goal conditions (which tiles need which color).
    - It builds a mapping from tile names ('tile_R_C') to (row, col) coordinates
      by parsing the tile names. This is used for Manhattan distance calculation.
    - It builds an adjacency map between tiles based on the 'up', 'down',
      'left', 'right' static predicates. This map is used to find adjacent tiles
      for movement cost calculation. It also collects all unique tile names.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h` to 0.
    2. Initialize sets to track `needed_colors` (distinct colors required by unpainted goals but not held by any robot) and `unpainted_goal_tiles` (set of (tile, color) tuples).
    3. Parse the current state to find:
       - Robot positions (`robot_positions`: map robot name to tile name).
       - Robot held colors (`robot_colors`: map robot name to color).
       - Clear tiles (`clear_tiles`: set of tile names).
       - Painted tiles (`painted_tiles`: set of (tile, color) tuples).
    4. Iterate through the precomputed `self.goal_conditions`. For each `(goal_tile, goal_color)`:
       - If `(goal_tile, goal_color)` is not present in `painted_tiles`:
         - Add `(goal_tile, goal_color)` to `unpainted_goal_tiles`.
         - Check if any robot in `robot_colors` has `goal_color`. If not, add `goal_color` to `needed_colors`.
    5. If `unpainted_goal_tiles` is empty, the state is a goal state, return 0.
    6. Add the number of `unpainted_goal_tiles` to `h`. This accounts for the paint actions (cost 1 each).
    7. Add the number of distinct `needed_colors` to `h`. This accounts for the color change actions (cost 1 each).
    8. Iterate through the `unpainted_goal_tiles` again. For each `(goal_tile, goal_color)`:
       - Determine the movement/clearing cost for this specific tile:
         - Check if `goal_tile` is in `clear_tiles`.
         - If `goal_tile` is NOT in `clear_tiles` (meaning a robot is on it):
           - The cost is 1 (to move the robot off). Add 1 to `h`.
         - If `goal_tile` IS in `clear_tiles`:
           - Need to find the minimum moves for *any* robot to reach *any* tile adjacent to `goal_tile`.
           - Get the set of adjacent tiles for `goal_tile` using the precomputed `self.adjacency` map.
           - Initialize `min_dist_to_adj_for_any_robot` to infinity.
           - Check if there are any robots or adjacent tiles. If not, the goal is unreachable from this state.
           - If not robot_positions or not adjacent_tiles:
                return math.inf

           - For each robot and its position (`r_pos`) in `robot_positions.items()`:
             # Ensure robot position is a known tile
             if r_pos not in self.all_tiles:
                  # This indicates an inconsistency in the problem definition or state
                  # For robustness, return infinity.
                  return math.inf

             for adj_tile in adjacent_tiles:
                  # Ensure adjacent tile is a known tile
                  if adj_tile not in self.all_tiles:
                       # This indicates an inconsistency in the problem definition
                       # For robustness, return infinity.
                       return math.inf

                  dist = self.manhattan_distance(r_pos, adj_tile)
                  min_dist_to_adj_for_any_robot = min(min_dist_to_adj_for_any_robot, dist)

           if min_dist_to_adj_for_any_robot != math.inf:
               h += min_dist_to_adj_for_any_robot
           else:
               # This happens if robots are on isolated tiles and cannot reach adjacent tiles
               return math.inf

    9. Return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic. Precomputes static information.

        Args:
            task: The planning task object (instance of Task class).
        """
        self.task = task
        self.goal_conditions = set()
        self.tile_coords = {} # Map tile name string to (row, col) tuple
        self.adjacency = {} # Map tile name to set of adjacent tile names
        self.all_tiles = set() # Set of all tile names

        # Extract goal conditions
        for goal_fact in task.goals:
            # Goal facts are like '(painted tile_R_C color)'
            parts = goal_fact.strip("()").split()
            if len(parts) == 3 and parts[0] == 'painted':
                tile_name = parts[1]
                color_name = parts[2]
                self.goal_conditions.add((tile_name, color_name))
                self.all_tiles.add(tile_name) # Add goal tiles to all_tiles

        # Extract tile names and build adjacency from static facts
        for fact in task.static:
             parts = fact.strip("()").split()
             if len(parts) >= 3 and parts[0] in ['up', 'down', 'left', 'right']:
                 # Predicate is (dir tile1 tile2)
                 tile1 = parts[1]
                 tile2 = parts[2]
                 self.all_tiles.add(tile1)
                 self.all_tiles.add(tile2)

                 # Build adjacency map (undirected graph)
                 self.adjacency.setdefault(tile1, set()).add(tile2)
                 self.adjacency.setdefault(tile2, set()).add(tile1)

        # Attempt to parse coordinates for all found tiles
        tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
        for tile_name in self.all_tiles:
            match = tile_pattern.match(tile_name)
            if match:
                r, c = int(match.group(1)), int(match.group(2))
                self.tile_coords[tile_name] = (r, c)
            # Note: If a tile name doesn't match tile_R_C, its coordinates won't be stored.
            # This will cause manhattan_distance to return inf for that tile.
            # This is acceptable if tile_R_C is guaranteed for all relevant tiles (like goal tiles).

        # Basic check: Ensure goal tiles have coordinates parsed
        for tile_name, _ in self.goal_conditions:
            if tile_name not in self.tile_coords:
                 # This indicates a potential issue with tile naming or parsing assumption
                 # For robustness, one might implement BFS here if coordinates fail,
                 # but for this problem, tile_R_C seems reliable based on examples.
                 pass


    def get_coords(self, tile_name):
        """Helper to get coordinates for a tile name."""
        return self.tile_coords.get(tile_name) # Returns None if not found

    def manhattan_distance(self, tile_name1, tile_name2):
        """Calculates Manhattan distance between two tiles using precomputed coordinates."""
        coords1 = self.get_coords(tile_name1)
        coords2 = self.get_coords(tile_name2)
        if coords1 is None or coords2 is None:
            # If coordinates are missing for any tile, distance is unknown/infinite
            return math.inf
        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            An integer estimate of the cost to reach the goal.
        """
        h = 0
        needed_colors = set()
        robot_positions = {} # Map robot name to tile name
        robot_colors = {} # Map robot name to color name
        clear_tiles = set() # Set of clear tiles
        painted_tiles = set() # Set of (tile, color) tuples that are painted

        # Parse current state facts
        for fact in state:
            parts = fact.strip("()").split()
            if not parts: continue # Skip empty strings from split

            if parts[0] == 'robot-at' and len(parts) == 3:
                robot_name = parts[1]
                tile_name = parts[2]
                robot_positions[robot_name] = tile_name
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot_name = parts[1]
                color_name = parts[2]
                robot_colors[robot_name] = color_name
            elif parts[0] == 'clear' and len(parts) == 2:
                tile_name = parts[1]
                clear_tiles.add(tile_name)
            elif parts[0] == 'painted' and len(parts) == 3:
                tile_name = parts[1]
                color_name = parts[2]
                painted_tiles.add((tile_name, color_name))

        unpainted_goal_tiles = set()

        # Identify unpainted goal tiles and required colors
        for goal_tile, goal_color in self.goal_conditions:
            if (goal_tile, goal_color) not in painted_tiles:
                unpainted_goal_tiles.add((goal_tile, goal_color))

                # Check if the required color is held by any robot
                has_color_robot = False
                for robot, color in robot_colors.items():
                    if color == goal_color:
                        has_color_robot = True
                        break
                if not has_color_robot:
                    needed_colors.add(goal_color)

        # If all goal tiles are painted correctly, we are in a goal state
        if not unpainted_goal_tiles:
            return 0

        # Heuristic calculation components:

        # 1. Cost for painting each unpainted goal tile
        h += len(unpainted_goal_tiles)

        # 2. Cost for changing color for each needed color not held by any robot
        h += len(needed_colors)

        # 3. Movement/Clearing cost for each unpainted goal tile
        for goal_tile, _ in unpainted_goal_tiles:
            # Find the robot currently on this tile, if any
            robot_on_tile = None
            # Iterate through robot positions to find if any robot is on goal_tile
            for robot, r_pos in robot_positions.items():
                if r_pos == goal_tile:
                    robot_on_tile = robot
                    break

            if robot_on_tile:
                # Tile is occupied. Cost to move robot off = 1.
                # After moving off, the robot is on an adjacent tile.
                # So, the cost to clear AND get a robot adjacent is 1.
                h += 1
            else:
                # Tile is clear. Need a robot to move adjacent.
                # Find the minimum moves for any robot to reach *any* adjacent tile.
                min_dist_to_adj_for_any_robot = math.inf
                adjacent_tiles = self.adjacency.get(goal_tile, set())

                # Check if the goal is unreachable from this state
                if not robot_positions or not adjacent_tiles:
                     # No robots or goal tile has no neighbors
                     return math.inf

                for robot, r_pos in robot_positions.items():
                    # Ensure robot position is a known tile
                    if r_pos not in self.all_tiles:
                         # This indicates an inconsistency in the problem definition or state
                         # For robustness, return infinity.
                         return math.inf

                    for adj_tile in adjacent_tiles:
                         # Ensure adjacent tile is a known tile
                         if adj_tile not in self.all_tiles:
                              # This indicates an inconsistency in the problem definition
                              # For robustness, return infinity.
                              return math.inf

                         dist = self.manhattan_distance(r_pos, adj_tile)
                         min_dist_to_adj_for_any_robot = min(min_dist_to_adj_for_any_robot, dist)

                if min_dist_to_adj_for_any_robot != math.inf:
                    h += min_dist_to_adj_for_any_robot
                else:
                    # This happens if robots are on isolated tiles and cannot reach adjacent tiles
                    return math.inf

    # The heuristic is already checked for the goal state at the beginning.
    # If we reach here, unpainted_goal_tiles is not empty.
    # The total h is accumulated in the loop.
    # No need for a final check here, as the check at the start handles the goal state.
    # If unpainted_goal_tiles was not empty, h will be > 0.
        return h
