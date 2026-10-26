import re
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    costs related to painting each goal tile that is not yet painted correctly.
    For each such tile, the estimated cost is the minimum number of actions
    required for any robot to reach a state where it can paint that tile.
    Additionally, it adds the cost for changing robot colors, counted once
    for each unique color required by the unpainted goal tiles that no robot
    currently possesses.

    Assumptions:
    - Tile names follow the format 'tile_R_C' where R and C are integers,
      allowing Manhattan distance calculation.
    - The grid structure implied by tile names is navigable (ignoring 'clear'
      constraints for movement estimation, which is a relaxation).
    - A tile painted with the wrong color represents a dead end.
    - Tiles that need painting are initially clear (or become clear via robot movement).
    - The 'free-color' predicate is not relevant for the heuristic calculation.

    Heuristic Initialization:
    The constructor pre-processes the task information to extract the goal
    facts and map tile names to their (row, column) coordinates based on
    the 'tile_R_C' naming convention. This mapping is used later to compute
    Manhattan distances. It iterates through all facts in the task to find
    all tile objects and their coordinates.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to identify:
       - The location of each robot (`robot-at`).
       - The color held by each robot (`robot-has`).
       - Which tiles are currently painted and with what color (`painted`).
    2. Parse the goal facts (`self.goals`) to identify which tiles need to be
       painted and with which color (`painted`). Store these as `goal_tiles`.
    3. Initialize the total heuristic value `h` to 0.
    4. Identify the set of unique colors required by the goal tiles that are
       *not* yet painted correctly in the current state.
    5. Identify the set of unique colors currently held by the robots.
    6. Determine the set of colors from step 4 that are not held by any robot
       (step 5). The size of this set is added to `h` as the estimated cost
       for color changes. This assumes one robot can acquire all needed colors
       sequentially.
    7. Iterate through each goal tile `T` that is not yet painted correctly
       with the required color `C`:
       a. If `T` is currently painted with a color *different* from `C`, the
          state is considered a dead end, and the heuristic returns infinity.
       b. If `T` is not painted with color `C` (either unpainted or painted
          with a different color, handled in 7a), it needs to be painted.
          i. Calculate the minimum cost for any robot to paint tile `T`.
             For each robot R at tile RT, calculate the Manhattan distance
             `dist` between RT and T.
             - If `dist == 0` (robot is on T), the robot must move away (1 action)
               making T clear, then paint from an adjacent tile (1 action). Total 2 actions.
             - If `dist > 0` (robot is not on T), the robot needs `dist - 1` moves
               to reach an adjacent tile, then 1 paint action. Total `(dist - 1) + 1 = dist` actions.
             The cost for robot R to paint T is `2 if dist == 0 else dist`.
             Find the minimum of this cost over all robots.
          ii. Add this minimum painting cost (including movement) to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.tile_coords = self._parse_tile_coordinates(task.facts)

    def _parse_tile_coordinates(self, facts):
        """Parses all facts to find tile names and their coordinates."""
        tile_coords = {}
        tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')
        # task.facts contains all fact names that are valid in the domain,
        # including those in init, goal, and static. This should cover all tiles.
        for fact_string in facts:
            # Extract all potential tokens that might be tile names
            # Use regex to find all occurrences of the tile pattern
            for match in tile_name_pattern.finditer(fact_string):
                 tile_name = match.group(0)
                 try:
                     row, col = int(match.group(1)), int(match.group(2))
                     tile_coords[tile_name] = (row, col)
                 except ValueError:
                     # Should not happen with the regex match, but good practice
                     pass # Ignore tokens that look like tiles but have non-integer coords
        return tile_coords

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding brackets and split by spaces
        # Example: '(robot-at robot1 tile_0_1)' -> ['robot-at', 'robot1', 'tile_0_1']
        # This assumes facts are well-formed with spaces separating elements
        parts = fact_string[1:-1].split()
        return parts

    def _manhattan_distance(self, coords1, coords2):
        """Calculates Manhattan distance between two (row, col) tuples."""
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def __call__(self, node):
        state = node.state

        # 1. Parse current state
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color}
        current_painted = {} # {tile_name: color}

        for fact_string in state:
            parts = self._parse_fact(fact_string)
            if not parts: continue # Handle empty fact string if any

            predicate = parts[0]
            if predicate == 'robot-at':
                # Ensure fact has enough parts before accessing indices
                if len(parts) > 2:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif predicate == 'robot-has':
                 if len(parts) > 2:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
            elif predicate == 'painted':
                 if len(parts) > 2:
                    tile, color = parts[1], parts[2]
                    current_painted[tile] = color
            # Ignore other predicates like 'clear', 'up', etc. for heuristic calculation

        # 2. Parse goal facts
        goal_tiles = {} # {tile_name: color}
        unpainted_goal_tiles_info = {} # {tile_name: color} for tiles needing paint
        colors_needed_by_unpainted_goal_tiles = set()

        for goal_fact_string in self.goals:
             parts = self._parse_fact(goal_fact_string)
             if not parts or parts[0] != 'painted':
                 # Ignore non-painted goal facts if any (domain only has painted goals)
                 continue
             if len(parts) > 2:
                 tile, goal_color = parts[1], parts[2]
                 goal_tiles[tile] = goal_color # Store all goal tiles

                 # 7a. Check if tile is painted with the wrong color
                 if tile in current_painted:
                     if current_painted[tile] != goal_color:
                         return float('inf') # Dead end
                     # Else: painted correctly, this goal fact is satisfied
                 else:
                     # 7b. Tile needs painting
                     unpainted_goal_tiles_info[tile] = goal_color
                     colors_needed_by_unpainted_goal_tiles.add(goal_color)


        # 3. Initialize heuristic
        h = 0

        # 5. Identify colors robots have
        colors_robots_have = set(robot_colors.values())

        # 6. Cost for color changes
        colors_to_acquire = colors_needed_by_unpainted_goal_tiles - colors_robots_have
        h += len(colors_to_acquire) # Add cost for acquiring colors

        # 7. Iterate through unpainted goal tiles
        if not robot_locations and unpainted_goal_tiles_info:
             # If there are tiles to paint but no robots, it's impossible
             return float('inf')

        for tile, goal_color in unpainted_goal_tiles_info.items():
            # 7b.i. Calculate minimum cost for any robot to paint this tile
            min_cost_to_paint_this_tile = float('inf')
            target_coords = self.tile_coords.get(tile)

            if target_coords is None:
                 # Goal tile not found in parsed coordinates - likely invalid problem
                 return float('inf')

            for robot, robot_tile in robot_locations.items():
                robot_coords = self.tile_coords.get(robot_tile)
                if robot_coords is None:
                    # Robot on a tile not found in parsed coordinates - likely invalid state
                    return float('inf')

                dist = self._manhattan_distance(robot_coords, target_coords)

                # Cost for this robot to paint tile T:
                # 2 actions if robot is on T (move away + paint)
                # dist actions if robot is not on T (dist-1 moves + paint)
                cost_for_this_robot = 2 if dist == 0 else dist

                min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)

            # 7b.ii. Add minimum painting cost to h
            # If min_cost_to_paint_this_tile is still inf, it means no robots could reach it
            # (this case should be covered by the check before the loop if robot_locations is empty)
            # but as a safeguard:
            if min_cost_to_paint_this_tile == float('inf'):
                 return float('inf')

            h += min_cost_to_paint_this_tile

        # 8. Return total heuristic value
        return h
