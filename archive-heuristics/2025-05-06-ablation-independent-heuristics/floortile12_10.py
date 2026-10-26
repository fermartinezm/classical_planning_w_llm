import math
from heuristics.heuristic_base import Heuristic
from task import Task # Used for type hinting and accessing task structure

class floortileHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the floortile domain. It estimates the cost
    to reach the goal by summing up the estimated costs for each unsatisfied
    goal tile. The estimated cost for a single unsatisfied goal tile is the sum
    of the paint action cost (1) and the minimum cost for any robot to get
    the required color and reach a tile adjacent to the target tile. The cost
    for a robot to get the color is 1 if it doesn't currently have it, and 0
    otherwise. The cost to reach the adjacent tile is estimated using Manhattan
    distance on the grid.

    Assumptions:
    1. The tile names are in the format 'tile_row_col', allowing extraction of
       grid coordinates (row, col).
    2. The grid structure defined by 'up', 'down', 'left', 'right' predicates
       corresponds to a grid where Manhattan distance is a reasonable estimate
       for movement cost between clear tiles.
    3. Solvable problem instances do not require painting a tile that is
       already painted with the wrong color. If an unsatisfied goal tile is
       not clear, the heuristic returns infinity.
    4. Robots always have exactly one color.
    5. All tiles, robots, and colors are declared in the task's objects.

    Heuristic Initialization:
    In the constructor, the heuristic processes the static facts and task objects
    to build data structures needed for efficient computation:
    - It identifies all robots, colors, and tiles from the task's objects.
    - It parses the tile names to build a coordinate map (row, col) for each tile.
    - it parses the 'up', 'down', 'left', 'right' static facts to build an
      adjacency map for tiles. This allows for quick lookup of adjacent
      tiles and calculation of Manhattan distances.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all goal facts of the form '(painted ?tile ?color)'.
    2. Filter these goal facts to find the 'UnsatisfiedGoals' - those that are
       not true in the current state.
    3. If 'UnsatisfiedGoals' is empty, the goal is reached, return 0.
    4. Extract the current location and color for each robot from the state.
    5. Check for impossible states: For each '(painted T C)' in 'UnsatisfiedGoals',
       check if '(clear T)' is true in the state. If not, the tile is painted
       with something else (or blocked), and we assume this means the goal is
       unreachable in solvable instances, so return infinity.
    6. Initialize the heuristic value `h = 0`.
    7. Iterate through each unsatisfied goal '(painted T C)' in 'UnsatisfiedGoals':
       a. This tile needs to be painted, which costs at least 1 action. Add 1 to `h`.
       b. This tile needs a robot with color C to be adjacent to it. Find the
          minimum cost for *any* robot R to achieve this state (being adjacent
          to T with color C).
          - Calculate the cost for robot R:
            - Color cost: 1 if R does not have color C in the current state, 0 otherwise.
            - Movement cost: Calculate the minimum Manhattan distance from R's
              current location to *any* tile X that is adjacent to T. Use the
              precomputed coordinate and adjacency maps.
            - Total cost for robot R for this tile = Color cost + Movement cost.
          - Find the minimum of this total cost over all robots R.
       c. Add this minimum robot-specific cost (color + movement) to `h`.
    8. Return the final calculated value of `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.robots = set()
        self.colors = set()
        self.tiles = set()
        self.coords = {} # tile_name -> (row, col)
        self.adj = {} # tile_name -> set of adjacent tile_names

        # Identify all robots, colors, tiles from task objects
        for obj_name, obj_type in task.objects.items():
            if obj_type == 'robot':
                self.robots.add(obj_name)
            elif obj_type == 'color':
                self.colors.add(obj_name)
            elif obj_type == 'tile':
                self.tiles.add(obj_name)
                self._parse_tile_coords(obj_name) # Parse coords for all tiles

        # Build adjacency map for all identified tiles
        for tile in self.tiles:
            self.adj[tile] = set()

        # Add adjacencies from static facts
        for fact_str in self.static:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            args = parts[1:]
            if predicate in ['up', 'down', 'left', 'right']:
                t1, t2 = args
                # Only add if both are known tiles (from task objects)
                if t1 in self.tiles and t2 in self.tiles:
                    self.adj[t1].add(t2)
                    self.adj[t2].add(t1)

    def _parse_tile_coords(self, tile_name):
        """Parses tile name like 'tile_row_col' to get coordinates (row, col)."""
        if tile_name in self.coords:
            return # Already parsed

        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                self.coords[tile_name] = (row, col)
            except ValueError:
                # Ignore tiles that don't match the pattern
                pass

    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles using precomputed coordinates."""
        if tile1_name not in self.coords or tile2_name not in self.coords:
            # Should not happen for tiles identified from task objects if names are consistent
            return float('inf') # Cannot calculate distance

        r1, c1 = self.coords[tile1_name]
        r2, c2 = self.coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """Computes the domain-dependent heuristic value for the given state."""
        state = node.state
        h = 0

        # 1. Identify unsatisfied goal facts (painted ?tile ?color)
        unsatisfied_goals = set()
        for goal_fact in self.goals:
            # Check if it's a painted goal fact
            if goal_fact.startswith('(painted '):
                 # Extract tile and color from goal fact string
                 parts = goal_fact.strip('()').split()
                 if len(parts) == 3: # Should be '(painted tile color)'
                     tile = parts[1]
                     color = parts[2]
                     # Check if the goal fact is NOT in the current state
                     if goal_fact not in state:
                         unsatisfied_goals.add((tile, color))

        # If no unsatisfied goals, the goal is reached
        if not unsatisfied_goals:
            return 0

        # Extract current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact_str in state:
            if fact_str.startswith('(robot-at '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif fact_str.startswith('(robot-has '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        # Check for impossible states (goal tile not clear)
        for tile, color in unsatisfied_goals:
            # If the goal is (painted T C) and (painted T C') is in state for C' != C,
            # or if T is blocked in some other way, it's unsolvable.
            # The domain implies a tile is either clear or painted.
            # If (painted T C) is a goal and not in state, then either (clear T) is in state
            # or (painted T C') for C' != C is in state.
            # We check if (clear T) is NOT in state. If so, it must be painted with the wrong color.
            if f'(clear {tile})' not in state:
                 return float('inf') # Tile is painted with wrong color or blocked

        # Calculate heuristic based on unsatisfied goals
        for tile_to_paint, needed_color in unsatisfied_goals:
            # Cost for this tile = 1 (paint action) + min_cost_for_robot_to_be_ready

            min_robot_ready_cost = float('inf')

            # Find adjacent tiles for the tile that needs painting
            target_adjacent_tiles = self.adj.get(tile_to_paint, set())

            # If the tile doesn't have adjacent tiles in our map, it cannot be painted.
            if not target_adjacent_tiles:
                 return float('inf') # Tile is isolated or not in the adjacency graph

            # If there are no robots but there are unsatisfied goals, it's unsolvable
            if not self.robots:
                 return float('inf')

            for robot in self.robots:
                if robot not in robot_locations:
                    # Robot location not known in state? Should not happen in valid states.
                    continue

                current_robot_location = robot_locations[robot]
                current_robot_color = robot_colors.get(robot) # Get color, assumes robot has one

                # Cost for this robot to get the needed color
                color_cost = 0
                if current_robot_color != needed_color:
                    color_cost = 1 # Assumes robot has some other color and can change

                # Cost for this robot to move to an adjacent tile
                min_move_cost_to_adjacent = float('inf')
                for adj_tile in target_adjacent_tiles:
                    move_cost = self._manhattan_distance(current_robot_location, adj_tile)
                    min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, move_cost)

                # Total cost for this robot to be ready to paint this tile
                # If min_move_cost_to_adjacent is inf (e.g., isolated tile or parsing error), this robot cannot reach it.
                if min_move_cost_to_adjacent != float('inf'):
                     robot_ready_cost = color_cost + min_move_cost_to_adjacent
                     min_robot_ready_cost = min(min_robot_ready_cost, robot_ready_cost)

            # If after checking all robots, none can reach an adjacent tile for this goal tile
            if min_robot_ready_cost == float('inf'):
                 return float('inf') # Tile is unreachable by any robot

            # Add cost for this tile: 1 (paint) + minimum cost for a robot to be ready
            h += 1 + min_robot_ready_cost

        return h
