import re
from collections import deque
import logging

from heuristics.heuristic_base import Heuristic
# from task import Task # Task class is used in __init__ for type hinting/context

# Configure logging (optional, but helpful for debugging)
# logging.basicConfig(level=logging.INFO)

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.
    """

    def __init__(self, task):
        """
        Heuristic Initialization:
        Stores the goal facts.
        Parses static facts ('up', 'down', 'left', 'right') to build an
        adjacency list representation of the grid graph and attempts to parse
        tile names into (row, col) coordinates.
        Stores available colors (though not directly used in the current heuristic logic).
        """
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # Data structures for grid graph and tile coordinates
        self.tile_coords = {} # tile_name -> (row, col)
        self.adjacency_list = {} # tile_name -> [adjacent_tile_name, ...]
        self._parse_static_info()

        # Store available colors (not strictly needed for this heuristic logic, but good practice)
        self.available_colors = set()
        for fact in self.static_facts:
            if fact.startswith('(available-color '):
                # Extract color name, removing trailing ')'
                parts = fact.split()
                if len(parts) > 1:
                    color = parts[1][:-1]
                    self.available_colors.add(color)

    def _parse_static_info(self):
        """
        Parses static facts to build the grid graph (adjacency list) and
        map tile names to coordinates based on the 'tile_row_col' pattern.
        """
        tile_names = set()
        # Collect all tile names mentioned in adjacency predicates
        for fact in self.static_facts:
            if fact.startswith('(up ') or fact.startswith('(down ') or \
               fact.startswith('(left ') or fact.startswith('(right '):
                parts = fact.split()
                if len(parts) > 2:
                    # Extract tile names, removing trailing ')' from the second one
                    tile1 = parts[1]
                    tile2 = parts[2][:-1]
                    tile_names.add(tile1)
                    tile_names.add(tile2)

        # Try to parse coordinates from tile names like 'tile_X_Y'
        for tile_name in tile_names:
            match = re.match(r'tile_(\d+)_(\d+)', tile_name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                self.tile_coords[tile_name] = (row, col)
            # Note: If tile names don't match the pattern, tile_coords will be incomplete.
            # The BFS relies only on the adjacency_list, which is built next.

        # Build adjacency list from static facts
        for tile_name in tile_names:
            self.adjacency_list[tile_name] = []

        # Add edges based on directional predicates. Each predicate implies a bidirectional edge.
        for fact in self.static_facts:
            if fact.startswith('(up '):
                parts = fact.split()
                if len(parts) > 2:
                    y, x = parts[1], parts[2][:-1]
                    if x in self.adjacency_list: self.adjacency_list[x].append(y) # Move from x to y (up)
                    if y in self.adjacency_list: self.adjacency_list[y].append(x) # Move from y to x (down)
            elif fact.startswith('(down '):
                parts = fact.split()
                if len(parts) > 2:
                    y, x = parts[1], parts[2][:-1]
                    # Edge (y, x) and (x, y) already added by the corresponding 'up' predicate
                    pass
            elif fact.startswith('(left '):
                parts = fact.split()
                if len(parts) > 2:
                    y, x = parts[1], parts[2][:-1]
                    if x in self.adjacency_list: self.adjacency_list[x].append(y) # Move from x to y (left)
                    if y in self.adjacency_list: self.adjacency_list[y].append(x) # Move from y to x (right)
            elif fact.startswith('(right '):
                parts = fact.split()
                if len(parts) > 2:
                    y, x = parts[1], parts[2][:-1]
                     # Edge (y, x) and (x, y) already added by the corresponding 'left' predicate
                    pass

    def _bfs_distance(self, start_tile, target_tiles):
        """
        Performs BFS on the grid graph to find the minimum distance
        (number of moves) from start_tile to any tile in target_tiles.

        Returns float('inf') if no target tile is reachable from start_tile.
        """
        if start_tile in target_tiles:
            return 0

        # Ensure start_tile is in the graph
        if start_tile not in self.adjacency_list:
             # This tile might not be part of the main grid defined by adjacency facts
             # (e.g., an isolated tile). Cannot reach anything from here.
             return float('inf')

        queue = deque([(start_tile, 0)])
        visited = {start_tile}

        while queue:
            current_tile, dist = queue.popleft()

            # Ensure current_tile is in the graph (should be if start was)
            if current_tile in self.adjacency_list:
                for neighbor_tile in self.adjacency_list[current_tile]:
                    if neighbor_tile not in visited:
                        visited.add(neighbor_tile)
                        if neighbor_tile in target_tiles:
                            return dist + 1 # Found the closest target
                        queue.append((neighbor_tile, dist + 1))

        return float('inf') # No target tile was reached

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic for the floortile domain.

        Summary:
        Estimates the cost to reach the goal state by summing the minimum estimated
        costs for each unsatisfied goal tile. For each unsatisfied goal tile,
        it calculates the minimum cost for any robot to move to an adjacent tile,
        change color if necessary, and paint the tile.

        Assumptions:
        - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
        - Tile names follow the pattern 'tile_row_col' allowing coordinate parsing
          (though coordinate parsing is not strictly required for the BFS part).
        - Robots always hold exactly one color.
        - If a goal tile is painted with the wrong color, the problem is unsolvable.
        - The cost of each action is 1.
        - A tile must be 'clear' to be painted.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify all unsatisfied goal facts of the form '(painted T C)'.
        2. Check for unsolvability: Iterate through all facts in the current state.
           If a fact is '(painted T C')' where '(painted T C)' is an unsatisfied
           goal and C' is different from C, return infinity.
        3. Initialize the total heuristic value h to 0.
        4. Identify the current location and color of each robot from the state.
        5. For each unsatisfied goal fact '(painted T C)':
           a. Check if the tile T is currently 'clear'. If it is not 'clear',
              and it wasn't painted with the wrong color (checked in step 2),
              this state is likely unsolvable as there's no action to make a
              non-clear, non-wrongly-painted tile clear. Return infinity.
           b. Find all tiles adjacent to T using the pre-computed adjacency list.
              If T has no adjacent tiles defined in the static facts, return infinity.
           c. Initialize minimum cost for this tile across all robots to infinity.
           d. For each robot R:
              i. Get the robot's current tile T_R and color C_R.
              ii. Calculate the minimum movement cost from T_R to any tile
                  adjacent to T using BFS on the grid graph. If unreachable,
                  this robot cannot paint the tile.
              iii. Calculate the color change cost: 1 if C_R is not C, 0 otherwise.
              iv. If the movement cost is finite, the estimated cost for robot R
                  to paint tile T is movement_cost + color_change_cost + 1
                  (for the paint action).
              v. Update the minimum cost for tile T with the calculated cost for robot R.
           e. After checking all robots, if the minimum cost for tile T is still
              infinity, it means no robot can reach and paint this clear goal tile.
              The state is unsolvable. Return infinity.
           f. Add the minimum cost for tile T to the total heuristic value h.
        6. Return the total heuristic value h.
        """
        state = node.state
        goals = self.goals

        # Identify current robot states
        robot_states = {} # robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            if fact.startswith('(robot-at '):
                parts = fact.split()
                if len(parts) > 2:
                    robot = parts[1]
                    tile = parts[2][:-1]
                    if robot not in robot_states:
                        robot_states[robot] = {}
                    robot_states[robot]['location'] = tile
            elif fact.startswith('(robot-has '):
                parts = fact.split()
                if len(parts) > 2:
                    robot = parts[1]
                    color = parts[2][:-1]
                    if robot not in robot_states:
                        robot_states[robot] = {}
                    robot_states[robot]['color'] = color

        # Identify painted and clear tiles
        painted_tiles = {} # tile_name -> color
        clear_tiles = set() # tile_name
        for fact in state:
            if fact.startswith('(painted '):
                parts = fact.split()
                if len(parts) > 2:
                    tile = parts[1]
                    color = parts[2][:-1]
                    painted_tiles[tile] = color
            elif fact.startswith('(clear '):
                parts = fact.split()
                if len(parts) > 1:
                    tile = parts[1][:-1]
                    clear_tiles.add(tile)

        # Identify unsatisfied goal facts
        unsatisfied_goals = [] # list of (tile_name, color) tuples
        for goal_fact in goals:
            if goal_fact.startswith('(painted '):
                parts = goal_fact.split()
                if len(parts) > 2:
                    goal_tile = parts[1]
                    goal_color = parts[2][:-1]
                    if goal_fact not in state:
                        unsatisfied_goals.append((goal_tile, goal_color))

        # Check for unsolvability (wrong color painted on a goal tile)
        # Iterate through state facts to find painted tiles
        for painted_fact in state:
             if painted_fact.startswith('(painted '):
                 parts = painted_fact.split()
                 if len(parts) > 2:
                     painted_tile = parts[1]
                     painted_color = parts[2][:-1]

                     # Check if this painted tile is an unsatisfied goal tile
                     for goal_tile, goal_color in unsatisfied_goals:
                         if painted_tile == goal_tile and painted_color != goal_color:
                             # Goal tile is painted with the wrong color
                             return float('inf')

        h_value = 0

        # Calculate cost for each unsatisfied goal tile
        for goal_tile, goal_color in unsatisfied_goals:
            # For an unsatisfied goal (painted T C), T must be clear to be paintable.
            # If T is not clear, and not painted with the wrong color (checked above),
            # it implies an unpaintable state -> unsolvable.
            if goal_tile not in clear_tiles:
                 # This state is likely unsolvable as the tile cannot be painted.
                 return float('inf')

            # Find tiles adjacent to the goal tile
            adjacent_to_goal = set()
            if goal_tile in self.adjacency_list:
                 adjacent_to_goal.update(self.adjacency_list[goal_tile])

            if not adjacent_to_goal:
                 # Should not happen in valid grid problems, but handle defensively
                 logging.warning(f"Goal tile {goal_tile} has no adjacent tiles defined.")
                 return float('inf') # Cannot paint if no adjacent tiles

            min_robot_cost_for_tile = float('inf')

            # Calculate min cost for any robot to paint this tile
            for robot_name, robot_info in robot_states.items():
                robot_location = robot_info.get('location')
                robot_color = robot_info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot state is incomplete (shouldn't happen in valid states)
                    continue

                # Calculate movement cost
                move_cost = self._bfs_distance(robot_location, adjacent_to_goal)

                if move_cost == float('inf'):
                    # Robot cannot reach any tile adjacent to the goal tile
                    continue

                # Calculate color change cost
                color_cost = 1 if robot_color != goal_color else 0

                # Total cost for this robot to paint this tile
                # move_cost: actions to get adjacent
                # color_cost: 1 action to change color if needed
                # paint_cost: 1 action to paint
                robot_total_cost = move_cost + color_cost + 1

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_total_cost)

            if min_robot_cost_for_tile == float('inf'):
                 # If no robot can reach and paint this specific clear goal tile,
                 # the state is unsolvable.
                 return float('inf')

            h_value += min_robot_cost_for_tile

        # If the loop finishes without returning infinity, h_value is the sum
        # of minimum costs for all unsatisfied, clear goal tiles.
        # If unsatisfied_goals was empty, h_value is 0.
        # If unsatisfied_goals was not empty, each tile added at least 1 to h_value.
        # Thus, h_value is 0 iff all goals are satisfied.
        return h_value
