# Required imports
from heuristics.heuristic_base import Heuristic
import re
import collections
import math # For float('inf')

# Define the heuristic class
class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing the minimum
    estimated costs for each unsatisfied goal painted tile. For a tile that
    needs to be painted with a specific color, the estimated cost is 1 (for
    the paint action) plus the minimum cost for any robot to reach an adjacent
    tile with the required color. The cost for a robot to reach an adjacent
    tile is the shortest path distance (BFS) on the tile grid. The cost for a
    robot to get the correct color is 0 if it already has it, or 1 if it needs
    to change color. The heuristic returns infinity if a goal tile is painted
    with the wrong color or cannot be painted (e.g., not clear).

    Assumptions:
    - Tile names are in the format 'tile_row_col' where row and col are integers.
    - The 'up', 'down', 'left', 'right' predicates define a grid structure.
    - Each robot always holds exactly one color.
    - Tiles are only painted if they are clear.
    - Tiles painted with the wrong color cannot be repainted (dead end).
    - A tile is either clear or painted with exactly one color.

    Heuristic Initialization:
    The constructor parses the static facts to build a representation of the
    tile grid. It extracts the adjacency information between tiles based on
    'up', 'down', 'left', 'right' predicates and stores it in an adjacency
    dictionary. It also extracts the goal facts, specifically the required
    '(painted tile color)' predicates. It also infers coordinates for tiles
    from their names for potential future use (though BFS distance is used here).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify all goal facts of the form '(painted tile color)'. Store them
       as a dictionary mapping tile to goal color.
    3. Initialize the total heuristic value to 0.
    4. Extract current robot locations, robot colors, currently painted tiles
       (with their colors), and currently clear tiles from the state.
    5. For each goal tile and its required goal color (goal_tile, goal_color)
       from the goal painted facts:
        a. Check if '(painted goal_tile goal_color)' is true in the current state.
           If yes, this goal is satisfied for this tile; continue to the next
           goal tile.
        b. Check if goal_tile is currently painted with a *different* color.
           This is true if goal_tile is a key in the current painted dictionary
           and the value is not goal_color. If yes, the tile is painted
           incorrectly; return float('inf') as it's likely a dead end.
        c. If the tile is not painted with the goal color, it needs painting.
           Check if '(clear goal_tile)' is true in the current state. If not,
           the tile cannot be painted (it's not clear and not painted with the
           goal color, which implies it's painted with the wrong color, already
           handled, or some other unpaintable state). Return float('inf').
        d. If the tile needs painting and is clear, calculate the minimum cost
           for any robot to paint it:
            i. Find all tiles adjacent to goal_tile using the precomputed
               adjacency information. If goal_tile has no adjacent tiles,
               return float('inf') (unreachable).
            ii. Initialize minimum robot cost for this tile (min_robot_cost_for_tile)
                to float('inf').
            iii. For each robot (robot_name, robot_location) and its color
                 (robot_color) from the current state:
                - Calculate color change cost (color_cost): 0 if robot_color
                  is the same as goal_color, otherwise 1.
                - Calculate minimum movement cost (move_cost): the shortest
                  path distance (using BFS on the tile grid) from the robot's
                  current_location to any tile in the set of adjacent tiles
                  found in step 5.d.i.
                - If move_cost is not float('inf') (i.e., an adjacent tile is
                  reachable):
                    - Calculate the total cost for this robot to paint the tile:
                      Cost_R = color_cost + move_cost + 1 (for the paint action).
                    - Update min_robot_cost_for_tile = min(min_robot_cost_for_tile, Cost_R).
            iv. After checking all robots, if min_robot_cost_for_tile is still
                float('inf'), it means no robot can reach an adjacent tile to
                paint the goal_tile. Return float('inf').
            v. Add min_robot_cost_for_tile to the total heuristic value.
    6. Return the total heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Data structures to build from static info
        self.adjacency = collections.defaultdict(list)
        self.tile_coords = {} # Map tile name to (row, col)
        self.available_colors = set()

        # Parse static facts to build adjacency and infer coordinates
        all_tiles_in_static = set()
        for fact_str in self.static:
            if fact_str.startswith('(up '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    tile2, tile1 = parts[1], parts[2] # tile2 is up from tile1
                    self.adjacency[tile1].append(tile2)
                    self.adjacency[tile2].append(tile1) # Adjacency is bidirectional for movement
                    all_tiles_in_static.add(tile1)
                    all_tiles_in_static.add(tile2)
            elif fact_str.startswith('(down '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    tile2, tile1 = parts[1], parts[2] # tile2 is down from tile1
                    self.adjacency[tile1].append(tile2)
                    self.adjacency[tile2].append(tile1)
                    all_tiles_in_static.add(tile1)
                    all_tiles_in_static.add(tile2)
            elif fact_str.startswith('(left '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    tile2, tile1 = parts[1], parts[2] # tile2 is left from tile1
                    self.adjacency[tile1].append(tile2)
                    self.adjacency[tile2].append(tile1)
                    all_tiles_in_static.add(tile1)
                    all_tiles_in_static.add(tile2)
            elif fact_str.startswith('(right '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3:
                    tile2, tile1 = parts[1], parts[2] # tile2 is right from tile1
                    self.adjacency[tile1].append(tile2)
                    self.adjacency[tile2].append(tile1)
                    all_tiles_in_static.add(tile1)
                    all_tiles_in_static.add(tile2)
            elif fact_str.startswith('(available-color '):
                parts = fact_str.strip('()').split()
                if len(parts) == 2:
                    self.available_colors.add(parts[1])

        # Infer coordinates for all tiles found in static adjacency facts
        for tile_name in all_tiles_in_static:
             self._infer_coords(tile_name)


        # Store goal painted facts
        self.goal_painted = {} # Map tile name to goal color
        for goal_fact in self.goals:
            if goal_fact.startswith('(painted '):
                parts = goal_fact.strip('()').split()
                # Ensure fact has correct number of parts
                if len(parts) == 3:
                    tile = parts[1]
                    color = parts[2]
                    self.goal_painted[tile] = color
                # else: Ignore malformed goal facts

    def _infer_coords(self, tile_name):
        """Infers coordinates for a tile based on its name tile_row_col."""
        if tile_name in self.tile_coords:
            return
        match = re.match(r'tile_(\d+)_(\d+)', tile_name)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            self.tile_coords[tile_name] = (row, col)
        # else: Ignore tile names not matching the format

    def _get_distance(self, start_tile, target_tiles):
        """
        Calculates the minimum BFS distance from start_tile to any tile
        in the set target_tiles.
        Returns float('inf') if no path exists.
        """
        if not target_tiles:
            return math.inf

        # Handle case where start_tile is not in the graph (e.g., isolated tile)
        # A robot should always be on a tile in the graph in valid problems.
        # This check is mostly defensive.
        if start_tile not in self.adjacency and start_tile not in target_tiles:
             return math.inf

        queue = collections.deque([(start_tile, 0)])
        visited = {start_tile}

        while queue:
            current_tile, dist = queue.popleft()

            if current_tile in target_tiles:
                return dist

            # Ensure current_tile is in adjacency map before accessing
            if current_tile in self.adjacency:
                for neighbor in self.adjacency[current_tile]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return math.inf # No path found

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        total_heuristic = 0

        # Extract current state information
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {}    # robot_name -> color_name
        current_painted = {} # tile_name -> color_name
        current_clear = set() # set of clear tile_names

        for fact_str in state:
            if fact_str.startswith('(robot-at '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3: robot_locations[parts[1]] = parts[2]
            elif fact_str.startswith('(robot-has '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3: robot_colors[parts[1]] = parts[2]
            elif fact_str.startswith('(painted '):
                parts = fact_str.strip('()').split()
                if len(parts) == 3: current_painted[parts[1]] = parts[2]
            elif fact_str.startswith('(clear '):
                parts = fact_str.strip('()').split()
                if len(parts) == 2: current_clear.add(parts[1])

        # Iterate through goal painted tiles
        for goal_tile, goal_color in self.goal_painted.items():
            # Check if goal is already satisfied for this tile
            if current_painted.get(goal_tile) == goal_color:
                continue # This tile is done

            # Check if tile is painted with the wrong color
            if goal_tile in current_painted and current_painted[goal_tile] != goal_color:
                # Tile is painted incorrectly, likely a dead end
                return math.inf

            # If the tile is not painted with the goal color, it needs painting.
            # It must be clear to be paintable.
            # A tile is either clear or painted. If it's not painted with the goal color,
            # and not clear, it must be painted with the wrong color (handled above)
            # or in an unpaintable state.
            if goal_tile not in current_clear:
                 # Tile needs painting but is not clear (and not painted with goal color).
                 return math.inf

            # Tile needs painting and is clear. Calculate minimum cost to paint it.
            min_robot_cost_for_tile = math.inf

            # Find tiles adjacent to the goal tile
            adjacent_tiles = self.adjacency.get(goal_tile, [])
            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles? Should not happen in valid grid problems.
                 # Treat as unreachable.
                 return math.inf

            # For each robot, calculate cost to paint this tile
            for robot_name, robot_location in robot_locations.items():
                robot_color = robot_colors.get(robot_name)

                # A robot must have a color to paint or change color.
                # The domain implies robots always have a color initially.
                # If for some reason a robot has no color, it cannot paint.
                if robot_color is None:
                    continue # This robot cannot paint

                # Cost to get the correct color
                color_cost = 0 if robot_color == goal_color else 1

                # Cost to move to an adjacent tile
                move_cost = self._get_distance(robot_location, set(adjacent_tiles))

                # Total cost for this robot to paint this tile
                # move_cost + color_cost + 1 (for paint action)
                # If move_cost is inf, total cost is inf
                if move_cost != math.inf:
                    total_robot_cost = move_cost + color_cost + 1
                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, total_robot_cost)
                # else: Robot cannot reach any adjacent tile, this robot is not an option for this tile

            # If no robot can paint this tile, the state is likely a dead end
            if min_robot_cost_for_tile == math.inf:
                return math.inf

            total_heuristic += min_robot_cost_for_tile

        return total_heuristic
