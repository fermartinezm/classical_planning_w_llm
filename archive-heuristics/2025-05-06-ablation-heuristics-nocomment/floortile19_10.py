from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
def get_parts(fact):
    """Extracts predicate and arguments from a fact string."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern."""
    parts = get_parts(fact)
    # Ensure we have enough parts to match against args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses 'tile_R_C' string into (R, C) tuple."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not a valid tile_R_C format
    return None # Or raise error

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum estimated costs for each unsatisfied goal tile. For each
        goal tile that needs to be painted, it calculates the minimum cost
        for any robot to paint that tile. This minimum cost is the sum of:
        1. Cost to change color (1 if the robot doesn't have the required color).
        2. Cost to move to a clear tile adjacent to the goal tile (calculated
           using BFS on the grid considering only clear tiles as traversable).
        3. Cost to paint the tile (1 action).
        The heuristic returns a large value if any goal tile is painted with
        the wrong color or if any unpainted goal tile is unreachable by any
        robot (e.g., no clear adjacent tiles or no path through clear tiles).

    Assumptions:
        - Tile names follow the format 'tile_R_C' where R and C are integers
          representing row and column.
        - The grid structure defined by 'up', 'down', 'left', 'right' predicates
          corresponds to a grid where 'up' means decreasing row, 'down' means
          increasing row, 'left' means decreasing column, and 'right' means
          increasing column.
        - Robots always have a color (no 'free-color' state initially or reachable).
        - All colors required for goals are available ('available-color') in the environment
          to be picked up via `change_color`.
        - Tiles are either 'clear' or 'painted'. If a tile is not 'clear' and
          not 'painted' with the goal color, and it is a goal tile, it is considered
          an unsolvable state for that tile.
        - The problem is solvable if and only if all goal tiles can eventually
          be painted with the correct color. States where a goal tile is painted
          with the wrong color or cannot be reached/painted are considered unsolvable
          and assigned a large heuristic value.

    Heuristic Initialization:
        - Parses static facts to build a mapping from tile names ('tile_R_C')
          to grid coordinates (R, C) and vice versa.
        - Builds an adjacency map for the grid based on 'up', 'down', 'left',
          and 'right' predicates, representing bidirectional movement possibilities.
        - Stores the goal requirements, mapping goal tile names to their target colors.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Parse the current state to determine:
           - The current location of each robot.
           - The current color held by each robot.
           - The clear/painted status and painted color for each tile.
           - Initialize all known tiles (from static facts) as not clear and not painted,
             then update based on state facts.
        3. Identify unsatisfied goal tiles: Iterate through the goal tiles stored
           during initialization. For each goal tile:
           - Check its current state (clear or painted color).
           - If it's painted with the correct color, it's satisfied; continue.
           - If it's painted with a different color, the state is likely unsolvable;
             return a large value (e.g., 1000000).
           - If it's clear, it's an unsatisfied goal tile that needs painting.
           - If it's not clear and not painted with the goal color, and it is a goal tile,
             it's considered an unsolvable state for that tile; return a large value.
        4. For each unsatisfied goal tile that needs painting (i.e., is currently clear):
           - Initialize `min_cost_for_tile` to infinity.
           - Find all tiles adjacent to the goal tile that are currently clear.
           - If there are no clear adjacent tiles, the goal tile cannot be painted
             in this state; return a large value.
           - For each robot:
             - Calculate the `color_cost`: 1 if the robot does not hold the
               required color for the goal tile, 0 otherwise. This assumes the
               required color is available in the environment or held by some robot
               (which is implicitly handled by the overall unsolvable check).
             - Calculate the `move_cost`: Perform a Breadth-First Search (BFS)
               starting from the robot's current location. The BFS explores
               neighboring tiles only if they are currently clear. The goal of
               the BFS is to find the shortest path to *any* of the clear tiles
               adjacent to the goal tile. The `move_cost` is the length of this
               shortest path. If no path exists, the `move_cost` is infinity.
             - If `move_cost` is infinity, this robot cannot reach a position
               to paint the tile; continue to the next robot.
             - The `paint_cost` is 1 (for the paint action).
             - The total cost for this robot to paint the tile is `color_cost + move_cost + paint_cost`.
             - Update `min_cost_for_tile` with the minimum cost found so far
               among all robots.
           - If, after checking all robots, `min_cost_for_tile` is still infinity,
             no robot can paint this tile; return a large value.
           - Add `min_cost_for_tile` to the total heuristic value `h`.
        5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.tile_coords = {}
        self.coords_tile = {}
        self.adj_map = {}
        self.available_colors = set()
        self.goal_tiles = {} # {tile_name: color}

        # Collect all tile names mentioned in static facts
        all_tile_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])
            elif parts[0] in ['up', 'down', 'left', 'right']:
                tile1_name = parts[1]
                tile2_name = parts[2]
                all_tile_names.add(tile1_name)
                all_tile_names.add(tile2_name)

                # Parse coordinates and build mappings
                coords1 = parse_tile_coords(tile1_name)
                coords2 = parse_tile_coords(tile2_name)

                if coords1:
                    self.tile_coords[tile1_name] = coords1
                    self.coords_tile[coords1] = tile1_name
                if coords2:
                    self.tile_coords[tile2_name] = coords2
                    self.coords_tile[coords2] = tile2_name

                # Build adjacency map (bidirectional)
                self.adj_map.setdefault(tile1_name, []).append(tile2_name)
                self.adj_map.setdefault(tile2_name, []).append(tile1_name)

        # Add any tiles from goals that might not be in static adjacency facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile_name = parts[1]
                color = parts[2]
                self.goal_tiles[tile_name] = color
                all_tile_names.add(tile_name) # Add goal tiles to known tiles

        # Ensure all tiles from all_tile_names are in tile_coords/coords_tile
        # This handles cases where a tile might be mentioned in init/goal but not adj facts.
        for tile_name in all_tile_names:
             if tile_name not in self.tile_coords:
                 coords = parse_tile_coords(tile_name)
                 if coords:
                     self.tile_coords[tile_name] = coords
                     self.coords_tile[coords] = tile_name


    def __call__(self, node):
        state = node.state

        # Parse current state
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        tile_clear_status = {} # {tile_name: bool}
        tile_painted_color = {} # {tile_name: color_name or None}

        # Initialize all known tiles from static facts/goals
        for tile_name in self.tile_coords:
             tile_clear_status[tile_name] = False
             tile_painted_color[tile_name] = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                tile_name = parts[1]
                if tile_name in self.tile_coords: # Only process known tiles
                    tile_clear_status[tile_name] = True
                    tile_painted_color[tile_name] = None # Ensure consistency
            elif parts[0] == 'painted':
                tile_name = parts[1]
                color = parts[2]
                if tile_name in self.tile_coords: # Only process known tiles
                    tile_painted_color[tile_name] = color
                    tile_clear_status[tile_name] = False # Painted tiles are not clear

        h = 0
        LARGE_VALUE = 1000000 # Represents unsolvable state or very high cost

        unsatisfied_goals = {} # {tile_name: goal_color}

        # Identify unsatisfied goals and check for unsolvable states (wrong color)
        for goal_tile, goal_color in self.goal_tiles.items():
            current_painted_color = tile_painted_color.get(goal_tile)

            if current_painted_color == goal_color:
                # Goal already satisfied for this tile
                continue
            elif current_painted_color is not None and current_painted_color != goal_color:
                # Tile painted with the wrong color - unsolvable
                return LARGE_VALUE
            else:
                 # Tile is clear or unpainted (and not clear) - needs painting
                 # If it's not painted with the goal color, it's unsatisfied.
                 unsatisfied_goals[goal_tile] = goal_color


        if not unsatisfied_goals:
            return 0 # All goals satisfied

        # Calculate cost for each unsatisfied goal tile
        for goal_tile, goal_color in unsatisfied_goals.items():
            min_cost_for_tile = float('inf')

            # Find clear tiles adjacent to the goal tile
            goal_coords = self.tile_coords.get(goal_tile)
            if goal_coords is None:
                 # Goal tile not found in static grid - should not happen in valid problems
                 return LARGE_VALUE

            potential_adj_coords = [
                (goal_coords[0]-1, goal_coords[1]), # Up
                (goal_coords[0]+1, goal_coords[1]), # Down
                (goal_coords[0], goal_coords[1]-1), # Left
                (goal_coords[0], goal_coords[1]+1)  # Right
            ]

            adjacent_clear_tiles = []
            for coord in potential_adj_coords:
                adj_tile_name = self.coords_tile.get(coord)
                if adj_tile_name and tile_clear_status.get(adj_tile_name, False):
                    adjacent_clear_tiles.append(adj_tile_name)

            # If no adjacent clear tiles, this goal tile cannot be painted in this state
            if not adjacent_clear_tiles:
                 return LARGE_VALUE # Unsolvable from this state

            # Calculate minimum cost for any robot to paint this tile
            # Iterate through all robots
            for robot_name in robot_locations.keys():
                robot_location = robot_locations.get(robot_name)
                if robot_location is None:
                    # Robot location unknown - this shouldn't happen in a valid state
                    continue # Or return LARGE_VALUE

                color_cost = 1 if robot_colors.get(robot_name) != goal_color else 0

                # BFS to find shortest path from robot_location to any adjacent_clear_tile
                move_cost = float('inf')
                queue = deque([(robot_location, 0)])
                visited = {robot_location}

                while queue:
                    current_tile, dist = queue.popleft()

                    if current_tile in adjacent_clear_tiles:
                        move_cost = dist
                        break # Found shortest path to a suitable adjacent tile

                    # Explore neighbors
                    for neighbor_tile in self.adj_map.get(current_tile, []):
                        # Robot can move to neighbor if neighbor is clear
                        if tile_clear_status.get(neighbor_tile, False) and neighbor_tile not in visited:
                            visited.add(neighbor_tile)
                            queue.append((neighbor_tile, dist + 1))

                # If robot cannot reach any clear adjacent tile
                if move_cost == float('inf'):
                    continue # Try next robot

                paint_cost = 1
                total_robot_cost = color_cost + move_cost + paint_cost
                min_cost_for_tile = min(min_cost_for_tile, total_robot_cost)

            # If no robot can paint this tile
            if min_cost_for_tile == float('inf'):
                 return LARGE_VALUE # Unsolvable from this state

            h += min_cost_for_tile

        return h
