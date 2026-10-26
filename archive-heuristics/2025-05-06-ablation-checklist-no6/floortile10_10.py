import re
from collections import deque
from fnmatch import fnmatch
# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts or len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on unsatisfied goal tiles, required color changes,
    and estimated movement costs for robots.

    Heuristic Components:
    1. Cost for painting: 1 action for each tile that needs to be painted.
    2. Cost for color changes: 1 action for each color required by an unpainted
       goal tile that no robot currently possesses.
    3. Cost for making tiles clear: 1 action for each goal tile needing paint
       that is currently occupied by a robot (requires the robot to move off).
    4. Cost for movement to position: For each goal tile needing paint, the
       minimum distance (in moves) from any robot's current location to any
       tile adjacent to the goal tile. These distances are summed up.

    This heuristic is non-admissible and aims to guide a greedy best-first search.
    Movement costs are estimated using precomputed shortest paths on the static
    grid graph, ignoring the dynamic 'clear' precondition for intermediate tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure and goal information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract all tile objects mentioned in static connectivity or goals
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                self.all_tiles.add(parts[1])
                self.all_tiles.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'painted':
                 self.all_tiles.add(parts[1])


        # Map tile names to coordinates (row, col) using regex
        self.tile_to_coord = {}
        self.coord_to_tile = {}
        tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
        for tile in self.all_tiles:
            match_obj = tile_pattern.match(tile)
            if match_obj:
                row, col = int(match_obj.group(1)), int(match_obj.group(2))
                self.tile_to_coord[tile] = (row, col)
                self.coord_to_tile[(row, col)] = tile
            # Assuming all relevant tile names follow the 'tile_row_col' pattern.

        # Build adjacency list based on coordinates from static facts
        self.adj = {} # coord -> list of adjacent coords
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                coord1 = self.tile_to_coord.get(tile1)
                coord2 = self.tile_to_coord.get(tile2)
                if coord1 and coord2: # Ensure both tiles were parsed
                    if coord1 not in self.adj:
                        self.adj[coord1] = []
                    if coord2 not in self.adj:
                        self.adj[coord2] = []

                    # Add bidirectional edges (assuming connectivity is symmetric)
                    self.adj[coord1].append(coord2)
                    self.adj[coord2].append(coord1)

        # Remove duplicate adjacencies if any
        for coord in self.adj:
             self.adj[coord] = list(set(self.adj[coord]))


        # Precompute all-pairs shortest paths using BFS on the static grid graph
        # This ignores the 'clear' precondition for movement.
        self.distances = {} # (coord1, coord2) -> distance
        # Only run BFS from/to coordinates that are part of the connected graph
        all_coords_in_graph = list(self.adj.keys())
        for start_coord in all_coords_in_graph:
            q = deque([(start_coord, 0)])
            visited = {start_coord}
            self.distances[(start_coord, start_coord)] = 0

            while q:
                current_coord, dist = q.popleft()

                for neighbor_coord in self.adj.get(current_coord, []):
                    if neighbor_coord not in visited:
                        visited.add(neighbor_coord)
                        self.distances[(start_coord, neighbor_coord)] = dist + 1
                        q.append((neighbor_coord, dist + 1))

        # Store goal painted facts for quick lookup
        self.goal_painted = {} # tile_name -> color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Identify adjacent tiles for each tile (using names) from static facts
        self.tile_adj = {} # tile_name -> list of adjacent tile_names
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                if tile1 not in self.tile_adj:
                    self.tile_adj[tile1] = []
                if tile2 not in self.tile_adj:
                    self.tile_adj[tile2] = []
                self.tile_adj[tile1].append(tile2)
                self.tile_adj[tile2].append(tile1)
        for tile in self.tile_adj:
             self.tile_adj[tile] = list(set(self.tile_adj[tile]))


    def get_distance(self, tile1, tile2):
        """Get precomputed distance between two tiles."""
        coord1 = self.tile_to_coord.get(tile1)
        coord2 = self.tile_to_coord.get(tile2)
        if coord1 is None or coord2 is None:
             # This happens if a tile exists but wasn't part of the connected grid graph
             # (e.g., a robot or goal is on an isolated tile). Should ideally not happen.
             return float('inf')
        return self.distances.get((coord1, coord2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        h = 0

        # Extract current state information
        current_painted = {} # tile_name -> color
        current_clear = set()
        robot_locations = {} # robot_name -> tile_name
        robot_colors = {} # robot_name -> color
        # all_robots = set() # Not strictly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                current_clear.add(parts[1])
            elif parts[0] == 'robot-at':
                robot_locations[parts[1]] = parts[2]
                # all_robots.add(parts[1])
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]
                # all_robots.add(parts[1])

        # Check for unsolvable states first (goal tile painted wrong color)
        for goal_tile, goal_color in self.goal_painted.items():
             if goal_tile in current_painted and current_painted[goal_tile] != goal_color:
                 # If a goal tile is painted with the wrong color, it's unsolvable
                 # in this domain as there's no unpaint action.
                 return float('inf') # Return a large value indicating unsolvability

        # Identify tiles that need painting
        tiles_to_paint = [] # List of (tile, color) that need painting
        needed_colors = set()

        for goal_tile, goal_color in self.goal_painted.items():
             # A tile needs painting if it's not painted correctly according to the goal.
             if goal_tile not in current_painted or current_painted[goal_tile] != goal_color:
                 tiles_to_paint.append((goal_tile, goal_color))
                 needed_colors.add(goal_color)

        if not tiles_to_paint:
            return 0 # Goal reached (all goal tiles are painted correctly)

        # Cost for colors: 1 action per color needed that no robot currently has.
        colors_robots_have = set(robot_colors.values())
        colors_to_acquire = needed_colors - colors_robots_have
        h += len(colors_to_acquire) # Add 1 for each color that needs to be acquired by *some* robot

        # Cost for painting and movement
        for tile, color in tiles_to_paint:
            h += 1 # Cost for the paint action itself for this tile

            # Cost to make tile clear if needed (i.e., if a robot is on it)
            # A tile is not clear if it's painted (ruled out already) or occupied by a robot.
            # If a goal tile needing paint is not clear, a robot must be on it.
            # That robot must move off (cost 1). This move also makes the tile clear.
            robot_on_tile = None
            for r, loc in robot_locations.items():
                if loc == tile:
                    robot_on_tile = r
                    break

            if robot_on_tile:
                 h += 1 # Add 1 action for the robot to move off this tile

            # Cost for movement: Minimum distance from any robot to any tile adjacent to the goal tile.
            # This estimates the cost to get a robot into a position where it can paint the tile.
            adjacent_tiles = self.tile_adj.get(tile, [])
            if not adjacent_tiles:
                 # If a goal tile has no adjacent tiles defined in static facts, it's unreachable.
                 return float('inf') # Unsolvable state

            min_dist_to_adjacent = float('inf')
            for robot_loc in robot_locations.values():
                for adj_tile in adjacent_tiles:
                    dist = self.get_distance(robot_loc, adj_tile)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            if min_dist_to_adjacent == float('inf'):
                 # If no path exists from any robot to any adjacent tile, it's unreachable.
                 return float('inf') # Unsolvable state

            h += min_dist_to_adjacent

        return h

