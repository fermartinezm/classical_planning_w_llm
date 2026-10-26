import math
import re
from collections import defaultdict, deque

# Helper function to parse PDDL fact string
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and its arguments.
    e.g., '(robot-at robot1 tile_0_1)' -> ('robot-at', ['robot1', 'tile_0_1'])
    """
    # Remove outer parentheses and split by space
    parts = fact_string.strip('()').split()
    if not parts: # Handle empty string or just ()
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to parse tile name into coordinates
def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_R_C' into (row, column) integers.
    Returns None if the name format doesn't match.
    """
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        try:
            row = int(match.group(1))
            col = int(match.group(2))
            return row, col
        except ValueError:
            # Should not happen if regex matched digits, but handle defensively
            pass
    return None # Not a standard tile_R_C name


class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up:
    1. The number of tiles that need to be painted (are clear in the state but
       require painting in the goal). Each such tile requires at least one
       paint action.
    2. The number of colors required by the unpainted goal tiles that are not
       currently held by any robot. Each such color requires at least one
       change_color action.
    3. For each tile that needs painting, the minimum grid distance from any
       robot's current location to any tile adjacent to the tile needing paint.
       This estimates the movement cost.

    Assumptions:
    - The problem is solvable.
    - Tiles that need painting in the goal are either clear or already painted
      with the correct color in the initial state. Tiles painted with the wrong
      color in the state that are goal tiles make the state unsolvable.
    - Tile names follow the format 'tile_R_C' where R and C are integers
      representing grid coordinates.
    - The grid defined by adjacency predicates is connected.
    - The heuristic calculation for movement ignores the 'clear' precondition
      for move actions (relaxation).

    Heuristic Initialization:
    The constructor precomputes static information from the task object:
    - Identifies all tile objects present in the initial state, goal, and static facts.
    - Parses tile names into grid coordinates (R, C) where possible.
    - Builds an adjacency graph representing the grid connectivity based on
      'up', 'down', 'left', 'right' predicates found in static facts.
    - Computes all-pairs shortest paths (grid distances) between all identified tiles
      using BFS.
    - Stores the goal facts, specifically identifying the target color for
      each tile that needs to be painted according to the goal.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check for unsolvable states: Iterate through the state's painted facts.
       If a tile is painted with color C' in the state, but the goal requires
       it to be painted with color C where C != C', the state is unsolvable.
       Return infinity.
    2. Identify 'TilesToPaint': Find all tiles T that are mentioned in the goal
       as needing to be painted with color C, such that the state does *not*
       contain the fact '(painted T C)' and the state *does* contain the fact
       '(clear T)'.
    3. If 'TilesToPaint' is empty, check if all goal painted facts are in the state.
       If yes, the goal is reached, return 0. Otherwise, the state is likely
       unsolvable (e.g., a goal tile is not clear and not painted correctly),
       return infinity.
    4. Initialize heuristic value `h = 0`.
    5. Add cost for paint actions: `h += len(TilesToPaint)`. Each tile needs one paint action.
    6. Identify 'ColorsNeeded': Collect the target colors for all tiles in
       'TilesToPaint'.
    7. Identify 'RobotsWithColor': Collect the colors currently held by robots
       in the state.
    8. Identify 'ColorsMissing': These are the colors in 'ColorsNeeded' that
       are not in 'RobotsWithColor'.
    9. Add cost for change_color actions: `h += len(ColorsMissing)`. Each missing
       color requires at least one robot to change to that color.
    10. Identify robot locations from the state.
    11. Add cost for movement: For each tile T in 'TilesToPaint':
        a. Find all tiles adjacent to T using the precomputed adjacency information.
        b. Calculate the minimum grid distance from *any* robot's current location
           to *any* tile adjacent to T. This is `min_{robot r} min_{adj_T adjacent to T} dist(location(r), adj_T)`.
           This calculation ignores the 'clear' precondition for movement.
        c. Add this minimum distance to `h`. If no adjacent tile is reachable from any robot (distance is infinity), the state is likely unsolvable, return infinity.
    12. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.task = task
        self.goal_facts = task.goals
        self.static_facts = task.static

        # 1a. Identify all tile objects from all facts
        self.tiles = set()
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact_string in all_facts:
             predicate, args = parse_fact(fact_string)
             for arg in args:
                  if arg.startswith('tile_'):
                       self.tiles.add(arg)

        # 1a. Parse coordinates and build adjacency list from static facts
        self.tile_coords = {} # tile_name -> (row, col)
        self.adj = defaultdict(dict) # tile_name -> {direction: neighbor_tile_name}

        for fact_string in self.static_facts:
            predicate, args = parse_fact(fact_string)
            if predicate in ['up', 'down', 'left', 'right']:
                y, x = args # Y is neighbor, X is source

                # Try to parse coordinates from names
                coord_x = parse_tile_name(x)
                coord_y = parse_tile_name(y)
                if coord_x: self.tile_coords[x] = coord_x
                if coord_y: self.tile_coords[y] = coord_y

                # Build adjacency mapping
                # The predicate name indicates the direction *from* X *to* Y
                # So, Y is in the 'predicate' direction relative to X
                self.adj[x][predicate] = y

        # 1b. Build grid graph (undirected adjacency list for BFS)
        self.graph = defaultdict(set)
        for tile, neighbors in self.adj.items():
            for neighbor in neighbors.values():
                self.graph[tile].add(neighbor)
                self.graph[neighbor].add(tile) # Assuming symmetric adjacency

        # Ensure all identified tiles are in the graph keys, even if isolated (though unlikely in a grid)
        for tile in self.tiles:
             if tile not in self.graph:
                  self.graph[tile] = set()


        # 1c. Precompute all-pairs shortest paths (distances)
        self.dist = defaultdict(lambda: defaultdict(lambda: float('inf')))
        for start_node in self.tiles:
            self._bfs(start_node)

        # 1d, 1e. Store goal painted tiles
        self.goal_painted_tiles = {} # tile_name -> target_color
        for fact_string in self.goal_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'painted':
                tile, color = args
                self.goal_painted_tiles[tile] = color

    def _bfs(self, start_node):
        """Helper to run BFS from a start node to compute distances."""
        # Only run BFS if the start_node is a known tile in the graph
        if start_node not in self.graph:
             # This tile might exist in facts but have no adjacency predicates
             # It's an isolated tile. Distance to/from it is inf unless it's itself.
             self.dist[start_node][start_node] = 0
             return

        q = deque([(start_node, 0)])
        self.dist[start_node][start_node] = 0
        visited = {start_node}

        while q:
            current_node, current_dist = q.popleft()

            # Get neighbors from the graph, which includes all tiles
            for neighbor in self.graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.dist[start_node][neighbor] = current_dist + 1
                    q.append((neighbor, current_dist + 1))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        state_facts = set(state) # Convert frozenset to set for easier checking

        # 1. Check for unsolvable states (tile painted with wrong color)
        for fact_string in state_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'painted':
                tile, painted_color = args
                if tile in self.goal_painted_tiles:
                    target_color = self.goal_painted_tiles[tile]
                    if painted_color != target_color:
                        # Tile is painted with the wrong color, unsolvable
                        return float('inf')

        # 2. Identify TilesToPaint
        tiles_to_paint = set()
        for tile, target_color in self.goal_painted_tiles.items():
            painted_fact = f'(painted {tile} {target_color})'
            clear_fact = f'(clear {tile})'
            # Tile needs painting if goal requires it, it's not painted correctly, and it's clear
            if painted_fact not in state_facts and clear_fact in state_facts:
                 tiles_to_paint.add(tile)

        # 3. Goal check / Unsolvable check for non-clear, non-painted-correctly tiles
        if not tiles_to_paint:
             # If there are no clear tiles that need painting, check if all goal tiles are met.
             # We already checked for wrong colors.
             # If a goal tile is not in state_facts as painted correctly, and it's not clear,
             # it must be something else (e.g., occupied, or some other state not defined).
             # Based on domain/examples, tiles are either clear or painted.
             # If a goal tile is not clear and not painted correctly, it's unsolvable.
             # We can check this by seeing if any goal painted fact is missing from state.
             for tile, target_color in self.goal_painted_tiles.items():
                  painted_fact = f'(painted {tile} {target_color})'
                  if painted_fact not in state_facts:
                       # This tile needs painting but is not in tiles_to_paint (so it's not clear).
                       # Unsolvable.
                       return float('inf')
             # If we iterated through all goal painted facts and they are all in state, goal reached.
             return 0


        # 4. Initialize heuristic
        h = 0

        # 5. Add cost for paint actions
        h += len(tiles_to_paint)

        # 6, 7, 8, 9. Add cost for change_color actions and get robot locations
        colors_needed = {self.goal_painted_tiles[tile] for tile in tiles_to_paint}
        robots_with_color = set()
        robot_locs = {} # robot_name -> tile_name

        for fact_string in state_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'robot-has':
                robot, color = args
                robots_with_color.add(color)
            elif predicate == 'robot-at':
                robot, tile = args
                robot_locs[robot] = tile

        colors_missing = colors_needed - robots_with_color
        h += len(colors_missing)

        # 10. Add cost for movement
        for tile_t in tiles_to_paint:
            min_dist_to_adj_t = float('inf')
            adj_tiles_t = self.adj.get(tile_t, {}).values() # Get neighbor tile names

            if not adj_tiles_t:
                 # Tile needs painting but has no neighbors. Unsolvable.
                 return float('inf')

            for robot, loc_r in robot_locs.items():
                # Ensure robot location is a known tile
                if loc_r not in self.dist:
                     # Should not happen in a valid problem instance if tile parsing is correct
                     return float('inf') # Robot is on an unknown tile?

                for adj_t in adj_tiles_t:
                    # Ensure adjacent tile is a known tile
                    if adj_t not in self.dist[loc_r]:
                         # Should not happen if BFS covered all tiles
                         return float('inf') # Adjacent tile is unknown?

                    # Relaxed movement: ignore clear precondition for adj_t
                    dist = self.dist[loc_r][adj_t]
                    min_dist_to_adj_t = min(min_dist_to_adj_t, dist)

            # If min_dist_to_adj_t is still inf, it means no robot can reach any adjacent tile.
            # This implies unsolvability in a connected grid.
            if min_dist_to_adj_t == float('inf'):
                 return float('inf')

            h += min_dist_to_adj_t

        return h
