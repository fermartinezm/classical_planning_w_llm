from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a given pattern
# Using manual check for potentially better performance than fnmatch
def manual_match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    for part, arg in zip(parts, args):
        if arg != '*' and part != arg:
            return False
    return True

# Helper function to build the tile graph from static adjacency facts
def build_tile_graph(static_facts):
    """
    Builds an adjacency list representation of the tile grid graph.
    Movement is possible between tiles connected by up, down, left, or right predicates.
    """
    graph = {}
    adj_preds = ["up", "down", "left", "right"]
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] in adj_preds:
            # Predicate is (direction tile1 tile2), meaning tile1 is in that direction from tile2.
            # This implies an edge between tile1 and tile2.
            tile1, tile2 = parts[1], parts[2]
            if tile1 not in graph:
                graph[tile1] = []
            if tile2 not in graph:
                graph[tile2] = []
            # Add bidirectional edges
            if tile2 not in graph[tile1]:
                 graph[tile1].append(tile2)
            if tile1 not in graph[tile2]:
                 graph[tile2].append(tile1)
    return graph

# Helper function to calculate shortest path distance using BFS
def bfs_distance(graph, start_tile, end_tile):
    """
    Calculates the shortest path distance (number of moves) between two tiles
    on the grid graph using Breadth-First Search.
    Assumes movement is possible between any adjacent tiles in the graph,
    ignoring 'clear' predicate constraints for heuristic calculation.
    """
    if start_tile == end_tile:
        return 0
    queue = deque([(start_tile, 0)])
    visited = {start_tile}
    
    while queue:
        current_tile, dist = queue.popleft()
        
        if current_tile == end_tile:
            return dist
            
        if current_tile in graph:
            for neighbor in graph[current_tile]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
    return float('inf') # No path found

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with their required colors. It sums the minimum estimated
    cost for each unpainted goal tile independently.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Builds a graph representation of the tile grid from static adjacency facts
      to calculate movement distances.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted according to the goal.
    2. Check if any tile is currently painted with a color different from its goal color.
       If so, the state is a dead end (unsolvable), return infinity.
    3. Identify the subset of goal tiles that are *not* currently painted correctly
       in the current state. These are the "unpainted goal tiles".
    4. If there are no unpainted goal tiles, the heuristic is 0 (goal reached).
    5. For each unpainted goal tile `T` requiring color `C`:
       a. Determine the minimum cost for *any* robot to paint this tile.
       b. The cost for a robot `R` to paint tile `T` with color `C` is estimated as:
          - Cost to get the correct color: 1 action if `R` currently has a different color, 0 otherwise.
          - Cost to move to a tile adjacent to `T`: Minimum BFS distance from `R`'s current location to *any* tile adjacent to `T`. (Relaxation: ignores 'clear' constraints for movement).
          - Cost to paint the tile: 1 action.
          Total for robot R: `(1 if R_color != C) + min_dist(R_loc, T_adj) + 1`.
       c. The minimum cost for tile `T` is the minimum of the above cost over all robots.
          (Relaxation: ignores robot coordination and resource contention).
    6. The total heuristic value is the sum of the minimum costs calculated for each
       unpainted goal tile.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Store goal locations and required colors for each tile
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

        # Build the tile graph for distance calculations
        self.tile_graph = build_tile_graph(static_facts)

        # Get all tile names from the graph nodes (or static facts)
        # This is useful for iterating through all possible tiles if needed,
        # but we primarily need it for BFS. The graph keys are sufficient.
        self.all_tiles = set(self.tile_graph.keys())

        # Pre-calculate adjacent tiles for all tiles for quick lookup
        self.adjacent_tiles_map = {
            tile: get_adjacent_tiles(self.tile_graph, tile)
            for tile in self.all_tiles
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Extract current state information
        robot_locations = {}
        robot_colors = {}
        painted_tiles = {}
        current_clear_tiles = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color
            elif parts[0] == "clear":
                 current_clear_tiles.add(parts[1])

        # Identify unpainted goal tiles and check for dead ends
        unpainted_goal_tiles = []
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            if goal_tile in painted_tiles:
                # Tile is painted, check if it's the correct color
                if painted_tiles[goal_tile] != goal_color:
                    # Painted the wrong color - this is a dead end in this domain
                    return float('inf')
                # Else: Tile is painted with the correct color, goal satisfied for this tile
            else:
                # Tile is not painted, it needs to be painted
                unpainted_goal_tiles.append((goal_tile, goal_color))

        # If all goal tiles are painted correctly, the goal is reached
        if not unpainted_goal_tiles:
            return 0

        total_heuristic = 0

        # Calculate minimum cost for each unpainted goal tile independently
        for tile_to_paint, required_color in unpainted_goal_tiles:
            min_cost_for_this_tile = float('inf')

            # Find adjacent tiles to the tile that needs painting
            adjacent_tiles_to_goal = self.adjacent_tiles_map.get(tile_to_paint, [])

            # If a tile has no adjacent tiles (shouldn't happen in valid grids, but defensive), it's unpaintable
            if not adjacent_tiles_to_goal:
                 return float('inf') # Cannot paint a tile if no robot can reach an adjacent tile

            # Consider each robot as a potential painter for this tile
            for robot, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot) # Get the color the robot has

                # Cost to get the required color
                color_cost = 0
                if robot_color != required_color:
                    color_cost = 1 # Need one change_color action

                # Cost to move to an adjacent tile
                min_move_cost = float('inf')
                for adj_tile in adjacent_tiles_to_goal:
                    # Calculate distance from robot's current location to the adjacent tile
                    # We use the pre-built graph which represents the grid connectivity
                    dist = bfs_distance(self.tile_graph, robot_loc, adj_tile)
                    min_move_cost = min(min_move_cost, dist)

                # If no adjacent tile is reachable, this robot cannot paint this tile
                if min_move_cost == float('inf'):
                    continue # Try next robot

                # Cost to paint
                paint_cost = 1 # One paint action

                # Total estimated cost for this robot to paint this tile
                robot_total_cost = color_cost + min_move_cost + paint_cost

                # Update minimum cost for this tile
                min_cost_for_this_tile = min(min_cost_for_this_tile, robot_total_cost)

            # If no robot can reach an adjacent tile (min_cost_for_this_tile is still inf), problem is likely unsolvable
            if min_cost_for_this_tile == float('inf'):
                 return float('inf')

            total_heuristic += min_cost_for_this_tile

        return total_heuristic

