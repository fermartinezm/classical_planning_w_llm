# Need to import Heuristic base class
# from heuristics.heuristic_base import Heuristic # Assuming this is available

from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle unexpected fact format, maybe log a warning or raise error
         # For robustness, return empty list or handle error appropriately.
         # Assuming valid PDDL fact string format for now.
         return [] # Or raise ValueError("Invalid fact format")
    return fact[1:-1].split()

# Assuming Heuristic base class is provided elsewhere
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with the required colors. It sums three components: the number of tiles
    that still need to be painted correctly, the number of colors required for
    these tiles that are not currently held by any robot, and an estimate of
    the total movement cost for robots to reach positions from which they can paint.

    # Assumptions
    - The grid structure is defined by up/down/left/right predicates and is static.
    - The grid is connected, allowing movement between most tiles (unless blocked by painted tiles or robots).
    - Tiles are named in the format 'tile_row_col', but the heuristic relies on
      the adjacency defined by predicates for movement distances.
    - A tile is either clear or painted. If painted with a color different from
      the goal color, the state is a dead end. There is no action to unpaint or repaint.
    - Robots can only move to clear tiles.
    - Robots can only paint clear tiles.

    # Heuristic Initialization
    - Parse the static facts to build the grid adjacency list, representing possible moves.
    - Compute all-pairs shortest paths (APSP) on the grid graph using BFS. This provides
      the minimum number of move actions between any two tiles assuming the path is clear.
      Note: This precomputation ignores dynamic obstacles (painted tiles, robots), which
      is a simplification.
    - Extract the goal conditions, specifically the required color for each goal tile,
      storing them in a dictionary mapping tile names to goal colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to identify robot locations, robot held colors,
       and the color of any painted tiles.
    2. Check for dead ends: Iterate through all tiles that are painted in the
       current state. If any painted tile is a goal tile and its current color
       does not match the required goal color, the state is unsolvable. Return
       infinity in this case.
    3. Identify unpainted goal tiles: Create a set of (tile, color) pairs for
       all goal tiles that are not currently painted with their required color.
       Let this set be `UnpaintedGoals`.
    4. Initialize the heuristic value `h = 0`.
    5. Add cost for paint actions: Each tile in `UnpaintedGoals` requires one
       `paint` action. Add `len(UnpaintedGoals)` to `h`.
    6. Add cost for changing colors: Determine the set of colors required by
       tiles in `UnpaintedGoals`. Determine the set of colors currently held
       by robots. The number of colors required but not currently held by any
       robot represents the minimum number of `change_color` actions needed
       across all robots. Add this number to `h`.
    7. Add cost for movement: For each `(tile, color)` pair in `UnpaintedGoals`,
       a robot needs to reach a tile adjacent to `tile` to perform the paint
       action. Calculate the minimum distance from *any* robot's current location
       to *any* tile adjacent to `tile` using the precomputed grid distances.
       Sum these minimum distances over all `(tile, color)` pairs in `UnpaintedGoals`.
       Add this sum to `h`.
       If a robot is currently *on* the goal tile `tile`, it must first move
       off (cost >= 1) before the tile can become clear and be painted. The
       minimum distance calculation from the robot's current location (on the
       tile) to an adjacent tile implicitly includes this initial move-off cost
       (distance is 1).
       If, for any unpainted goal tile, no robot can reach any adjacent tile
       based on the precomputed distances (e.g., due to a disconnected grid
       definition or tiles not included in the grid), the state is likely
       unsolvable. Return infinity.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing grid distances and storing goal info.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build grid adjacency list from static facts
        self.adj = {}
        all_tiles = set()
        # Collect all tiles mentioned in static facts defining connectivity
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                # Predicate format is (direction tile_neighbor tile_current)
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # Robot moves from tile_current to tile_neighbor
                # So, tile_neighbor is adjacent to tile_current
                tile_neighbor, tile_current = parts[1], parts[2]
                all_tiles.add(tile_neighbor)
                all_tiles.add(tile_current)
                if tile_current not in self.adj: self.adj[tile_current] = set()
                if tile_neighbor not in self.adj: self.adj[tile_neighbor] = set()
                # Add bidirectional edges for movement
                self.adj[tile_current].add(tile_neighbor)
                self.adj[tile_neighbor].add(tile_current)

        # Collect all tiles mentioned in goal facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'painted':
                 tile = parts[1]
                 all_tiles.add(tile)
                 if tile not in self.adj: self.adj[tile] = set() # Add goal tiles that might be isolated in static facts

        self.all_tiles = list(all_tiles) # Store list for consistent indexing/iteration

        # 2. Compute All-Pairs Shortest Paths (APSP) using BFS
        self.dist = {}
        for start_tile in self.all_tiles:
            self.dist[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}
            while queue:
                current_tile, d = queue.popleft()
                self.dist[start_tile][current_tile] = d
                # Check if current_tile has neighbors before iterating
                if current_tile in self.adj:
                    for neighbor in self.adj[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))
            # If any tile is unreachable from start_tile, its distance remains undefined.
            # This is handled during the heuristic calculation by checking if dist[r_loc][adj_tile] exists.


        # 3. Extract goal conditions (painted tiles)
        self.goal_painted_dict = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted_dict[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state information
        robot_locs = {}
        robot_colors = {}
        painted_tiles_state = {}
        # clear_tiles_state = set() # Not strictly needed if we check painted state directly

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'robot-at':
                robot, loc = parts[1], parts[2]
                robot_locs[robot] = loc
            elif predicate == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == 'painted':
                tile, color = parts[1], parts[2]
                painted_tiles_state[tile] = color
            # elif predicate == 'clear':
            #     tile = parts[1]
            #     clear_tiles_state.add(tile)

        # 1. Check for dead ends
        for goal_tile, required_color in self.goal_painted_dict.items():
            if goal_tile in painted_tiles_state:
                current_color = painted_tiles_state[goal_tile]
                if current_color != required_color:
                    # Tile is painted with the wrong color, cannot reach goal
                    return float('inf')

        # 2. Identify unpainted goal tiles
        # These are goal tiles that are not currently painted with the required color.
        unpainted_goal_tiles = [] # List of (tile, color) tuples
        for goal_tile, required_color in self.goal_painted_dict.items():
             if goal_tile not in painted_tiles_state or painted_tiles_state[goal_tile] != required_color:
                 unpainted_goal_tiles.append((goal_tile, required_color))

        # If all goal tiles are painted correctly, we are at the goal state.
        if not unpainted_goal_tiles:
            return 0

        # 3. Initialize heuristic value
        h = 0

        # 4. Add cost for paint actions
        h += len(unpainted_goal_tiles)

        # 5. Add cost for changing colors
        needed_colors = {color for tile, color in unpainted_goal_tiles}
        available_colors = set(robot_colors.values())
        colors_to_acquire = needed_colors - available_colors
        h += len(colors_to_acquire)

        # 6. Add cost for movement
        # Sum of minimum distances from any robot to any adjacent tile for each unpainted goal tile
        movement_cost = 0
        for goal_tile, required_color in unpainted_goal_tiles:
            min_dist_to_adj = float('inf')
            adjacent_tiles = self.adj.get(goal_tile, set())

            if not adjacent_tiles:
                 # Goal tile has no adjacent tiles defined in the grid. Cannot be painted.
                 return float('inf')

            # Check if there are any robots
            if not robot_locs:
                 # No robots to paint the tile
                 return float('inf')

            for robot, r_loc in robot_locs.items():
                # Check if robot location is valid and reachable in our distance map
                if r_loc in self.dist:
                    for adj_tile in adjacent_tiles:
                        # Check if adjacent tile is reachable from robot location
                        if adj_tile in self.dist[r_loc]:
                            min_dist_to_adj = min(min_dist_to_adj, self.dist[r_loc][adj_tile])

            # If after checking all robots and all adjacent tiles, min_dist_to_adj is still inf,
            # it means no robot can reach any tile adjacent to this goal tile.
            # This tile cannot be painted. The state is likely unsolvable.
            if min_dist_to_adj == float('inf'):
                 return float('inf')

            movement_cost += min_dist_to_adj

        h += movement_cost

        # 7. Return h
        return h
