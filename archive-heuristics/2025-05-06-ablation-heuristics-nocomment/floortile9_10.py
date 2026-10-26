# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Example: '(predicate arg1 arg2)' -> ['predicate', 'arg1', 'arg2']
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    # Example: match('(at obj rooma)', 'at', '*', 'rooma') -> True
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    Estimates the cost to reach the goal state by summing up several components:
    1. The number of goal tiles that are not yet painted correctly (representing paint actions).
    2. The number of colors required by unpainted goal tiles that no robot currently possesses (representing change_color actions).
    3. The number of robots currently occupying a goal tile that needs painting (representing move actions to unblock).
    4. The sum, for each unpainted goal tile, of the minimum distance (in moves) from any robot
       to any tile adjacent to that goal tile (representing movement towards painting positions).

    This heuristic is non-admissible but aims to guide a greedy best-first search
    towards states where goal tiles are painted, robots are near unpainted tiles,
    and robots have the necessary colors. It is designed to be efficiently computable
    by precomputing tile distances.

    Assumptions:
    - If a goal tile is painted with a color different from the goal color, the state is considered unsolvable.
    - If a goal tile is not painted with the goal color, it is assumed to be 'clear' for painting purposes,
      unless a robot is currently occupying it.
    - The tile grid graph defined by adjacency predicates is connected.
    - All tiles and robots mentioned in the problem are part of the connected grid.

    Heuristic Initialization:
    The __init__ method performs the following steps:
    1. Parses the goal facts to create a dictionary mapping each goal tile to its required color (self.goal_paint).
    2. Parses the static facts (up, down, left, right predicates) to build an undirected adjacency graph of tiles (self.adj).
       All tiles mentioned in static facts or goals are included.
    3. Computes the shortest path distance between every pair of tiles in the graph using Breadth-First Search (BFS).
       These distances are stored in a dictionary of dictionaries (self.dist) for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state:
    1. Extract current robot locations, robot colors, and painted tiles from the state facts. Also identify which tiles are occupied by robots.
    2. Check for unsolvable states: Iterate through goal tiles. If any goal tile is currently painted with a color different from its required goal color, return float('inf').
    3. Identify unpainted goal tiles: Create a list of goal tiles that are not currently painted with their required color. Also, collect the set of colors needed for these unpainted tiles.
    4. If there are no unpainted goal tiles, the state is a goal state, so return 0.
    5. Initialize the heuristic value `h` to 0.
    6. Add Component A (Paint Actions): Add the total number of unpainted goal tiles to `h`. This represents the minimum number of paint actions required.
    7. Add Component B (Color Changes): Determine which colors are needed by at least one unpainted tile but are not currently held by any robot. For each such color, add 1 to `h`, representing the cost of one change_color action.
    8. Add Component C (Unblocking Moves): Identify unpainted goal tiles that are currently occupied by a robot. For each such tile, add 1 to `h`, representing the cost of moving the robot off the tile to allow painting access.
    9. Add Component D (Robot Movement): For each unpainted goal tile, find the minimum distance from *any* robot's current location to *any* tile adjacent to the goal tile. Sum these minimum distances over all unpainted goal tiles and add the sum to `h`. This estimates the movement cost to get robots into painting positions. If any unpainted goal tile has no adjacent tile reachable by any robot, the state is considered unsolvable, and float('inf') is returned.
    10. Return the total computed heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object containing goals, initial state,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization:
        # 1. Parse goal facts to identify which tiles need which colors.
        self.goal_paint = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paint[tile] = color

        # 2. Parse static facts to build the tile adjacency graph.
        # The graph represents possible moves between tiles.
        self.tiles = set()
        self.adj = {} # Adjacency list: tile -> [neighbor1, neighbor2, ...]

        for fact in static_facts:
            parts = get_parts(fact)
            # Predicates like (up t1 t2) mean t1 is above t2.
            # Movement is possible between adjacent tiles in any direction.
            if parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add bidirectional edges as movement is possible both ways
                self.adj.setdefault(t1, []).append(t2)
                self.adj.setdefault(t2, []).append(t1)

        # Ensure all tiles mentioned in goals are included, even if they have no static adjacency facts (unlikely in grid)
        for tile in self.goal_paint:
             self.tiles.add(tile)
             self.adj.setdefault(tile, []) # Ensure tile exists in adj dict

        # 3. Precompute all-pairs shortest paths (distances) between all tiles
        # using BFS. This allows quick lookup of movement costs.
        self.dist = {}
        for start_tile in self.tiles:
            self.dist[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}

            while queue:
                current_tile, d = queue.popleft()
                self.dist[start_tile][current_tile] = d

                # Handle tiles with no neighbors gracefully (though unexpected in grid)
                if current_tile in self.adj:
                    for neighbor in self.adj[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An estimate of the remaining cost to reach the goal, or float('inf')
            if the state is detected as unsolvable.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic:

        # 1. Extract relevant information from the current state.
        robot_loc = {}
        robot_color = {}
        current_paint = {}
        robot_on_tile = set() # Keep track of tiles occupied by robots

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
                robot_on_tile.add(tile)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_paint[tile] = color

        # 2. Check for unsolvable states: If any goal tile is painted with the wrong color.
        # Painted tiles cannot be cleared or repainted in this domain.
        for tile, goal_c in self.goal_paint.items():
            if tile in current_paint and current_paint[tile] != goal_c:
                # Goal tile is painted with the wrong color, this state is unsolvable.
                return float('inf')

        # 3. Identify unpainted goal tiles and the colors needed for them.
        unpainted_goal_tiles = [] # List of (tile, color) tuples
        needed_colors = set()     # Set of colors required for unpainted tiles

        for tile, goal_c in self.goal_paint.items():
            if tile not in current_paint: # Assumed clear if not painted with goal color
                unpainted_goal_tiles.append((tile, goal_c))
                needed_colors.add(goal_c)

        # If there are no unpainted goal tiles, the goal is reached.
        if not unpainted_goal_tiles:
            return 0

        # 4. Calculate heuristic components based on the identified tasks.
        h = 0

        # Component A: Cost for paint actions.
        # Each unpainted goal tile requires one paint action.
        h += len(unpainted_goal_tiles)

        # Component B: Cost for color changes.
        # For each color needed by at least one unpainted tile, if no robot
        # currently possesses that color, we need at least one change_color action.
        held_colors = set(robot_color.values())
        for color in needed_colors:
            if color not in held_colors:
                h += 1 # Add 1 for the cost of changing one robot's color

        # Component C: Cost for robots to move off goal tiles they are blocking.
        # If a robot is on an unpainted goal tile, it must move off before
        # another robot can potentially move adjacent to paint it.
        # We count the number of such blocking robots/tiles.
        blocking_goal_tiles = {tile for tile, _ in unpainted_goal_tiles if tile in robot_on_tile}
        h += len(blocking_goal_tiles)


        # Component D: Cost for robot movement to get into painting position.
        # For each unpainted goal tile, we need *some* robot to move to an adjacent tile.
        # We sum the minimum distances from *any* robot's current location
        # to *any* tile adjacent to the goal tile. This is a relaxation that
        # ignores robot assignment and potential conflicts.
        total_min_movement = 0
        for tile, goal_c in unpainted_goal_tiles:
            min_dist_to_tile_adj = float('inf')
            found_reachable_adj = False # Flag to check if any adjacent tile is reachable by any robot

            # Check if tile has neighbors in the graph (should be true for valid grid tiles)
            if tile in self.adj:
                for adj_tile in self.adj[tile]:
                    # Ensure the adjacent tile exists in the distance map (should be true if graph is built correctly)
                    if adj_tile in self.dist:
                        for robot, r_loc in robot_loc.items():
                             # Check if distance is known (i.e., adj_tile is reachable from r_loc)
                             if r_loc in self.dist and adj_tile in self.dist[r_loc]:
                                min_dist_to_tile_adj = min(min_dist_to_tile_adj, self.dist[r_loc][adj_tile])
                                found_reachable_adj = True # At least one adjacent tile is reachable

            # If after checking all robots and all adjacent tiles, min_dist_to_tile_adj is still inf,
            # it means no robot can reach any adjacent tile of this goal tile.
            if not found_reachable_adj:
                 # This tile is unreachable by any robot. Problem likely unsolvable from here.
                 return float('inf')

            total_min_movement += min_dist_to_tile_adj

        h += total_min_movement

        return h
