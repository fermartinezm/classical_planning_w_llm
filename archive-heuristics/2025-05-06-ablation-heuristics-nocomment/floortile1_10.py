from collections import deque
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# Define the heuristic class, potentially inheriting from Heuristic
# class floortileHeuristic(Heuristic):
class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    This heuristic estimates the number of actions required to reach the
    goal state. It calculates this estimate by summing two main components:
    1. The number of unpainted goal tiles (each requiring one paint action).
    2. For each color required by the unpainted goal tiles, the minimum
       cost for any robot to acquire that color (if needed) and reach
       the closest tile adjacent to any goal tile requiring that color.
       This captures the cost of getting robots to the "work areas" for
       each required color.

    Assumptions:
    - Action costs are uniform (implicitly 1 per action).
    - The grid structure defined by the static 'up', 'down', 'left',
      'right' predicates is consistent and represents the possible
      movements and painting positions.
    - Problem instances are solvable from the initial state; specifically,
      no goal tile is painted with an incorrect color in any reachable state.
    - Robots always possess a color (the 'free-color' predicate is not
      used in a way that requires picking up paint from a source).

    Heuristic Initialization:
    The `__init__` method performs necessary pre-calculations based on
    the static information from the planning task:
    1. Stores the set of goal facts.
    2. Builds an undirected graph representing the grid connectivity,
       where nodes are tiles and edges are defined by the static
       'up', 'down', 'left', 'right' predicates.
    3. Computes the shortest path distance between all pairs of tiles
       in the grid graph using Breadth-First Search (BFS). These distances
       represent the minimum number of move actions between tiles.
    4. Stores the required color for each tile specified in the goal.
    5. For each goal tile, determines and stores the set of adjacent
       tiles where a robot must be positioned to paint that goal tile.
       Based on the PDDL actions, these are the direct neighbors in the grid.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1. Parse the current state to extract the current location and color
       of each robot, and the painted/clear status of all tiles.
    2. Identify all goal tiles that are not currently painted with their
       required goal color.
    3. Check for unsolvable conditions: If any goal tile is found to be
       painted with a color different from its required goal color, the
       state is a dead end, and the heuristic returns infinity.
    4. From the tiles identified in step 2, filter down to the set of
       unpainted goal tiles that are currently 'clear'. These are the
       tiles that still need to be painted. Let this set be U.
    5. If the set U is empty, all goal painting conditions are met,
       and the heuristic returns 0.
    6. Initialize the heuristic value `h`. The minimum number of paint
       actions required is equal to the number of tiles in U. So,
       `h = len(U)`.
    7. Determine the set of distinct colors required by the tiles in U.
    8. For each color C in the set of required colors:
        a. Identify the "work area" for this color: the set of all tiles
           that are adjacent to *any* unpainted goal tile requiring color C.
           Let this set be S_C.
        b. Calculate the minimum cost for *any* robot to reach *any* tile
           in S_C while possessing color C. This cost is computed by
           considering each robot R:
           - The cost to acquire color C: 1 if Robot R's current color
             is different from C, otherwise 0.
           - The minimum movement cost for Robot R to reach any tile
             in S_C (using the pre-calculated distances).
           The minimum cost for color C is the minimum of (color acquisition
           cost + movement cost) over all robots R.
        c. Add this minimum cost calculated in step 8b to the total
           heuristic value `h`.
    9. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating static information.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the grid graph for movement and painting adjacency
        self.graph = {}
        self.all_tiles = set()

        # Static predicates define grid connectivity
        # (up y x), (down y x), (left y x), (right y x)
        # These imply an edge between x and y.
        # Robot at x can paint y if (direction y x) is true.
        # So, the tiles adjacent for painting a tile T are its neighbors in the grid.
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)
                self.graph.setdefault(tile1, set()).add(tile2)
                self.graph.setdefault(tile2, set()).add(tile1) # Grid is generally traversable both ways

        # Ensure all tiles mentioned in goals are included, even if isolated
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'painted':
                 self.all_tiles.add(parts[1])
                 self.graph.setdefault(parts[1], set()) # Add tile even if no connections

        self.all_tiles = list(self.all_tiles) # Convert to list for consistent iteration

        # Compute all-pairs shortest paths (distances) using BFS
        self.dist = {}
        for start_node in self.all_tiles:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, d = q.popleft()
                self.dist[start_node][current_node] = d
                for neighbor in self.graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # Store goal painted requirements: {tile: color}
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Store adjacent tiles for each goal tile (tiles where robot must be to paint)
        # This is the set of neighbors in the grid graph.
        self.adj_to_goal_tile = {}
        for goal_tile in self.goal_painted:
             self.adj_to_goal_tile[goal_tile] = list(self.graph.get(goal_tile, []))


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # 1. Parse current state
        robot_loc = {}
        robot_color = {}
        current_painted = {}
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_loc[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == 'clear':
                tile = parts[1]
                current_clear.add(tile)

        # 2. & 3. Identify unpainted goal tiles and check for unsolvable state
        unpainted_goal_tiles = [] # List of (tile, color) tuples
        for goal_tile, required_color in self.goal_painted.items():
            current_color_on_tile = current_painted.get(goal_tile)

            if current_color_on_tile is None: # Tile is not in current_painted facts
                 # If it's a goal tile and not painted, it must be clear to be paintable
                 if goal_tile in current_clear:
                     unpainted_goal_tiles.append((goal_tile, required_color))
                 # If it's not painted and not clear, this shouldn't happen in valid states
                 # based on domain actions (painted tiles become not clear, moving makes tile clear)
                 # We assume valid states are reachable.
            elif current_color_on_tile != required_color:
                # Goal tile is painted with the wrong color -> unsolvable
                return float('inf')
            # If current_color_on_tile == required_color, the goal is met for this tile.

        # 5. If U is empty, goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 6. Base heuristic: number of paint actions needed
        h = len(unpainted_goal_tiles)

        # 7. Find required colors among unpainted goal tiles
        required_colors = {color for tile, color in unpainted_goal_tiles}

        # 8. Add cost for movement and color changes per required color area
        for color in required_colors:
            # 8a. Determine the work area S_C for this color
            S_C = set()
            for tile, required_color in unpainted_goal_tiles:
                if required_color == color:
                    # Add adjacent tiles for this goal tile to S_C
                    S_C.update(self.adj_to_goal_tile.get(tile, []))

            # S_C should not be empty if color is in required_colors and unpainted_goal_tiles is not empty

            # 8b. Calculate minimum cost for any robot to reach S_C with color C
            min_cost_to_serve_color_C = float('inf')

            for robot, current_loc in robot_loc.items():
                current_color = robot_color[robot]

                # Cost to acquire the required color C
                color_change_cost = 1 if current_color != color else 0

                # Minimum moves for this robot to reach any tile in S_C
                min_moves_to_Sc = float('inf')
                for adj_tile in S_C:
                    # Check if distance is known (i.e., tile is reachable)
                    if current_loc in self.dist and adj_tile in self.dist[current_loc]:
                         min_moves_to_Sc = min(min_moves_to_Sc, self.dist[current_loc][adj_tile])
                    # If adj_tile is not in dist[current_loc], it's unreachable from current_loc,
                    # min_moves_to_Sc remains inf for this adj_tile.

                # If min_moves_to_Sc is still inf after checking all tiles in S_C,
                # this robot cannot reach the work area for color C.
                if min_moves_to_Sc == float('inf'):
                    cost_R_serve_C = float('inf')
                else:
                    # Total cost for robot R to be ready to paint color C tiles
                    cost_R_serve_C = color_change_cost + min_moves_to_Sc

                # Update the minimum cost across all robots for this color area
                min_cost_to_serve_color_C = min(min_cost_to_serve_color_C, cost_R_serve_C)

            # If no robot can reach the area for this color, the problem is unsolvable
            if min_cost_to_serve_color_C == float('inf'):
                 return float('inf')

            # 8c. Add this minimum cost to the heuristic
            h += min_cost_to_serve_color_C

        # 9. Return total heuristic value
        return h

