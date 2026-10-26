# Required imports
import collections
import math # For infinity

# Helper functions (as discussed)
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove leading/trailing brackets and split by space
    # Handle potential empty facts or malformed strings gracefully
    if not fact_string or not isinstance(fact_string, str) or fact_string[0] != '(' or fact_string[-1] != ')':
        return None # Indicate failure
    parts = fact_string[1:-1].split()
    if not parts:
        return None # Empty fact ()
    return tuple(parts)

def build_tile_graph_and_distances(static_facts):
    """
    Builds the tile adjacency graph and computes all-pairs shortest paths.

    Args:
        static_facts: A frozenset of static fact strings.

    Returns:
        A tuple containing:
        - graph: Adjacency dictionary {tile: [adjacent_tiles]}
        - distances: Dictionary {start_tile: {end_tile: distance}}
        - all_tiles: List of all tile names.
    """
    graph = collections.defaultdict(list)
    tiles = set()
    for fact_str in static_facts:
        fact = parse_fact(fact_str)
        if fact and fact[0] in {'up', 'down', 'left', 'right'} and len(fact) == 3:
            # Fact is (direction neighbor current_tile)
            neighbor_tile = fact[1]
            current_tile = fact[2]
            graph[current_tile].append(neighbor_tile)
            graph[neighbor_tile].append(current_tile) # Graph is undirected for movement
            tiles.add(neighbor_tile)
            tiles.add(current_tile)

    # Ensure unique neighbors
    for tile in graph:
        graph[tile] = list(set(graph[tile]))

    distances = {}
    # Perform BFS from each tile
    for start_node in tiles:
        distances[start_node] = {}
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        distances[start_node][start_node] = 0 # Distance from node to itself is 0

        while queue:
            current_node, dist = queue.popleft()

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start_node][neighbor] = dist + 1 # Store distance
                    queue.append((neighbor, dist + 1))

    return graph, distances, list(tiles)

def extract_goal_painted_tiles(goal_facts):
    """
    Extracts the required painted tiles and colors from goal facts.

    Args:
        goal_facts: A frozenset of goal fact strings.

    Returns:
        A dictionary mapping tile names to required color names.
    """
    goal_painted = {}
    for goal_fact_str in goal_facts:
        fact = parse_fact(goal_fact_str)
        if fact and fact[0] == 'painted' and len(fact) == 3:
            tile = fact[1]
            color = fact[2]
            goal_painted[tile] = color
    return goal_painted

# The heuristic class
class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each goal tile that is not yet painted correctly. The estimated
    cost for a single unpainted goal tile is the minimum cost among all
    robots to get adjacent to the tile with the required color, plus one
    action for painting.

    Assumptions:
    - The tile grid structure is defined by 'up', 'down', 'left', 'right'
      predicates in the static facts, forming a connected graph.
    - Goal tiles are either 'clear' or 'painted' with the correct color
      in any solvable state. If a goal tile is painted with the wrong color,
      the state is considered unsolvable (heuristic returns infinity).
    - The 'change_color' action is always possible if the robot has a color
      and the target color is available (guaranteed by domain preconditions).
    - All tiles mentioned in 'up', 'down', 'left', 'right' facts are part
      of the grid.
    - Robots always have a color (no 'free-color' state that requires an action to get a color).

    Heuristic Initialization:
    - Parses the static facts to build the tile graph based on adjacency
      predicates ('up', 'down', 'left', 'right').
    - Computes all-pairs shortest paths between tiles using BFS on the
      constructed graph. Stores these distances.
    - Parses the goal facts to identify which tiles need to be painted
      and with which color. Stores this mapping.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Extract relevant information from the current state: robot positions,
       robot colors, painted tiles, and clear tiles.
    4. Iterate through each goal tile `t` and its required color `c` from the
       preprocessed goal information (`self.goal_painted_tiles`).
    5. For the current goal tile `t`:
       a. Check if `(painted t c)` is true in the current state. If yes, this
          goal for tile `t` is satisfied. Continue to the next goal tile.
       b. Check if `t` is painted with a different color `c'` (`(painted t c')`
          where `c' != c`). If yes, the state is likely unsolvable. Return a
          large value (infinity).
       c. Check if `(clear t)` is true in the current state. If not, and it's
          not painted correctly (checked above), the tile is blocked or in an
          unexpected state for a solvable problem. Return a large value (infinity).
       d. If the tile needs painting (it's clear and not painted correctly),
          calculate the minimum cost to paint it:
          - Initialize `min_cost_to_paint_this_tile` to infinity.
          - Find all tiles `x` that are adjacent to `t` in the grid. These are
            the possible locations a robot can be *at* to paint tile `t`.
            These are the neighbors of `t` in the undirected graph.
          - For each robot `r` whose position `r_pos` is known:
             - Find the robot's current color `r_color`.
             - Calculate the minimum distance from `r_pos` to any tile `x`
               adjacent to `t` using the preprocessed shortest path distances.
               Let this be `min_dist_to_adj`.
             - If `min_dist_to_adj` is finite (meaning the robot can reach an
               adjacent tile):
                - Calculate the cost for the robot to get the correct color:
                  `color_cost = 0` if `r_color == c`, `color_cost = 1` otherwise
                  (assuming `required_color` is available, which is static).
                - The cost for this robot to be ready to paint `t` is
                  `min_dist_to_adj + color_cost`.
                - Update `min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)`.
          - If `min_cost_to_paint_this_tile` is still infinity after checking
            all robots, the state is likely unsolvable. Return a large value (infinity).
          - Add `min_cost_to_paint_this_tile + 1` (for the paint action) to the
            total heuristic `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        Args:
            task: The planning task object containing initial_state, goals,
                  operators, and static facts.
        """
        self.task = task
        self.static_facts = task.static
        self.goal_facts = task.goals

        # Preprocess static information
        self.tile_graph, self.tile_distances, self.all_tiles = build_tile_graph_and_distances(self.static_facts)

        # Preprocess goal information
        self.goal_painted_tiles = extract_goal_painted_tiles(self.goal_facts)

        # Define a large value for unsolvable states
        self.UNSOLVABLE = math.inf # Use math.inf for float infinity

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            An integer or float representing the estimated cost to reach the goal.
        """
        # If the state is a goal state, heuristic is 0
        if self.task.goal_reached(state):
            return 0

        h = 0

        # Extract relevant information from the current state
        robot_positions = {}
        robot_colors = {}
        painted_tiles_state = {} # {tile: color}
        clear_tiles_state = set()

        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact is None: # Handle parsing errors
                continue
            if fact[0] == 'robot-at' and len(fact) == 3:
                robot, tile = fact[1], fact[2]
                robot_positions[robot] = tile
            elif fact[0] == 'robot-has' and len(fact) == 3:
                robot, color = fact[1], fact[2]
                robot_colors[robot] = color
            elif fact[0] == 'painted' and len(fact) == 3:
                tile, color = fact[1], fact[2]
                painted_tiles_state[tile] = color
            elif fact[0] == 'clear' and len(fact) == 2:
                tile = fact[1]
                clear_tiles_state.add(tile)

        # Iterate through goal tiles that need painting
        for goal_tile, required_color in self.goal_painted_tiles.items():
            # Check if the goal for this tile is already satisfied
            if painted_tiles_state.get(goal_tile) == required_color:
                continue # This tile is already painted correctly

            # Check if the tile is painted with the wrong color
            if goal_tile in painted_tiles_state and painted_tiles_state[goal_tile] != required_color:
                # Tile is painted with the wrong color - likely unsolvable
                return self.UNSOLVABLE

            # If we reach here, the tile needs painting (it must be clear or not painted)
            # It must be clear for the paint action to be applicable.
            # The domain definition says paint requires (clear ?y).
            # If a goal tile is not clear and not painted correctly, it's unsolvable.
            # Let's explicitly check if it's clear.
            if goal_tile not in clear_tiles_state:
                 # If it's not clear and not painted correctly (checked above), it's blocked.
                 # This case shouldn't happen in solvable instances if our assumption holds.
                 # But to be safe, treat as unsolvable.
                 return self.UNSOLVABLE


            # Calculate minimum cost to paint this goal_tile with required_color
            min_cost_to_paint_this_tile = self.UNSOLVABLE

            # Find adjacent tiles for the goal tile. These are the tiles a robot must be AT to paint goal_tile.
            # The graph stores neighbors, which are exactly these tiles.
            paint_from_tiles = self.tile_graph.get(goal_tile, [])
            if not paint_from_tiles:
                 # Goal tile has no adjacent tiles? Should not happen in valid grid problems.
                 # Treat as unsolvable if it needs painting.
                 return self.UNSOLVABLE

            # Consider each robot
            for robot, r_pos in robot_positions.items():
                # Ensure robot's position is known and is a valid tile in the graph
                if r_pos not in self.tile_distances:
                    continue # Robot is in an unknown location, cannot paint

                r_color = robot_colors.get(robot) # Get robot's current color

                # Find minimum distance from robot's position to any tile it can paint from
                min_dist_to_paint_pos = self.UNSOLVABLE
                for paint_pos in paint_from_tiles:
                    if paint_pos in self.tile_distances[r_pos]:
                         min_dist_to_paint_pos = min(min_dist_to_paint_pos, self.tile_distances[r_pos][paint_pos])

                # If the robot can reach a paintable position
                if min_dist_to_paint_pos != self.UNSOLVABLE:
                    # Cost to get the correct color
                    color_cost = 0
                    # Check if robot has the required color
                    if r_color != required_color:
                         # Robot needs to change color. This costs 1 action.
                         # We assume the required_color is available (checked in init).
                         color_cost = 1

                    # Total cost for this robot to be ready to paint this tile
                    cost_for_this_robot = min_dist_to_paint_pos + color_cost

                    # Update minimum cost across all robots for this tile
                    min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)

            # If no robot can reach a paintable position, the state is unsolvable
            if min_cost_to_paint_this_tile == self.UNSOLVABLE:
                 return self.UNSOLVABLE

            # Add the minimum cost to get a robot ready + 1 for the paint action
            h += min_cost_to_paint_this_tile + 1

        return h
