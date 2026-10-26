from collections import deque

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Removes parentheses and splits by space."""
    # Handle potential empty string or malformed fact
    if not fact_string or not fact_string.strip():
        return None, []
    # Strip outer parentheses and split
    parts = fact_string.strip().strip('()').split()
    if not parts:
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class floortileHeuristic:
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the minimum estimated costs for each tile that needs to be painted
        according to the goal but is currently not painted with the required color.
        The estimated cost for painting a single tile is the minimum cost over
        all robots, considering the cost to change color (if needed), the cost
        to move to an adjacent tile, and the cost of the paint action itself.
        The movement cost is calculated as the shortest path distance on the
        static grid graph, ignoring the 'clear' precondition for movement.

    Assumptions:
        - The grid defined by 'up', 'down', 'left', 'right' predicates is
          connected.
        - Robots always possess a color (as per domain initialization and actions).
        - Tiles that are painted in the goal state are initially clear or
          already painted with the correct color. Repainting or unpainting
          is not possible in this domain.

    Heuristic Initialization:
        The constructor processes the static information from the task.
        - It identifies the goal conditions, specifically which tiles need
          to be painted and with which color, storing this in `self.goal_painted`.
        - It builds an undirected adjacency list representation of the grid
          graph based on the 'up', 'down', 'left', and 'right' predicates,
          storing this in `self.grid_adj`. This graph is used for calculating
          shortest path distances between tiles.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to determine robot locations and robot colors.
           (Other state facts like 'clear' and 'painted' are checked directly
           against the goal).
        2. Identify the set of 'unpainted goal tiles'. These are the tiles
           specified in the goal for which the corresponding '(painted tile color)'
           fact is not present in the current state.
        3. Initialize the total heuristic value to 0.
        4. If there are no robots but there are unpainted goal tiles, the problem
           is likely unsolvable from this state. Return a large penalty.
        5. For each tile `t` that needs to be painted with color `c` (i.e.,
           `(painted t c)` is a goal fact and the fact is not in the state):
            a. Find the set of tiles `adj_t` that are adjacent to `t` based on
               the pre-calculated grid graph (`self.grid_adj`).
            b. If `t` has no adjacent tiles in the graph, it cannot be painted.
               Add a large penalty for this tile and continue to the next.
            c. Initialize the minimum cost to paint tile `t` (`min_cost_for_tile`)
               to infinity.
            d. For each robot `r`:
                i. Get the robot's current location `r_loc` and color `r_color`
                   from the current state.
                ii. Calculate the cost to change the robot's color to `c`:
                    This is 1 if `r_color` is not `c`, and 0 otherwise.
                iii. Calculate the shortest path distances from `r_loc` to all
                     other reachable tiles using Breadth-First Search (BFS)
                     on the grid graph (`self.grid_adj`).
                iv. Find the minimum distance from `r_loc` to any tile in the
                    set `adj_t`. This is the estimated movement cost (`move_cost`).
                    If no tile in `adj_t` is reachable from `r_loc` (based on BFS),
                    the move cost is considered infinite for this robot and tile.
                v. If a path exists (`move_cost` is finite), calculate the total
                   estimated cost for robot `r` to paint tile `t`:
                   `cost_for_robot = color_cost + move_cost + 1` (where 1 is the
                   cost of the paint action).
                vi. Update `min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)`.
            e. If `min_cost_for_tile` is still infinity after checking all robots,
               it means no robot can reach an adjacent tile to paint `t`. This
               tile is considered unreachable. Add a large penalty (e.g., 1000)
               to the total heuristic.
            f. Otherwise, add `min_cost_for_tile` to the total heuristic value.
        6. Return the total heuristic value. If the set of unpainted goal tiles
           was empty, the heuristic is 0, correctly identifying a goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.goal_painted = {}  # {tile: color} for goal tiles
        self.grid_adj = {}      # {tile: set of adjacent tiles}

        # Parse goal facts
        for fact_string in task.goals:
            predicate, args = parse_fact(fact_string)
            if predicate == 'painted':
                if len(args) == 2:
                    tile, color = args
                    self.goal_painted[tile] = color

        # Parse static facts to build grid graph
        for fact_string in task.static:
            predicate, args = parse_fact(fact_string)
            if predicate in {'up', 'down', 'left', 'right'}:
                if len(args) == 2:
                    tile1, tile2 = args
                    self.grid_adj.setdefault(tile1, set()).add(tile2)
                    self.grid_adj.setdefault(tile2, set()).add(tile1) # Grid is undirected for movement

    def bfs(self, start_node, graph):
        """
        Performs Breadth-First Search to find shortest distances from a start node.

        Args:
            start_node: The tile to start the BFS from.
            graph: The adjacency list representation of the grid.

        Returns:
            A dictionary mapping reachable tiles to their shortest distance
            from the start_node. Returns an empty dictionary if start_node
            is not in the graph.
        """
        if start_node not in graph:
             return {} # Start node is isolated or doesn't exist in graph

        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            if current_node in graph: # Check again, just in case graph is inconsistent
                for neighbor in graph[current_node]:
                    if neighbor not in distances: # Check if visited
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach a goal state.
        """
        # Parse current state
        current_robot_locations = {}
        current_robot_colors = {}

        # Use a set for faster lookup of state facts
        state_facts = set(state)

        for fact_string in state_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'robot-at':
                if len(args) == 2:
                    robot, tile = args
                    current_robot_locations[robot] = tile
            elif predicate == 'robot-has':
                 if len(args) == 2:
                    robot, color = args
                    current_robot_colors[robot] = color

        # Identify unpainted goal tiles
        unpainted_goal_tiles = []
        for tile, required_color in self.goal_painted.items():
            # A tile needs painting if the required painted fact is NOT in the state
            if f'(painted {tile} {required_color})' not in state_facts:
                 unpainted_goal_tiles.append((tile, required_color))

        # If all goal tiles are painted correctly, it's a goal state
        if not unpainted_goal_tiles:
            return 0

        total_heuristic = 0
        UNREACHABLE_PENALTY = 1000 # Penalty for a tile that no robot can paint

        # If there are no robots but tiles need painting, they are unreachable
        if not current_robot_locations:
             return len(unpainted_goal_tiles) * UNREACHABLE_PENALTY


        for tile_to_paint, required_color in unpainted_goal_tiles:
            min_cost_for_tile = float('inf')

            # Find tiles adjacent to tile_to_paint
            adj_tiles_to_paint = self.grid_adj.get(tile_to_paint, set())

            # If tile_to_paint has no adjacent tiles (unlikely in grid), it's impossible to paint
            if not adj_tiles_to_paint:
                # This tile cannot be painted. Problem likely unsolvable.
                total_heuristic += UNREACHABLE_PENALTY
                continue # Move to next unpainted tile

            # Iterate through all robots to find the minimum cost to paint this tile
            for robot_name, robot_loc in current_robot_locations.items():
                robot_color = current_robot_colors.get(robot_name) # Get color

                # Cost to get the required color
                # Assumes robot always has a color based on domain init/actions
                # If robot_color is None, it implies the robot-has fact is missing,
                # which shouldn't happen in a valid state based on the domain.
                # Treat None color as needing a change.
                color_cost = 1 if robot_color != required_color else 0

                # Calculate distances from robot's current location using BFS
                dist_from_robot = self.bfs(robot_loc, self.grid_adj)

                # Find minimum move cost to any adjacent tile
                move_cost = float('inf')
                # Check if robot_loc is actually in the grid graph and reachable (should be distance 0)
                if robot_loc in dist_from_robot:
                    for adj_tile in adj_tiles_to_paint:
                        if adj_tile in dist_from_robot: # Check reachability of adjacent tile
                            move_cost = min(move_cost, dist_from_robot[adj_tile])

                # If robot can reach an adjacent tile
                if move_cost != float('inf'):
                    # Total cost for this robot to paint this tile
                    # color_cost (change color) + move_cost (to adjacent tile) + 1 (paint action)
                    cost_for_robot = color_cost + move_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # Add the minimum cost found for this tile to the total heuristic
            if min_cost_for_tile != float('inf'):
                total_heuristic += min_cost_for_tile
            else:
                # No robot can reach any adjacent tile of this unpainted goal tile.
                # This tile is unreachable. Add a large penalty.
                total_heuristic += UNREACHABLE_PENALTY

        return total_heuristic
