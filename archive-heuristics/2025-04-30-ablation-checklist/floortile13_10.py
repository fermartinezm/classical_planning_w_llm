import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument (with wildcard support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, adj_list, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all other reachable nodes in a graph.

    Args:
        start_node: The node to start the BFS from.
        adj_list: A dictionary representing the graph's adjacency list.
        all_nodes: A set of all nodes in the graph.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from start_node.
    """
    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in adj_list: # Ensure the node has neighbors defined
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with their target colors. It sums the estimated cost for
    each unpainted goal tile independently. The cost for a single tile
    is the minimum cost for any robot to reach an adjacent tile, change
    color if necessary, and paint the tile.

    # Assumptions
    - The grid is connected.
    - Tiles that need painting according to the goal are currently 'clear'.
    - Robots always have a color initially and can change it if needed.
    - The cost of moving, changing color, and painting is 1 action each.

    # Heuristic Initialization
    - Precomputes shortest path distances between all pairs of tiles using BFS.
    - Stores the goal conditions (which tiles need which color).
    - Stores the adjacency information for each tile (which tiles are neighbors).

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location and color of each robot.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal predicate of the form `(painted ?tile ?color)`.
    4. For each such goal predicate:
       a. Check if the predicate `(painted ?tile ?color)` is already true in the current state. If it is, this goal is satisfied, and the cost for this tile is 0. Continue to the next goal tile.
       b. If the goal predicate is not true, the tile `?tile` needs to be painted with `?color`.
       c. Calculate the minimum cost for *any* robot to paint this tile:
          i. Initialize `min_cost_for_tile = infinity`.
          ii. For each robot:
              - Get the robot's current location `robot_loc` and color `robot_color`.
              - Determine the cost to change color if needed: `color_cost = 1` if `robot_color` is not `?color`, otherwise `0`.
              - To paint `?tile`, the robot must be on an *adjacent* tile. Find all neighbors of `?tile`.
              - Calculate the minimum distance from the robot's current location `robot_loc` to *any* of the neighbors of `?tile`. Let this be `min_dist_to_neighbor`.
              - The estimated cost for this robot to paint this tile is `min_dist_to_neighbor + color_cost + 1` (move to neighbor + change color + paint).
              - Update `min_cost_for_tile = min(min_cost_for_tile, estimated_cost_for_this_robot)`.
       d. Add `min_cost_for_tile` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and storing goal info.
        """
        super().__init__(task)
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Identify all tiles
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                all_tiles.add(parts[1])
                all_tiles.add(parts[2])

        self.all_tiles = list(all_tiles) # Convert to list for consistent ordering if needed, though not strictly necessary here

        # 2. Build adjacency list and neighbor mapping from static facts
        self.adj = {tile: [] for tile in self.all_tiles}
        self.neighbors = {tile: set() for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                # PDDL relation (dir tile1 tile2) means tile1 is dir from tile2
                # So, tile2 is a neighbor of tile1, and tile1 is a neighbor of tile2
                # And robot moves from tile2 to tile1 using move_dir action
                # And robot paints tile1 while at tile2 using paint_dir action
                # The graph edges are for movement: robot moves between adjacent tiles.
                # (up y x) means y is up from x. Robot at x can move to y. Robot at y can move to x (down).
                # Adjacency for movement: x <-> y if (up y x) or (down y x) or (left y x) or (right y x)
                # Neighbors for painting: to paint y, robot must be at x if (up y x) etc.
                # Let's build adjacency for movement and store painting neighbors separately.

                # Adjacency for movement (bidirectional)
                self.adj[tile1].append(tile2)
                self.adj[tile2].append(tile1)

                # Painting neighbors: to paint tile1, robot must be at tile2
                self.neighbors[tile1].add(tile2)


        # Remove duplicate edges in adj list
        for tile in self.adj:
             self.adj[tile] = list(set(self.adj[tile]))


        # 3. Precompute all-pairs shortest paths using BFS from each tile
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = bfs(start_tile, self.adj, self.all_tiles)

        # 4. Identify robots (assuming they are objects of type robot)
        # We can get robots from the initial state or static facts if available,
        # or just look for objects involved in robot-at/robot-has predicates in goals/initial state.
        # A robust way is to parse the initial state or goal for robot predicates.
        self.robots = set()
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if parts[0] in ["robot-at", "robot-has"]:
                  self.robots.add(parts[1])
        self.robots = list(self.robots) # Convert to list


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                robot_colors[parts[1]] = parts[2]

        total_cost = 0  # Initialize action cost counter.

        # 2. Iterate through goal painted tiles
        goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                goal_painted_tiles[(tile, color)] = True # Mark this tile/color as a goal

        # Check which goal painted tiles are NOT satisfied
        unsatisfied_goals = set(goal_painted_tiles.keys())
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "painted":
                  tile, color = parts[1], parts[2]
                  if (tile, color) in unsatisfied_goals:
                       unsatisfied_goals.remove((tile, color))

        # 3. Calculate cost for each unsatisfied goal tile
        for tile, target_color in unsatisfied_goals:
            # Tile needs to be painted with target_color

            min_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot_name in self.robots:
                robot_loc = robot_locations.get(robot_name)
                robot_color = robot_colors.get(robot_name)

                # If robot info is missing (shouldn't happen in valid states, but safety)
                if robot_loc is None or robot_color is None:
                    continue

                # Cost to change color if needed
                color_cost = 0 if robot_color == target_color else 1

                # Find minimum distance from robot_loc to any neighbor of the target tile
                min_dist_to_neighbor = float('inf')
                painting_neighbors = self.neighbors.get(tile, set()) # Tiles robot must be at to paint 'tile'

                if not painting_neighbors:
                    # This tile has no defined painting neighbors in static facts,
                    # which shouldn't happen for goal tiles in a valid problem.
                    # Treat as unreachable or assign a very high cost.
                    # For now, assume it's reachable via some neighbor.
                    # If the grid is regular, neighbors can be inferred, but relying on PDDL is safer.
                    # If painting_neighbors is empty, this tile cannot be painted by any robot.
                    # This state is likely on an unsolvable path or the problem is malformed.
                    # A high heuristic value is appropriate.
                    min_dist_to_neighbor = float('inf') # Cannot paint this tile

                for neighbor_loc in painting_neighbors:
                    if robot_loc in self.distances and neighbor_loc in self.distances[robot_loc]:
                         dist = self.distances[robot_loc][neighbor_loc]
                         min_dist_to_neighbor = min(min_dist_to_neighbor, dist)
                    # else: neighbor_loc is unreachable from robot_loc (dist remains inf)


                # Estimated cost for this robot to paint this tile:
                # move_cost + color_cost + paint_cost
                # move_cost is min_dist_to_neighbor
                # paint_cost is 1
                if min_dist_to_neighbor != float('inf'):
                    cost_this_robot = min_dist_to_neighbor + color_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

            # If the tile is unreachable by any robot, min_cost_for_tile remains inf.
            # This state is likely a dead end or part of an unsolvable problem.
            # Returning inf signals this.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            total_cost += min_cost_for_tile

        return total_cost

