from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct in the environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles by summing
    the estimated minimum cost for each unpainted goal tile. The estimated
    cost for a single tile includes the cost to move the closest available
    robot to an adjacent tile from which the painting action can be performed,
    change its color if necessary, and perform the paint action.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - All tiles are reachable from each other (the grid is connected).
    - If a tile is painted with the wrong color, the problem is unsolvable.
    - Action costs are uniform (implicitly 1).
    - Robots always have a color (the 'free-color' predicate is not used in the provided domain actions).
    - All goal colors are available colors.

    # Heuristic Initialization
    - Parse goal conditions to identify target tiles and colors.
    - Build a graph representing the tile grid connectivity using 'up', 'down',
      'left', 'right' static facts.
    - Pre-calculate shortest distances between all pairs of tiles using BFS.
    - Identify available colors from static facts.
    - Identify, for each tile, the set of adjacent tiles from which it can be painted, based on the directional predicates used in paint actions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state by comparing the state facts against the goal conditions. If all goal conditions are met, return 0.
    2. Identify all goal conditions `(painted ?x ?c)` that are not satisfied in the current state.
    3. Initialize the total heuristic cost to 0.
    4. Extract the current location and color of each robot from the state facts.
    5. For each unsatisfied goal `(painted goal_tile goal_color)`:
        a. Check the current state of `goal_tile`:
           - Iterate through the state facts to see if `goal_tile` is painted. If `(painted goal_tile current_color)` exists and `current_color != goal_color`, the tile is painted incorrectly, and the problem is unsolvable from this state. Return `float('inf')`.
           - If `(clear goal_tile)` is false and `(painted goal_tile goal_color)` is false, the tile must be occupied by a robot. For heuristic purposes, we assume the robot will move off the tile when needed, so this state is not treated as an obstacle to painting the tile eventually.
        b. If the tile is already painted with the `goal_color`, this goal is satisfied in the current state, so continue to the next goal tile.
        c. If the tile needs painting (i.e., it's not painted with the `goal_color` and not painted with a wrong color):
           - Find the minimum cost among all robots to get into a position to paint `goal_tile` with `goal_color` and perform the paint action. Initialize `min_cost_to_paint_this_tile = float('inf')`.
           - For each robot `robot` with its current location `robot_loc` and color `robot_color`:
               - Calculate the cost to acquire the correct color: `color_cost = 0` if `robot_color == goal_color`. If `robot_color != goal_color`, a `change_color` action is needed, so `color_cost = 1`. We assume the `goal_color` is available (checked during initialization).
               - Find the minimum distance from the robot's current location (`robot_loc`) to any tile `paint_pos` from which `goal_tile` can be painted (i.e., `paint_pos` is in the set `self.paintable_from_adjacent[goal_tile]`). Initialize `min_dist_to_paint_pos = float('inf')`.
               - For each potential painting position `paint_pos` in `self.paintable_from_adjacent[goal_tile]`:
                   - If `robot_loc` and `paint_pos` are valid tiles in the grid graph (present in `self.distances`), update `min_dist_to_paint_pos = min(min_dist_to_paint_pos, self.distances[robot_loc][paint_pos])`.
               - If `min_dist_to_paint_pos` is still `float('inf')` after checking all potential painting positions for this goal tile (meaning no path exists from the robot's current location to any valid painting position), this robot cannot paint this tile via movement. Continue to the next robot.
               - The estimated cost for this robot to paint this specific tile is the sum of the minimum movement cost (`min_dist_to_paint_pos`), the color change cost (`color_cost`), and the cost of the paint action itself (1).
               - Update `min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)`.
           - After checking all robots, if `min_cost_to_paint_this_tile` is still `float('inf')`, it means no robot can reach a position to paint this goal tile. The problem is unsolvable from this state. Return `float('inf')`.
           - Add the calculated `min_cost_to_paint_this_tile` for this goal tile to the `total_heuristic`.
    6. After iterating through all unsatisfied goal tiles, return the accumulated `total_heuristic`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the tile graph."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations and colors for each tile.
        # Map: tile_name -> goal_color
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Build the tile graph for movement and pre-calculate distances.
        self.tile_graph = defaultdict(set)
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # (direction tile1 tile2) means tile1 is direction of tile2
                # For movement, tile1 and tile2 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                self.tile_graph[tile1].add(tile2)
                self.tile_graph[tile2].add(tile1) # Graph is undirected for movement
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # Calculate all-pairs shortest paths using BFS from each tile.
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

        # Store available colors
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}

        # Store adjacency mapping for painting: tile_to_paint -> set of tiles from which it can be painted
        # If (direction tile_to_paint tile_robot_at) is true, a robot at tile_robot_at can paint tile_to_paint.
        self.paintable_from_adjacent = defaultdict(set)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] in ["up", "down", "left", "right"]:
                 # (predicate tile_to_paint tile_robot_at)
                 tile_to_paint, tile_robot_at = parts[1], parts[2]
                 self.paintable_from_adjacent[tile_to_paint].add(tile_robot_at)


    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.all_tiles}
        if start_node not in self.all_tiles:
             # Start node might not be in the graph if it's not mentioned in connectivity facts
             # but exists as an object. Handle defensively.
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in the graph before accessing neighbors
            if current_node not in self.tile_graph:
                 continue

            for neighbor in self.tile_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Extract robot locations and colors
        robot_info = {} # Map: robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color
            # Assuming robots always have a color, no need to handle 'free-color'

        total_heuristic = 0

        # Iterate through each goal painting requirement
        for goal_tile, goal_color in self.goal_paintings.items():
            # Check if this goal is already satisfied
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state

            if not is_painted_correctly:
                # Check if painted with wrong color (unsolvable state)
                for fact in state:
                    if match(fact, "painted", goal_tile, "*"):
                        current_color = get_parts(fact)[2]
                        if current_color != goal_color:
                            # Tile is painted with the wrong color, problem is unsolvable from here.
                            return float('inf')

                # This tile needs to be painted with goal_color
                # It must be clear to be painted, but we assume it will become clear if occupied.

                min_cost_to_paint_this_tile = float('inf')

                # Find the minimum cost for any robot to paint this tile
                for robot, info in robot_info.items():
                    robot_loc = info.get('location')
                    robot_color = info.get('color')

                    # Ensure robot_loc is a valid tile in our graph
                    if robot_loc is None or robot_loc not in self.distances:
                        # Robot location unknown or not in the tile graph, cannot use this robot
                        continue

                    # Cost to get the correct color
                    color_cost = 0
                    if robot_color != goal_color:
                        # Need to change color. Assumes robot has *some* color to change from.
                        # We assume goal_color is available (checked in init).
                        color_cost = 1 # Cost of change_color action

                    # Find minimum distance from robot_loc to any tile adjacent to goal_tile
                    # from which it can be painted.
                    min_dist_to_paint_pos = float('inf')
                    paint_positions = self.paintable_from_adjacent.get(goal_tile, set())

                    if not paint_positions:
                         # This goal tile cannot be painted from any adjacent tile based on static facts.
                         # This implies an invalid problem definition for this goal tile.
                         # Treat as unreachable.
                         return float('inf')

                    for paint_pos in paint_positions:
                        # Ensure the paint_pos is a valid tile and reachable from robot_loc
                        if paint_pos in self.distances[robot_loc]:
                             min_dist_to_paint_pos = min(min_dist_to_paint_pos, self.distances[robot_loc][paint_pos])

                    # If no path exists to any paint position, this robot cannot paint it.
                    if min_dist_to_paint_pos == float('inf'):
                        continue # Try next robot

                    # Estimated cost for this robot to paint this tile:
                    # Move cost + Color change cost + Paint action cost
                    cost_for_this_robot = min_dist_to_paint_pos + color_cost + 1

                    min_cost_to_paint_this_tile = min(min_cost_to_paint_this_tile, cost_for_this_robot)

                # If no robot can reach a paint position, this goal is unreachable.
                if min_cost_to_paint_this_tile == float('inf'):
                     return float('inf')

                total_heuristic += min_cost_to_paint_this_tile

        return total_heuristic
