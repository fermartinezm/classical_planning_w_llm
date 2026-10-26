from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(predicate arg1 arg2)" -> ["predicate", "arg1", "arg2"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for calculating shortest path distance on the tile grid
def bfs_distance(start_node, end_node, graph):
    """
    Calculates the shortest path distance between two nodes in a graph using BFS.
    Graph is represented as {node: set(neighbors)}.
    Returns distance or None if not reachable.
    """
    if start_node == end_node:
        return 0

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, distance = queue.popleft()

        if current_node == end_node:
            return distance

        # Check if current_node exists in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return None # Not reachable

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the minimum estimated cost for each unpainted
    goal tile, considering all robots. The cost for a single tile includes the cost
    for a robot blocking the tile to move off (if necessary), the minimum cost for
    any robot to reach an adjacent tile with the correct color, and the paint action.

    # Assumptions
    - The grid is connected, allowing movement between any two tiles (ignoring 'clear' for intermediate steps in BFS distance calculation).
    - Unpainted goal tiles are assumed to be paintable once cleared (i.e., not painted with the wrong color). If a goal tile is found painted with the wrong color, the state is considered unsolvable (heuristic returns infinity).
    - Robots can reach any adjacent tile to paint a goal tile (reachability is assumed in a connected grid).
    - The cost for each unpainted goal tile is estimated independently, ignoring potential conflicts (like multiple robots needing the same adjacent tile or path).
    - The 'change_color' action cost is 1.
    - Move actions cost 1.
    - Paint actions cost 1.
    - If an unpainted goal tile is not clear, it is assumed a robot is on it and needs 1 move to leave.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Build the tile adjacency graph from the static 'up', 'down', 'left', 'right' facts to enable distance calculations. The graph is built symmetrically (if A is up from B, B is down from A, etc.).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted ?tile ?color)`. Store these target colors for each goal tile in `self.goal_paintings`.
    2. Build the tile graph: Parse static facts `(up y x)`, `(down y x)`, `(left y x)`, `(right y x)`. For each such fact, add an edge between `x` and `y` in the graph. Store this in `self.tile_graph`.
    3. For the current state:
        a. Check if the current state is a goal state. If yes, return 0.
        b. Identify the current location of each robot (`(robot-at ?robot ?tile)`).
        c. Identify the current color held by each robot (`(robot-has ?robot ?color)`). Store this robot information.
        d. Identify which goal tiles are *not* yet painted correctly. These are the unpainted goal tiles. If any goal tile is painted with a color different from its target color, return infinity (unsolvable).
        e. Identify which tiles are currently clear (`(clear ?tile)`).
    4. Initialize the total heuristic cost to 0.
    5. For each unpainted goal tile `T` requiring color `C`:
        a. Calculate the cost to clear tile `T` if it's not already clear. If `(clear T)` is not in the state, add 1 to the cost for this tile (assuming a robot is on it and needs 1 move to leave).
        b. Find all tiles `AdjT` that are adjacent to `T` in the grid graph. These are the tiles from which a robot can paint `T`.
        c. Initialize `min_cost_to_get_ready_at_adjacent = infinity`. This variable will store the minimum cost for any robot to reach an adjacent tile with the correct color.
        d. For each robot `R`:
            i. Get robot `R`'s current location `RL` and color `RC`.
            ii. Calculate the cost to change color if needed: `color_cost = 1` if `RC != C`, otherwise `0`.
            iii. Calculate the minimum move cost for robot `R` to reach *any* tile `AdjT` adjacent to `T`. This is done by finding `min(bfs_distance(RL, AdjT, tile_graph))` over all `AdjT` adjacent to `T`. The BFS is performed on the full grid graph, ignoring the 'clear' predicate for intermediate steps for simplicity and speed.
            iv. If a reachable adjacent tile `AdjT` is found (distance is not None):
                Calculate the estimated cost for robot `R` to get ready to paint tile `T` (reach adjacent tile with correct color): `robot_ready_cost = min_move_cost + color_cost`.
                Update `min_cost_to_get_ready_at_adjacent = min(min_cost_to_get_ready_at_adjacent, robot_ready_cost)`.
        e. If `min_cost_to_get_ready_at_adjacent` is not infinity (meaning at least one robot can reach an adjacent tile):
            # Total estimated cost for this tile = cost to clear it (if needed) + min cost for a robot to get ready + paint action cost (1)
            total_cost_for_T = (1 if f"(clear {tile_to_paint})" not in state else 0) + min_cost_to_get_ready_at_adjacent + 1
            total_cost += total_cost_for_T
        else:
            # No robot can reach any adjacent tile of this unpainted goal tile.
            # This state is likely unsolvable.
            return float('inf')

    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal conditions: which tiles need which color.
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # 2. Build the tile adjacency graph from static facts.
        self.tile_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for directional predicates that define grid connectivity
            if parts[0] in ["up", "down", "left", "right"]:
                # Fact is typically (direction tile1 tile2) meaning tile1 is direction of tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies movement is possible between tile1 and tile2 in both directions
                tile1, tile2 = parts[1], parts[2]
                self.tile_graph.setdefault(tile1, set()).add(tile2)
                self.tile_graph.setdefault(tile2, set()).add(tile1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 3a. Check if it's a goal state
        # A state is a goal state if all tiles in self.goal_paintings
        # are painted with the correct color in the state.
        is_goal_state = True
        for tile, target_color in self.goal_paintings.items():
            if f"(painted {tile} {target_color})" not in state:
                is_goal_state = False
                break
        if is_goal_state:
             return 0

        # 3b, 3c. Identify robot locations and colors in the current state.
        robot_info = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robot_info.setdefault(robot, {})['location'] = location
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_info.setdefault(robot, {})['color'] = color

        # 3d. Identify unpainted goal tiles.
        unpainted_goals = {}
        for tile, target_color in self.goal_paintings.items():
            # Check if the tile is already painted with the target color
            is_painted_correctly = f"(painted {tile} {target_color})" in state
            if not is_painted_correctly:
                 # Check if it's painted with the wrong color
                 # This heuristic assumes solvable states don't have goal tiles painted wrongly.
                 # If we find one, it's likely unsolvable from here.
                 is_painted_wrongly = any(match(fact, "painted", tile, "*") for fact in state if fact != f"(painted {tile} {target_color})")
                 if is_painted_wrongly:
                     # This state is likely unsolvable.
                     return float('inf')

                 # If not painted correctly (and not painted wrongly), it needs painting
                 unpainted_goals[tile] = target_color

        # If all goal tiles are painted correctly, it's a goal state (already checked above, but double check)
        if not unpainted_goals:
             return 0 # Should be caught by the initial check, but safe check

        # 4. Initialize total heuristic cost.
        total_cost = 0

        # 5. For each unpainted goal tile...
        for tile_to_paint, target_color in unpainted_goals.items():
            # 5a. Calculate the cost to clear tile_to_paint if it's not already clear.
            # Assumes if not clear, a robot is on it and needs 1 move to leave.
            cost_to_clear_tile = 1 if f"(clear {tile_to_paint})" not in state else 0

            # 5b. Find adjacent tiles to the tile_to_paint.
            # These are the tiles from which a robot can paint tile_to_paint.
            adjacent_tiles = self.tile_graph.get(tile_to_paint, set())

            if not adjacent_tiles:
                 # Goal tile has no neighbors? Unlikely in grid domains.
                 # This tile cannot be painted. Problem unsolvable from here.
                 return float('inf')

            # 5c. Initialize min cost for any robot to get ready at an adjacent tile.
            min_cost_to_get_ready_at_adjacent = float('inf')

            # 5d. For each robot...
            for robot, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                # Ensure robot info is complete
                if robot_location is None or robot_color is None:
                    continue

                # 5d.ii. Calculate the cost to change color if needed.
                color_cost = 1 if robot_color != target_color else 0

                # 5d.iii. Calculate minimum move cost for robot R to reach *any* tile AdjT adjacent to T.
                min_move_cost = float('inf')
                for adj_tile in adjacent_tiles:
                    # Calculate distance from robot's current location to the adjacent tile.
                    # Using BFS on the full graph, ignoring 'clear' for intermediate steps.
                    d = bfs_distance(robot_location, adj_tile, self.tile_graph)
                    if d is not None:
                        min_move_cost = min(min_move_cost, d)

                # 5d.iv. Calculate cost for this robot if reachable.
                if min_move_cost != float('inf'):
                    # Cost for this robot to reach adjacent tile and be ready to paint: move_cost + color_change_cost
                    robot_ready_cost = min_move_cost + color_cost
                    min_cost_to_get_ready_at_adjacent = min(min_cost_to_get_ready_at_adjacent, robot_ready_cost)

            # 5e. Add the total estimated cost for this tile to the total heuristic.
            if min_cost_to_get_ready_at_adjacent == float('inf'):
                 # No robot can reach any adjacent tile of this unpainted goal tile.
                 # This state is likely unsolvable.
                 return float('inf')

            # Total estimated cost for this tile = cost to clear it (if needed) + min cost for a robot to get ready + paint action cost (1)
            total_cost_for_T = cost_to_clear_tile + min_cost_to_get_ready_at_adjacent + 1
            total_cost += total_cost_for_T

        # 6. Return the total heuristic cost.
        return total_cost
