from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Function to build adjacency map from static facts
def build_adjacency_map(static_facts):
    """
    Builds a graph representing the tile grid connectivity based on
    up/down/left/right predicates.
    """
    adj_map = {}
    all_tiles = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
            tile1, tile2 = parts[1], parts[2]
            all_tiles.add(tile1)
            all_tiles.add(tile2)
            if tile1 not in adj_map:
                adj_map[tile1] = set()
            if tile2 not in adj_map:
                adj_map[tile2] = set()
            # Add symmetric connections
            adj_map[tile1].add(tile2)
            adj_map[tile2].add(tile1)

    # Ensure all tiles mentioned in static facts are keys in the map, even if isolated
    for tile in all_tiles:
        if tile not in adj_map:
            adj_map[tile] = set()

    return adj_map

# Function to calculate shortest path distances using BFS
def bfs_distances(start_node, adj_map):
    """
    Computes shortest path distances from a start_node to all reachable nodes
    in the graph defined by adj_map using Breadth-First Search.
    """
    # Initialize distances for all nodes known in the map, plus the start_node if it's not already there.
    all_nodes = set(adj_map.keys())
    if start_node is not None:
        all_nodes.add(start_node)

    distances = {node: float('inf') for node in all_nodes}

    if start_node is None or start_node not in all_nodes:
        # Start node is invalid or not part of the known tile set.
        # Cannot compute distances from it.
        return distances # All distances remain inf

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is in adj_map keys before iterating neighbors
        # It should be, based on how all_nodes was built, unless adj_map[current_node] is empty.
        if current_node in adj_map:
            for neighbor in adj_map[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the number of tiles that need painting,
    the cost to acquire the necessary colors, and an estimate of the movement cost.
    It is designed for greedy best-first search and is not admissible.

    # Assumptions
    - Tiles are arranged in a grid-like structure defined by up/down/left/right predicates.
    - Tiles painted with the wrong color in the initial state make the problem unsolvable
      as there is no action to unpaint or repaint a tile that is not clear.
    - A tile must be clear to be painted or moved onto. If a robot is on a goal tile that needs painting,
      it must move off first.
    - Movement cost is estimated using shortest path distance on the tile graph.
    - Color change cost is incurred if a robot needs a color that is not currently held by any robot.

    # Heuristic Initialization
    - Extract goal conditions (`painted` facts) to identify target paintings for tiles.
    - Build the tile adjacency map from static facts (`up`, `down`, `left`, `right`) to represent the grid structure.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal tiles and their target colors from the task goals stored during initialization.
    2. For each goal tile `T` requiring color `C`:
       - Check the current state of tile `T`:
         - If `(painted T C')` is in the state where `C' != C`, the tile is painted incorrectly. Based on domain rules (no unpaint/repaint), this state is likely unsolvable. Return `float('inf')`.
         - If `(painted T C)` is already in the state, this goal is satisfied for this tile. Ignore it for heuristic calculation.
         - If `T` is not painted correctly and not painted wrong, it needs painting. Add 1 to the total heuristic for the paint action itself.
         - Check if `T` is part of the tile grid graph (i.e., mentioned in static adjacency facts). If not, and it needs painting, it's unreachable. Return `float('inf')`.
         - Check if a robot is currently on tile `T`. If `any(match(fact, "robot-at", "*", T) for fact in state)`, the robot must move off before painting. Add 1 to the total heuristic for this move-off action. After the move-off, the tile is assumed clear for painting.
         - Estimate the movement cost to get a robot adjacent to `T`. Calculate the shortest path distance from *each* robot's current location to *each* tile adjacent to `T` (using the precomputed adjacency map and BFS). Find the minimum of these distances over all robots and all adjacent tiles. If no adjacent tile is reachable from any robot, the tile cannot be painted. Return `float('inf')`. Add this minimum distance to the total heuristic.
       - Add the required color `C` to a set of colors needed for unpainted tiles.

    3. After processing all goal tiles, consider the colors needed for the unpainted tiles. For each color `C` in the set of needed colors, check if any robot currently possesses color `C`. If no robot has color `C`, add 1 to the total heuristic for the cost of a robot changing to that color. This is a simplified cost for color acquisition.

    4. The total heuristic value is the sum of the paint costs, estimated movement costs, and estimated color change costs calculated in the previous steps. If any unsolvable condition was detected, `float('inf')` is returned. If no goal tiles need painting, the heuristic is 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal paintings: Map tile name to the required color.
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Build the adjacency map (graph) of tiles from static facts.
        self.adj_map = build_adjacency_map(static_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Find current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, location = get_parts(fact)
                robot_locations[robot] = location
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # Precompute shortest path distances from each robot's current location
        # This is done per state as robot locations change.
        robot_distances = {
            robot: bfs_distances(loc, self.adj_map)
            for robot, loc in robot_locations.items()
        }

        # Identify colors currently held by robots
        colors_held = set(robot_colors.values())

        # Identify colors needed for unpainted goal tiles
        colors_needed_for_unpainted = set()

        # Iterate through goal tiles to find those not painted correctly
        unpainted_goal_tiles_info = [] # List of (tile, color) pairs that need painting

        for goal_tile, goal_color in self.goal_paintings.items():
            is_painted_correctly = False
            is_painted_wrong = False
            is_robot_on_tile = False

            # Check the state of the goal tile
            for fact in state:
                if match(fact, "painted", goal_tile, goal_color):
                    is_painted_correctly = True
                    break # This tile is done
                elif match(fact, "painted", goal_tile, "*"):
                    is_painted_wrong = True
                    break # This tile is painted wrong
                elif match(fact, "robot-at", "*", goal_tile):
                    is_robot_on_tile = True

            if is_painted_correctly:
                continue # This goal is satisfied

            # If painted wrong, assume unsolvable in this domain
            if is_painted_wrong:
                return float('inf')

            # If the tile needs painting but is not in the adjacency map (disconnected), it's unsolvable
            if goal_tile not in self.adj_map or not self.adj_map[goal_tile]:
                 return float('inf')

            # Tile needs painting
            unpainted_goal_tiles_info.append((goal_tile, goal_color))
            colors_needed_for_unpainted.add(goal_color)

            # Add cost for the paint action itself
            total_cost += 1

            # If a robot is on the tile, it needs to move off first
            if is_robot_on_tile:
                 total_cost += 1 # Cost for the robot to move off

            # Add estimated movement cost for this tile
            min_move_cost_to_adjacent = float('inf')

            # Find adjacent tiles for the goal tile
            adjacent_tiles = self.adj_map[goal_tile] # We already checked if goal_tile is in adj_map

            # Calculate min distance from any robot to any adjacent tile
            for robot, dists in robot_distances.items():
                for adj_tile in adjacent_tiles:
                    if adj_tile in dists: # Check if adj_tile is a known node and reachable
                         min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, dists[adj_tile])

            # If min_move_cost_to_adjacent is still inf, it means no robot can reach any adjacent tile.
            # This tile cannot be painted. Problem is unsolvable.
            if min_move_cost_to_adjacent == float('inf'):
                 return float('inf')

            total_cost += min_move_cost_to_adjacent

        # Add cost for color changes
        for needed_color in colors_needed_for_unpainted:
            if needed_color not in colors_held:
                total_cost += 1 # Cost to change one robot's color to this needed color

        # If unpainted_goal_tiles_info is empty, all goals are met, total_cost is 0.
        # Otherwise, total_cost > 0.

        return total_cost

