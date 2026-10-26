from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to check fact pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute distances from a source node
def bfs(start_node, adj_map, all_nodes):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in all_nodes}
    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        dist = distances[current_node]

        # Get neighbors, handling cases where a node might not be in adj_map keys (e.g., isolated)
        neighbors = adj_map.get(current_node, [])

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append(neighbor)

    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the cost of painting, changing colors,
    moving robots to adjacent tiles, and clearing blocked tiles.

    # Assumptions
    - Tiles are arranged in a grid structure defined by 'up', 'down', 'left', 'right' predicates.
    - Robots can move between adjacent clear tiles.
    - Robots can change color if the new color is available.
    - Robots can paint an adjacent clear tile if they have the correct color.
    - Unpainted goal tiles that are not clear are assumed to be blocked by a robot
      that needs to move off.
    - The grid is connected, meaning all tiles are reachable from each other.

    # Heuristic Initialization
    - Parses goal conditions to identify target tile colors.
    - Parses static facts ('up', 'down', 'left', 'right') to build the grid graph (adjacency map).
    - Identifies all tile objects in the problem.
    - Computes all-pairs shortest paths between tiles using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not currently painted with the correct color.
    2. Initialize the heuristic value `h` to 0.
    3. Add 1 to `h` for each identified unsatisfied goal tile (representing the paint action).
    4. Determine the set of colors required by the unsatisfied goal tiles.
    5. Determine the set of colors currently held by the robots.
    6. Add the number of required colors not currently held by any robot to `h`. This estimates the minimum number of color changes needed across the robot fleet to introduce all necessary colors.
    7. For each unsatisfied goal tile `T` requiring color `C`:
        a. Find all tiles adjacent to `T` based on the precomputed grid structure.
        b. Find the minimum shortest path distance from *any* robot's current location to *any* tile adjacent to `T`. Add this minimum distance to `h`. This estimates the movement cost for the closest robot to get into a painting position for this tile.
        c. Check if tile `T` is currently clear in the state. If it is not clear, add 1 to `h`. This estimates the cost to clear the tile (e.g., by moving a robot off it).
    8. The total value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the grid graph,
        identifying tiles, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Parse goal facts to get target tile colors
        self.goal_painted_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_painted_tiles[tile] = color

        # 2. Identify all tile names
        self.all_tiles = set()
        # Infer tile names from predicates that involve tiles in initial state and static facts
        facts_to_parse = set(initial_state) | set(static_facts)
        for fact in facts_to_parse:
             parts = get_parts(fact)
             # Predicates involving tiles: robot-at, up, down, left, right, clear, painted
             if parts and parts[0] in ['robot-at', 'up', 'down', 'left', 'right', 'clear', 'painted']:
                 # Check parts length before accessing indices
                 if parts[0] == 'robot-at' and len(parts) == 3:
                     if parts[2].startswith('tile_'): self.all_tiles.add(parts[2])
                 elif parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                     if parts[1].startswith('tile_'): self.all_tiles.add(parts[1])
                     if parts[2].startswith('tile_'): self.all_tiles.add(parts[2])
                 elif parts[0] in ['clear', 'painted'] and len(parts) >= 2: # painted has 2 args, clear has 1
                     if parts[1].startswith('tile_'): self.all_tiles.add(parts[1])


        # 3. Build the grid graph (adjacency map) from static facts
        self.adj_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                # Ensure both tiles are recognized tiles before adding adjacency
                if len(parts) == 3 and parts[1] in self.all_tiles and parts[2] in self.all_tiles:
                    tile1, tile2 = parts[1], parts[2]
                    # Adjacency is bidirectional for movement
                    self.adj_map.setdefault(tile1, []).append(tile2)
                    self.adj_map.setdefault(tile2, []).append(tile1)

        # Ensure all tiles found are in the adj_map keys, even if they have no neighbors listed in static (unlikely in grid)
        for tile in self.all_tiles:
             self.adj_map.setdefault(tile, [])


        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_tiles:
            node_distances = bfs(start_node, self.adj_map, self.all_tiles)
            for end_node, dist in node_distances.items():
                 self.distances[(start_node, end_node)] = dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Convert state frozenset to set for potentially faster lookups
        state_set = set(state)

        # 1. Identify unsatisfied goal tiles
        unsatisfied_goals = {} # {tile: color}
        for tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is painted with the correct color in the current state
            # The goal is (painted T C). If this exact fact is not in state, it's unsatisfied.
            if f"(painted {tile} {goal_color})" not in state_set:
                 unsatisfied_goals[tile] = goal_color


        # 2. Initialize heuristic value
        h = 0

        # 3. Add cost for paint actions
        h += len(unsatisfied_goals)

        # 4. Identify needed colors
        needed_colors = set(unsatisfied_goals.values())

        # 5. Identify held colors by robots and robot locations
        held_colors = set()
        robot_locations = {} # {robot_name: tile_name}
        for fact in state_set:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                held_colors.add(color)
            elif parts and parts[0] == "robot-at" and len(parts) == 3:
                 robot, location = parts[1], parts[2]
                 robot_locations[robot] = location

        # 6. Add cost for acquiring new colors
        colors_to_acquire = needed_colors - held_colors
        h += len(colors_to_acquire) # Minimum number of color changes needed across fleet

        # 7. Add cost for movement and clearing for each unsatisfied goal tile
        for tile, color in unsatisfied_goals.items():
            # Find tiles adjacent to the target tile
            adjacent_tiles = self.adj_map.get(tile, [])

            # Find minimum distance from any robot to any adjacent tile
            min_dist_to_adj = float('inf')
            
            # Only calculate distance if there are robots and adjacent tiles
            if robot_locations and adjacent_tiles:
                for robot_loc in robot_locations.values():
                    for adj_tile in adjacent_tiles:
                        dist = self.distances.get((robot_loc, adj_tile), float('inf'))
                        min_dist_to_adj = min(min_dist_to_adj, dist)

            # Add movement cost (if reachable). If min_dist_to_adj is inf, it means
            # the tile is unreachable from any robot location, which implies unsolvability
            # or a dead end state in a connected grid. Add a large penalty.
            if min_dist_to_adj == float('inf'):
                 # This tile is unreachable by any robot. This state is likely a dead end.
                 # Add a large penalty.
                 h += 1000 # Arbitrary large number
            else:
                 h += min_dist_to_adj


            # 8. Add cost for clearing the tile if it's not clear
            # Check if '(clear tile_X_Y)' is NOT in the state
            is_clear = False
            # Check if the specific clear fact exists in the state set
            if f"(clear {tile})" in state_set:
                 is_clear = True

            if not is_clear:
                 h += 1 # Cost to move a robot off this tile (simplified)

        return h
