from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
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
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their target colors. It sums the estimated cost for each
    unpainted goal tile independently. The cost for a single tile is estimated
    as the minimum cost for any robot to reach an adjacent tile with the correct
    color, plus the cost of the paint action itself.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates is static.
    - Robots can move between adjacent tiles if the destination is clear.
    - A tile is clear if no robot is on it and it is not painted.
    - Robots can change color if the target color is available.
    - All colors mentioned in the problem are available.
    - The heuristic calculates movement cost based on the static grid graph,
      ignoring dynamic 'clear' constraints on intermediate tiles, but requiring
      the target tile for painting to be clear (which is true for unpainted
      goal tiles not occupied by a robot).
    - The cost of each action (move, change_color, paint) is assumed to be 1 for heuristic calculation.

    # Heuristic Initialization
    - Extracts the goal conditions to identify target tiles and colors.
    - Builds the tile adjacency graph from the static up/down/left/right facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task's goals.
    2. In the constructor, build the adjacency list representation of the tile grid
       using the static up/down/left/right predicates. This graph represents
       possible movements between tiles.
    3. In the __call__ method, for the given state:
       - Identify which goal tiles are not yet painted correctly. Check also if any goal tile
         is painted with the wrong color, indicating a likely dead end (return infinity).
       - If there are no unpainted goal tiles, the heuristic value is 0.
       - Find the current location and color of each robot in the state.
       - Initialize the total heuristic cost to 0.
       - Pre-compute shortest distances from each robot's current location to all
         reachable tiles using BFS on the tile graph.
       - For each unpainted goal tile (T, C) requiring color C:
         - Find all tiles adjacent to T using the pre-built adjacency graph.
         - Calculate the minimum cost to get *any* robot adjacent to T with color C:
           - Initialize minimum cost for this tile to infinity.
           - For each robot R:
             - Retrieve the pre-computed distance map for robot R.
             - Find the minimum distance (min_dist_R) from R's current location
               to *each* tile adjacent to T that is reachable according to the distance map.
             - Calculate the color change cost for R: 1 if R's current color is
               not C, otherwise 0.
             - The cost for robot R to prepare for painting T is min_dist_R + color_change_cost.
             - Update the minimum cost for tile T with the minimum cost found
               across all robots.
           - If no robot can reach an adjacent tile (min_cost_for_tile is still infinity),
             return infinity as the problem is likely unsolvable from this state.
         - Add the minimum cost for tile T (movement + color) plus 1 (for the paint action)
           to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the tile adjacency graph.
        """
        super().__init__(task) # Call base class constructor

        # Store goal locations and colors {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Ensure goal fact has correct structure before accessing args
                if len(args) == 2:
                    tile, color = args
                    self.goal_paintings[tile] = color
                # else: ignore malformed goal fact


        # Build the tile adjacency graph
        self.adj = {}
        all_tiles = set()

        for fact in self.static:
            parts = get_parts(fact)
            # Check if the fact has at least 3 parts for relation predicates
            if len(parts) >= 3 and parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                all_tiles.add(tile1)
                all_tiles.add(tile2)
                self.adj.setdefault(tile1, set()).add(tile2)
                self.adj.setdefault(tile2, set()).add(tile1)

        # Ensure all tiles mentioned in static facts are in the adjacency map
        # This handles potential isolated tiles, although unlikely in typical grids.
        for tile in all_tiles:
             self.adj.setdefault(tile, set())


    def _bfs(self, start_node):
        """
        Perform BFS to find shortest distances from start_node to all reachable nodes
        within the static grid graph.
        Returns a dictionary {node: distance}.
        """
        # If start_node is not in the graph (e.g., malformed state), return empty map
        if start_node not in self.adj:
             return {}

        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Get neighbors, handle case where current_node might not have entries (shouldn't happen with init logic)
            neighbors = self.adj.get(current_node, set())

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles and check for wrong colors
        unpainted_goal_tiles = {} # {tile: color}
        for goal_tile, goal_color in self.goal_paintings.items():
            is_painted_correctly = False
            is_painted_wrongly = False
            for fact in state:
                parts = get_parts(fact)
                # Check if fact is a painted predicate with the correct number of arguments
                if len(parts) == 3 and parts[0] == "painted" and parts[1] == goal_tile:
                    if parts[2] == goal_color:
                        is_painted_correctly = True
                        break # Found correct painting
                    else:
                        is_painted_wrongly = True
                        break # Found wrong painting

            if is_painted_wrongly:
                # Goal tile is painted with the wrong color, likely unsolvable
                return float('inf') # Use infinity to signal a dead end

            if not is_painted_correctly:
                # Goal tile is not painted correctly (and not wrongly), needs painting
                unpainted_goal_tiles[goal_tile] = goal_color

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goal_tiles:
            return 0

        # 2. Find current robot locations and colors
        robots_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robots_info.setdefault(robot, {})['location'] = location
            elif len(parts) == 3 and parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robots_info.setdefault(robot, {})['color'] = color

        # If no robots found (malformed state?), return infinity
        if not robots_info:
             return float('inf')

        total_heuristic = 0

        # Pre-compute BFS distances for all robots from their current locations
        robot_dist_maps = {}
        for robot_name, info in robots_info.items():
             # Ensure robot has a location before running BFS
             if 'location' in info:
                robot_dist_maps[robot_name] = self._bfs(info['location'])
             else:
                # Robot exists but has no location? Malformed state.
                return float('inf')


        # 3. Calculate cost for each unpainted goal tile
        for tile_T, color_C in unpainted_goal_tiles.items():
            min_cost_for_tile = float('inf')

            # Find tiles adjacent to tile_T
            # If tile_T is not in the graph (malformed problem?), it cannot be painted.
            adjacent_to_T = self.adj.get(tile_T, set())
            if not adjacent_to_T:
                 # Goal tile has no adjacent tiles in the graph. Cannot be painted.
                 return float('inf')

            for robot_name, robot_info in robots_info.items():
                # Ensure robot has color info (should be true based on domain init)
                if 'color' not in robot_info:
                    # Malformed state? Robot exists but has no color.
                    return float('inf')

                robot_location = robot_info['location']
                robot_color = robot_info['color']
                robot_dist_map = robot_dist_maps.get(robot_name, {}) # Get pre-computed map

                # Find minimum distance from robot's location to any tile adjacent to T
                min_dist_R = float('inf')
                for adj_tile in adjacent_to_T:
                    if adj_tile in robot_dist_map:
                        min_dist_R = min(min_dist_R, robot_dist_map[adj_tile])

                # If robot cannot reach any adjacent tile (disconnected components?)
                # If min_dist_R is still infinity, this robot cannot reach the tile.
                # We continue to check other robots.

                # Calculate color change cost
                color_cost = 1 if robot_color != color_C else 0

                # Cost for this robot to prepare for painting tile_T
                # Only consider this robot if it can actually reach an adjacent tile
                if min_dist_R != float('inf'):
                    cost_R = min_dist_R + color_cost
                    # Update minimum cost for tile_T
                    min_cost_for_tile = min(min_cost_for_tile, cost_R)

            # If after checking all robots, none can reach an adjacent tile with the right color
            if min_cost_for_tile == float('inf'):
                 # No robot can paint this tile. Problem likely unsolvable.
                 return float('inf')

            # Add cost for this tile: minimum preparation cost + paint action (cost 1)
            total_heuristic += min_cost_for_tile + 1

        return total_heuristic
