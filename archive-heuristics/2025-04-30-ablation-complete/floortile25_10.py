from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# If running standalone, you might need a dummy Heuristic class definition
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not found
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.initial_state = task.initial_state # Added for floortile heuristic needs

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list or handle error appropriately for malformed input
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is sufficient for the pattern
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are not yet
    painted correctly. It sums the estimated cost for each unpainted goal tile.
    The estimated cost for a single tile includes:
    1. The cost of the paint action (always 1).
    2. The minimum movement cost for any robot to reach a tile adjacent to the
       target tile from which it can be painted (either above or below).
    3. A cost of 1 for each distinct color required by unpainted goal tiles that
       is not currently held by any robot. This color cost is counted once per
       needed color globally, not per tile.

    # Assumptions
    - Tiles are arranged in a grid structure defined by `up`, `down`, `left`, `right` predicates.
    - Robots can only paint tiles directly above or below their current position
      using `paint_down` and `paint_up` actions, respectively.
    - All action costs are 1.
    - If a tile is painted with the wrong color, the problem is unsolvable.
    - Robots always have a color initially (as per example instances).

    # Heuristic Initialization
    - Extracts goal conditions to identify target tiles and their required colors.
    - Builds a graph representation of the tile grid based on `up`, `down`, `left`,
      `right` static facts.
    - Computes all-pairs shortest paths between tiles using BFS on the grid graph.
    - Identifies, for each tile that needs to be painted, the possible locations
      a robot must be at to paint it (i.e., the tiles directly above or below it).
    - Extracts robot names from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to get robot locations, robot colors, and painted/clear tiles.
    2. Identify all goal tiles that are not currently painted with the correct color.
    3. Check if any goal tile is painted with an incorrect color (different from the goal color). If so, the problem is unsolvable, return infinity.
    4. If there are no unpainted goal tiles, the goal is reached, return 0.
    5. Calculate the "color cost":
       a. Determine the set of distinct colors required by the identified unpainted goal tiles.
       b. Determine the set of colors currently held by all robots.
       c. The color cost is the number of required colors that are not present in the set of colors held by robots.
    6. Calculate the "movement and paint cost":
       a. Initialize a running total for this cost to 0.
       b. For each unpainted goal tile:
          i. Find the set of tiles where a robot must be located to paint this tile (these are the tiles directly above or below it, based on the static `up` and `down` facts parsed during initialization).
          ii. Find the minimum grid distance from any robot's current location to any of these required painting locations using the precomputed shortest paths.
          iii. If no robot can reach any required painting location for this tile, the problem is unsolvable, return infinity.
          iv. Add this minimum distance plus 1 (for the paint action) to the running total.
    7. The total heuristic value is the sum of the color cost and the movement and paint cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        tile grid graph, and computing shortest paths.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract goal painted tiles and their colors
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    self.goal_painted[parts[1]] = parts[2]
                else:
                    # Handle unexpected goal format if necessary
                    # print(f"Warning: Unexpected goal format: {goal}")
                    pass # Skip malformed goal facts

        # Build tile graph and adjacency list for movement
        self.adj = {}
        self.tiles = set()

        # Collect all tiles first and initialize adjacency
        for fact in self.static:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                 tile1, tile2 = parts[1], parts[2]
                 self.tiles.add(tile1)
                 self.tiles.add(tile2)
                 if tile1 not in self.adj: self.adj[tile1] = []
                 if tile2 not in self.adj: self.adj[tile2] = []

        # Add edges for movement (grid connectivity)
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, tile1, tile2 = parts
                if pred in ["up", "down", "left", "right"]:
                     # Movement is bidirectional on the grid
                     self.adj[tile1].append(tile2)
                     self.adj[tile2].append(tile1)

        # Build paint adjacency: tile_to_paint -> list of robot_locations to paint it from
        # A robot at X can paint Y if (up Y X) or (down Y X) is true.
        # So, to paint Y, the robot must be at X where (up Y X) or (down Y X).
        self.paint_adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, tile_to_paint, robot_loc = parts
                if pred in ["up", "down"]:
                    if tile_to_paint not in self.paint_adj:
                        self.paint_adj[tile_to_paint] = []
                    self.paint_adj[tile_to_paint].append(robot_loc)

        # Calculate all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.tiles:
            self.dist[start_node] = self._bfs(start_node)

        # Get robot names from initial state
        self.robots = {parts[1] for fact in self.initial_state if match(fact, "robot-at", "*", "*")}


    def _bfs(self, start_node):
        """
        Perform Breadth-First Search to find shortest paths from start_node
        to all other reachable nodes in the tile grid.
        Returns a dictionary mapping reachable nodes to their distance from start_node.
        """
        distances = {node: float('inf') for node in self.tiles}
        if start_node not in self.tiles:
             # Start node is not a known tile, cannot compute distances
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Ensure current_node has neighbors defined, even if it's an isolated tile
            neighbors = self.adj.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        robot_location = {}
        robot_color = {}
        current_painted = {}
        current_clear = set() # Not strictly needed for this heuristic logic, but good practice

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == "robot-at" and len(parts) == 3:
                robot, loc = parts[1], parts[2]
                robot_location[robot] = loc
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif predicate == "clear" and len(parts) == 2:
                tile = parts[1]
                current_clear.add(tile)

        # Identify unpainted goal tiles and needed colors
        unpainted_goal_tiles = []
        needed_colors = set()

        for tile, goal_color in self.goal_painted.items():
            if tile in current_painted:
                if current_painted[tile] != goal_color:
                    # Tile is painted with the wrong color - unsolvable
                    return float('inf')
                # Tile is painted with the correct color - goal achieved for this tile
                pass
            else:
                # Tile is not painted (implicitly clear if not painted or wrongly painted)
                unpainted_goal_tiles.append(tile)
                needed_colors.add(goal_color)

        # If no unpainted goal tiles, we are at the goal
        if not unpainted_goal_tiles:
            return 0

        # Calculate color cost
        robot_colors_held = {robot_color.get(r) for r in self.robots if r in robot_color}
        # Remove None in case a robot somehow doesn't have a color initially or in state
        robot_colors_held.discard(None)

        colors_to_get = needed_colors - robot_colors_held
        color_cost = len(colors_to_get)

        # Calculate movement and paint cost
        move_paint_cost = 0
        current_robot_locs = list(robot_location.values())

        if not current_robot_locs:
             # No robots to paint! Unsolvable.
             return float('inf')

        for tile_T in unpainted_goal_tiles:
            # Get locations a robot must be at to paint tile_T
            paint_locs_T = self.paint_adj.get(tile_T, [])

            if not paint_locs_T:
                # This tile cannot be painted from any adjacent tile defined in static facts.
                # Problem is likely unsolvable.
                return float('inf')

            min_dist_T = float('inf')

            # Find the minimum distance from any robot to any valid paint location for tile_T
            for rl in current_robot_locs:
                 if rl not in self.dist:
                     # Robot is at an unknown location? Unsolvable.
                     return float('inf')
                 for pl in paint_locs_T:
                     if pl in self.dist[rl]:
                         min_dist_T = min(min_dist_T, self.dist[rl][pl])

            if min_dist_T == float('inf'):
                 # No robot can reach any valid paint location for this tile
                 return float('inf')

            # Cost for this tile: min distance to a paint location + 1 (for the paint action)
            move_paint_cost += min_dist_T + 1

        # Total heuristic is the sum of color cost and movement/paint cost
        total_heuristic = color_cost + move_paint_cost

        return total_heuristic
