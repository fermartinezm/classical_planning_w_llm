from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty or invalid fact strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to check if a PDDL fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if parts match args, allowing '*' wildcard
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint all
    goal tiles with the correct color. It sums the estimated cost for each
    unpainted goal tile, considering the minimum cost for any robot to reach
    an adjacent painting position, change color if needed, and perform the paint action.

    # Assumptions
    - Tiles are arranged in a grid structure defined by 'up', 'down', 'left', 'right' predicates.
    - A robot at tile Z can paint tile X if (up X Z) or (down X Z) is true.
    - Painting a tile requires it to be 'clear'. If a goal tile is not clear
      and not painted with the goal color, the state is considered unreachable
      (heuristic returns infinity).
    - The cost of moving between adjacent tiles is 1.
    - The cost of changing color is 1.
    - The cost of painting is 1.
    - The heuristic sums the minimum costs for each unpainted goal tile independently,
      which is an admissible (or at least consistent for greedy search) lower bound
      if robots and colors were infinitely available and movement/color changes
      didn't interact. Since they do interact, it's not strictly admissible but
      provides a reasonable estimate for greedy search.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Builds the tile adjacency graph for movement from static 'up', 'down', 'left', 'right' facts.
    - Computes all-pairs shortest paths between tiles using BFS.
    - Identifies for each tile, the set of adjacent tiles from which it can be painted.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted tileX colorY)`.
    2. For each such goal condition:
       a. Check if the condition `(painted tileX colorY)` is already true in the current state. If yes, this goal is satisfied, contribute 0 to the heuristic for this tile.
       b. If the condition is not true, check if `(clear tileX)` is false in the current state. If `tileX` is not clear and not painted with the goal color, the goal is unreachable for this tile. Return infinity for the heuristic.
       c. If the condition is not true and `tileX` is clear (or not painted at all), calculate the minimum cost to paint `tileX` with `colorY`.
          i. Find the current location and color of each robot from the state.
          ii. For each robot, calculate the cost to paint `tileX`:
              - Minimum number of moves from the robot's current location to any tile from which `tileX` can be painted (using precomputed shortest paths).
              - Add 1 if the robot's current color is not `colorY` (cost of `change_color`).
              - Add 1 for the `paint` action itself.
          iii. The minimum cost for `tileX` is the minimum of these costs over all robots.
       d. Add this minimum cost for `tileX` to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Get all tile objects
        # We can find all tiles by looking at the arguments of spatial predicates
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) > 1: all_tiles.add(parts[1])
                if len(parts) > 2: all_tiles.add(parts[2])
        self.all_tiles = list(all_tiles) # Keep a list for consistent indexing if needed, or just use the set

        # 2. Build adjacency list for movement graph
        self.adj = {tile: [] for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) == 3:
                    tileY, tileX = parts[1], parts[2]
                    # Movement is possible from X to Y and Y to X if clear
                    # The graph for distance calculation should be undirected
                    self.adj[tileX].append(tileY)
                    self.adj[tileY].append(tileX)

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_tile in self.all_tiles:
            self.dist[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            while q:
                current_tile, d = q.popleft()
                self.dist[start_tile][current_tile] = d
                for neighbor in self.adj.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # 4. Identify paintable neighbors for each tile
        # Robot at Z can paint X if (up X Z) or (down X Z)
        self.PaintAdj = {tileX: set() for tileX in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down']:
                 if len(parts) == 3:
                    tileX, tileZ = parts[1], parts[2] # (up tileX tileZ) means tileX is up from tileZ
                    self.PaintAdj[tileX].add(tileZ) # Robot at tileZ can paint tileX

        # Store goal tiles and their target colors
        self.goal_tiles = {} # {tile: color}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot locations and colors
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}
        all_robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at':
                if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
                    all_robots.add(robot)
            elif parts and parts[0] == 'robot-has':
                if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                    all_robots.add(robot)

        total_heuristic = 0

        # Check each goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if the goal for this tile is already satisfied
            goal_satisfied = False
            for fact in state:
                 if match(fact, "painted", goal_tile, goal_color):
                     goal_satisfied = True
                     break

            if goal_satisfied:
                continue # This goal is met, no cost

            # If goal is not satisfied, check if it's unreachable
            # Goal tile must be clear to be painted
            # Check if it's painted with a *wrong* color
            painted_wrong_color = False
            for fact in state:
                 if match(fact, "painted", goal_tile, "*") and not match(fact, "painted", goal_tile, goal_color):
                     painted_wrong_color = True
                     break

            # If the tile is painted with wrong color, it cannot be repainted (requires clear)
            if painted_wrong_color:
                 # This state is likely a dead end
                 return float('inf') # Indicate unreachable goal

            # Calculate minimum cost to paint this tile with the goal color
            min_cost_for_tile = float('inf')

            # Find tiles adjacent to the goal tile from which painting is possible
            paint_adj_tiles = self.PaintAdj.get(goal_tile, set())

            # If there are no tiles from which this goal tile can be painted, it's unreachable
            if not paint_adj_tiles:
                 return float('inf')

            # If there are no robots, the goal is unreachable
            if not all_robots:
                 return float('inf')

            for robot in all_robots:
                r_loc = robot_locations.get(robot)
                r_color = robot_colors.get(robot)

                # If robot location or color is unknown (shouldn't happen in valid states)
                if r_loc is None or r_color is None:
                    continue # Skip this robot

                # Find minimum moves from robot's current location to any paintable adjacent tile
                min_moves_to_paint_pos = float('inf')
                for adj_tile in paint_adj_tiles:
                    if r_loc in self.dist and adj_tile in self.dist[r_loc]:
                         min_moves_to_paint_pos = min(min_moves_to_paint_pos, self.dist[r_loc][adj_tile])

                # If robot cannot reach any paintable adjacent tile (e.g., disconnected grid)
                if min_moves_to_paint_pos == float('inf'):
                     continue # This robot cannot paint this tile

                # Cost for color change
                color_change_cost = 1 if r_color != goal_color else 0

                # Total cost for this robot to paint this tile: moves + change_color + paint
                cost_r = min_moves_to_paint_pos + color_change_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_r)

            # If no robot can paint this tile (e.g., disconnected grid, or no robots could reach)
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            total_heuristic += min_cost_for_tile

        return total_heuristic
