from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False # Basic check for arity
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums the estimated cost for each
    unpainted goal tile that is currently clear. The cost for a single tile is
    estimated as 1 (for the paint action) plus the minimum cost to get a robot
    with the correct color adjacent to the tile. The cost to get a robot with
    the correct color adjacent considers both moving a robot that already has
    the color and moving any robot and changing its color. Movement cost is
    estimated using precomputed BFS distances on the static grid, ignoring
    the 'clear' precondition for movement.

    # Assumptions
    - The grid structure (up, down, left, right) is static.
    - Tiles are named in a way that allows identifying neighbors from static facts.
    - If a goal tile is already painted with the wrong color, the state is considered unreachable (heuristic returns infinity).
    - Movement cost is based on shortest path on the static grid, ignoring the dynamic 'clear' predicate for movement. This is a relaxation.
    - Each paint action requires the robot to be on a tile adjacent to the target tile.

    # Heuristic Initialization
    - Build the grid graph (adjacency list) from the static 'up', 'down', 'left', 'right' facts.
    - Compute all-pairs shortest paths (BFS distances) on this static grid.
    - Extract the set of goal 'painted' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that need to be painted with a specific color from the task's goal conditions.
    2. For each such goal tile (tile T needing color C):
       - Check the current state of tile T.
       - If T is already painted with color C in the current state, it is satisfied and contributes 0 to the heuristic.
       - If T is painted with a *different* color (C' != C) in the current state, the goal is unreachable according to the domain rules (paint actions require the tile to be clear). The heuristic returns infinity.
       - If T is currently 'clear' and needs to be painted with color C:
         - This tile requires at least 1 paint action (cost 1).
         - A robot must be moved to a tile adjacent to T.
         - The robot performing the paint action must have color C.
         - Estimate the minimum cost to get *any* robot with color C adjacent to T:
           - Find the set of tiles adjacent to T using the precomputed grid graph (Adj(T)).
           - For each robot R, calculate the minimum BFS distance from its current location (R_curr) to any tile A in Adj(T). Let this be dist(R, T).
           - Calculate the minimum of dist(R, T) for all robots R that *already* have color C in the current state (Option 1 cost). If no robot has color C, this cost is infinity.
           - Calculate the minimum of dist(R, T) for *all* robots R, and add 1 for a potential `change_color` action (Option 2 cost).
           - The minimum cost to get a robot with color C adjacent to T is the minimum of Option 1 and Option 2 costs.
         - The estimated cost for this specific goal tile T is 1 (paint action) + the minimum cost calculated above.
    3. The total heuristic value is the sum of the estimated costs for all clear goal tiles that need painting.
    4. If the set of unsatisfied goal tiles is empty, the heuristic is 0 (goal state).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the grid graph, computing distances,
        and extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract goal painted tiles
        self.goal_painted_tiles = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted_tiles.add((tile, color))

        # Build grid graph (adjacency list) and collect all tile names
        self.adj = {}
        self.tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies tile_0_1 is adjacent to tile_1_1
                tile1, tile2 = parts[1], parts[2]
                self.tiles.add(tile1)
                self.tiles.add(tile2)
                if tile1 not in self.adj:
                    self.adj[tile1] = set()
                if tile2 not in self.adj:
                    self.adj[tile2] = set()
                self.adj[tile1].add(tile2)
                self.adj[tile2].add(tile1) # Grid is undirected for movement

        # Compute all-pairs BFS distances
        self.bfs_dist = {}
        for start_node in self.tiles:
            self.bfs_dist[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.tiles}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.adj: # Check if node has neighbors in the graph
                for neighbor in self.adj[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current state information
        current_painted = set()
        current_clear = set()
        robot_locs = {}
        robot_colors = {}
        all_robots = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                current_painted.add((parts[1], parts[2]))
            elif parts[0] == "clear":
                current_clear.add(parts[1])
            elif parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locs[robot] = tile
                all_robots.add(robot)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                all_robots.add(robot)

        total_cost = 0

        # Identify goal tiles that are not satisfied
        unsatisfied_goals = {
            (tile, color) for (tile, color) in self.goal_painted_tiles
            if (tile, color) not in current_painted
        }

        for tile, goal_color in unsatisfied_goals:
            # Check if the tile is painted with the wrong color
            is_wrongly_painted = False
            # A tile is wrongly painted if it's painted, but not with the goal color
            # We don't need to iterate through all facts, just check if it's in current_painted
            # but with a different color.
            # A simpler check: if the tile is NOT clear, and NOT painted with the goal color, it's wrongly painted.
            if tile not in current_clear and (tile, goal_color) not in current_painted:
                 is_wrongly_painted = True

            if is_wrongly_painted:
                 # Cannot repaint a non-clear tile according to domain rules
                 return float('inf')

            # If not wrongly painted, it must be clear (or already painted correctly, which is handled by unsatisfied_goals)
            # We only process tiles that are clear and need painting.
            if tile in current_clear:
                # This tile needs to be painted with goal_color
                cost_for_tile = 1 # Cost of the paint action

                # Find tiles adjacent to the goal tile
                adjacent_to_goal_tile = self.adj.get(tile, set())
                if not adjacent_to_goal_tile:
                    # Should not happen in a valid grid where goal tiles are paintable
                    return float('inf') # Cannot paint if no adjacent tiles

                # Calculate minimum distance from each robot to any adjacent tile
                dist_to_adj = {}
                for robot in all_robots:
                    robot_curr_loc = robot_locs.get(robot)
                    if robot_curr_loc is None or robot_curr_loc not in self.bfs_dist:
                         # Robot location unknown or not in the grid graph - indicates a problem state
                         return float('inf') # Cannot move robot if its location is invalid
                    else:
                        min_dist = math.inf
                        for adj_tile in adjacent_to_goal_tile:
                            if adj_tile in self.bfs_dist[robot_curr_loc]:
                                min_dist = min(min_dist, self.bfs_dist[robot_curr_loc][adj_tile])
                            # else: adj_tile not in graph? Should not happen if adj is built from static facts.
                        dist_to_adj[robot] = min_dist


                # Option 1: Use a robot that already has the goal color
                robots_with_goal_color = {r for r, c in robot_colors.items() if c == goal_color}
                option1_cost = math.inf
                if robots_with_goal_color:
                    # Ensure robots in robots_with_goal_color are also in dist_to_adj (should be true if all_robots check passed)
                    option1_cost = min(dist_to_adj.get(r, math.inf) for r in robots_with_goal_color)

                # Option 2: Use any robot and change color
                min_dist_any_robot = math.inf
                if all_robots:
                     min_dist_any_robot = min(dist_to_adj.values())

                option2_cost = 1 + min_dist_any_robot # 1 for change_color action

                # Minimum cost to get a robot with the correct color adjacent
                min_cost_get_robot_adjacent_with_color = min(option1_cost, option2_cost)

                # If min_cost_get_robot_adjacent_with_color is still infinity, it means no robot can reach an adjacent tile.
                if min_cost_get_robot_adjacent_with_color == math.inf:
                     return float('inf')

                # Add movement and color change cost to the tile cost
                cost_for_tile += min_cost_get_robot_adjacent_with_color

                total_cost += cost_for_tile

        # Heuristic is 0 only if all goal painted tiles are satisfied (unsatisfied_goals is empty)
        # If unsatisfied_goals is empty, the loop is skipped and total_cost remains 0. This is correct.

        return total_cost
