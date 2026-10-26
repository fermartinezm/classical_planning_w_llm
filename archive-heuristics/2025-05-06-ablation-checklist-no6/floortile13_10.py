# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment
from fnmatch import fnmatch
from collections import deque


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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

    Estimates the cost to paint all required tiles by summing the cost
    for each unpainted goal tile. The cost for a single tile is estimated
    as 1 (paint action) + minimum cost for any robot to reach an adjacent
    tile with the correct color. Robot movement cost is estimated using
    shortest path on the static grid graph, ignoring 'clear' preconditions
    for simplicity and efficiency. Color change cost is 1 if the robot
    doesn't have the color, 0 otherwise.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the tile adjacency graph.
        """
        # Assuming Heuristic base class is initialized correctly by the framework
        # super().__init__(task)

        # Extract goal tiles and their required colors
        self.goal_tiles_colors = {}
        for goal in task.goals: # Use task.goals directly
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles_colors[tile] = color

        # Build tile adjacency graph from static facts
        self.adj_graph = {}
        self.all_tiles = set()

        # Define directions for building symmetric graph
        directions = {"up", "down", "left", "right"}

        for fact in task.static: # Use task.static directly
            predicate, *args = get_parts(fact)
            if predicate in directions:
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # This implies tile_1_1 and tile_0_1 are adjacent.
                tile1, tile2 = args

                # Ensure both tiles are in the graph
                self.adj_graph.setdefault(tile1, set()).add(tile2)
                self.adj_graph.setdefault(tile2, set()).add(tile1) # Add reverse direction

                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # Precompute all-pairs shortest paths on the static grid graph
        self.distances = self._precompute_distances()


    def _precompute_distances(self):
        """
        Computes shortest path distances between all pairs of tiles
        using BFS on the static adjacency graph.
        Ignores 'clear' predicate for simplicity and efficiency.
        """
        distances = {}
        for start_tile in self.all_tiles:
            distances[start_tile] = self._bfs_single_source(start_tile)
        return distances

    def _bfs_single_source(self, start_tile):
        """
        Performs BFS starting from start_tile to find distances to all other tiles.
        """
        dist = {tile: float('inf') for tile in self.all_tiles}
        dist[start_tile] = 0
        queue = deque([start_tile])

        while queue:
            current_tile = queue.popleft()

            # Handle tiles with no connections (shouldn't happen in valid problems)
            if current_tile not in self.adj_graph:
                 continue

            for neighbor_tile in self.adj_graph[current_tile]:
                if dist[neighbor_tile] == float('inf'):
                    dist[neighbor_tile] = dist[current_tile] + 1
                    queue.append(neighbor_tile)
        return dist

    def get_distance(self, start_tile, target_tiles):
        """
        Gets the minimum precomputed distance from start_tile to any tile in target_tiles.
        Returns float('inf') if no target tile is reachable from start_tile.
        """
        min_dist = float('inf')
        if start_tile not in self.distances:
             return min_dist # Should not happen if all_tiles is correct

        for target_tile in target_tiles:
             # Check if target_tile is in the precomputed distances for start_tile
             # (it might not be if the graph is disconnected, though unlikely for tiles)
             if target_tile in self.distances[start_tile]:
                  min_dist = min(min_dist, self.distances[start_tile][target_tile])
        return min_dist


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Extract current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        current_painted_tiles_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "painted":
                 tile, color = parts[1], parts[2]
                 current_painted_tiles_colors[tile] = color

        total_cost = 0

        # Iterate through goal conditions to find unsatisfied painted tiles
        for goal_tile, goal_color in self.goal_tiles_colors.items():
            # Check if the tile is already painted with the correct color
            if goal_tile in current_painted_tiles_colors and current_painted_tiles_colors[goal_tile] == goal_color:
                continue # Goal already satisfied for this tile

            # Check if the tile is painted with the wrong color (unsolvable state)
            # We also check if the goal_tile is even in the adjacency graph to avoid errors
            if goal_tile in current_painted_tiles_colors and goal_tile in self.adj_graph and current_painted_tiles_colors[goal_tile] != goal_color:
                 # This state is likely unsolvable in this domain as there's no unpaint action.
                 # Return a very large number to prune this branch.
                 return float('inf') # Or a large constant like 1000000

            # The tile needs to be painted with goal_color
            # Cost includes 1 for the paint action itself
            tile_cost = 1

            # Find adjacent tiles for the goal tile
            # A robot must be AT an adjacent tile to paint the goal_tile.
            adjacent_to_goal_tile = self.adj_graph.get(goal_tile, set())

            if not adjacent_to_goal_tile:
                 # Goal tile has no adjacent tiles? Should not happen in valid problems.
                 # Treat as unreachable or very high cost.
                 return float('inf') # Or a large constant

            # Find the minimum cost for any robot to get the correct color and reach an adjacent tile
            min_robot_prep_cost = float('inf')

            # If there are no robots, this is an unsolvable state (or initial state with no robots?)
            if not robot_locations:
                 return float('inf') # Or a large constant

            for robot, robot_loc in robot_locations.items():
                # Cost to get the correct color
                # Assumes change_color is always possible if the color is available (which is a static fact)
                robot_current_color = robot_colors.get(robot) # .get handles case where robot has no color
                color_cost = 0 if robot_current_color == goal_color else 1

                # Cost to move from robot_loc to any tile adjacent to goal_tile
                # Use precomputed distances, ignoring 'clear' precondition for intermediate tiles
                # The destination adjacent tile *must* be clear to move onto it.
                # However, the precomputed distance ignores this. This is a simplification.
                move_cost = self.get_distance(robot_loc, adjacent_to_goal_tile)

                # Total cost for this robot to prepare for painting this tile
                robot_prep_cost = color_cost + move_cost

                min_robot_prep_cost = min(min_robot_prep_cost, robot_prep_cost)

            # If no robot can reach an adjacent tile (shouldn't happen in connected grid):
            if min_robot_prep_cost == float('inf'):
                 # This tile is unreachable by any robot. The state is unsolvable.
                 return float('inf') # Or a large constant

            # Add the minimum preparation cost for this tile to the total
            tile_cost += min_robot_prep_cost

            total_cost += tile_cost

        return total_cost
