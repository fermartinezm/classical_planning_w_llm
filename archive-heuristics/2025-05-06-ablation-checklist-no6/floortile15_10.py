# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone for testing, define a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic arity check
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is imported from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost based on:
    1. The number of unpainted goal tiles that are currently clear.
    2. The number of distinct colors needed for these tiles that no robot currently has.
    3. The minimum distance from any robot to any paint spot for any of these tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, distances,
        paint spots, and goal conditions from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        # Parse goal facts to get target painted states {tile: color}
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                if len(args) == 2:
                    tile, color = args
                    self.goal_tiles[tile] = color
                # else: malformed goal fact, ignore

        # Extract all tile names and build grid structure
        self.all_tiles = set()
        self.grid_adj = {} # Adjacency list for grid graph (undirected)
        self.paint_spots = {} # Map tile -> set of tiles from which it can be painted

        # Iterate through static facts to build grid and paint spots
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                dir_pred, tile_y, tile_x = parts # (Dir y x) means y is Dir from x
                self.all_tiles.add(tile_x)
                self.all_tiles.add(tile_y)

                # Build undirected grid graph
                if tile_x not in self.grid_adj:
                    self.grid_adj[tile_x] = set()
                if tile_y not in self.grid_adj:
                    self.grid_adj[tile_y] = set()
                self.grid_adj[tile_x].add(tile_y)
                self.grid_adj[tile_y].add(tile_x)

                # Build paint spots: robot at x paints y if (Dir y x)
                if tile_y not in self.paint_spots:
                    self.paint_spots[tile_y] = set()
                self.paint_spots[tile_y].add(tile_x)

        # Compute all-pairs shortest paths on the grid
        self.grid_dist = {}
        for start_tile in self.all_tiles:
            self.grid_dist[start_tile] = self._bfs(start_tile, self.grid_adj)

    def _bfs(self, start_node, graph):
        """Performs BFS to find distances from start_node to all reachable nodes."""
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            if u in graph: # Ensure node exists in graph keys
                for v in graph[u]:
                    if v not in distances:
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Identify current state facts
        current_painted = {}
        current_clear = set()
        robot_locations = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                if len(parts) == 3:
                    current_painted[parts[1]] = parts[2]
            elif parts[0] == "clear":
                 if len(parts) == 2:
                    current_clear.add(parts[1])
            elif parts[0] == "robot-at":
                 if len(parts) == 3:
                    robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has":
                 if len(parts) == 3:
                    robot_colors[parts[1]] = parts[2]

        # Identify unpainted clear goal tiles
        unpainted_clear_goal_tiles = [] # List of (tile, color) tuples
        needed_colors = set()

        for tile, goal_color in self.goal_tiles.items():
            is_painted_correctly = tile in current_painted and current_painted[tile] == goal_color
            is_clear = tile in current_clear

            if not is_painted_correctly and is_clear:
                 unpainted_clear_goal_tiles.append((tile, goal_color))
                 needed_colors.add(goal_color)
            # If tile is painted with the wrong color, it's not clear and not painted correctly.
            # These tiles are effectively "blocked" unless there's an unpaint action (which there isn't).
            # We assume solvable problems don't reach states with wrongly painted goal tiles.
            # If they do, the heuristic might return 0, which is acceptable for GBFS.

        # If all goal tiles are painted correctly or are not clear (and thus presumably wrongly painted),
        # and there are no clear unpainted goal tiles left, the heuristic is 0.
        # The goal check will handle the case where all are painted correctly.
        if not unpainted_clear_goal_tiles:
            return 0

        h = 0

        # 1. Add cost for paint actions
        # Each unpainted clear goal tile needs one paint action.
        h += len(unpainted_clear_goal_tiles)

        # 2. Add cost for color changes
        # Estimate the number of color changes needed.
        # A lower bound is the number of distinct colors required for unpainted tiles
        # that no robot currently possesses.
        robot_current_colors = set(robot_colors.values())
        missing_colors_count = sum(1 for color in needed_colors if color not in robot_current_colors)
        h += missing_colors_count

        # 3. Add cost for movement
        # Estimate the movement cost to get a robot to a position where it can start painting.
        # This is the minimum distance from any robot's current location
        # to any valid paint spot for any of the unpainted clear goal tiles.
        min_movement = math.inf

        if not robot_locations:
             # No robots? Problem likely unsolvable. Return large value.
             return math.inf

        for robot, robot_loc in robot_locations.items():
            # Ensure robot_loc is a valid tile in our grid graph
            if robot_loc not in self.grid_dist:
                 # If robot is at an unknown location, it can't move. Problem likely unsolvable.
                 return math.inf

            for tile, goal_color in unpainted_clear_goal_tiles:
                if tile in self.paint_spots: # Check if paint spots exist for this tile
                    for paint_spot in self.paint_spots[tile]:
                        # Ensure paint_spot is a valid tile in our grid graph and reachable from robot_loc
                        if robot_loc in self.grid_dist and paint_spot in self.grid_dist[robot_loc]:
                             min_movement = min(min_movement, self.grid_dist[robot_loc][paint_spot])
                        # else: paint_spot is not reachable from robot_loc, or paint_spot is not in grid_dist (invalid tile)
                # else: If a goal tile has no paint spots defined in static facts, it's unsolvable.
                # min_movement will remain inf, leading to inf heuristic.

        if min_movement != math.inf:
            h += min_movement
        else:
             # If min_movement is still infinity, it means no robot can reach any paint spot
             # for any unpainted goal tile. This indicates an unsolvable state.
             return math.inf

        return h
