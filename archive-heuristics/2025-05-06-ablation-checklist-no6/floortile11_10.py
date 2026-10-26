# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch
import collections # Needed for BFS queue
import math # For math.inf

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Remove parentheses and split
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
        # Handle cases that might not be standard facts, though unlikely for predicates
        return fact_str.split()
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, considering wildcards
    # A simple zip and fnmatch check is sufficient and robust.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles that are not yet painted
    correctly. For each unpainted goal tile, it finds the minimum cost
    for any robot to reach a clear adjacent tile from which it can paint
    the goal tile, with the required color, and then paint it.

    Cost for a single unpainted goal tile needing color C:
    min_over_robots_r [ (1 if robot_r does not have color C) +
                        min_over_clear_paint_from_tiles_x [ BFS_dist(robot_r_location, x, state) ]
                      ] + 1 (for the paint action)

    BFS_dist calculates the shortest path distance considering only moves
    to 'clear' tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid adjacency list and paint-from map from static facts.
        """
        super().__init__(task)

        # Store goal tiles and their required colors: {(tile_name, color_name), ...}
        self.goal_painted_tiles = set()
        for goal in self.goals:
            # Goal facts are typically (painted tile color)
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_painted_tiles.add((tile, color))

        # Build the grid adjacency list from static facts
        # adj_list[from_tile] = list of to_tile where robot can move from from_tile to to_tile
        self.adj_list = {}
        # paint_from_map[painted_tile] = list of tiles robot can paint from
        self.paint_from_map = {}

        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Predicate is (direction tile_y tile_x) meaning robot at tile_x can move to tile_y
                direction, tile_y, tile_x = parts
                # Add to adjacency list: robot at tile_x can move to tile_y
                self.adj_list.setdefault(tile_x, []).append(tile_y)
                # Add to paint_from map: robot at tile_x can paint tile_y
                self.paint_from_map.setdefault(tile_y, []).append(tile_x)


    def _bfs_dist(self, start_tile, end_tile, clear_tiles):
        """
        Calculate the shortest path distance from start_tile to end_tile
        moving only through tiles present in clear_tiles.
        Returns math.inf if end_tile is unreachable via clear tiles.
        """
        # BFS requires start_tile to be reachable, which it is (robot is there).
        # The destination end_tile must be clear to move *to* it.
        # The BFS explores paths where every step lands on a clear tile.

        q = collections.deque([(start_tile, 0)]) # Use deque for efficient popping
        visited = {start_tile}

        while q:
            current_tile, dist = q.popleft() # Use popleft for BFS

            if current_tile == end_tile:
                return dist

            # Explore neighbors reachable by one move
            for next_tile in self.adj_list.get(current_tile, []):
                # A move from current_tile to next_tile is only possible if next_tile is clear
                if next_tile in clear_tiles and next_tile not in visited:
                    visited.add(next_tile)
                    q.append((next_tile, dist + 1))

        # Destination unreachable via clear tiles
        return math.inf


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Extract current state information
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        clear_tiles = set()  # {tile_name, ...}
        painted_tiles = set() # {(tile_name, color_name), ...}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                _, robot, tile = parts
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                _, robot, color = parts
                robot_colors[robot] = color
            elif predicate == "clear" and len(parts) == 2:
                _, tile = parts
                clear_tiles.add(tile)
            elif predicate == "painted" and len(parts) == 3:
                 _, tile, color = parts
                 painted_tiles.add((tile, color))

        # Identify unpainted goal tiles
        unpainted_goals = self.goal_painted_tiles - painted_tiles

        total_heuristic = 0

        # Compute cost for each unpainted goal tile independently
        for goal_tile, goal_color in unpainted_goals:

            # Check if the goal tile itself is clear (required for painting)
            # Paint actions require (clear ?y) where ?y is the tile being painted.
            if goal_tile not in clear_tiles:
                 # This tile needs painting but is not clear. Cannot paint it.
                 # This state is likely unsolvable or requires actions not modeled.
                 # Returning infinity is a strong signal that this path is bad.
                 return math.inf

            # Find tiles from which the robot can paint the goal tile
            paint_from_tiles = self.paint_from_map.get(goal_tile, [])

            # Find clear tiles among the paint_from tiles (robot must move *to* a clear tile)
            clear_paint_from_tiles = [t for t in paint_from_tiles if t in clear_tiles]

            # If no clear tile exists from which the robot can paint the goal tile
            if not clear_paint_from_tiles:
                 # Cannot reach a clear position to paint this tile
                 return math.inf

            min_robot_cost_for_tile = math.inf

            # Find the minimum cost for any robot to paint this tile
            for robot_name, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot_name) # None if robot has no color (not possible based on domain)

                # Cost to get the correct color
                color_cost = 0 if robot_color == goal_color else 1

                # Minimum moves to reach any clear paint_from tile
                min_move_cost_to_paint_from_clear = math.inf
                for paint_from_tile in clear_paint_from_tiles:
                    # BFS distance from robot's current location to the clear paint_from tile
                    # The BFS path must only go through clear tiles.
                    dist = self._bfs_dist(robot_loc, paint_from_tile, clear_tiles)
                    min_move_cost_to_paint_from_clear = min(min_move_cost_to_paint_from_clear, dist)

                # If the robot can reach a clear paint_from tile
                if min_move_cost_to_paint_from_clear != math.inf:
                    robot_cost_for_tile = color_cost + min_move_cost_to_paint_from_clear
                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost_for_tile)

            # If no robot can reach a clear paint_from tile
            if min_robot_cost_for_tile == math.inf:
                 # Tile is currently unreachable/unpaintable by any robot
                 return math.inf

            # Add the minimum cost to reach and get color, plus the paint action cost (1)
            total_heuristic += min_robot_cost_for_tile + 1

        return total_heuristic
