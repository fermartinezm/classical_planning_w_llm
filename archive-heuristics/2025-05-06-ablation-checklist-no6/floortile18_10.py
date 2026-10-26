# Assuming Heuristic base class is available in the execution environment
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
# import re # Not strictly needed for this implementation


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
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def bfs(start_node, graph):
    """
    Performs Breadth-First Search to find shortest distances from a start node.
    Returns a dictionary mapping reachable nodes to their distances.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


# The Heuristic base class is expected to be provided by the environment.
# If running this code standalone, you would need a mock definition like:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal by summing the minimum costs
    to paint each unsatisfied goal tile independently. The minimum cost
    for a tile includes getting a robot with the correct color to an
    adjacent tile and performing the paint action, plus a cost if the
    tile is currently blocked by a robot.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions,
        building the grid graph, and identifying available colors.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adj_list = {}
        all_tiles = set()

        # Build adjacency list from directional predicates
        # Predicates are (dir Y X), meaning Y is reached by moving dir from X.
        # Robot is at X and paints Y. So robot needs to be at a tile X adjacent to Y.
        # The graph is undirected for movement.
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                dir_pred, tile_y, tile_x = parts
                # Add edge X -> Y and Y -> X (grid is undirected for movement)
                self.adj_list.setdefault(tile_x, set()).add(tile_y)
                self.adj_list.setdefault(tile_y, set()).add(tile_x)
                all_tiles.add(tile_x)
                all_tiles.add(tile_y)

        # Build map from tile to its adjacent tiles (tiles from which it can be painted)
        # A robot at tile X can paint tile Y if (dir Y X) is true.
        # So, for a goal tile T (which is Y), we need a robot at a tile X such that (dir T X) is true.
        # This means X is adjacent to T in the graph.
        self.adjacent_tiles_map = {}
        for tile in all_tiles:
             self.adjacent_tiles_map[tile] = set(self.adj_list.get(tile, []))

        # Identify available colors
        self.available_colors = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Identify unsatisfied goal tiles (tile, color)
        unsatisfied_goals = set()
        for goal in self.goals:
            # Goal facts are typically (painted tile color)
            if match(goal, "painted", "*", "*"):
                goal_parts = get_parts(goal)
                goal_tile = goal_parts[1]
                goal_color = goal_parts[2]
                if goal not in state:
                    unsatisfied_goals.add((goal_tile, goal_color))

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        # Identify robot locations and colors
        robot_locs = {}
        robot_colors = {}
        robots = set()
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                r, loc = get_parts(fact)[1:]
                robot_locs[r] = loc
                robots.add(r)
            elif match(fact, "robot-has", "*", "*"):
                r, color = get_parts(fact)[1:]
                robot_colors[r] = color
                robots.add(r)

        # Compute distances from each robot's current location to all reachable tiles
        dist_from_robots = {}
        for robot in robots:
            if robot in robot_locs: # Ensure robot location is known
                 dist_from_robots[robot] = bfs(robot_locs[robot], self.adj_list)
            else:
                 # Robot location unknown? Should not happen in valid states.
                 dist_from_robots[robot] = {}


        total_h = 0

        # Calculate cost for each unsatisfied goal tile independently
        for goal_tile, goal_color in unsatisfied_goals:

            # Check if the required color is available at all
            if goal_color not in self.available_colors:
                 # This goal is impossible to satisfy
                 return float('inf')

            cost_for_tile = 0

            # Check if the tile is clear. If not painted, it must be occupied by a robot.
            is_clear = (f"(clear {goal_tile})" in state)
            is_painted_in_state = any(match(fact, "painted", goal_tile, "*") for fact in state)

            if not is_clear and not is_painted_in_state:
                 # Tile is not clear and not painted -> occupied by a robot.
                 # This robot must move off (1 action) before painting is possible.
                 cost_for_tile += 1

            # Calculate minimum cost to get a robot with the right color adjacent to the tile
            min_cost_to_get_robot_in_pos = float('inf')
            adjacent_to_goal_tile = self.adjacent_tiles_map.get(goal_tile, set())

            if not adjacent_to_goal_tile:
                 # Goal tile has no adjacent tiles, impossible to paint.
                 return float('inf')

            for robot in robots:
                if robot not in robot_locs: continue # Skip robots with unknown location

                current_tile_R = robot_locs[robot]
                robot_has_goal_color = (robot in robot_colors and robot_colors[robot] == goal_color)
                # Cost to change color if needed. Assumes robot always has *some* color.
                color_change_cost = 0 if robot_has_goal_color else 1

                # Find minimum distance from robot's current location to any adjacent tile
                min_dist_R_to_adjacent_T = float('inf')
                if current_tile_R in dist_from_robots[robot]: # Check if robot's tile is in the graph
                    for adj_tile in adjacent_to_goal_tile:
                        if adj_tile in dist_from_robots[robot]:
                            min_dist_R_to_adjacent_T = min(min_dist_R_to_adjacent_T, dist_from_robots[robot][adj_tile])

                # Calculate cost for this robot to get into position and have the right color
                if min_dist_R_to_adjacent_T is not float('inf'):
                    cost_R = min_dist_R_to_adjacent_T + color_change_cost
                    min_cost_to_get_robot_in_pos = min(min_cost_to_get_robot_in_pos, cost_R)

            # If no robot can reach an adjacent tile with the right color (or acquire it)
            if min_cost_to_get_robot_in_pos == float('inf'):
                 return float('inf') # This goal is impossible to satisfy

            # Add the cost to get a robot in position + the paint action cost
            cost_for_tile += min_cost_to_get_robot_in_pos + 1

            total_h += cost_for_tile

        return total_h
