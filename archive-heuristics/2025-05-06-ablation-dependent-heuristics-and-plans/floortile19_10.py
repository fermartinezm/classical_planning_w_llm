from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility functions (can be inside the class or outside)
def get_parts(fact_string):
    """Extract parts from a PDDL fact string like '(pred arg1 arg2)'."""
    # Remove leading/trailing parentheses and split by space
    parts = fact_string[1:-1].split()
    return parts

def fact_matches(fact_string, predicate, *args):
    """Check if a PDDL fact string matches a given pattern."""
    parts = get_parts(fact_string)
    if not parts or parts[0] != predicate:
        return False
    # Check arguments, allowing wildcards '*'
    if len(parts) - 1 != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts[1:], args))

def bfs_distances(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their correct colors. It sums the estimated minimum cost
    for each unpainted goal tile independently. The cost for a single tile
    includes moving a robot to an adjacent clear tile, changing the robot's
    color if necessary, and performing the paint action.

    # Assumptions
    - Tiles are arranged in a grid structure defined by up/down/left/right facts.
    - Painted tiles cannot be repainted or moved into.
    - Robots can only move to clear tiles.
    - A tile is clear if it is not painted and not occupied by another robot.
    - The heuristic assumes the necessary adjacent tiles are reachable and clear
      when calculating movement costs using grid distances. Specifically, it checks
      if an adjacent tile is clear in the current state before considering it as a
      potential destination for movement calculation for that specific paint task.
    - If a goal tile is painted with the wrong color, the problem is considered unsolvable
      from that state, and a very large heuristic value is returned.

    # Heuristic Initialization
    - Parses the static facts to build the grid adjacency graph based on
      up/down/left/right relationships.
    - Extracts the goal conditions to determine which tiles need to be painted
      and with which colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task definition.
    2. In the current state, find the location and color of each robot.
    3. Identify the set of "unpainted goal tiles". These are goal tiles that are
       not currently painted with their required color. If a goal tile is found
       to be painted with a *different* color, return a very large heuristic value
       as the state is likely unsolvable.
    4. If there are no unpainted goal tiles, the goal is reached, return 0.
    5. For each robot, compute the shortest path distances from its current location
       to all other tiles on the grid graph using Breadth-First Search (BFS).
    6. Initialize the total heuristic value to 0.
    7. For each unpainted goal tile `T` that needs color `C`:
        a. Find all tiles `Adj_T` that are adjacent to `T` based on the grid graph.
        b. Determine the minimum cost for *any* robot to paint tile `T`. This cost is calculated as:
           - Minimum moves for the robot to reach *any* tile `X` in `Adj_T` that is currently `clear`.
           - Plus 1 if the robot's current color is not `C` (cost of `change_color`).
           - Plus 1 for the `paint` action itself.
        c. The minimum moves to reach a clear adjacent tile `X` from the robot's current location `R_loc` is the BFS distance `dist(R_loc, X)`. We take the minimum distance over all clear `X` in `Adj_T`.
        d. If no robot can reach a clear adjacent tile to `T`, the state is likely unsolvable, return a very large value.
        e. Add this minimum cost for tile `T` to the total heuristic value.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure and goal conditions.
        """
        # The base class constructor stores task.goals and task.static
        super().__init__(task)

        # Build the grid adjacency graph from static facts
        self.adj_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] in ["up", "down", "left", "right"]:
                # Format is (direction tile1 tile2) where robot at tile2 can paint/move to tile1
                # This means tile1 and tile2 are adjacent.
                if len(parts) == 3: # Ensure fact has correct number of parts
                    tile1, tile2 = parts[1], parts[2]
                    self.adj_graph.setdefault(tile1, []).append(tile2)
                    self.adj_graph.setdefault(tile2, []).append(tile1)

        # Ensure adjacency lists are unique
        for tile in self.adj_graph:
             self.adj_graph[tile] = list(set(self.adj_graph[tile]))

        # Store goal colors for tiles
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot locations and colors
        robot_locations = {}
        robot_colors = {}
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)

        # Identify unpainted goal tiles and check for unsolvable states
        unpainted_goal_tiles = {} # {tile: required_color}
        large_value = 1000000 # Use a large number for unsolvable states

        for tile, required_color in self.goal_colors.items():
            is_painted_correctly = f"(painted {tile} {required_color})" in state
            
            # Check if painted with *any* color
            is_painted_at_all = any(
                fact_matches(fact, "painted", tile, "*")
                for fact in state
            )

            if is_painted_correctly:
                continue # Goal met for this tile
            elif is_painted_at_all:
                 # Tile is painted, but not with the correct color. Cannot be repainted. Unsolvable.
                 return large_value
            else:
                # Tile is not painted correctly (implies it's clear or occupied).
                # It needs painting.
                unpainted_goal_tiles[tile] = required_color

        if not unpainted_goal_tiles:
            return 0 # Goal reached

        # Compute distances from each robot's current location
        robot_distances = {}
        for robot, loc in robot_locations.items():
            # Only compute BFS if the robot's location is in the graph (should always be)
            if loc in self.adj_graph:
                 robot_distances[robot] = bfs_distances(self.adj_graph, loc)
            else:
                 # Robot is at a location not in the grid graph? Problem setup issue.
                 # Treat as unreachable for now.
                 robot_distances[robot] = {}


        total_h = 0

        # Calculate cost for each unpainted goal tile
        for tile, required_color in unpainted_goal_tiles.items():
            adjacent_tiles = self.adj_graph.get(tile, [])
            if not adjacent_tiles:
                 # Should not happen in valid problems, but handle defensively
                 # A tile with no adjacent tiles cannot be painted by moving to an adjacent tile.
                 return large_value # Cannot paint this tile

            min_cost_for_tile = large_value # Initialize with a large value

            for robot in robots:
                if robot not in robot_locations or robot not in robot_colors:
                    # Should not happen in valid states, but defensive check
                    continue

                R_loc = robot_locations[robot]
                R_color = robot_colors[robot]

                min_moves_to_adjacent_clear = large_value

                # Find minimum moves to any *clear* adjacent tile
                for adj_tile in adjacent_tiles:
                    # Check if the adjacent tile is clear in the current state
                    # A tile is clear if the predicate (clear tile) is true.
                    is_adj_tile_clear = f"(clear {adj_tile})" in state

                    if is_adj_tile_clear:
                        dist = large_value # Default if unreachable
                        if R_loc == adj_tile:
                            dist = 0 # Robot is already at a clear adjacent tile
                        elif robot in robot_distances and adj_tile in robot_distances[robot]:
                            dist = robot_distances[robot][adj_tile]
                        # else: dist remains large_value (unreachable by BFS)

                        min_moves_to_adjacent_clear = min(min_moves_to_adjacent_clear, dist)

                # If no clear adjacent tile is reachable by this robot, this robot cannot paint this tile yet.
                if min_moves_to_adjacent_clear == large_value:
                    continue # Try next robot

                color_change_cost = 1 if R_color != required_color else 0

                # Cost for this robot to paint this tile: moves + color change + paint action
                cost_for_this_robot = min_moves_to_adjacent_clear + color_change_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            # If no robot can paint this tile (all min_cost_for_tile remained large_value)
            if min_cost_for_tile == large_value:
                 # This tile cannot be painted by any robot from the current state
                 # (e.g., all adjacent tiles are blocked/painted, or robots are isolated)
                 return large_value # Unsolvable state

            total_h += min_cost_for_tile

        return total_h
