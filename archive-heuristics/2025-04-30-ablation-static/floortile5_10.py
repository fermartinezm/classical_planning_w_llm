# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For math.inf

# Utility functions (copied from Logistics example, slightly adjusted match)
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
    # Check if each part matches the corresponding arg pattern
    # This handles cases where len(parts) != len(args) if '*' is used appropriately.
    # e.g., match("(at obj room)", "at", "*", "*") works.
    # match("(at obj)", "at", "*", "*") would fail the all() check.
    # match("(at obj room extra)", "at", "*", "*") would fail the all() check.
    # This seems sufficient for the patterns used in this heuristic.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS helper function
def bfs_shortest_path(start_tile, target_tiles, clear_tiles, adj_list):
    """
    Finds the shortest distance (number of moves) from start_tile to any tile in target_tiles.
    Movement is restricted to tiles in clear_tiles.
    Returns distance or math.inf if no path exists.
    """
    # If the robot is already at a target tile (which must be clear, handled by caller), 0 moves are needed.
    if start_tile in target_tiles:
        return 0

    q = deque([(start_tile, 0)])
    visited = {start_tile} # Robot is at start_tile, can start BFS from here.

    while q:
        curr_tile, dist = q.popleft()

        # Explore neighbors
        for neighbor in adj_list.get(curr_tile, []):
            # Can only move *to* a clear tile
            if neighbor in clear_tiles and neighbor not in visited:
                if neighbor in target_tiles:
                    return dist + 1 # Found shortest path to a target tile
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

    return math.inf # No path found

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles that are not yet painted correctly. It calculates the minimum
    cost for each unpainted goal tile independently, considering the distance
    for the closest robot to reach an adjacent clear tile, the cost to change
    color if necessary, and the paint action itself.

    # Assumptions
    - Tiles specified in the goal that are not marked as 'painted' in the
      current state are assumed to be 'clear' and need painting.
    - If a goal tile is painted with the wrong color, the goal is unreachable.
      The heuristic returns infinity in this case.
    - Robots can only move onto 'clear' tiles.
    - Painting a tile makes it 'not clear'.
    - The grid structure is defined by 'up', 'down', 'left', 'right' facts.
    - The cost of each action (move, change_color, paint) is 1.
    - The heuristic sums the minimum costs for each unpainted goal tile,
      ignoring potential resource contention (multiple robots/tiles).

    # Heuristic Initialization
    - Extracts the grid adjacency list from static 'up', 'down', 'left',
      'right' facts.
    - Extracts the target color for each goal tile from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task's goals.
    2. In the current state, identify the location and color of each robot.
    3. In the current state, identify which tiles are 'clear' and which are 'painted' (and their color).
    4. Identify which goal tiles are not yet painted with the correct color.
    5. For each unpainted goal tile (let's call it `goal_tile`) with target color `target_color`:
        a. Check if `goal_tile` is currently 'clear'. If not (i.e., it's painted, possibly the wrong color), the goal is unreachable from this state. Return infinity immediately.
        b. If `goal_tile` is clear, find all tiles adjacent to `goal_tile` using the pre-computed grid adjacency list. These are the potential locations (`adjacent_tiles`) from which a robot can paint `goal_tile`.
        c. Initialize the minimum cost to paint this `goal_tile` (`min_cost_for_tile`) to infinity.
        d. For each robot:
            i. Calculate the shortest distance from the robot's current location (`robot_loc`) to *any* tile in `adjacent_tiles`. This distance calculation must only consider paths through tiles that are currently 'clear', *including* the final adjacent tile where the robot will stand to paint. Use a Breadth-First Search (BFS) where the start node is `robot_loc`, the target nodes are `adjacent_tiles`, and movement is restricted to `clear_tiles`.
            ii. If a path exists (BFS returns a finite distance `dist`):
                Calculate the cost for this robot to paint `goal_tile`: `dist + (1 if robot needs to change color to target_color else 0) + 1 (for the paint action)`.
                Update `min_cost_for_tile = min(min_cost_for_tile, cost)`.
        e. If after checking all robots, `min_cost_for_tile` is still infinity, it means no robot can reach a clear adjacent tile to paint the clear `goal_tile`. The goal is unreachable. Return infinity immediately.
        f. Add `min_cost_for_tile` to the total heuristic value.
    6. If the loop finishes without returning infinity, return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure and goal tiles.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build grid adjacency list from static facts
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                # Fact (dir tile1 tile2) means tile1 is dir from tile2.
                # Robot at tile2 can move dir to tile1.
                # So tile1 is a neighbor of tile2, and tile2 is a neighbor of tile1.
                self.adj.setdefault(tile1, []).append(tile2)
                self.adj.setdefault(tile2, []).append(tile1)

        # Store goal tiles and their target colors
        self.goal_tiles = {}
        for goal in self.goals:
            # Assuming goals are conjunctions of (painted tile color)
            if match(goal, "painted", "*", "*"):
                 _, tile, color = get_parts(goal)
                 self.goal_tiles[tile] = color
            # Note: This assumes task.goals is a set of ground facts like `(painted tile_1_1 white)`.
            # If it's a complex PDDL goal structure like `(and ...)`, the task parser should handle it
            # and provide task.goals as the set of ground facts that must be true.
            # The provided Task class seems to handle this correctly.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        robot_locs = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles = {} # {tile: color}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locs[robot] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "clear":
                tile = parts[1]
                clear_tiles.add(tile)
            elif parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        total_heuristic_cost = 0

        # For each goal tile, check if it's painted correctly or needs painting
        for goal_tile, target_color in self.goal_tiles.items():
            is_painted_correctly = goal_tile in painted_tiles and painted_tiles[goal_tile] == target_color

            if not is_painted_correctly:
                # This goal tile needs painting.
                # First, check if it's clear (precondition for paint).
                if goal_tile not in clear_tiles:
                    # Goal tile is painted the wrong color. Unsolvable.
                    # print(f"Warning: Goal tile {goal_tile} needs {target_color} but is painted {painted_tiles.get(goal_tile, 'unknown')}. Unsolvable.")
                    return math.inf # Problem is unsolvable from this state

                # Goal tile is clear and needs painting.
                # Find tiles adjacent to the goal tile (where robot must stand).
                adjacent_tiles = set(self.adj.get(goal_tile, []))

                # The robot must move to an adjacent tile that is *clear*.
                # So, the target tiles for BFS are adjacent_tiles that are also clear.
                reachable_adjacent_tiles = adjacent_tiles & clear_tiles

                if not reachable_adjacent_tiles:
                    # No clear adjacent tile exists. Cannot paint this tile. Unsolvable.
                    # print(f"Warning: No clear adjacent tile for {goal_tile}. Unsolvable.")
                    return math.inf

                min_cost_for_tile = math.inf

                # Find the minimum cost for any robot to paint this tile
                for robot, robot_loc in robot_locs.items():
                    # Calculate shortest distance from robot_loc to any reachable_adjacent_tile
                    # The BFS searches through clear tiles. The start tile robot_loc doesn't need to be clear.
                    dist_to_any_adjacent_clear = bfs_shortest_path(robot_loc, reachable_adjacent_tiles, clear_tiles, self.adj)

                    if dist_to_any_adjacent_clear != math.inf:
                        # Cost to reach an adjacent clear tile + change color (if needed) + paint
                        color_change_cost = 1 if robot_colors.get(robot) != target_color else 0 # Use .get in case robot_has fact is missing (though domain implies it's always there)
                        cost = dist_to_any_adjacent_clear + color_change_cost + 1
                        min_cost_for_tile = min(min_cost_for_tile, cost)

                # If after checking all robots, none can paint this tile, it's unsolvable.
                if min_cost_for_tile == math.inf:
                     # print(f"Warning: No robot can reach a clear adjacent tile for {goal_tile}. Unsolvable.")
                     return math.inf

                total_heuristic_cost += min_cost_for_tile

        # If we processed all goal tiles and didn't return infinity, return the total cost.
        return total_heuristic_cost
