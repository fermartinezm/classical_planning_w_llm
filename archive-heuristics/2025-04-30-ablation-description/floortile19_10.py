from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for shortest path calculation using BFS
def shortest_path_distance(start_node, target_nodes_set, graph):
    """
    Finds the shortest path distance from start_node to any node in target_nodes_set
    in an unweighted graph using BFS.
    Returns infinity if no target node is reachable.
    """
    if not target_nodes_set:
        # If there are no target nodes, distance is effectively infinite
        return float('inf')

    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft()

        if current_node in target_nodes_set:
            return dist

        # Ensure current_node is in the graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Target nodes are unreachable

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles that are not yet painted correctly. It sums the estimated
    cost for each unpainted goal tile, considering the closest robot, the
    distance to an adjacent tile, and the need to change color.

    # Assumptions:
    - The grid of tiles is connected.
    - Tiles that need to be painted according to the goal are initially clear
      or remain clear until painted.
    - Tiles painted with the wrong color cannot be repainted (unsolvable state
      if a goal tile is painted incorrectly).
    - All actions (move, change_color, paint) have a cost of 1.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color.
    - Builds a graph representing the adjacency relationships between tiles
      based on the static `up`, `down`, `left`, and `right` facts. This graph
      is used for shortest path calculations.
    - Pre-calculates the set of adjacent tiles for each goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Extract the current location and color of each robot from the state.
    3. Identify which goal tiles are not yet painted correctly.
       - Iterate through the `goal_tiles_spec` (pre-calculated in `__init__`).
       - For each goal tile `T` needing color `C`:
         - Check if `(painted T C)` is in the current state. If yes, this goal is met for this tile.
         - If not, check if `(painted T C')` for any `C' != C` is in the state. If yes, the state is likely unsolvable; return a large heuristic value (infinity).
         - If `(clear T)` is in the state and the tile is not painted correctly, it needs painting. Add it to a list of `tiles_to_paint`.
    4. Initialize `total_heuristic_cost` to 0.
    5. For each tile `T` in the `tiles_to_paint` list:
       a. Get the required color `C` for `T` from `goal_tiles_spec`.
       b. Get the pre-calculated set of tiles `Adjacent(T)` that are adjacent to `T`.
       c. Initialize `min_cost_for_tile` to infinity.
       d. For each robot `R`:
          i. Get the robot's current location `L_R`.
          ii. Calculate the shortest path distance `dist` from `L_R` to any
              tile in `Adjacent(T)` using the pre-built tile graph and BFS.
          iii. Determine the color cost: 0 if robot `R` currently has color `C`,
               1 otherwise (assuming a `change_color` action is needed and possible).
          iv. The estimated cost for robot `R` to paint tile `T` is
              `dist + color_cost + 1` (where +1 is for the `paint` action itself).
          v. Update `min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)`.
       e. If `min_cost_for_tile` is still infinity after checking all robots,
          it means no robot can reach an adjacent tile to paint `T`. The state
          is likely unsolvable; return a large heuristic value (infinity).
       f. Add `min_cost_for_tile` to `total_heuristic_cost`.
    6. Return `total_heuristic_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the tile graph."""
        self.goals = task.goals
        self.static_facts = task.static

        # Build the tile graph from static adjacency facts
        self.tile_graph = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Facts are like (direction tile_A tile_B) meaning tile_A is in that direction from tile_B
                # So, tile_A and tile_B are adjacent.
                tile_a, tile_b = parts[1], parts[2]
                # Add bidirectional edges
                self.tile_graph.setdefault(tile_a, []).append(tile_b)
                self.tile_graph.setdefault(tile_b, []).append(tile_a)

        # Store goal tiles and their required colors
        self.goal_tiles_spec = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles_spec[tile] = color

        # Pre-calculate adjacent tiles for each goal tile
        self.goal_tile_adjacencies = {}
        for goal_tile in self.goal_tiles_spec:
             # Find all tiles X such that X is adjacent to goal_tile
             adjacent_to_goal = set()
             # Neighbors of the goal tile in the graph are its adjacent tiles
             if goal_tile in self.tile_graph:
                 adjacent_to_goal.update(self.tile_graph[goal_tile])
             self.goal_tile_adjacencies[goal_tile] = adjacent_to_goal


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Extract current robot locations and colors
        current_robot_locations = {}
        current_robot_colors = {}
        robots = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                current_robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                current_robot_colors[robot] = color
                robots.add(robot)

        # 3. Identify goal tiles that need painting
        tiles_to_paint = {} # {tile: required_color}
        for goal_tile, required_color in self.goal_tiles_spec.items():
            is_painted_correctly = False
            is_painted_wrongly = False
            is_clear = False

            # Check the state for the status of the goal tile
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "painted" and parts[1] == goal_tile:
                    if parts[2] == required_color:
                        is_painted_correctly = True
                        break # Goal met for this tile
                    else:
                        is_painted_wrongly = True
                        break # Painted incorrectly
                elif parts[0] == "clear" and parts[1] == goal_tile:
                    is_clear = True

            if is_painted_wrongly:
                 # If a goal tile is painted with the wrong color, it's likely unsolvable
                 # given the domain actions. Return a large heuristic value.
                 return float('inf')

            if not is_painted_correctly and is_clear:
                tiles_to_paint[goal_tile] = required_color

        # If all goal tiles are painted correctly (and none painted wrongly),
        # the heuristic should be 0. This check is already done at the start,
        # but this handles the case where some goal tiles were already correct
        # and others were not needed (not in goal_tiles_spec).
        if not tiles_to_paint:
             return 0 # Should be covered by the initial check, but good safety.


        # 4. Initialize total heuristic cost
        total_heuristic_cost = 0

        # 5. Calculate minimum cost for each tile that needs painting
        for tile, required_color in tiles_to_paint.items():
            adjacent_to_tile = self.goal_tile_adjacencies.get(tile, set())

            if not adjacent_to_tile:
                 # This goal tile has no adjacent tiles? Unreachable?
                 # Or maybe the graph wasn't built correctly?
                 # Assuming valid problems, this shouldn't happen for goal tiles.
                 # Treat as unsolvable for safety.
                 return float('inf')

            min_cost_for_tile = float('inf')

            # Consider each robot as a potential candidate to paint this tile
            for robot in robots:
                robot_location = current_robot_locations.get(robot)
                robot_color = current_robot_colors.get(robot)

                if robot_location is None:
                    # Robot location unknown? Should not happen in valid states.
                    continue # Skip this robot

                # Calculate distance from robot's current location to any adjacent tile
                dist = shortest_path_distance(robot_location, adjacent_to_tile, self.tile_graph)

                if dist == float('inf'):
                    # Robot cannot reach any tile adjacent to the goal tile
                    continue # Skip this robot for this tile

                # Calculate color change cost
                # Assumes available-color predicate is handled by the planner's applicability check
                # and that changing color is always possible if the color is available.
                color_cost = 1 if robot_color != required_color else 0

                # Total cost for this robot to paint this tile:
                # moves + color change (if needed) + paint action
                cost_for_robot = dist + color_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            if min_cost_for_tile == float('inf'):
                 # No robot can paint this tile
                 return float('inf') # State is likely unsolvable

            total_heuristic_cost += min_cost_for_tile

        # 6. Return total heuristic cost
        return total_heuristic_cost
