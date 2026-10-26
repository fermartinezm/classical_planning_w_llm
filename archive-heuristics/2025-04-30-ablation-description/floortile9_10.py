# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment
from fnmatch import fnmatch
from collections import deque
import math # Import math for infinity

# Helper functions (assuming they are available or defined elsewhere)
# def get_parts(fact): ...
# def match(fact, *args): ...

# Dummy Heuristic base class for standalone testing
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError

# Dummy Task class for standalone testing
class Task:
    def __init__(self, name, facts, initial_state, goals, operators, static):
        self.name = name
        self.facts = facts
        self.initial_state = initial_state
        self.goals = goals
        self.operators = operators
        self.static = static

# Helper functions from Logistics example
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
    # Using zip handles cases where parts is shorter than args (mismatch)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    The heuristic estimates the total cost to paint all goal tiles that are not
    yet painted correctly. For each unpainted goal tile, it sums the estimated
    cost to get a robot to the correct adjacent location, change to the required
    color (if necessary), and perform the paint action. It considers the minimum
    cost over all available robots for each goal tile.

    # Assumptions
    - Tiles are arranged in a grid, and adjacency is defined by up/down/left/right predicates.
    - Only tiles that are 'clear' can be painted.
    - If a goal tile is painted with a color different from the goal color, the state is unsolvable (heuristic returns infinity).
    - Action costs are uniform (cost 1).
    - All colors required for goals are available.

    # Heuristic Initialization
    - Extract goal conditions (which tiles need which color).
    - Build the grid graph (adjacency list) from static facts (up, down, left, right). The graph is treated as undirected for movement.
    - Identify all tile objects mentioned in adjacency facts.
    - Pre-compute shortest path distances between all pairs of tiles using BFS.
    - Map each tile to the set of tiles where a robot must be located to paint it, based on the paint action preconditions and static adjacency facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize total heuristic cost to 0.
    2. Identify the current location and color of each robot in the state by examining `(robot-at ...)` and `(robot-has ...)` facts. Store this information in a dictionary mapping robot names to their location and color.
    3. Iterate through each goal condition `(painted ?tile ?color)` stored during initialization.
    4. For the current goal tile `T` and required color `C`:
       a. Check the state of tile `T` by looking for `(painted T ...)` or `(clear T)` facts in the current state:
          - If `(painted T C)` is true, this goal is satisfied. Continue to the next goal tile.
          - If `(painted T C')` is true where `C' != C`, the state is likely unsolvable as tiles cannot be repainted according to the domain rules. Return `float('inf')`.
          - If `(clear T)` is true, this tile needs painting. Proceed to step 4b.
          - If `T` is neither painted correctly nor clear, this indicates an issue (e.g., painted incorrectly, or domain/problem inconsistency). Assuming valid problems, this case should be covered by the `is_painted_wrong` check.
       b. If `(clear T)` is true, calculate the minimum cost for *any* robot to paint tile `T` with color `C`:
          - Find the set of tiles `Paint_Locs_T` where a robot must be located to paint tile `T`. This set was pre-computed during initialization (`self.paint_locations[T]`).
          - Initialize `min_robot_cost_for_tile = float('inf')`.
          - For each robot `r` at `R_loc_r` with color `R_color_r` (identified in step 2):
             - Calculate the cost for this specific robot `r` to paint tile `T`:
               - Color change cost: 1 if `R_color_r != C` else 0.
               - Movement cost: Find the minimum grid distance from the robot's current location `R_loc_r` to any tile in the set `Paint_Locs_T`. Use the pre-computed distances (`self.distances`). If `R_loc_r` is not in the grid or no tile in `Paint_Locs_T` is reachable from `R_loc_r`, this robot cannot reach the paint location.
               - Paint action cost: 1 (for the `paint_...` action itself).
               - Total cost for robot `r` to paint `T`: `color_change_cost + movement_cost + paint_action_cost`.
             - Update `min_robot_cost_for_tile = min(min_robot_cost_for_tile, total_cost_for_robot_r)`.
          - After checking all robots, if `min_robot_cost_for_tile` is still `float('inf')`, it means no robot can reach and paint this tile from the current state, so the state is unsolvable. Return `float('inf')`.
          - Otherwise, add `min_robot_cost_for_tile` to the `total_cost`.
    5. After iterating through all goal tiles, return the final `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        # Heuristic base class initialization
        # super().__init__(task) # Uncomment this line in the actual environment
        self.goals = task.goals
        static_facts = task.static

        # Store goal paintings: {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Build the grid graph (adjacency list) and collect all tile names
        self.adj = {}
        self.tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                self.tiles.add(t1)
                self.tiles.add(t2)
                # Add bidirectional edges for movement
                self.adj.setdefault(t1, set()).add(t2)
                self.adj.setdefault(t2, set()).add(t1)

        # Pre-compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.tiles:
            self.distances[start_tile] = self._bfs(start_tile)

        # Map tile to its adjacent tiles where robot must be located to paint it
        # Based on paint action preconditions: (paint_DIR ?r ?y ?x ?c) requires (DIR ?y ?x)
        # Robot at ?x paints ?y. So, if goal is to paint ?y, robot must be at ?x.
        self.paint_locations = {} # {goal_tile: set_of_robot_locations_to_paint_it}
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] in ["up", "down", "left", "right"]:
                 goal_tile = parts[1] # The tile being painted (?y)
                 robot_loc = parts[2] # The tile the robot must be at (?x)
                 self.paint_locations.setdefault(goal_tile, set()).add(robot_loc)


    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.tiles}
        if start_node not in self.tiles:
             # Start node is not part of the known grid, cannot reach anything
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.adj: # Handle tiles with no connections (shouldn't happen if tiles come from adj)
                for neighbor in self.adj[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot locations and colors
        robots_info = {} # {robot_name: {'location': tile, 'color': color}}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "robot-at":
                  robot, location = parts[1], parts[2]
                  robots_info.setdefault(robot, {})['location'] = location
             elif parts[0] == "robot-has":
                  robot, color = parts[1], parts[2]
                  robots_info.setdefault(robot, {})['color'] = color

        total_cost = 0

        # Check each goal painting
        for goal_tile, goal_color in self.goal_paintings.items():
            is_painted_correctly = False
            is_painted_wrong = False
            is_clear = False

            # Check the state of the goal tile
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "painted" and parts[1] == goal_tile:
                    if parts[2] == goal_color:
                        is_painted_correctly = True
                        break # Goal satisfied for this tile
                    else:
                        is_painted_wrong = True
                        break # Painted with wrong color
                elif parts[0] == "clear" and parts[1] == goal_tile:
                    is_clear = True
                    # Don't break, ensure we check for painted status too

            if is_painted_correctly:
                continue # This goal is already met

            if is_painted_wrong:
                 # This state is likely unsolvable based on domain rules (cannot repaint)
                 return float('inf') # Return infinity

            if is_clear:
                # Tile needs painting. Calculate min cost for any robot.
                min_robot_cost_for_tile = float('inf')

                # Find valid robot locations to paint this tile
                valid_paint_robot_locations = self.paint_locations.get(goal_tile, set())

                if not valid_paint_robot_locations:
                    # This goal tile cannot be painted based on static adjacency facts.
                    # This implies an unsolvable problem instance.
                     return float('inf')

                for robot_name, info in robots_info.items():
                    robot_loc = info.get('location')
                    robot_color = info.get('color')

                    if robot_loc is None or robot_color is None:
                         # Robot state is incomplete? Should not happen in valid states.
                         continue # Skip this robot

                    # Cost for color change
                    color_cost = 1 if robot_color != goal_color else 0

                    # Cost for movement to a paint location
                    min_dist_to_paint_loc = float('inf')
                    if robot_loc in self.distances: # Ensure robot location is a known tile in the grid
                        for paint_loc in valid_paint_robot_locations:
                            if paint_loc in self.distances.get(robot_loc, {}): # Ensure paint location is a known tile reachable from robot_loc
                                dist = self.distances[robot_loc][paint_loc]
                                if dist != float('inf'): # Ensure paint_loc is reachable from robot_loc
                                    min_dist_to_paint_loc = min(min_dist_to_paint_loc, dist)

                    if min_dist_to_paint_loc == float('inf'):
                         # Robot cannot reach any paint location for this tile
                         continue # Skip this robot

                    # Total cost for this robot to paint this tile
                    # Cost = (change color if needed) + (move to paint location) + (paint action)
                    current_robot_cost = color_cost + min_dist_to_paint_loc + 1

                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, current_robot_cost)

                if min_robot_cost_for_tile == float('inf'):
                    # No robot can reach and paint this tile
                    return float('inf') # State is unsolvable

                total_cost += min_robot_cost_for_tile

            # If tile is not painted correctly, not painted wrong, and not clear,
            # this is an unexpected state. Treat as unsolvable.
            # This case should ideally be covered by the `is_painted_wrong` check
            # if the domain implies tiles are either clear or painted.
            # Adding an explicit check just in case, though it might overlap.
            # If not is_painted_correctly and not is_painted_wrong and not is_clear:
            #     # Tile exists but is in an unknown state?
            #     return float('inf')


        return total_cost
