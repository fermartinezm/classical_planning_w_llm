from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with the correct colors. It sums the estimated minimum cost for
    any robot to paint each unpainted goal tile independently.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates is static.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1 action.
    - Painting a tile costs 1 action.
    - The heuristic ignores the 'clear' predicate for calculating movement costs
      (i.e., assumes movement is always possible between adjacent tiles in the static grid).
    - If a goal tile is already painted with a color different from the goal color,
      the problem is considered unsolvable from that state, as there's no action
      to unpaint or repaint a non-clear tile.
    - All goal colors must be among the available colors.

    # Heuristic Initialization
    - Builds a static undirected graph of tiles based on up/down/left/right facts
      to calculate shortest path distances for movement.
    - Stores the set of available colors.
    - Stores the goal tile-color mappings.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is the sum of estimated costs for
    each goal tile that is not yet painted with the correct color:

    1. Identify all goal facts of the form `(painted tile_Y color_C)`.
    2. For each such goal fact:
       a. Check if the fact `(painted tile_Y color_C)` is already true in the current state. If yes, the cost for this tile is 0.
       b. If the goal fact is not true:
          i. Check if `tile_Y` is currently painted with *any* color. If it is painted with a color *different* from `color_C`, the goal is unreachable from this state (as paint requires clear and there's no unpaint action). Return infinity.
          ii. Check if `color_C` is an available color. If not, the goal is unreachable. Return infinity.
          iii. Initialize the minimum cost for this tile (`min_robot_cost_for_tile`) to infinity.
          iv. Identify all tiles adjacent to `tile_Y` (these are the locations a robot must be at to paint `tile_Y`). These are the tiles `Z` such that `(up tile_Y Z)`, `(down tile_Y Z)`, `(left tile_Y Z)`, or `(right tile_Y Z)` holds.
          v. If `tile_Y` has no adjacent tiles, the problem is likely unsolvable; return infinity.
          vi. For each robot `R` in the state:
              - Get the robot's current location (`robot_loc`).
              - Calculate the cost for `R` to acquire the required color `color_C`: 1 if `R` does not have `color_C`, otherwise 0.
              - Calculate the minimum movement cost for `R` to reach *any* of the tiles adjacent to `tile_Y`. This is done using BFS on the static tile graph.
              - If no adjacent tile is reachable by `R`, this robot cannot paint `tile_Y` (in this relaxed view).
              - If reachable, the cost for `R` to paint `tile_Y` is: (cost to get color) + (minimum movement cost) + 1 (cost of paint action).
              - Update `min_robot_cost_for_tile` with the minimum cost found among all robots.
          vii. If `min_robot_cost_for_tile` is still infinity after checking all robots, the problem is likely unsolvable; return infinity.
          viii. Add `min_robot_cost_for_tile` to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Tile graph for movement distances.
        - Available colors.
        - Goal tile-color mappings.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the static tile graph (undirected) and store neighbors
        self.tile_graph = {}
        self.tile_neighbors = {} # Stores neighbors for painting locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                # Undirected graph for movement
                self.tile_graph.setdefault(t1, set()).add(t2)
                self.tile_graph.setdefault(t2, set()).add(t1)
                # Neighbors for painting (a robot at t2 can paint t1 if (up/down/left/right t1 t2))
                # This means t2 is a paint location for t1.
                self.tile_neighbors.setdefault(t1, set()).add(t2)


        # Store available colors
        self.available_colors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])

        # Store goal tile-color mappings
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color


    def shortest_path_distance(self, start_tile, end_tile):
        """
        Computes the shortest path distance between two tiles using BFS on the static graph.
        Returns float('inf') if no path exists.
        """
        if start_tile == end_tile:
            return 0
        queue = deque([(start_tile, 0)])
        visited = {start_tile}
        while queue:
            current_tile, dist = queue.popleft()
            if current_tile not in self.tile_graph: # Should not happen if start_tile is valid
                 return float('inf')
            for neighbor in self.tile_graph.get(current_tile, []):
                if neighbor == end_tile:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf') # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        robots = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot) # Ensure all robots are found

        total_heuristic = 0

        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if goal is already satisfied
            is_painted_correctly = False
            is_painted_wrongly = False

            # Iterate through state facts to find painted status
            for fact in state:
                if match(fact, "painted", goal_tile, goal_color):
                    is_painted_correctly = True
                    break
                # Check if painted with *any* color other than the goal color
                if match(fact, "painted", goal_tile, "*"):
                     if not match(fact, "painted", goal_tile, goal_color):
                         is_painted_wrongly = True
                         break # Found wrongly painted, no need to check further for this tile

            if is_painted_wrongly:
                 # If painted with the wrong color, it's not clear and cannot be repainted.
                 # This goal is unreachable.
                 return float('inf')

            if not is_painted_correctly:
                # Check if goal color is available - if not, problem is unsolvable
                if goal_color not in self.available_colors:
                     return float('inf')

                min_robot_cost_for_tile = float('inf')

                # Find tiles where a robot must be AT to paint the goal tile
                paint_locations_for_tile = self.tile_neighbors.get(goal_tile, set())

                if not paint_locations_for_tile:
                     # This tile has no neighbors, cannot be painted. Unsolvable.
                     return float('inf')

                # If there are no robots, no tile can be painted. Unsolvable if goals exist.
                if not robots:
                    return float('inf')

                for robot in robots: # Iterate through all robots
                    robot_loc = robot_locations.get(robot)
                    if robot_loc is None:
                         # Robot location is unknown? Should not happen in valid states.
                         # This might indicate an issue with state representation or parsing.
                         # For robustness, assume this state is problematic.
                         return float('inf')

                    cost_get_color = 1 if robot_colors.get(robot) != goal_color else 0

                    min_move_cost = float('inf')
                    for paint_loc in paint_locations_for_tile:
                        dist = self.shortest_path_distance(robot_loc, paint_loc)
                        if dist is not float('inf'):
                            min_move_cost = min(min_move_cost, dist)

                    if min_move_cost is not float('inf'):
                        # Cost for this robot to get color, move to paint location, and paint
                        robot_cost = cost_get_color + min_move_cost + 1 # +1 for paint action
                        min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)

                if min_robot_cost_for_tile is float('inf'):
                     # No robot can reach any valid paint location for this tile. Unsolvable.
                     return float('inf')

                total_heuristic += min_robot_cost_for_tile

        return total_heuristic
