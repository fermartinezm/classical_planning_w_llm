from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing, for each unpainted goal tile,
    the minimum estimated cost for any robot to paint that tile. The estimated cost for a single tile
    includes the minimum moves for a robot to reach a painting position, plus the cost of changing
    color if needed, and the paint action itself.

    # Assumptions
    - The goal state consists of specific tiles being painted with specific colors.
    - Tiles that need to be painted according to the goal are initially in a 'clear' state.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates is consistent and connected
      such that robots can move between adjacent clear tiles.
    - All necessary colors are available.
    - Problem instances are solvable (i.e., goal tiles are reachable and paintable by at least one robot).
      If a tile is unreachable or unpaintable, the heuristic returns infinity.

    # Heuristic Initialization
    - Extracts the target color for each goal tile from the task's goal conditions.
    - Builds a graph representing the tile grid based on 'up', 'down', 'left', 'right' static facts.
    - Determines, for each tile, the set of adjacent tiles from which a robot can paint it.
    - Pre-calculates the shortest path distances between all pairs of tiles using Breadth-First Search (BFS)
      on the grid graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are currently in the 'clear' state but need to be painted
       with a specific color according to the goal conditions. These are the 'unpainted goal tiles'.
    2. If there are no unpainted goal tiles, the heuristic value is 0 (goal state reached).
    3. If there are unpainted goal tiles, initialize the total heuristic cost to 0.
    4. For each unpainted goal tile `T` that needs color `C`:
       a. Find the set of tiles `Painters_T` from which tile `T` can be painted (based on the
          pre-calculated `tile_to_painters` map). If `Painters_T` is empty, return infinity
          as the tile cannot be painted.
       b. Initialize the minimum cost to paint tile `T` (`min_cost_for_this_tile`) to infinity.
       c. For each robot `R` currently in the state:
          i. Get the robot's current location `R_loc` and color `R_color`.
          ii. Calculate the cost of changing color: 1 if `R_color` is not `C`, otherwise 0.
          iii. Find the minimum distance from `R_loc` to any tile in `Painters_T` using the
               pre-calculated shortest path distances. If no painter tile is reachable, this
               minimum distance is infinity.
          iv. If a painter tile is reachable, calculate the total cost for robot `R` to paint
              tile `T`: `minimum_distance + color_change_cost + 1` (for the paint action).
          v. Update `min_cost_for_this_tile` with the minimum cost found so far across all
             painter tiles for robot `R`.
       d. After checking all robots, if `min_cost_for_this_tile` is still infinity, return
          infinity as the tile cannot be painted by any robot.
       e. Add `min_cost_for_this_tile` to the total heuristic cost.
    5. Return the calculated total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal paintings: {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Goal fact is like "(painted tile_1_1 white)"
                if len(args) == 2:
                    tile, color = args
                    self.goal_paintings[tile] = color
                # Handle potential goal groups like (and (...)) - the task parser should handle this,
                # but being robust doesn't hurt. Assuming goals are flattened into a set of facts.

        # 2. Build grid graph and tile_to_painters map
        self.grid_graph = defaultdict(set)
        self.tile_to_painters = defaultdict(set)
        self.all_tiles = set() # Keep track of all tiles mentioned in adjacency facts

        # Adjacency facts define both movement and painting positions
        # (direction tile1 tile2) means tile1 is direction from tile2
        # Robot at tile2 can move to tile1 (if clear)
        # Robot at tile2 can paint tile1 (if clear)
        # So, tile1 is adjacent to tile2 for movement, and tile1 can be painted from tile2.
        adjacency_predicates = ["up", "down", "left", "right"]
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in adjacency_predicates:
                direction, tile1, tile2 = parts[0], parts[1], parts[2]
                
                # Add bidirectional edge for movement
                self.grid_graph[tile1].add(tile2)
                self.grid_graph[tile2].add(tile1)
                
                # Add tile2 as a painter location for tile1
                self.tile_to_painters[tile1].add(tile2)
                
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)

        # 3. Calculate all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            self.distances[start_tile][start_tile] = 0

            while q:
                curr_tile, dist = q.popleft()

                # Check neighbors for movement
                for neighbor in self.grid_graph.get(curr_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_tile][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify robot locations and colors in the current state
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                # Fact is like "(robot-at robot1 tile_0_4)"
                if len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                 # Fact is like "(robot-has robot1 white)"
                 if len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color

        total_heuristic = 0
        infinity = float('inf')

        # Identify unpainted goal tiles that are currently clear
        unpainted_goals = {} # {tile: required_color}
        for tile, goal_color in self.goal_paintings.items():
            # Check if the tile is already painted correctly
            is_painted_correctly = f"(painted {tile} {goal_color})" in state

            if not is_painted_correctly:
                # If it's not painted correctly, is it clear?
                is_clear = f"(clear {tile})" in state
                if is_clear:
                    # This tile needs painting
                    unpainted_goals[tile] = goal_color
                # Note: If it's not clear and not painted correctly, it must be
                # painted with the wrong color. Assuming solvable instances,
                # this case implies an unsolvable path from the initial state,
                # but the heuristic doesn't explicitly handle this by returning
                # infinity here. It just won't count this tile as an unpainted_goal.
                # A more robust heuristic might check for wrong colors and return infinity.
                # For this problem, we follow the assumption that unmet goals are clear.


        if not unpainted_goals:
            # All painting goals are met
            return 0

        # Calculate sum of minimum costs for each unpainted goal tile
        for tile_to_paint, required_color in unpainted_goals.items():
            min_cost_for_this_tile = infinity

            # Find potential painting locations for this tile
            painters = self.tile_to_painters.get(tile_to_paint, set())
            if not painters:
                # This tile cannot be painted from any adjacent tile according to static facts.
                # This indicates a potentially invalid problem instance or an unreachable goal.
                # Return infinity as it's likely unsolvable.
                return infinity

            # Find the minimum cost for any robot to paint this tile
            for robot, robot_loc in robot_locations.items():
                robot_color = robot_colors.get(robot) # Get robot's current color

                # Cost to change color if needed
                color_change_cost = 0 if robot_color == required_color else 1

                # Find min distance from robot_loc to any painter tile for tile_to_paint
                min_dist_to_painter = infinity
                if robot_loc in self.distances: # Ensure robot_loc is a known tile
                    for painter_loc in painters:
                        if painter_loc in self.distances[robot_loc]:
                            dist = self.distances[robot_loc][painter_loc]
                            min_dist_to_painter = min(min_dist_to_painter, dist)

                # If a painter location is reachable by this robot
                if min_dist_to_painter != infinity:
                    # Total cost for this robot to paint this tile = moves + color_change + paint_action
                    cost = min_dist_to_painter + color_change_cost + 1
                    min_cost_for_this_tile = min(min_cost_for_this_tile, cost)

            # If after checking all robots, no robot can reach a painter location for this tile
            if min_cost_for_this_tile == infinity:
                # This tile is unreachable/unpaintable by any robot in the current state
                return infinity # Unsolvable state from here

            total_heuristic += min_cost_for_this_tile

        return total_heuristic
