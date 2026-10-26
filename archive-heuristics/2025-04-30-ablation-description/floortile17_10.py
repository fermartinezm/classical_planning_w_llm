# Import necessary modules
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import collections # Used for deque in BFS
import math # For infinity

# Helper functions for PDDL fact parsing
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential non-string input or malformed facts
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # print(f"Warning: get_parts received malformed fact: {fact}")
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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

    # Summary
    The heuristic estimates the minimum number of actions required to paint
    all goal tiles with their desired colors. It sums the estimated cost
    for each unpainted goal tile independently, considering the closest
    robot's movement and color change needs.

    # Assumptions
    - All actions have a unit cost of 1.
    - Tiles needing painting are either clear or painted with the wrong color.
      If painted with the wrong color, the problem is unsolvable (heuristic is infinity).
    - Robot movement is possible between adjacent tiles (up, down, left, right).
    - Painting a tile requires the robot to be on an adjacent tile (up or down).
    - Robot conflicts (multiple robots wanting the same tile) and the 'clear'
      precondition for painting (beyond the wrong color check) are ignored
      in the cost estimation (relaxation).
    - Robots always have a color.

    # Heuristic Initialization
    - Extract goal conditions to identify target tiles and colors.
    - Build the tile grid graph from static `up`, `down`, `left`, `right` facts.
    - Compute all-pairs shortest path distances between tiles using BFS.
    - Identify valid painting positions for each tile based on static `up`/`down` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted T C)`.
    2. Determine which of these goal facts are not satisfied in the current state.
       Store these as `unpainted_goals = {tile: goal_color, ...}`.
    3. For each `tile` in `unpainted_goals`, check if the state contains
       `(painted tile C')` for any color `C'` that is *not* the `goal_color`.
       If such a fact exists, the state is unsolvable, return infinity.
    4. If `unpainted_goals` is empty, the state is a goal state, return 0.
    5. For each robot, find its current location and color from the state.
       Store this information, e.g., in a dictionary `{robot_name: {'location': tile_name, 'color': color_name}}`.
    6. Initialize total heuristic cost `h = 0`.
    7. For each `tile_to_paint, goal_color` in `unpainted_goals.items()`:
       a. Find the valid painting positions `X` for `tile_to_paint`. These are tiles
          `X` such that `(up tile_to_paint X)` or `(down tile_to_paint X)` is a static fact.
       b. Find the minimum estimated cost for *any* robot to paint `tile_to_paint`.
          Initialize `min_cost_for_tile = infinity`.
       c. For each robot `R`:
          i. Get robot `R`'s current location `robot_loc_R` and color `robot_color_R`.
          ii. Calculate the minimum movement cost for `R` to reach *any* of the
              valid painting positions `X` for `tile_to_paint`. This is `min_{X} distance(robot_loc_R, X)`.
              Use the precomputed distances. If a painting position is unreachable, its distance is infinity.
          iii. Calculate the color change cost for `R`: 1 if `robot_color_R != goal_color` else 0.
          iv. The estimated cost for robot `R` to paint `tile_to_paint` is
              `min_movement_cost_R + color_change_cost_R + 1` (movement + color change + paint action).
          v. Update `min_cost_for_tile = min(min_cost_for_tile, estimated_cost_R)`.
       d. If `min_cost_for_tile` is still infinity after checking all robots, the tile is unreachable, return infinity.
       e. Add `min_cost_for_tile` to the total heuristic cost `h`.
    8. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the tile graph."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts
        
        # Store goal locations and colors for each tile.
        # {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_paintings[tile] = color

        # Build the tile graph and compute distances
        self.tile_graph = collections.defaultdict(list)
        self.all_tiles = set()

        # Find all tiles by looking at initial state, static facts, and goals
        # This is a heuristic way to find all objects of type tile
        # It assumes tile names start with 'tile_'
        for fact in initial_state | static_facts | self.goals:
             parts = get_parts(fact)
             # Check parts for potential tile names (starting with 'tile_')
             for part in parts:
                 if isinstance(part, str) and part.startswith("tile_"):
                     self.all_tiles.add(part)

        # Add edges based on adjacency predicates
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate in ["up", "down", "left", "right"] and len(args) == 2:
                # These predicates are (dir tileA tileB) meaning tileA is dir of tileB
                # Movement is bidirectional, so add edges both ways
                tile_a, tile_b = args
                # Ensure tiles are in our known set before adding edges
                if tile_a in self.all_tiles and tile_b in self.all_tiles:
                    self.tile_graph[tile_a].append(tile_b)
                    self.tile_graph[tile_b].append(tile_a)
                # else:
                    # print(f"Warning: Adjacency fact {fact} involves unknown tile(s).")


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_tile in self.all_tiles:
            self.distances[start_tile] = self._bfs(start_tile)

        # Store valid painting positions for each tile
        # {tile_to_paint: [robot_pos1, robot_pos2, ...]}
        self.painting_positions = collections.defaultdict(list)
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            # (up tile_to_paint robot_pos) or (down tile_to_paint robot_pos)
            if predicate in ["up", "down"] and len(args) == 2:
                 tile_to_paint, robot_pos = args
                 # Ensure tiles are in our known set before storing
                 if tile_to_paint in self.all_tiles and robot_pos in self.all_tiles:
                    self.painting_positions[tile_to_paint].append(robot_pos)
                 # else:
                    # print(f"Warning: Painting position fact {fact} involves unknown tile(s).")


    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all other nodes."""
        distances = {node: math.inf for node in self.all_tiles}
        # Only start BFS if the start_node is a known tile
        if start_node not in self.all_tiles:
             # print(f"Warning: BFS started from unknown node {start_node}.")
             return distances # Return distances dictionary with all infinities

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node exists in the graph before accessing neighbors
            # (This check is mostly for safety if all_tiles included nodes not in graph keys)
            if current_node not in self.tile_graph:
                 continue

            for neighbor in self.tile_graph[current_node]:
                # Ensure neighbor is a known tile before processing
                if neighbor in self.all_tiles and distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        unpainted_goals = {} # {tile: goal_color}
        current_paintings = {} # {tile: current_color}

        # First, build a map of current paintings
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                current_paintings[tile] = color

        # Identify unpainted goals and check for unsolvable states
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile)

            if current_color is not None and current_color != goal_color:
                 # Tile is painted with the wrong color - unsolvable
                 return math.inf

            if current_color is None: # Tile is not painted
                 unpainted_goals[tile] = goal_color
            # If current_color == goal_color, the goal is met for this tile, do nothing.


        # If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goals:
            return 0

        # Find robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "robot-at" and len(args) == 2:
                robot, location = args
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif predicate == "robot-has" and len(args) == 2:
                 robot, color = args
                 if robot not in robot_info:
                    robot_info[robot] = {}
                 robot_info[robot]['color'] = color

        total_cost = 0

        # For each tile that needs to be painted
        for tile_to_paint, goal_color in unpainted_goals.items():
            valid_painting_positions = self.painting_positions.get(tile_to_paint, [])

            # If a goal tile has no valid painting positions defined in static facts,
            # it might be an invalid problem instance or unsolvable.
            # Return infinity in this case.
            if not valid_painting_positions:
                 # print(f"Warning: Goal tile {tile_to_paint} has no defined painting positions.")
                 return math.inf

            min_cost_for_tile = math.inf

            # Consider each robot
            for robot_name, info in robot_info.items():
                robot_loc = info.get('location')
                robot_color = info.get('color')

                # A robot must have a location and a color to be useful for painting
                if robot_loc is None or robot_color is None:
                    # print(f"Warning: Robot {robot_name} missing location or color in state.")
                    continue # Skip robots not fully defined in state

                # Calculate minimum movement cost to any valid painting position for this tile
                min_movement_cost = math.inf
                # Ensure robot_loc is a known tile before looking up distances
                if robot_loc in self.distances:
                    for paint_pos in valid_painting_positions:
                        # Ensure paint_pos is a known tile and distance is known
                        if paint_pos in self.distances[robot_loc]:
                             min_movement_cost = min(min_movement_cost, self.distances[robot_loc][paint_pos])
                        # else:
                             # print(f"Warning: Painting position {paint_pos} for tile {tile_to_paint} is not a known tile or unreachable from {robot_loc}.")
                # else:
                    # print(f"Warning: Robot location {robot_loc} for robot {robot_name} is not a known tile.")


                # If no path exists from the robot's location to any painting position, this robot can't paint this tile
                if min_movement_cost == math.inf:
                    continue

                # Calculate color change cost
                color_cost = 1 if robot_color != goal_color else 0

                # Estimated cost for this robot to paint this tile: move + change color (if needed) + paint
                estimated_cost = min_movement_cost + color_cost + 1

                # Update minimum cost for this tile across all robots
                min_cost_for_tile = min(min_cost_for_tile, estimated_cost)

            # If no robot can reach a painting position for this tile, it's unsolvable
            if min_cost_for_tile == math.inf:
                 # print(f"Warning: No robot can reach a painting position for tile {tile_to_paint}.")
                 return math.inf

            total_cost += min_cost_for_tile

        return total_cost
