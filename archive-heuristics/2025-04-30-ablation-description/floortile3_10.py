from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact is at least the number of arguments in the pattern
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint
    all goal tiles with the correct color. It sums the estimated cost for
    each goal tile that is not yet painted correctly. The cost for a single
    tile includes the paint action itself, the minimum movement cost for
    any robot to reach a position adjacent to the tile, and the cost for
    that robot to have the required color. Movement cost is estimated using
    shortest path on the grid graph, ignoring the 'clear' precondition.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' static facts.
    - Robots always have a color (no 'free-color' state transition is possible
      via 'change_color' action alone).
    - Tiles are either 'clear' or 'painted'. A tile cannot be both clear and painted,
      nor can it be neither clear nor painted (unless it's a dead end).
    - If a goal tile is painted with the wrong color, the problem is unsolvable
      (no unpaint action). The heuristic returns infinity in this case.
    - All tiles are connected in the grid graph.

    # Heuristic Initialization
    - Extracts goal conditions (which tiles need which color).
    - Parses static facts to build the grid graph (adjacency list for movement).
    - Identifies all unique tiles in the domain.
    - Computes all-pairs shortest paths on the grid graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value `h` to 0.
    2. Extract the current location and color of each robot from the current state.
    3. Iterate through each goal tile and its required color stored during initialization (`self.goal_painted_tiles`).
    4. For the current goal tile `tile_G` and required color `color_C`:
       a. Check if the fact `(painted tile_G color_C)` is already present in the current state. If yes, this goal is satisfied for this tile; continue to the next goal.
       b. If the goal is not satisfied, check if the tile `tile_G` is painted with a *different* color `color_W` in the current state. This indicates a dead end. Iterate through the state facts to find if any fact `(painted tile_G color_W)` exists where `color_W` is not `color_C`. If found, return `float('inf')`.
       c. If the tile `tile_G` is not painted with the correct color and is not painted with a wrong color (implying it must be clear, based on domain effects), it needs to be painted.
       d. Add 1 to the heuristic for the painting action itself.
       e. Determine the possible locations `required_robot_locs` where a robot could be positioned to paint `tile_G`. These are the tiles `tile_R` that are directly adjacent to `tile_G` in the grid graph (i.e., neighbors of `tile_G` in `self.adj`).
       f. Initialize `min_cost_for_this_tile = float('inf')`. This will track the minimum cost for *any* robot to paint this specific tile.
       g. For each robot `r` and its current location `robot_loc`:
          i. Get the robot's current color `robot_color`.
          ii. Calculate the color change cost: `color_cost = 1` if `robot_color` is not `color_C`, otherwise `0`. (Assumes robot always has a color).
          iii. Find the minimum shortest path distance `min_dist_to_required_loc` from the robot's current location `robot_loc` to any of the tiles in `required_robot_locs` using the precomputed shortest paths (`self.shortest_paths`). If `robot_loc` or any `required_loc` is not in the precomputed paths (e.g., disconnected grid), the distance will remain `float('inf')`.
          iv. Calculate the total cost for this specific robot `r` to paint this tile: `robot_cost_for_tile = min_dist_to_required_loc + color_cost`.
          v. Update `min_cost_for_this_tile = min(min_cost_for_this_tile, robot_cost_for_tile)`.
       h. If `min_cost_for_this_tile` is still `float('inf')` after checking all robots, it means no robot can reach a position to paint this goal tile. This state is unsolvable. Return `float('inf')`.
       i. Add `min_cost_for_this_tile` to the total heuristic `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals  # Goal conditions: set of facts like '(painted tile_1_1 white)'
        static_facts = task.static  # Static facts: frozenset of facts like '(up tile_1_1 tile_0_1)'

        # Store goal locations and colors for easy lookup
        self.goal_painted_tiles = {} # Map tile -> color for goal state
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

        # Build the grid graph (adjacency list for movement)
        self.adj = defaultdict(set)
        all_tiles = set()

        # Directions and their opposites for building bidirectional graph
        directions = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in directions:
                direction = parts[0]
                tile_y, tile_x = parts[1], parts[2] # (direction tile_Y tile_X) means tile_Y is direction from tile_X
                all_tiles.add(tile_x)
                all_tiles.add(tile_y)

                # Add edges for movement (bidirectional)
                self.adj[tile_x].add(tile_y)
                self.adj[tile_y].add(tile_x)

        self.all_tiles = list(all_tiles) # Get a list of all unique tiles

        # Compute all-pairs shortest paths on the grid graph using BFS
        self.shortest_paths = {}
        for start_node in self.all_tiles:
            self.shortest_paths[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all other nodes."""
        distances = {node: float('inf') for node in self.all_tiles}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in self.adj (should be if from self.all_tiles)
            if current_node in self.adj:
                for neighbor in self.adj[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Extract current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_heuristic = 0

        # Iterate through goal tiles and their required colors
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the goal for this tile is already satisfied
            goal_satisfied = f"(painted {goal_tile} {goal_color})" in state

            if not goal_satisfied:
                # Check for dead end: is the tile painted with the wrong color?
                is_clear = f"(clear {goal_tile})" in state
                if not is_clear:
                    # Tile is not clear, check if it's painted with *any* color
                    painted_with_wrong_color = False
                    for fact in state:
                        parts = get_parts(fact)
                        if parts and parts[0] == "painted" and parts[1] == goal_tile and parts[2] != goal_color:
                             painted_with_wrong_color = True
                             break
                    if painted_with_wrong_color:
                         # Tile is painted with the wrong color, this is a dead end
                         return float('inf')

                # If not a dead end and goal not satisfied, the tile needs painting
                # Add cost for the paint action itself
                cost_for_this_tile = 1

                # Find potential robot locations adjacent to the goal tile
                # These are the tiles tile_R that are directly adjacent to goal_tile
                required_robot_locs = self.adj.get(goal_tile, set())

                # If the goal tile has no neighbors in the grid, it's likely an invalid problem setup
                # or unreachable. Treat as unsolvable from this state.
                if not required_robot_locs:
                     return float('inf')


                min_robot_cost_for_tile = float('inf')

                # Find the minimum cost for any robot to get into position and have the right color
                for robot, robot_loc in robot_locations.items():
                    robot_color = robot_colors.get(robot) # Get robot's current color

                    # Cost to get the correct color
                    # Assumes robot always has a color if not free-color (which isn't used)
                    color_cost = 1 if robot_color != goal_color else 0

                    # Find minimum distance from robot's current location to any required painting location
                    min_dist_to_required_loc = float('inf')

                    # Ensure robot_loc is a valid tile in our graph
                    if robot_loc in self.shortest_paths:
                        for required_loc in required_robot_locs:
                            # Ensure required_loc is a valid tile in our graph
                            if required_loc in self.shortest_paths[robot_loc]:
                                dist = self.shortest_paths[robot_loc][required_loc]
                                min_dist_to_required_loc = min(min_dist_to_required_loc, dist)

                    # If the robot cannot reach any required location, its cost is infinity
                    if min_dist_to_required_loc == float('inf'):
                         robot_cost = float('inf')
                    else:
                         robot_cost = min_dist_to_required_loc + color_cost

                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)

                # If no robot can reach a painting position for this tile, the problem is unsolvable from this state
                if min_robot_cost_for_tile == float('inf'):
                     return float('inf')

                # Add the minimum robot cost to the tile cost (1 for paint action)
                cost_for_this_tile += min_robot_cost_for_tile
                total_heuristic += cost_for_this_tile

        return total_heuristic
