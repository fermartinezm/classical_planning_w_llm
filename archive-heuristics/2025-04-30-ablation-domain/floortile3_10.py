from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in the environment or imported elsewhere
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (copied from Logistics example)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path on the tile grid
def bfs_distance(start_tile, end_tile, graph):
    """
    Computes the shortest path distance between two tiles in the grid graph.
    Returns float('inf') if the end_tile is unreachable.
    """
    if start_tile == end_tile:
        return 0

    queue = deque([(start_tile, 0)]) # Use deque for efficiency
    visited = {start_tile}

    while queue:
        current_tile, dist = queue.popleft() # Use popleft() for deque

        if current_tile == end_tile:
            return dist

        if current_tile in graph:
            for neighbor in graph[current_tile]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Should not happen in connected grids for solvable problems

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums, for each unpainted goal tile, the minimum
    distance from any robot with the required color to that tile, plus one action
    for painting.

    # Assumptions
    - Robots are permanently assigned a color (they cannot pick up or change colors).
    - Tiles that need painting are assumed to be clear or become clear implicitly
      when painted correctly (i.e., no explicit 'clear' action is needed for
      wrongly painted tiles).
    - The grid of tiles is connected.
    - All necessary colors are available via at least one robot.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Builds the tile grid graph from static connectivity facts (up, down, left, right).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted tile color)`.
    2. For each such goal condition `(painted T C)`:
        a. Check if the state already contains `(painted T C)`. If yes, this goal is satisfied for this tile, and we move to the next goal tile.
        b. If the goal `(painted T C)` is not satisfied, find all robots `R` in the current state that have the color `C` (i.e., `(robot-has R C)` is true).
        c. If no robot has the required color `C`, the problem might be unsolvable (or the instance is malformed for this heuristic's assumptions). Return a large value (infinity).
        d. For each robot `R` found in step 2b, determine its current location `L_R` from the state (i.e., `(robot-at R L_R)` is true).
        e. Calculate the shortest path distance from each robot location `L_R` to the target tile `T` using BFS on the pre-built tile grid graph.
        f. Find the minimum distance among all robots capable of painting tile `T`: `min_dist = min_{R | (robot-has R C)} distance(L_R, T)`.
        g. Add `min_dist + 1` (for the movement and the paint action) to the total heuristic value.
    3. The total heuristic value is the sum accumulated in step 2g over all unsatisfied goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        # Call the base class constructor to set self.goals and self.static
        super().__init__(task)

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are expected to be in the form '(painted tile color)'
            if len(parts) == 3 and parts[0] == "painted":
                 tile, color = parts[1], parts[2]
                 self.goal_tiles[tile] = color

        # Build the tile grid graph from static connectivity facts
        self.tile_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            # Connectivity predicates are up, down, left, right
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                self.tile_graph.setdefault(tile1, []).append(tile2)
                self.tile_graph.setdefault(tile2, []).append(tile1) # Grid is undirected

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all goal tiles.
        """
        state = node.state

        # Extract current robot locations and colors, and painted tiles
        robot_locations = {}
        robot_colors = {}
        current_painted = {}

        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                if parts[0] == "robot-at":
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
                elif parts[0] == "robot-has":
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                elif parts[0] == "painted":
                    tile, color = parts[1], parts[2]
                    current_painted[tile] = color

        total_cost = 0

        # Iterate through each goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly
            if goal_tile in current_painted and current_painted[goal_tile] == goal_color:
                continue # This goal is satisfied

            # Find robots that have the required color
            robots_with_color = [
                robot for robot, color in robot_colors.items() if color == goal_color
            ]

            # If no robot has the required color, this goal is likely unreachable.
            # Return infinity.
            if not robots_with_color:
                return float('inf')

            # Find the minimum distance from any robot with the correct color
            # to the goal tile.
            min_dist_to_tile = float('inf')
            for robot in robots_with_color:
                robot_loc = robot_locations.get(robot) # Get robot's current location
                if robot_loc: # Robot must have a location
                    # Ensure the robot's location and the goal tile are valid nodes in the graph
                    if robot_loc in self.tile_graph and goal_tile in self.tile_graph:
                         dist = bfs_distance(robot_loc, goal_tile, self.tile_graph)
                         min_dist_to_tile = min(min_dist_to_tile, dist)
                    else:
                         # This indicates an inconsistency (e.g., robot at a non-existent tile)
                         # Treat as unreachable.
                         return float('inf')
                else:
                    # Robot has no location fact? Should not happen in valid states.
                    return float('inf')


            # If the tile is unreachable from any robot with the correct color, return infinity.
            if min_dist_to_tile == float('inf'):
                 return float('inf')

            # Add the cost for this tile: minimum travel distance + 1 (for paint action)
            total_cost += min_dist_to_tile + 1

        return total_cost
