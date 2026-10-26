from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to parse the current state
def parse_state(state_frozenset):
    """
    Parses the state frozenset to extract relevant information.
    Returns robot locations, robot colors, currently painted tiles, and clear tiles.
    """
    robot_locations = {}
    robot_colors = {}
    current_painted = set()
    current_clear = set()

    for fact in state_frozenset:
        parts = get_parts(fact)
        if not parts: continue

        predicate = parts[0]
        if predicate == 'robot-at':
            robot, tile = parts[1], parts[2]
            robot_locations[robot] = tile
        elif predicate == 'robot-has':
            robot, color = parts[1], parts[2]
            robot_colors[robot] = color
        elif predicate == 'painted':
            tile, color = parts[1], parts[2]
            current_painted.add((tile, color))
        elif predicate == 'clear':
            tile = parts[1]
            current_clear.add(tile)

    return robot_locations, robot_colors, current_painted, current_clear

# Helper function to parse static facts
def parse_static(static_frozenset):
    """
    Parses static facts to build an adjacency map for the grid.
    Returns a dictionary mapping each tile to a list of its adjacent tiles.
    """
    adjacency = {}
    # Build bidirectional adjacency list from up/down/left/right facts
    for fact in static_frozenset:
        parts = get_parts(fact)
        if not parts: continue

        predicate = parts[0]
        if predicate in ['up', 'down', 'left', 'right']:
            # (up tile_y tile_x) means tile_y is up from tile_x.
            # This implies tile_y is adjacent to tile_x, and tile_x is adjacent to tile_y.
            tile1, tile2 = parts[1], parts[2]
            adjacency.setdefault(tile1, []).append(tile2)
            adjacency.setdefault(tile2, []).append(tile1)

    # Remove duplicates from adjacency lists
    for tile in adjacency:
        adjacency[tile] = list(set(adjacency[tile]))

    return adjacency

# Helper function for BFS movement distance
def bfs_movement_distance(start_tile, target_adj_tiles, current_clear, adjacency_info):
    """
    Performs BFS from start_tile to find the minimum distance (number of moves)
    to any tile in target_adj_tiles that is currently clear.
    Movement is only possible to clear tiles.
    Returns the minimum distance, or float('inf') if no such tile is reachable.
    """
    distances = {start_tile: 0}
    queue = deque([start_tile])
    min_dist = float('inf')

    # BFS explores the graph where nodes are tiles and edges exist from A to B
    # if B is adjacent to A and B is currently clear.
    # The start node is the robot's current location (which is NOT clear).
    # The BFS finds the shortest path from the start_tile to any reachable tile.
    # The distance is the number of moves.

    while queue:
        current_tile = queue.popleft()
        current_dist = distances[current_tile]

        # If the current tile is one of the target adjacent tiles AND is clear, update min_dist
        # This check is done *after* moving to current_tile.
        if current_tile in target_adj_tiles and current_tile in current_clear:
             min_dist = min(min_dist, current_dist)
             # Continue BFS to potentially find shorter paths to *other* target_adj_tiles

        neighbors = adjacency_info.get(current_tile, [])
        for next_tile in neighbors:
            # Can move to next_tile if it is clear AND we haven't found a shorter path to it yet
            # The `distances` dict serves as the visited set for shortest paths in BFS
            if next_tile in current_clear and next_tile not in distances:
                distances[next_tile] = current_dist + 1
                queue.append(next_tile)

    # After BFS, `distances` contains the shortest path from `start_tile` to
    # any reachable clear tile. We need the minimum distance among the target_adj_tiles
    # that were reachable and clear.
    # The `min_dist` variable tracked this during the BFS.

    return min_dist


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles that are not yet painted correctly. It sums the minimum estimated
    cost for each individual unpainted goal tile, considering the costs of
    changing color, moving a robot to an adjacent clear tile, and painting.

    # Assumptions
    - Tiles that need to be painted are initially clear or become clear if
      they were painted incorrectly (although the domain doesn't support
      unpainting/repainting, so we assume goal tiles are initially clear
      if not already painted correctly).
    - Robots can only move onto tiles that are currently clear.
    - The cost of moving between adjacent clear tiles is 1.
    - The cost of changing color is 1.
    - The cost of painting is 1.
    - The heuristic assumes that each unpainted goal tile can be painted
      independently by the "best" robot for that tile, ignoring potential
      conflicts or synergies when a single robot paints multiple tiles.
      This is a relaxation.

    # Heuristic Initialization
    - Parses the goal conditions to identify the set of tiles that need to be
      painted and their target colors.
    - Parses the static facts to build an adjacency map representing the grid
      structure, allowing for movement path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`. Store these
       as the set of target paintings.
    2. In the `__call__` method for a given state:
       a. Parse the current state to determine:
          - The current location of each robot.
          - The current color held by each robot.
          - The set of tiles that are currently painted (and their colors).
          - The set of tiles that are currently clear.
       b. Identify the set of "unpainted goal tiles": these are tiles specified
          in the goal that are not currently painted with the correct color.
       c. If there are no unpainted goal tiles, the heuristic value is 0 (goal state).
       d. Initialize the total heuristic value `h` to 0.
       e. For each unpainted goal tile `(tile_to_paint, goal_color)`:
          i. Calculate the minimum cost required for *any* robot to paint this specific tile. Initialize `min_cost_for_tile` to infinity.
          ii. Get the set of tiles adjacent to `tile_to_paint` using the pre-computed adjacency map. These are the potential locations from which a robot can paint `tile_to_paint`.
          iii. For each robot `R`:
              - Get the robot's current location `L_R`.
              - Get the robot's current color `C_R`.
              - Calculate the `color_cost`: 1 if `C_R` is different from `goal_color` else 0.
              - Calculate the `move_cost`: This is the minimum number of moves required for robot `R` to get from `L_R` to *any* tile `L_Adj` adjacent to `tile_to_paint`, provided `L_Adj` is currently `clear`. This is computed using a BFS on the grid where movement is restricted to clear tiles.
              - If a reachable clear adjacent tile exists (`move_cost` is not infinity):
                  - The total cost for this robot to paint this tile is `color_cost + move_cost + 1` (where 1 is the cost of the paint action itself).
                  - Update `min_cost_for_tile = min(min_cost_for_tile, robot_cost_for_tile)`.
          iv. If `min_cost_for_tile` is still infinity after checking all robots, it means this goal tile is unreachable in the current state. Return infinity for the total heuristic.
          v. Add `min_cost_for_tile` to the total heuristic `h`.
       f. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal painted facts
        self.goal_painted = set()
        # task.goals is a set of facts that must be true.
        for fact_str in self.goals:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted.add((tile, color))
        # Ignore other goal types if any (like robot-at specific location)

        # Build adjacency map from static facts
        self.adjacency = parse_static(self.static)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state.
        """
        state = node.state

        # Parse current state
        robot_locations, robot_colors, current_painted, current_clear = parse_state(state)

        # Identify unpainted goal tiles
        unpainted_goals = {
            (t, c) for (t, c) in self.goal_painted
            if (t, c) not in current_painted
        }

        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goals:
            return 0

        total_h = 0

        # Calculate minimum cost for each unpainted goal tile independently
        for tile_to_paint, goal_color in unpainted_goals:
            min_cost_for_tile = float('inf')

            # Get potential painting locations (adjacent tiles)
            adj_tiles = self.adjacency.get(tile_to_paint, [])

            # If no adjacent tiles found (malformed problem?), this tile is unreachable
            if not adj_tiles:
                 # This indicates a problem with the domain/instance definition if a goal tile has no adjacent tiles
                 # In a real scenario, this might mean the state is unsolvable.
                 return float('inf')

            # Consider each robot as a potential painter for this tile
            if not robot_locations: # No robots in the state
                 return float('inf') # Cannot paint anything

            for robot in robot_locations:
                L_R = robot_locations[robot]
                C_R = robot_colors.get(robot) # Use .get() in case robot_has fact is missing (shouldn't happen)

                # Cost to get the right color
                # Assuming robot always has a color based on domain
                color_cost = 1 if C_R != goal_color else 0

                # Cost to move from robot's current location to a clear tile adjacent to the target tile
                # BFS finds shortest path from L_R to any tile in adj_tiles that is clear
                move_cost = bfs_movement_distance(L_R, adj_tiles, current_clear, self.adjacency)

                # If a reachable clear adjacent tile exists
                if move_cost != float('inf'):
                    # Total cost for this robot to paint this tile = color change + movement + paint action
                    robot_cost_for_tile = color_cost + move_cost + 1
                    min_cost_for_tile = min(min_cost_for_tile, robot_cost_for_tile)

            # If no robot can reach a clear adjacent tile for this goal tile, it's unreachable
            if min_cost_for_tile == float('inf'):
                return float('inf') # State is likely unsolvable

            total_h += min_cost_for_tile

        return total_h
