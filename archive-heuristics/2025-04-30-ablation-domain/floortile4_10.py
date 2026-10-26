from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed fact string gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest path on the tile grid
def bfs(start_node, graph):
    """
    Performs Breadth-First Search starting from start_node on the given graph.
    Returns a dictionary of shortest distances from start_node to all reachable nodes.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in distances: # Ensure start_node is in the graph
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current = queue.popleft()

            if current not in graph: # Should not happen if graph is built correctly from all_tiles
                 continue

            for neighbor in graph.get(current, []): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

    return distances

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions (clear, paint, move) required
    to satisfy the goal conditions for each tile. It sums the estimated costs
    for each tile that is not yet painted with its goal color. The cost for a
    tile includes the actions needed at the tile (clear if necessary, paint)
    plus the estimated minimum movement cost for a robot with the correct color
    to reach that tile.

    # Assumptions
    - Tiles are arranged in a grid connected by up/down/left/right relations.
    - Movement between adjacent tiles costs 1 action.
    - Painting a clear tile costs 1 action and requires a robot with the target color at the tile.
    - Clearing a wrongly painted tile costs 1 action and requires any robot at the tile.
    - A wrongly painted tile must be cleared before it can be painted correctly.
    - Robots have colors, and these colors might change (heuristic uses current robot color).
    - To paint a tile with color C, a robot *currently* holding color C must be used.
    - The heuristic estimates movement cost as the shortest path distance on the tile grid.
    - The heuristic assumes that for any required goal color, there is at least one robot that can eventually acquire/has that color (otherwise the problem might be unsolvable, and the heuristic returns infinity).

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and what color.
    - Parses static facts to build the adjacency graph of tiles based on spatial relations (up, down, left, right).
    - Computes and stores the shortest path distances between all pairs of tiles using BFS on the graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot from the state facts.
    2. Compute the shortest path distance from each robot's current location to every other tile using the precomputed distances. (Note: This step is implicitly handled by looking up distances from the robot's current tile in the precomputed all-pairs distances).
    3. Initialize the total heuristic cost `h` to 0.
    4. Iterate through each tile `T` and its required goal color `C` as specified in the task goals.
    5. Check if tile `T` is already painted with color `C` in the current state. If yes, this goal is met for this tile; continue to the next goal tile.
    6. If the goal `(painted T C)` is not met:
        a. Determine the current status of tile `T`: Is it `(clear T)` or `(painted T C')` for some `C' != C`?
        b. Find all robots that currently possess the color `C`.
        c. If no robot currently possesses color `C`, the state might be unsolvable (under the assumption that painting requires a robot with the color). Return infinity.
        d. Calculate the minimum distance from any robot possessing color `C` to tile `T` using the precomputed distances. Let this be `min_dist_C`.
        e. If tile `T` is `(clear T)`: The estimated cost for this tile is `min_dist_C` (movement) + 1 (paint action).
        f. If tile `T` is `(painted T C')` where `C' != C`: The estimated cost for this tile is `min_dist_C` (movement for painting) + 1 (clear action) + 1 (paint action). This simplifies the movement cost by assuming the robot that paints also handles clearing or that movement for clearing is covered.
        g. Add the estimated cost for tile `T` to the total heuristic cost `h`.
    7. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, tile grid structure,
        and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and colors for each tile that needs painting.
        self.goal_tiles = {}
        # Assuming goals are always conjunctions of (painted tile color)
        # If task.goals is a single fact or a complex structure, this might need adjustment.
        # Based on example, task.goals is a set of facts.
        for goal_fact in self.goals:
             if match(goal_fact, "painted", "*", "*"):
                parts = get_parts(goal_fact)
                if len(parts) == 3: # (painted tile color)
                    tile, color = parts[1], parts[2]
                    self.goal_tiles[tile] = color
             # Add handling for other potential goal types if necessary, though problem implies only painted goals.


        # Collect all unique tile names from initial state and static facts
        all_objects = set()
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if not parts: continue
             # Collect all arguments of relevant predicates as potential objects
             if parts[0] in {'robot-at', 'clear', 'painted', 'up', 'down', 'left', 'right', 'robot-has', 'available-color'}:
                 for obj in parts[1:]:
                     all_objects.add(obj)

        # Filter objects to identify tiles. Assuming tiles are named like 'tile_R_C'
        # A more robust way would be to parse the :objects section if available,
        # but relying on naming convention or appearance in tile-specific predicates is common.
        # Let's refine based on predicates that *only* take tiles or relate tiles.
        tile_candidates = set()
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] in {'clear', 'painted'}: # These predicates apply directly to tiles
                 if len(parts) > 1: tile_candidates.add(parts[1])
             elif parts[0] in {'up', 'down', 'left', 'right'}: # These predicates relate two tiles
                 if len(parts) > 2:
                     tile_candidates.add(parts[1])
                     tile_candidates.add(parts[2])
             elif parts[0] == 'robot-at': # Robot is at a tile
                 if len(parts) > 2: tile_candidates.add(parts[2]) # The second argument is the tile

        self.all_tiles = list(tile_candidates) # Store as list

        # Build adjacency graph from static facts
        self.graph = {tile: [] for tile in self.all_tiles}
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] in {'up', 'down', 'left', 'right'}:
                if len(parts) == 3:
                    t1, t2 = parts[1], parts[2]
                    # Ensure both tiles are in our collected list before adding edge
                    if t1 in self.graph and t2 in self.graph:
                        self.graph[t1].append(t2)
                        self.graph[t2].append(t1) # Assuming symmetric relations

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_tiles:
            self.distances[start_node] = bfs(start_node, self.graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract robot locations and colors from the current state
        robot_locs = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locs[robot] = tile
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_cost = 0  # Initialize action cost counter.

        for tile, goal_color in self.goal_tiles.items():
            # Check if the goal for this tile is already met
            if f"(painted {tile} {goal_color})" in state:
                continue

            # Find current status of the tile
            is_clear = f"(clear {tile})" in state
            is_wrongly_painted = False
            current_painted_fact = None
            for fact in state:
                if match(fact, "painted", tile, "*"):
                    current_painted_fact = fact
                    # Check if it's painted with a different color
                    if get_parts(fact)[2] != goal_color:
                         is_wrongly_painted = True
                    break # A tile should only be painted one color

            # Find robots that have the required color
            robots_with_goal_color = [
                robot for robot, color in robot_colors.items() if color == goal_color
            ]

            # If no robot has the required color, this goal is unreachable by painting
            if not robots_with_goal_color:
                 # Return infinity as a conservative estimate for unsolvable states.
                 return float('inf')

            # Calculate minimum distance from a robot with the goal color to the tile
            min_dist_C = float('inf')
            for robot in robots_with_goal_color:
                if robot in robot_locs: # Ensure robot location is known
                    robot_current_tile = robot_locs[robot]
                    # Look up distance using precomputed values
                    if robot_current_tile in self.distances and tile in self.distances[robot_current_tile]:
                         min_dist_C = min(min_dist_C, self.distances[robot_current_tile][tile])
                    # else: # This case indicates an issue (e.g., robot at unknown location or tile not in graph)
                    #      print(f"Warning: Distance lookup failed for robot {robot} at {robot_current_tile} to tile {tile}")


            # If min_dist_C is still infinity, it means no robot with the color can reach the tile
            # (e.g., disconnected graph, or robot location unknown/not in graph)
            if min_dist_C == float('inf'):
                 return float('inf') # Unreachable tile for painting

            # Estimate cost based on tile status
            if is_clear:
                # Needs 1 paint action + movement for a robot with color C
                total_cost += min_dist_C + 1
            elif is_wrongly_painted:
                # Needs 1 clear action + 1 paint action + movement for a robot with color C
                # We simplify movement cost to just the trip for the painting robot.
                total_cost += min_dist_C + 2
            # else: # Tile is neither clear nor painted, or already painted correctly (handled by continue)
                # This case should ideally not be reached for unpainted goal tiles in valid states.
                # If it happens, it might indicate a state where the tile is painted the *correct* color
                # but the initial check failed, or an invalid state representation.
                # The initial check `if f"(painted {tile} {goal_color})" in state:` handles the correct color case.
                # Any other state (neither clear nor painted wrong) is unexpected.

        # Heuristic is 0 if all goal tiles are painted correctly (total_cost remains 0)
        return total_cost

