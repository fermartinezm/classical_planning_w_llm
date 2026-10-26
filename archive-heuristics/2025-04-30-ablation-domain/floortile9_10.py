# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# If running standalone for testing, define a dummy class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, adj):
    """
    Performs BFS starting from start_node on the given adjacency list.
    Returns a dictionary of distances from start_node to all reachable nodes.
    """
    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node exists in adj to avoid KeyError for isolated nodes
        if current_node in adj:
            for neighbor in adj[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    by summing the estimated costs for each unpainted goal tile. The cost for a
    single tile includes the minimum movement cost for any robot to reach it,
    the cost to pick up the required color if no robot currently has it, and the
    paint action itself.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring the `clear` predicate for other robots).
    - Picking up a color replaces the robot's current color.
    - The cost of picking up a color is incurred once per color if no robot currently holds it and it's needed for any unpainted goal tile.
    - The cost for painting a tile is 1 action.
    - The cost for moving between adjacent tiles is 1 action.
    - The grid is connected for all relevant tiles.

    # Heuristic Initialization
    - Extracts goal conditions to identify target tiles and colors.
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency list representing the grid connectivity. This is used for calculating shortest path distances between tiles using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of goal tiles that are not yet painted with the correct color in the current state (`U`).
    2. If `U` is empty, the state is a goal state (or satisfies all painted goals), return 0.
    3. Identify the current location and color of each robot in the state.
    4. Determine the set of colors required by the unpainted goal tiles (`colors_needed`).
    5. Determine the set of colors currently held by robots (`colors_held`).
    6. Calculate the set of colors that are needed but not held (`colors_to_pick`). The number of actions to pick up these colors is estimated as the size of this set. Add this to the total heuristic cost.
    7. For each robot, compute the shortest distance from its current location to all other tiles using BFS on the precomputed grid adjacency list. Store these distances.
    8. For each unpainted goal tile `(T, C)` in `U`:
       - Find the minimum distance from any robot's current location to tile `T`.
       - Add this minimum distance to the total heuristic cost (representing the movement cost).
       - Add 1 to the total heuristic cost (representing the `paint` action).
    9. The total accumulated cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal tiles and their required colors
        self.goal_tiles = set()
        for goal in self.goals:
            # Goal is typically (painted tile color)
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles.add((tile, color))

        # Build adjacency list for the grid graph from static facts
        self.adj = {}
        all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                t1, t2 = parts[1], parts[2]
                all_tiles.add(t1)
                all_tiles.add(t2)
                # Add bidirectional edges
                self.adj.setdefault(t1, []).append(t2)
                self.adj.setdefault(t2, []).append(t1)

        # Ensure all tiles mentioned in goals or static facts are in adj keys,
        # even if they have no neighbors listed (unlikely in grid problems)
        for tile in all_tiles:
             self.adj.setdefault(tile, [])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        current_painted = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "painted":
                 current_painted.add((parts[1], parts[2]))

        for goal_tile, goal_color in self.goal_tiles:
            if (goal_tile, goal_color) not in current_painted:
                 unpainted_goal_tiles.add((goal_tile, goal_color))

        # 2. If U is empty, return 0.
        if not unpainted_goal_tiles:
            return 0

        # 3. Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif len(parts) == 3 and parts[0] == "robot-has":
                 robot, color = parts[1], parts[2]
                 robot_colors[robot] = color

        robot_info = {}
        for robot in robot_locations:
            if robot in robot_colors: # Should always be true in valid states
                robot_info[robot] = (robot_locations[robot], robot_colors[robot])

        # 4. Determine colors needed and 5. colors held
        colors_needed = {color for tile, color in unpainted_goal_tiles}
        colors_held = {color for loc, color in robot_info.values()}

        # 6. Calculate colors to pick cost
        colors_to_pick = colors_needed - colors_held
        total_cost = len(colors_to_pick) # Cost for picking up colors

        # 7. Calculate distances from each robot location
        robot_distances = {}
        for robot, (location, color) in robot_info.items():
             # Ensure robot location is a valid tile in the graph before BFS
             if location in self.adj:
                robot_distances[robot] = bfs(location, self.adj)
             # else: Robot is at an unknown location, cannot compute distances from it.
             #      This robot won't be considered for reaching tiles.

        # 8. Calculate sum of minimum distances and paint costs for unpainted tiles
        tiles_cost = 0
        for tile, color in unpainted_goal_tiles:
            min_dist_to_tile = float('inf')
            reachable_from_any_robot = False
            for robot, (location, robot_color) in robot_info.items():
                # Check if robot's distances were computed (i.e., robot is on the grid)
                # and if the target tile is reachable from this robot
                if robot in robot_distances and tile in robot_distances[robot]:
                    min_dist_to_tile = min(min_dist_to_tile, robot_distances[robot][tile])
                    reachable_from_any_robot = True

            if not reachable_from_any_robot:
                 # Tile is unreachable from any robot on the grid - problem is likely unsolvable
                 return float('inf')

            tiles_cost += min_dist_to_tile + 1 # Move cost + Paint cost

        # 9. Total heuristic
        total_cost += tiles_cost

        return total_cost
