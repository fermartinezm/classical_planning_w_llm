from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle potentially different lengths gracefully
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_node, target_nodes, graph, state):
    """
    Find the shortest path distance from start_node to any node in target_nodes
    on the grid graph, considering only 'clear' tiles as traversable destinations.
    Returns infinity if no path exists.
    """
    q = deque([(start_node, 0)])
    visited = {start_node}

    # Handle the case where start_node is already a target node
    if start_node in target_nodes:
        return 0

    while q:
        current_node, dist = q.popleft()

        # Check neighbors
        for neighbor in graph.get(current_node, []):
            # Robot can only move to a neighbor if the neighbor is clear
            # and has not been visited in this BFS search.
            if f"(clear {neighbor})" in state and neighbor not in visited:
                if neighbor in target_nodes:
                    return dist + 1 # Found a target node, return distance + 1 (cost of the move)
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

    return float('inf') # Use float('inf') for infinity

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It sums the estimated minimum cost for each
    unpainted goal tile independently. The cost for a single tile includes:
    1. Cost to make the tile clear if it's not (only if a robot is on it).
    2. Cost to get a robot with the required color to the correct adjacent location from which to paint the tile.
    3. The paint action itself.

    # Assumptions
    - Tiles needing painting are initially clear or become clear only when a robot moves off them.
    - Tiles painted with the wrong color represent an unsolvable state for that tile (heuristic returns infinity).
    - Robots can only paint tiles directly above or below them.
    - Movement is restricted to adjacent tiles connected by up/down/left/right relations, and only to tiles that are currently clear.
    - The cost of each action (move, change_color, paint) is 1.

    # Heuristic Initialization
    - Extract the goal painting requirements (tile and color).
    - Build the grid graph (adjacency list) from the static up/down/left/right facts.
    - Build a map from a tile to the locations from which a robot can paint it (based on up/down facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are not currently painted with the correct color in the state.
    2. If all goal tiles are painted correctly, the heuristic is 0.
    3. For each unpainted goal tile (T, C):
        a. Initialize the base cost for this tile to 1 (for the paint action).
        b. Check if tile T is clear in the current state. If not:
           - If a robot is currently at T, add 1 to the base cost (for the robot to move off).
           - If T is painted with the wrong color (checked by verifying it's not clear, not painted correctly, and no robot is on it), the state is unsolvable for this tile; return infinity.
           - If T is not clear for other reasons (not expected in this domain), this heuristic might be inaccurate; return infinity as a safeguard.
        c. Find the set of possible locations P from which a robot can paint tile T (i.e., (up T P) or (down T P) is true). If P is empty, the tile cannot be painted; return infinity.
        d. Initialize the minimum additional cost (color change + movement) for a robot to infinity.
        e. For each robot R:
            i. Find the robot's current location R_loc and color R_color.
            ii. Calculate the color change cost: 1 if R_color is not C, 0 otherwise.
            iii. Calculate the minimum movement cost from R_loc to any location in P using BFS on the grid graph, considering only clear tiles as traversable destinations.
            iv. If a path exists (distance is not infinity), calculate the additional cost for this robot: (color change cost) + (movement cost).
            v. Update the minimum additional cost for the tile with the minimum found among all robots.
        f. If the minimum additional cost is still infinity after checking all robots, the tile is unreachable; return infinity.
        g. Add the base cost (including clear_cost) and the minimum additional cost to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal paintings: map tile -> required color
        self.goal_paintings = {}
        for goal in self.goals:
            # Goal is typically (painted tile_X_Y color)
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings[tile] = color

        # Build grid graph (adjacency list)
        self.grid_adj = {}
        # Build paint_from_map: map tile_to_paint -> set of tiles_robot_must_be_at
        self.paint_from_map = {}

        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate in ["up", "down", "left", "right"]:
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # Robot can move from tile_0_1 to tile_1_1 using move_up
                # Robot can move from tile_1_1 to tile_0_1 using move_down
                # The relation is symmetric for movement
                loc1, loc2 = parts[1], parts[2]
                self.grid_adj.setdefault(loc1, []).append(loc2)
                self.grid_adj.setdefault(loc2, []).append(loc1)

            if predicate in ["up", "down"]:
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # Robot at tile_0_1 can paint tile_1_1 (using paint_up)
                # e.g., (down tile_0_1 tile_1_1) means tile_0_1 is down from tile_1_1
                # Robot at tile_1_1 can paint tile_0_1 (using paint_down)
                tile_to_paint, tile_robot_at = parts[1], parts[2]
                self.paint_from_map.setdefault(tile_to_paint, set()).add(tile_robot_at)

        # Remove duplicate neighbors from grid_adj (due to symmetric relations)
        for tile in self.grid_adj:
             self.grid_adj[tile] = list(set(self.grid_adj[tile]))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goals = {} # tile -> required_color
        for tile, required_color in self.goal_paintings.items():
            if f"(painted {tile} {required_color})" not in state:
                unpainted_goals[tile] = required_color

        # 2. If all goal tiles are painted correctly, the heuristic is 0.
        if not unpainted_goals:
            return 0

        # Get robot info
        robot_info = {} # robot_name -> (location, color)
        robots = set()
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, loc = get_parts(fact)
                robots.add(robot)
                if robot not in robot_info:
                    robot_info[robot] = [None, None] # [location, color]
                robot_info[robot][0] = loc
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robots.add(robot)
                if robot not in robot_info:
                    robot_info[robot] = [None, None] # [location, color]
                robot_info[robot][1] = color

        total_heuristic = 0

        # 3. For each unpainted goal tile (T, C)
        for tile_to_paint, required_color in unpainted_goals.items():
            # a. Initialize the base cost for this tile to 1 (for the paint action).
            tile_base_cost = 1

            # b. Check if tile T is clear. Add cost if robot is on it.
            is_clear = f"(clear {tile_to_paint})" in state
            if not is_clear:
                 robot_on_tile = None
                 for robot, (loc, color) in robot_info.items():
                     if loc == tile_to_paint:
                         robot_on_tile = robot
                         break

                 if robot_on_tile:
                     tile_base_cost += 1 # Cost to move the robot off
                 else:
                     # Not clear, not painted correctly (otherwise wouldn't be in unpainted_goals),
                     # and no robot on it. It must be painted with a different color.
                     # This state is likely unsolvable for this tile.
                     # Check explicitly to be safe, though the logic implies it.
                     is_painted_wrongly = False
                     for fact in state:
                         if match(fact, "painted", tile_to_paint, "*"):
                             _, painted_tile, painted_color = get_parts(fact)
                             if painted_color != required_color:
                                 is_painted_wrongly = True
                                 break
                     if is_painted_wrongly:
                         # Unsolvable tile
                         return float('inf')
                     else:
                         # This case shouldn't happen in valid states based on domain rules.
                         # It implies the tile is not clear but also not painted and not occupied.
                         # Assume valid states and return inf as a safeguard.
                         return float('inf')


            # c. Find paintable locations P for tile T
            paint_from_locs = self.paint_from_map.get(tile_to_paint, set())

            if not paint_from_locs:
                 # This tile cannot be painted according to static facts.
                 # Problem is likely unsolvable. Return infinity.
                 return float('inf')

            # d. Initialize the minimum additional cost (color change + movement) for a robot to infinity.
            min_robot_action_cost = float('inf')

            # e. For each robot R
            for robot, (robot_loc, robot_color) in robot_info.items():
                # ii. Calculate color change cost
                color_change_cost = 1 if robot_color != required_color else 0

                # iii. Calculate minimum movement cost to a paintable location
                # BFS finds distance from robot_loc to a *clear* paint_loc
                min_move_cost_to_paint_loc = bfs(robot_loc, paint_from_locs, self.grid_adj, state)

                # iv. Calculate additional cost for this robot
                if min_move_cost_to_paint_loc != float('inf'):
                    robot_action_cost = color_change_cost + min_move_cost_to_paint_loc
                    # v. Update the minimum additional cost for the tile
                    min_robot_action_cost = min(min_robot_action_cost, robot_action_cost)

            # f. If the minimum additional cost is still infinity after checking all robots,
            # it means no robot can reach a paintable location for this tile.
            # This state is likely unsolvable.
            if min_robot_action_cost == float('inf'):
                 return float('inf') # Return infinity if any goal tile is unreachable

            # g. Add the base cost (including clear_cost) and the minimum additional cost
            # to the total heuristic value.
            total_heuristic += tile_base_cost + min_robot_action_cost

        # 4. Return the total heuristic value.
        return total_heuristic
