from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For float('inf')

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity mismatch unless using wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if parts match args, allowing wildcards
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the minimum number of actions required to paint all
    goal tiles with their target colors. It considers the cost of painting,
    changing robot color, and moving robots through clear tiles to reach
    positions adjacent to the target tiles.

    # Assumptions
    - Goal tiles are either initially clear or already correctly painted. If a goal
      tile is not painted correctly and is not clear, the problem is considered
      unsolvable from that state (heuristic returns infinity).
    - Robots can only move onto clear tiles.
    - Robots are always located on non-clear tiles after a move action.
    - Painting a tile requires the robot to be at a specific adjacent tile
      depending on the paint direction (e.g., below for paint_up).
    - The heuristic sums the minimum cost for each unpainted goal tile independently,
      minimizing over available robots. This ignores potential synergies (one move
      helping multiple tiles, one color change helping multiple tiles) and is thus non-admissible.

    # Heuristic Initialization
    - Extracts the set of goal painted tiles and their target colors.
    - Builds the full tile adjacency graph from static facts (up, down, left, right).
    - Builds a mapping from each tile to the set of tiles where a robot must be
      located to paint it (required adjacent tiles).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not yet painted with the correct color.
       Let U be the set of (tile, color) pairs for these unsatisfied goal conditions.
    2. Check if any tile in U is not currently `clear`. If so, return infinity (unsolvable).
    3. If U is empty, the heuristic is 0 (goal state reached).
    4. Identify the current location and color of each robot.
    5. Identify the set of tiles that are currently `clear`.
    6. Build a movement graph where nodes are all tiles, and a directed edge exists
       from tile A to tile B if A and B are adjacent in the full grid and tile B is `clear`.
       This represents possible moves for a robot (from A onto clear tile B).
    7. For each robot, perform a Breadth-First Search (BFS) starting from its current
       location on the movement graph built in step 6. This computes the shortest
       distance (number of moves) from the robot's current location to all reachable tiles
       via paths consisting of moves onto clear tiles.
    8. Initialize the total heuristic value `h` to 0.
    9. For each unsatisfied goal tile `(T, C)` in U:
        a. Add 1 to `h` for the paint action itself.
        b. Determine the set of required adjacent tiles `ReqAdj(T)` where a robot must
           be positioned to paint tile `T`.
        c. Calculate the minimum cost for *any* robot to reach *any* tile in `ReqAdj(T)`
           with color `C`. Let this be `min_robot_cost_for_tile`.
           For each robot R:
             i. Calculate the color change cost: 1 if Robot R's current color is not C else 0.
             ii. Calculate the minimum movement cost for Robot R to reach any tile in `ReqAdj(T)`.
                 For each `t_req` in `ReqAdj(T)`:
                   - The cost is the shortest distance from Robot R's current location to `t_req`
                     as computed by the BFS in step 7. If `t_req` is unreachable via clear tiles,
                     this cost is infinity.
                 Take the minimum of these movement costs over all `t_req` in `ReqAdj(T)`.
             iii. The total cost for robot R for this tile is (color change cost + minimum movement cost).
             iv. Update `min_robot_cost_for_tile` with the minimum cost found across all robots.
        d. If `min_robot_cost_for_tile` is infinity (no robot can reach a required adjacent tile),
           the state is likely unsolvable; return infinity.
        e. Add `min_robot_cost_for_tile` to `h`.
    10. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Builds the full tile graph and the required adjacent tile mapping.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations and colors for painted tiles.
        self.goal_painted_tiles = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted_tiles.add((tile, color))

        # Build the full tile adjacency graph.
        # graph[tile] = {neighbor_tile1, neighbor_tile2, ...}
        self.full_tile_graph = {}
        # Build the required adjacent tile map:
        # req_adj_map[tile_to_paint] = {tile_robot_must_be_at}
        self.req_adj_map = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                direction, tile1, tile2 = parts
                # (direction tile1 tile2) means tile1 is 'direction' from tile2.
                # Robot at tile2 can paint tile1 using the corresponding paint action.
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1.
                # Robot at tile_0_1 can paint tile_1_1 using paint_up.
                # So, tile_0_1 is a required adjacent tile for tile_1_1.

                # Full graph: Add edges in both directions
                self.full_tile_graph.setdefault(tile1, set()).add(tile2)
                self.full_tile_graph.setdefault(tile2, set()).add(tile1)

                # Required adjacent tile map:
                # To paint tile1, robot must be at tile2.
                self.req_adj_map.setdefault(tile1, set()).add(tile2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        clear_tiles = set()
        robot_locations = {}
        robot_colors = {}
        robots = set()
        current_painted_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "clear":
                clear_tiles.add(parts[1])
            elif parts[0] == "painted":
                current_painted_facts.add(fact)
            elif parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                if goal_fact not in current_painted_facts:
                    tile, color = get_parts(goal_fact)[1:]
                    unpainted_goal_tiles.add((tile, color))
                    # 2. Check if any tile in U is not currently clear. If so, return infinity.
                    # If a tile is not painted correctly and is not clear, it must be painted
                    # with the wrong color. The domain has no unpaint action, so it's unsolvable.
                    if tile not in clear_tiles:
                         return float('inf')


        # 3. If U is empty, the heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # 6. Build a movement graph and 7. Run BFS for each robot
        # Graph: Nodes are ALL tiles. Edge u -> v exists if u, v are adjacent in full_tile_graph AND v is clear.
        dist_from_robot = {}
        for robot, start_tile in robot_locations.items():
            dist_from_robot[robot] = {}
            q = deque([(start_tile, 0)])
            visited = {start_tile}
            dist_from_robot[robot][start_tile] = 0

            while q:
                current_tile, d = q.popleft()

                # Consider all neighbors in the full grid
                if current_tile in self.full_tile_graph:
                    for next_tile in self.full_tile_graph[current_tile]:
                        # Can move to next_tile if next_tile is clear
                        if next_tile in clear_tiles:
                            if next_tile not in visited:
                                visited.add(next_tile)
                                dist_from_robot[robot][next_tile] = d + 1
                                q.append((next_tile, d + 1))

        # 8. Initialize the total heuristic value h
        h = 0

        # 9. For each unsatisfied goal tile (T, C) in U
        for tile_to_paint, target_color in unpainted_goal_tiles:
            # a. Add 1 for the paint action
            h += 1

            # b. Determine required adjacent tiles
            req_adj_tiles = self.req_adj_map.get(tile_to_paint, set())

            min_robot_cost_for_tile = float('inf')

            # c. Calculate min cost for any robot
            for robot in robots:
                robot_current_location = robot_locations[robot]
                robot_current_color = robot_colors[robot]

                color_change_cost = 1 if robot_current_color != target_color else 0

                min_movement_cost = float('inf')

                # Find min movement cost for this robot to any required adjacent tile
                for t_req in req_adj_tiles:
                    # Cost to move from Loc(R) to t_req through clear tiles.
                    # dist_from_robot[robot][t_req] is the shortest path using moves onto clear tiles.
                    if t_req in dist_from_robot[robot]:
                         movement_cost = dist_from_robot[robot][t_req]
                         min_movement_cost = min(min_movement_cost, movement_cost)

                # If the robot can reach at least one required adjacent tile
                if min_movement_cost == float('inf'):
                    # This robot cannot reach any required adjacent tile for this goal tile
                    pass # Check next robot
                else:
                    robot_cost = color_change_cost + min_movement_cost
                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)


            # d. If no robot can reach a required adjacent tile for this goal tile
            if min_robot_cost_for_tile == float('inf'):
                # This goal tile is unreachable for all robots in the current clear-tile configuration.
                return float('inf')

            # e. Add min robot cost to h
            h += min_robot_cost_for_tile

        # 10. Return h
        return h
