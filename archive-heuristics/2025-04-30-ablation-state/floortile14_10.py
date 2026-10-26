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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums three components:
    1. The number of tiles that still need to be painted correctly.
    2. The number of distinct colors needed for painting that no robot currently holds.
    3. An estimate of the total movement cost, calculated as the sum, for each tile
       needing painting, of the minimum distance from any robot to a tile adjacent
       to the target tile.
    If a tile required by the goal is painted with the wrong color, the heuristic
    returns infinity, as the problem is likely unsolvable from that state.

    # Assumptions
    - Tiles needing painting are initially clear or painted with the wrong color.
      The heuristic assumes they are clear for the paint action precondition, unless
      they are painted with the wrong color, in which case it assumes unsolvability.
    - Movement is possible between adjacent tiles, ignoring the 'clear' precondition
      for the tiles being moved *to*. This is a relaxation.
    - The grid structure is defined by the static 'up', 'down', 'left', 'right' facts.
    - All tiles involved in connectivity facts are the relevant tiles in the grid.
    - All robots can perform all actions (move, paint, change_color).

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Builds an undirected graph representing the grid connectivity based on static
      'up', 'down', 'left', 'right' facts.
    - Computes all-pairs shortest paths on this grid graph using BFS from each tile.
    - Identifies the set of tiles adjacent to each tile (from which it can be painted)
      based on the paint action preconditions.
    - Identifies all robot objects from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check for unsolvability: If any tile required by the goal is currently painted
       with a color different from the goal color, return infinity.
    2. Identify the set of goal facts `(painted T C)` that are not true in the current state. Let this set be `unsatisfied_painted_goals`.
    3. If `unsatisfied_painted_goals` is empty, the heuristic value is 0 (goal state).
    4. Calculate `paint_cost`: This is simply the number of unsatisfied painted goals: `|unsatisfied_painted_goals|`. Each requires at least one paint action.
    5. Calculate `color_cost`:
       - Determine the set of distinct colors `C` required by the tiles in `unsatisfied_painted_goals`.
       - Determine the set of colors currently held by robots in the state.
       - The `color_cost` is the number of colors required that are not currently held by any robot. Each such color requires at least one `change_color` action by some robot.
    6. Calculate `movement_cost`:
       - For each `(T, C)` in `unsatisfied_painted_goals`:
         - Find the set of tiles `AdjacentTiles(T)` from which a robot can paint `T`. These are the tiles `X` such that `(up T X)`, `(down T X)`, `(left T X)`, or `(right T X)` is true.
         - Find the minimum distance from *any* robot's current location to *any* tile in `AdjacentTiles(T)`, using the precomputed grid distances.
       - The `movement_cost` is the sum of these minimum distances over all `(T, C)` in `unsatisfied_painted_goals`. This estimates the movement needed to get a robot into position for each painting task, independently.
    7. The total heuristic value is the sum: `paint_cost + color_cost + movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the grid graph,
        and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build the grid graph and identify all tiles
        self.adj = {}
        self.tiles = set()
        self.adjacent_paint_pos = {} # Map tile T to set of tiles X where robot at X can paint T

        # Collect all tiles mentioned in connectivity facts
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                dir_pred, tile1, tile2 = parts
                self.tiles.add(tile1)
                self.tiles.add(tile2)

                # Add undirected edge for movement
                self.adj.setdefault(tile1, []).append(tile2)
                self.adj.setdefault(tile2, []).append(tile1)

                # Identify paint positions: robot at tile2 paints tile1 if (dir tile1 tile2)
                # paint_up ?r ?y ?x ?c requires (up ?y ?x) -> robot at ?x paints ?y
                # paint_down ?r ?y ?x ?c requires (down ?y ?x) -> robot at ?x paints ?y
                # paint_left ?r ?y ?x ?c requires (left ?y ?x) -> robot at ?x paints ?y
                # paint_right ?r ?y ?x ?c requires (right ?y ?x) -> robot at ?x paints ?y
                # So, if (dir tile1 tile2) is true, robot at tile2 can paint tile1.
                self.adjacent_paint_pos.setdefault(tile1, set()).add(tile2)


        # Ensure all tiles mentioned in goals are included, even if isolated
        for goal in self.goals:
             g_parts = get_parts(goal)
             if g_parts[0] == 'painted' and len(g_parts) == 3:
                 self.tiles.add(g_parts[1])
                 # Ensure isolated tiles have an entry in adj and adjacent_paint_pos
                 self.adj.setdefault(g_parts[1], [])
                 self.adjacent_paint_pos.setdefault(g_parts[1], set())


        # 2. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.tiles:
            self.dist[start_node] = {tile: float('inf') for tile in self.tiles}
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])

            while queue:
                curr = queue.popleft()
                # Use .get(curr, []) to handle potential isolated tiles added from goals
                for neighbor in self.adj.get(curr, []):
                    if self.dist[start_node][neighbor] == float('inf'):
                        self.dist[start_node][neighbor] = self.dist[start_node][curr] + 1
                        queue.append(neighbor)

        # 3. Store goal painted facts for quick lookup
        self.goal_painted = {}
        for goal in self.goals:
            g_parts = get_parts(goal)
            if g_parts[0] == 'painted' and len(g_parts) == 3:
                tile, color = g_parts[1], g_parts[2]
                self.goal_painted[tile] = color

        # 4. Identify all robot objects (assuming they are in initial state facts)
        self.robots = set()
        for fact in task.initial_state:
             f_parts = get_parts(fact)
             if f_parts[0] == 'robot-at' and len(f_parts) == 3:
                 self.robots.add(f_parts[1])
             elif f_parts[0] == 'robot-has' and len(f_parts) == 3:
                 self.robots.add(f_parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot locations and colors
        robot_locations = {}
        robot_colors = set()
        for fact in state:
            f_parts = get_parts(fact)
            if f_parts[0] == 'robot-at' and len(f_parts) == 3:
                robot, location = f_parts[1], f_parts[2]
                robot_locations[robot] = location
            elif f_parts[0] == 'robot-has' and len(f_parts) == 3:
                 robot, color = f_parts[1], f_parts[2]
                 robot_colors.add(color)

        # Identify current painted tiles and check for unsolvability
        current_painted = {}
        for fact in state:
             f_parts = get_parts(fact)
             if f_parts[0] == 'painted' and len(f_parts) == 3:
                 tile, color = f_parts[1], f_parts[2]
                 current_painted[tile] = color
                 # Check if this painted tile conflicts with a goal
                 if tile in self.goal_painted and self.goal_painted[tile] != color:
                     # Tile is painted with the wrong color - likely unsolvable
                     return float('inf') # Return infinity

        # Identify unsatisfied painted goals
        unsatisfied_painted_goals = set()
        needed_colors = set()
        for tile, goal_color in self.goal_painted.items():
            if tile not in current_painted or current_painted[tile] != goal_color:
                 # This tile needs to be painted with goal_color
                 unsatisfied_painted_goals.add((tile, goal_color))
                 needed_colors.add(goal_color)

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_painted_goals:
            return 0

        # 1. Calculate paint_cost
        paint_cost = len(unsatisfied_painted_goals)

        # 2. Calculate color_cost
        colors_to_acquire = needed_colors - robot_colors
        color_cost = len(colors_to_acquire)

        # 3. Calculate movement_cost
        movement_cost = 0
        for tile, color in unsatisfied_painted_goals:
            min_dist_to_paint_tile = float('inf')
            # Find the set of tiles from which this tile can be painted
            possible_paint_positions = self.adjacent_paint_pos.get(tile, set())

            # Find the minimum distance from any robot to any valid paint position for this tile
            for robot_loc in robot_locations.values():
                if robot_loc not in self.dist:
                     # Robot is at a location not in our distance map (e.g., isolated tile)
                     # This shouldn't happen in valid problems, but handle defensively.
                     continue

                for paint_pos in possible_paint_positions:
                    if paint_pos in self.dist[robot_loc]:
                         min_dist_to_paint_tile = min(min_dist_to_paint_tile, self.dist[robot_loc][paint_pos])
                    # else: paint_pos is not reachable from robot_loc (disconnected graph?)
                    # If dist remains inf, it means unreachable.

            # Add the minimum distance required to get *a* robot into position for this tile
            # If min_dist_to_paint_tile is still inf, it means no robot can reach a paint position.
            # This contributes infinity to the total movement_cost, correctly reflecting unsolvability.
            movement_cost += min_dist_to_paint_tile


        # Total heuristic value
        # If movement_cost is inf (due to unreachable paint positions), total_heuristic will be inf.
        total_heuristic = paint_cost + color_cost + movement_cost

        return total_heuristic
