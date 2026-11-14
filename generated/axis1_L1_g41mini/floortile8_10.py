from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(yh6q5j8r tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to reach a goal state by focusing
    on the tiles that still need to be painted with their goal colors. 
    It takes into account the robot's current location and held color, the mapping between tiles 
    (adjacencies), and the fact that painting a tile requires the robot to be on the corresponding 
    tile and holding the right color.
    The heuristic approximates the minimal effort to get each tile painted with the goal color,
    including moves to tiles and color switches.

    Assumptions:
    -------------
    - The goal facts are always of the form (yh6q5j8r tile color), meaning "tile painted with color".
    - The robot moves among tiles via the bidirectional adjacency relations (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
    - The robot holds exactly one color at a time (i4u94rl9 predicate).
    - Some tiles are "empty" (d7ekk7vb fact), meaning unpainted and ready to be painted.
    - Painting consumes the tile's emptiness (removes d7ekk7vb) and produces yh6q5j8r.
    - Color changes happen by the z7pcmkeu action which swaps the current color (i4u94rl9) to another available color.
    - Moving the robot between tiles is done by the f2q9mxxi, cf57hd49, tgataxn3, ylvh432k actions based on adjacency predicates.
    - The heuristic does not need to be admissible but should be efficiently computable.

    Heuristic Initialization:
    --------------------------
    - Extract the goal tiles and their required colors.
    - Extract all adjacency relations between tiles (xr2t1c1d, oh0smogc, w224no0l, g7ya35is), combining them into a single undirected graph.
    - Extract the available colors (objects appearing in cdb9op3o facts).
    - Extract static info does not change over search: color availability and tile adjacency graph.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify unpainted tiles that have a color requirement in the goal but are not yet painted.
       (tiles that lack (yh6q5j8r tile color) matching the goal, or have (d7ekk7vb tile) meaning still empty)
    2. From the current state, determine:
       - The robot's current tile location (gk9u3vg3 robot tile).
       - The robot's currently held color (i4u94rl9 robot color).
    3. For each tile that needs painting with a color:
       - Estimate cost components:
         a) Move cost: shortest path distance in tile graph from current robot location or previous painted tile.
         b) Color-switch cost: if current held color differs from required color, count 1 action to switch.
         c) Paint cost: 1 action to paint (via c1qh1p5g or cadwr4wd).
    4. Since painting a tile consumes the emptiness of that tile, each tile must be visited by the robot holding the correct color once.
    5. To not overestimate, assume the robot can paint tiles in any order—approximate by greedily summing minimal distances between tiles to paint, considering color switches.
    6. Account for the initial position and color of the robot.
    7. Return the sum of all estimated moves, color switches, and painting actions for all tiles yet to be painted.

    The heuristic is zero if all goal painting facts are satisfied (goal reached).
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Parse the goal: map tiles -> required color for painting
        self.goal_tile_colors = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "yh6q5j8r" and len(parts) == 3:
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        # Extract known tiles and colors from static facts
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "cdb9op3o", "*"):
                # cdb9op3o defines available colors
                color = get_parts(fact)[1]
                self.available_colors.add(color)

        # Extract tile adjacency: combine all adjacency predicates into undirected graph
        # Adjacency predicates: xr2t1c1d, oh0smogc, w224no0l, g7ya35is (given from domain predicates)
        self.adjacency = {}
        def add_edge(t1, t2):
            self.adjacency.setdefault(t1, set()).add(t2)
            self.adjacency.setdefault(t2, set()).add(t1)

        for fact in static_facts:
            for pred in ("xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"):
                if match(fact, pred, "*", "*"):
                    parts = get_parts(fact)
                    add_edge(parts[1], parts[2])

    def shortest_path_length(self, start, goal):
        """Compute shortest path length from start tile to goal tile in adjacency graph using BFS."""
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return float('inf')
        from collections import deque
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def __call__(self, node):
        state = node.state

        # If goal is reached (all goal facts satisfied), h=0
        if self.goals <= state:
            return 0

        # Extract robot current tile location (gk9u3vg3 robot tile)
        robot_tile = None
        for fact in state:
            if match(fact, "gk9u3vg3", "*", "*"):
                parts = get_parts(fact)
                robot_tile = parts[1]
                break
        if robot_tile is None:
            # Should not happen, but fail safe
            return 0xFFFF

        # Extract robot current held color (i4u94rl9 robot color)
        robot_color = None
        for fact in state:
            if match(fact, "i4u94rl9", "*", "*"):
                parts = get_parts(fact)
                robot_color = parts[2]
                break
        if robot_color is None:
            # Should not happen, but fail safe
            return 0xFFFF

        # Tiles that still require painting: those tiles in goal_tile_colors not (yh6q5j8r tile color) in state
        tiles_to_paint = []
        for tile, color in self.goal_tile_colors.items():
            goal_fact = f"(yh6q5j8r {tile} {color})"
            if goal_fact not in state:
                tiles_to_paint.append((tile, color))

        # Also ensure tiles are empty (d7ekk7vb tile) -- painting only applies if tile is empty
        # But painting actions remove d7ekk7vb
        # Only consider tiles that are empty or not painted (since painting a non-empty tile likely invalid).
        tiles_ready_to_paint = []
        empty_tiles = {get_parts(f)[1] for f in state if match(f, "d7ekk7vb", "*")}
        for (tile, color) in tiles_to_paint:
            if tile in empty_tiles:
                tiles_ready_to_paint.append((tile, color))
            else:
                # Tile not empty, assume must be painted anyway; keep it for heuristic
                tiles_ready_to_paint.append((tile, color))

        if not tiles_ready_to_paint:
            # No tiles require painting, h=0
            return 0

        # To estimate heuristic, apply a greedy path & color cost approximation:
        # Cost components:
        # - Move between tiles (use shortest path of tiles)
        # - Color switch cost: 1 per color change (switch action)
        # - Paint cost: 1 per paint action to place color on tile

        total_cost = 0
        current_location = robot_tile
        current_color = robot_color

        # Keep track of which tiles remain to paint
        remaining = tiles_ready_to_paint.copy()

        while remaining:
            # Find tile in remaining with minimal cost:
            # cost = move distance from current_location + (1 if current_color != tile_color else 0) + 1 paint action

            best_cost = float('inf')
            best_index = None
            for i, (tile, color) in enumerate(remaining):
                dist = self.shortest_path_length(current_location, tile)
                if dist == float('inf'):
                    continue  # cannot reach, ignore for now

                color_switch = 1 if current_color != color else 0
                cost = dist + color_switch + 1  # move + color switch + paint
                if cost < best_cost:
                    best_cost = cost
                    best_index = i

            if best_index is None:
                # No reachable tile; assign high cost
                total_cost += 1000
                break

            # Pick the best tile to paint next
            tile, color = remaining.pop(best_index)
            total_cost += best_cost
            current_location = tile
            current_color = color

        return total_cost
