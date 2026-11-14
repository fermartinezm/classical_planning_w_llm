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
    A domain-dependent heuristic for the secret_domain (w133ormt) domain.

    Summary:
    ----------
    This heuristic estimates the number of actions needed to achieve the goal predicates of the
    form (yh6q5j8r tile color) for each tile by considering:
    - Tiles that still need to have their color set correctly.
    - The shortest path (in tile steps) for the robot to reach those tiles from its current position.
    - Whether the robot needs to switch color before coloring a tile.
    The heuristic returns 0 iff the goal is reached and otherwise provides an estimate of the
    minimal steps to paint all goal tiles correctly by planning a cost-effective painting route that
    accounts for movement and recoloring actions.

    Assumptions:
    ------------
    - The robot can carry exactly one color at a time (represented by the predicate (i4u94rl9 robot color)).
    - Movement between tiles follows directional predicates (xr2t1c1d) and their inverse (oh0smogc).
    - Tiles are colored by a single action (c1qh1p5g, cadwr4wd), which require the robot
      to be on the tile and carry the color already.
    - Changing the color the robot carries is done by a single action (z7pcmkeu).
    - Each painting and recoloring action costs 1; each movement between tiles costs 1.
    - The heuristic does NOT guarantee admissibility but aims to guide greedy best-first search efficiently.

    Heuristic Initialization:
    --------------------------
    - Collect all color objects (vfphwgc6) from static facts via (cdb9op3o color).
    - Build adjacency maps for tiles from static facts (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
      These predicates form various directional edges; all are considered bidirectional.
    - Store the set of tiles that appear in goals as targets to be painted with specific colors.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Extract robot's current tile and carried color from the current state.
    2. Extract the current colors of all tiles from (yh6q5j8r tile color) facts in the state.
    3. Identify which tiles still need repainting as per the goal conditions:
       those tiles whose (yh6q5j8r tile color) goal predicate is not yet satisfied in the current state.
    4. For each tile to paint:
       - Compute the shortest tile-distance from the robot's current position or from previously painted tiles.
       - Account for recoloring if carried color differs from the target tile color.
    5. Use a greedy approximation:
       - Start at the robot's tile and color.
       - Repeatedly pick the closest tile to paint (using shortest path distances on the tile adjacency graph).
       - Add cost for moving to tile, recoloring if needed, and painting.
       - Update current robot position and carried color accordingly.
    6. Sum these costs as an estimate for the remaining steps to goal.
    7. Return 0 if all goals are already satisfied.

    This heuristic tries to balance movement and recoloring costs and thus guides the search
    towards efficient painting sequences while being feasible to compute quickly.
    """

    def __init__(self, task):
        """
        Initialize by extracting:
        - color objects
        - tile adjacency graph from directional static predicates
        - goal color assignments per tile
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract all colors from static (cdb9op3o color)
        self.colors = set()
        for fact in static_facts:
            if match(fact, "cdb9op3o", "*"):
                color = get_parts(fact)[1]
                self.colors.add(color)

        # Build tile adjacency: consider all directional predicates that connect tiles bidirectionally
        # Predicates with known edges: xr2t1c1d, oh0smogc, w224no0l, g7ya35is
        self.adjacency = {}
        def add_edge(t1, t2):
            self.adjacency.setdefault(t1, set()).add(t2)

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"):
                t1 = parts[1]
                t2 = parts[2]
                add_edge(t1, t2)
                add_edge(t2, t1)

        # Extract goal tile-color assignments from goals of type (yh6q5j8r tile color)
        self.goal_tile_colors = {}
        for goal in self.goals:
            if match(goal, "yh6q5j8r", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_tile_colors[tile] = color

    def __call__(self, node):
        """
        Estimate the number of primitive actions required to reach the goal from the current state:

        Steps:
        1. If all goals are satisfied, return 0.
        2. Extract the robot's current tile (from gk9u3vg3 robot tile).
        3. Extract the robot's carried color (from i4u94rl9 robot color).
        4. Extract the current colored tiles (yh6q5j8r tile color).
        5. Determine tiles still requiring painting (goal tiles not matching current colors).
        6. Using a greedy approach:
           - Compute shortest path distances between tiles via BFS on adjacency graph.
           - Start at robot's position and carried color.
           - Iteratively pick the closest tile yet to paint, add painting and recoloring cost, move current position.
        7. Sum all costs and return.

        Note: Each action (move, recolor, paint) counts as 1.
        """

        state = node.state

        # Quick goal check
        if self.goals <= state:
            return 0

        # Helper to find robot's tile and carried color
        robot_tile = None
        robot_color = None
        for fact in state:
            if match(fact, "gk9u3vg3", "robot1", "*") or match(fact, "gk9u3vg3", "robot2", "*") or match(fact, "gk9u3vg3", "robot3", "*"):
                parts = get_parts(fact)
                robot_tile = parts[2]
            if match(fact, "i4u94rl9", "robot1", "*") or match(fact, "i4u94rl9", "robot2", "*") or match(fact, "i4u94rl9", "robot3", "*"):
                parts = get_parts(fact)
                robot_color = parts[2]

        # If robot_tile or color not found (should not happen), fallback heuristic value:
        if robot_tile is None or robot_color is None:
            # Conservative fallback: number of tiles to paint * 3 (assign some cost)
            tiles_to_paint = [
                tile for tile, color in self.goal_tile_colors.items()
                if not match_in_state(state, tile, color)
            ]
            return len(tiles_to_paint) * 3

        # Extract current tile colors (yh6q5j8r tile color)
        current_tile_colors = {}
        for fact in state:
            if match(fact, "yh6q5j8r", "*", "*"):
                _, tile, color = get_parts(fact)
                current_tile_colors[tile] = color

        # Determine tiles still to paint (goal color != current color)
        tiles_to_paint = []
        for tile, goal_color in self.goal_tile_colors.items():
            current_color = current_tile_colors.get(tile)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))

        if not tiles_to_paint:
            return 0  # Already achieved

        # Precompute shortest paths from any tile to any tile using BFS on adjacency graph.
        # But since we only need shortest distances from robot tile and between candidate tiles,
        # we'll compute distances on demand with BFS for efficiency.

        def bfs_distance(start, goal):
            if start == goal:
                return 0
            visited = {start}
            frontier = [(start, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                for neighbor in self.adjacency.get(current, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, dist + 1))
            # If no path found (should not happen), return large distance
            return 10000

        # Greedy heuristic: start at robot_tile and robot_color
        total_cost = 0
        curr_tile = robot_tile
        curr_color = robot_color

        # Copy list to modify
        remaining = tiles_to_paint[:]

        while remaining:
            # Find closest tile to current position in terms of movement cost + recoloring cost
            best_choice = None
            best_cost = None
            for idx, (tile, color) in enumerate(remaining):
                dist = bfs_distance(curr_tile, tile)
                recolor_cost = 0 if curr_color == color else 1  # recoloring cost
                paint_cost = 1  # paint on tile
                cost = dist + recolor_cost + paint_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_choice = (idx, tile, color, dist, recolor_cost, paint_cost)

            # Apply cost and update current position/color
            idx, tile, color, dist, recolor_cost, paint_cost = best_choice
            total_cost += dist + recolor_cost + paint_cost
            curr_tile = tile
            curr_color = color
            remaining.pop(idx)

        return total_cost


def match_in_state(state, tile, color):
    """Helper to check if a goal colored tile predicate is satisfied in state."""
    target_fact = f"(yh6q5j8r {tile} {color})"
    return target_fact in state
