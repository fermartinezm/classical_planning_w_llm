from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal facts of the form
    (yh6q5j8r <tile> <color>) starting from the current state.
    The domain involves robots located on tiles, each tile having a "d7ekk7vb" property when
    unpainted (or paintable), and tiles eventually need to be "painted" with colors (vfphwgc6) by
    delivering color facts (yh6q5j8r tile color). Robots can move colors around the tiles by
    carrying one color at a time and switching colors in their possession.

    The heuristic roughly measures:
    - How many tiles still need to be painted.
    - For each such tile, estimate the minimal movement cost for a robot from its current tile to the destination tile and
      the cost to "apply" the paint.
    - The cost also incorporates how many color swaps the robot must do to get the right paint color,
      as robots carry one color at a time.
    - Implements a simplified shortest-path graph on tiles using static adjacency predicates,
      assuming the robot's color can always be swapped efficiently.

    Assumptions
    -----------
    - Only one color can be carried by a robot at a time (carried color indicated by predicate i4u94rl9).
    - Tiles needing paint are indicated by the presence of predicate d7ekk7vb (paintable).
    - The robot moves between tiles along adjacency relations (xr2t1c1d, oh0smogc, w224no0l, g7ya35is),
      which form a graph on tiles.
    - The heuristic assumes all goals involve tiles painted with a certain color (yh6q5j8r).
    - Multiple robots are considered; heuristic picks the cheapest cost for painting all required tiles.
    - The heuristic is not guaranteed admissible but aims to guide search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract from static facts the adjacency graph between tiles, building a combined undirected graph for movement.
    - Extract the set of color objects available via predicate cdb9op3o.
    - Extract the goal tiles and their required colors from goal facts (predicate yh6q5j8r).
    - Extract the objects of each type, especially:
      * tiles (type bffr59sc)
      * robots (type ioytrsy9)
      * colors (type vfphwgc6)
    - Prepare lookup dictionaries for adjacency and colors.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify tiles that still require painting by checking which goal painting facts (yh6q5j8r tile color)
       are not yet satisfied in the current state.
    2. Identify robots' current locations (gk9u3vg3 robot tile) and the current color each robot holds (i4u94rl9 robot color).
    3. For each tile that requires painting, estimate the minimal movement cost:
       - For each robot, compute the shortest distance from the robot's current tile to the tile needing paint.
       - Add cost for switching robot's color to the required color if different.
       - Add cost for applying paint (probably 1 action).
       - Keep the minimal such cost over all robots.
    4. Sum these minimum costs for all unpainted tiles to get the heuristic value.
    5. If no tiles need painting (all goals met), heuristic is zero.
    6. The heuristic is efficiently computable by using precomputed adjacency and simple breadth-first search for shortest paths.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static adjacency info, colors, and goal paint assignments.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract tiles (bffr59sc), robots (ioytrsy9), colors (vfphwgc6) from static info heuristically.
        # Because static facts only include predicates, extract used objects in domain objects:
        # We assume objects given from facts.

        # Extract adjacency relations between tiles from static facts:
        # Adjacency predicates: (xr2t1c1d tile1 tile2),
        #                       (oh0smogc tile1 tile2),
        #                       (w224no0l tile1 tile2),
        #                       (g7ya35is tile1 tile2)
        # All represent neighbor relations (undirected for movement).

        # Build undirected adjacency graph for tiles
        self.adjacency = dict()  # tile -> set of neighbor tiles

        def add_edge(t1, t2):
            self.adjacency.setdefault(t1, set()).add(t2)
            self.adjacency.setdefault(t2, set()).add(t1)

        for fact in self.static:
            if match(fact, "xr2t1c1d", "*", "*"):
                _, t1, t2 = get_parts(fact)
                add_edge(t1, t2)
            elif match(fact, "oh0smogc", "*", "*"):
                _, t1, t2 = get_parts(fact)
                add_edge(t1, t2)
            elif match(fact, "w224no0l", "*", "*"):
                _, t1, t2 = get_parts(fact)
                add_edge(t1, t2)
            elif match(fact, "g7ya35is", "*", "*"):
                _, t1, t2 = get_parts(fact)
                add_edge(t1, t2)

        # Extract known colors from static predicate (cdb9op3o <color>)
        self.colors = set()
        for fact in self.static:
            if match(fact, "cdb9op3o", "*"):
                _, color = get_parts(fact)
                self.colors.add(color)

        # Extract goal paint predicates (yh6q5j8r <tile> <color>)
        self.goal_paints = {}
        for goal in self.goals:
            if match(goal, "yh6q5j8r", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paints[tile] = color

    def shortest_path_length(self, start, target):
        """
        Computes shortest path length between two tiles using BFS on adjacency.
        Returns integer distance or a large number if unreachable.
        """
        if start == target:
            return 0
        visited = {start}
        frontier = [start]
        distance = 0
        while frontier:
            distance += 1
            next_frontier = []
            for u in frontier:
                neighbors = self.adjacency.get(u, ())
                for v in neighbors:
                    if v == target:
                        return distance
                    if v not in visited:
                        visited.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
        # If unreachable, return a large cost (domain should be connected)
        return 1000

    def __call__(self, node):
        """
        Compute a heuristic estimate of actions to reach the goal from the current state.

        Steps:
        - Identify which tiles still need painting by comparing current state with goals.
        - For each robot, find its current tile and carried color.
        - For each tile needing paint, calculate minimal cost:
            Movement cost : shortest tile path distance for robot to tile.
            Color swap cost: cost 1 if robot color differs from needed color (for painting).
            Painting cost: cost 1 for applying paint.
        - Sum minimal costs over all tiles needing painting.
        - Return 0 if all tiles already painted as required.
        """
        state = node.state

        # Identify tiles that still need painting
        unpainted_tiles = []
        for tile, needed_color in self.goal_paints.items():
            fact = f"(yh6q5j8r {tile} {needed_color})"
            if fact not in state:
                unpainted_tiles.append((tile, needed_color))

        if not unpainted_tiles:
            # All tiles already painted as goal demands
            return 0

        # Extract robots' current locations and colors:
        # (gk9u3vg3 robot tile), (i4u94rl9 robot color)
        robot_positions = {}  # robot -> tile
        robot_colors = {}     # robot -> color

        for fact in state:
            if match(fact, "gk9u3vg3", "*", "*"):
                _, robot, tile = get_parts(fact)
                robot_positions[robot] = tile
            elif match(fact, "i4u94rl9", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # If no robots found, heuristic cannot progress properly, return high cost
        if not robot_positions:
            return 10000

        total_cost = 0

        # For each tile needing paint, estimate minimal cost over all robots
        for tile, needed_color in unpainted_tiles:
            minimal_cost = 10000  # large initial cost

            for robot, robot_tile in robot_positions.items():
                robot_color = robot_colors.get(robot)
                # Movement cost = shortest path distance robot_tile -> tile
                move_cost = self.shortest_path_length(robot_tile, tile)
                # Color swap cost: 1 if different, 0 if same
                color_swap_cost = 0 if robot_color == needed_color else 1
                # Painting cost (applying paint) = 1
                paint_cost = 1

                cost = move_cost + color_swap_cost + paint_cost

                if cost < minimal_cost:
                    minimal_cost = cost

            total_cost += minimal_cost

        return total_cost
