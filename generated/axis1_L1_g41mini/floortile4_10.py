from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact string, e.g. "(predicate arg1 arg2)".
    - `args`: pattern to match, wildcards '*' allowed.
    Returns True if fact matches pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (domain w133ormt).

    Summary:
    This heuristic estimates the number of actions required to achieve all goal predicates (yh6q5j8r ?tile ?color) from the current state.
    It leverages the static connectivity relations between tiles to estimate the minimal movement and painting steps required, 
    considering the robot's location and color it carries.

    Assumptions:
    - The goal is exclusively conjunctions of facts of form (yh6q5j8r ?tile ?color).
    - Tiles that require painting are those goal tiles missing the goal color in current state.
    - Robot can move by swapping its assigned tile (gk9u3vg3 robot tile) via operators that move it along adjacent tiles connected by predicates (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
    - Only one robot exists (or heuristic is computed per robot if multiple present).
    - Changing color held by robot employs z7pcmkeu action; painting tile is done by c1qh1p5g, cadwr4wd actions.
    - d7ekk7vb tile indicates a tile is free to be painted (unpainted).
    - The robot must be positioned on or adjacent to a tile to paint it.
    - Heuristic is not required to be admissible - can optimistic or approximate.

    Heuristic Initialization:
    - Extract adjacency relations of tiles from static facts to build a connectivity graph for robot movement.
    - Extract all colors available (identified by predicate cdb9op3o).
    - Extract initial robot location (tile) and color held in current state during each call.
    - Identify all tiles and build quick lookup tables for connections.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify goal painting facts (yh6q5j8r tile color) from task goals.
    2. Identify which goal painting facts are not yet satisfied in current state.
    3. Determine robot’s current location tile and current held color.
    4. For each tile that requires painting, estimate:
       a) Minimal robot movement cost from the robot’s current tile to the tile to be painted, using BFS on adjacency graph.
       b) If robot does not hold the required color, add at least one color change action cost (z7pcmkeu).
       c) Add one painting action cost.
    5. Because robot can only paint one tile at a time and must move between tiles, 
       approximate cost by a minimal spanning path covering all tiles to paint:
       to keep efficient calculation, sum minimal distances from robot to each tile requiring painting
       plus painting and color change costs per tile as a simplistic approximation.
    6. Return cumulative estimate.
    7. If no tiles require painting (state satisfies goals), heuristic is zero.

    Note:
    This heuristic is fast to compute and guides the search by approximating total moves + paint actions + color changes.
    It may underestimate moves required for traveling between multiple tiles (no exact traveling salesman approximation),
    but sufficiently guides greedy search to minimize expansions.
    """

    def __init__(self, task):
        """
        Extract static connectivity graph and colors.
        """
        self.goals = task.goals
        self.static = task.static

        # Build adjacency map between tiles for movement
        # Tiles are bffr59sc type objects, identified in adjacency predicates:
        # xr2t1c1d, oh0smogc, w224no0l, g7ya35is (all describe adjacency but in different directions)
        self.adj_map = dict()  # tile -> set of adjacent tiles

        def add_edge(a, b):
            if a not in self.adj_map:
                self.adj_map[a] = set()
            self.adj_map[a].add(b)

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}:
                # Add edges in both directions to treat undirected connectivity graph
                t1, t2 = parts[1], parts[2]
                add_edge(t1, t2)
                add_edge(t2, t1)

        # Extract all colors by cdb9op3o predicate
        self.colors = set()
        for fact in self.static:
            if match(fact, "cdb9op3o", "*"):
                color = get_parts(fact)[1]
                self.colors.add(color)

        # Extract goal facts of form (yh6q5j8r ?tile ?color)
        self.goal_paint = []
        for goal_fact in self.goals:
            if match(goal_fact, "yh6q5j8r", "*", "*"):
                parts = get_parts(goal_fact)
                tile = parts[1]
                color = parts[2]
                self.goal_paint.append((tile, color))

    def tile_distance(self, start_tile, goal_tile):
        """Compute shortest distance between two tiles in adjacency graph using BFS."""
        if start_tile == goal_tile:
            return 0
        visited = {start_tile}
        frontier = [start_tile]
        distance = 0
        while frontier:
            next_frontier = []
            distance += 1
            for current in frontier:
                for neighbor in self.adj_map.get(current, []):
                    if neighbor == goal_tile:
                        return distance
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        # If no path found, return large cost (infeasible)
        return 1000

    def __call__(self, node):
        state = node.state

        # Quick lookup sets for facts
        facts = state

        # If goal already reached, heuristic = 0
        if self.goals <= facts:
            return 0

        # Find robot(s) current tile(s) and held color(s)
        # From predicates:
        # (gk9u3vg3 robot tile) - robot on tile
        # (i4u94rl9 robot color) - robot holds color
        # We assume one robot, find its location and held color
        robot_tile = None
        robot_color = None
        for fact in facts:
            if match(fact, "gk9u3vg3", "robot*", "*"):
                robot_tile = get_parts(fact)[2]
            elif match(fact, "i4u94rl9", "robot*", "*"):
                robot_color = get_parts(fact)[2]
        if robot_tile is None:
            # No robot found - large cost to discourage
            return 10000
        if robot_color is None:
            # No color held - large cost
            return 10000

        # Determine which tiles require painting (goal paint facts not yet satisfied)
        tiles_to_paint = []
        for tile, color in self.goal_paint:
            if not match_exists(facts, "yh6q5j8r", tile, color):
                tiles_to_paint.append((tile, color))

        # Early exit if nothing to paint
        if not tiles_to_paint:
            return 0

        # Cost settings (rough unit costs based on actions):
        # - Move robot one tile: cost 1
        # - Change held color: cost 1
        # - Paint tile: cost 1
        MOVE_COST = 1
        COLOR_CHANGE_COST = 1
        PAINT_COST = 1

        total_cost = 0

        # To approximate total cost we:
        # For each tile to paint,
        # sum shortest path from current robot position or last painted tile (we approximate no ordering here),
        # plus color change if needed, plus painting action.

        # For efficiency, we approximate by summing from the robot start position to each tile independently,
        # ignoring route optimization among paint tiles.
        for tile, required_color in tiles_to_paint:
            dist = self.tile_distance(robot_tile, tile)
            color_change = 0 if robot_color == required_color else COLOR_CHANGE_COST
            total_cost += dist * MOVE_COST + color_change + PAINT_COST

        # This underestimates real cost of traveling between paint tiles,
        # but guides the search to reduce remaining painting tasks quickly.

        return total_cost


def match_exists(facts, *pattern):
    """Return True if any fact in facts matches the pattern."""
    for fact in facts:
        if match(fact, *pattern):
            return True
    return False
