from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: '(painted tile_1_1 white)' -> ['painted', 'tile_1_1', 'white']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary
    -------
    Estimates the number of actions required to reach a goal state from a given state.
    It considers the cost to paint all unpainted or incorrectly painted tiles, the cost to move the robots to the correct positions,
    and the cost to change robots' held colors when needed. The heuristic guides a greedy best-first search efficiently without requiring admissibility.

    Assumptions
    -----------
    - Multiple robots may exist, each holding exactly one color at a time.
    - Tiles to be painted have goal colors defined in the goal conjunction.
    - Painting a tile requires the robot to be adjacent (up, down, left, or right) to the tile and holding the correct color.
    - A tile can be painted only if it is clear.
    - Tiles may already be painted, partially painted with correct or incorrect color.
    - Robots can move only onto clear tiles.
    - Changing color held by a robot requires one color change action.
    - The heuristic counts painting actions, movements to adjacent tiles, and color changes, but ignores finer path details to maintain efficiency.

    Heuristic Initialization
    ------------------------
    - Extract goal painted color assignments per tile from the goal conditions.
    - Build adjacency mappings between tiles using static facts (up, down, left, right).
    - Identify all tiles and robots from the static facts.
    - Store available colors (though color availability doesn't impact heuristic directly here).
    - The adjacency dictionary maps each tile to its neighbors (up, down, left, right).
    - Initialize sets of all colors and robots for internal bookkeeping.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Identify goal tiles that require painting with a specific color.
    2. From the current state, determine which tiles are already correctly painted.
       - Tiles that are painted with the correct color do not contribute to the heuristic.
       - Tiles not yet correctly painted need at least one paint action.
    3. Calculate the number of tiles to paint (tiles that must be painted or repainted).
    4. Extract the robots' current locations and held colors.
    5. For each tile to be painted, estimate the minimal cost for some robot to:
       a) Move from its current position to a tile adjacent to that tile (to allow painting).
          - Use a BFS on the adjacency graph considering clear tiles to estimate minimal moves.
          - Efficiently approximate distance ignoring dynamic obstacles except clear tiles.
       b) Possibly change held color if the robot's current color differs from the required color of the tile.
          - Count one color change action if needed.
       c) One paint action always needed.
    6. For efficiency, approximate by:
       - For each tile needed to paint, find the minimum distance from any robot to an adjacent tile.
       - Add cost for color change if robot color differs.
       - Add 1 for paint action.
       - Sum these minimal costs for all tiles needing painting, choosing assignments greedily.
    7. Return the total as heuristic estimate.
    8. If no tiles require painting, heuristic returns 0 (goal state).

    Note:
    - The heuristic assumes clear tiles remain clear for movement.
    - It underestimates the impact of competing robots and dynamic tile states but is a practical guiding heuristic.
    """

    def __init__(self, task):
        """Initialize all required data structures from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal painting requirements: tile -> color
        self.goal_painted = {}  # tile -> color
        for fact in self.goals:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                self.goal_painted[tile] = color

        # Extract adjacency between tiles from static facts: build neighbors per tile
        self.adj = {}  # tile -> set of adjacent tiles
        # Initialize adjacency dict with empty sets for all relevant tiles
        tiles = set()
        for fact in self.static:
            # Collect all tiles statically from adjacency relations
            pred, a, b = get_parts(fact) if len(get_parts(fact)) == 3 else (None, None, None)
            if pred in {"up", "down", "left", "right"}:
                tiles.add(a)
                tiles.add(b)
        for tile in tiles:
            self.adj[tile] = set()
        for fact in self.static:
            pred, a, b = get_parts(fact) if len(get_parts(fact)) == 3 else (None, None, None)
            # Relations are directed, but for adjacency, links are bidirectional
            if pred in {"up", "down", "left", "right"}:
                if a in self.adj:
                    self.adj[a].add(b)
                if b in self.adj:
                    self.adj[b].add(a)

        # Extract all robots and their available colors (static)
        self.robots = set()
        self.colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                _, color = get_parts(fact)
                self.colors.add(color)
        # Robots are not static, but they appear in initial_state and states as predicates:
        # No direct static info; they will be extracted in heuristic call from state

    def __call__(self, node):
        """
        Compute heuristic value: estimated actions remaining to reach goal.

        Returns 0 if goal is reached.
        """
        state = node.state

        # Quick goal check: if all goals satisfied, return 0
        if self.goals <= state:
            return 0

        # Extract painted tiles currently correct and tiles needing painting or repainting
        painted_correct = set()
        painted_wrong = set()
        painted_any = set()
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                painted_any.add(tile)
                if tile in self.goal_painted:
                    if color == self.goal_painted[tile]:
                        painted_correct.add(tile)
                    else:
                        painted_wrong.add(tile)  # painted with wrong color

        # Tiles that must be painted (because different or unpainted but goal demands paint)
        tiles_to_paint = set(self.goal_painted.keys()) - painted_correct

        if not tiles_to_paint:
            # All tiles painted correctly, heuristic cost zero
            return 0

        # Extract robot locations and held colors from current state
        robot_positions = {}  # robot -> current tile
        robot_colors = {}     # robot -> color held

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robot_positions[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # Extract clear tiles from current state (robots can only move to clear tiles)
        clear_tiles = set()
        for fact in state:
            if match(fact, "clear", "*"):
                _, tile = get_parts(fact)
                clear_tiles.add(tile)

        # Precompute accessible tiles for movement: assumption = only clear tiles are movable destinations.
        # Tiles occupied by robots currently are not clear, so robots cannot move onto them.
        # We'll allow robots to stand on their own tile even if it is not marked clear (the definitions imply robot-at moves onto clear tiles, but robot's tile itself may not be clear)
        # So for each robot, start at their tile (even if not marked clear) and move only to clear adjacent tiles.
        # We'll allow start position in BFS even if not in clear_tiles.

        # Build a set of reachable tiles via BFS for pathfinding per robot efficiently.
        # To approximate distances, BFS will count steps in adjacency on clear tiles.

        def bfs_dist(start):
            # Returns dict tile -> distance from start
            dist = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in self.adj.get(current, []):
                    # neighbor must be clear or equal to start (robots can stand on their tile)
                    if neighbor in clear_tiles and neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            return dist

        # For each robot, compute distances to all tiles reachable by moving only on clear tiles,
        # starting from their current position (which might not be clear)
        robot_dist_maps = {}
        for robot, pos in robot_positions.items():
            robot_dist_maps[robot] = bfs_dist(pos)

        # For painting a tile, robot must be adjacent (up/down/left/right) to that tile.
        # For each tile to paint, find all adjacent tiles (neighboring) where robot can move and paint.
        # For each robot, find minimal moves to get to ANY adjacent tile of the tile to paint,
        # then add color change if needed, then add 1 for paint action.

        total_cost = 0

        # To avoid assigning multiple robots to the same tile and overestimate,
        # we greedily assign tiles to robots with minimal cost per tile
        # using a list of (cost, robot, tile). Then sum minimal cost assignments without double counting.

        tile_robot_costs = []  # (cost, robot, tile)

        for tile in tiles_to_paint:
            required_color = self.goal_painted[tile]
            adjacent_tiles = self.adj.get(tile, set())
            # For each robot compute minimal distance to any adjacent tile of this tile
            best_cost = None
            best_robot = None
            for robot, dist_map in robot_dist_maps.items():
                # Find minimal distance to any adjacent tile of `tile`
                min_dist_to_adj = None
                for adj_tile in adjacent_tiles:
                    if adj_tile in dist_map:
                        d = dist_map[adj_tile]
                        if min_dist_to_adj is None or d < min_dist_to_adj:
                            min_dist_to_adj = d
                if min_dist_to_adj is None:
                    # robot cannot reach adjacent tile (blocked)
                    continue
                # Calculate color change cost
                robot_color = robot_colors.get(robot)
                color_cost = 1 if robot_color != required_color else 0
                cost = min_dist_to_adj + color_cost + 1  # moves + color change + paint
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_robot = robot
            if best_cost is None:
                # no robot can paint this tile currently (e.g., isolated)
                # return a large estimate (but finite) to avoid infinite heuristics
                # or skip? We'll add a large fixed penalty
                best_cost = 50
                best_robot = None
            tile_robot_costs.append((best_cost, best_robot, tile))

        # Greedy assignment of tiles to robots with minimal cost (simple sum here suffices)
        # (Not solving assignment optimally to keep heuristic lightweight)
        cost_sum = sum(cost for cost, _, _ in tile_robot_costs)

        return cost_sum
