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
    A domain-dependent heuristic for the secret_domain (w133ormt).

    Summary:
    This heuristic estimates the number of actions required to paint all target tiles with the goal colors.
    The domain involves a robot moving on a grid of tiles, each tile has a color which can be changed by the robot.
    The robot can move between adjacent tiles (using directional connectivity predicates) and paint tiles with specific colors.
    The heuristic measures the distance and painting effort required to cover all goal paint assignments from the current state.

    Assumptions:
    - The robot occupies exactly one tile at any time.
    - The robot has an assigned current paint color.
    - Tiles are identified as either "empty" (d7ekk7vb predicate true) or painted with some color (yh6q5j8r predicate).
    - Painting an unpainted tile or repainting a tile with another color requires one paint action per tile.
    - The robot can move along the directional graph defined by predicates (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
      These represent directional edges, allowing adjacency exploration.
    - The heuristic counts minimal moves to reach tiles needing painting plus one paint action per tile.
    - It does not require admissibility but tries to provide an informed estimate that reduces node expansions.
    - Multiple robot objects may be present, but the heuristic considers only the first encountered robot's position and color.

    Heuristic Initialization:
    - Extract goal tile-color assignments from the task goals condition (yh6q5j8r tile color).
    - Extract static adjacency relations from static predicates for efficient path cost estimation.
      Specifically, build a directed graph mapping each tile to adjacent tiles reachable by one move.
    - Extract all colors available (cdb9op3o).
    - Store object types for tiles (bffr59sc) and robots (ioytrsy9).
    - Prepare fast lookup structures for:
       - Tiles that must be painted with a target color.
       - Adjacencies (graph) between tiles for robot movement.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the robot's current tile and color from the state.
       (From predicates gk9u3vg3 robot tile and i4u94rl9 robot color)
    2. For each tile in the goal with a required color:
       a) Check if the tile already has the goal color painted (yh6q5j8r tile color).
          If yes, no action needed there.
          If no (unpainted or painted differently), mark tile as needing painting.
    3. Compute shortest path distances on the tile adjacency graph from the robot's current tile to each tile needing painting.
       Use a breadth-first search on the directed graph.
    4. The heuristic estimate sums:
       - The minimal number of moves to reach each tile that needs painting starting from the robot's position,
         approximated by the sum of distances visiting each tile in an optimistic incremental manner.
       - One paint action per tile needing painting.
       - One color change action if robot's color differs from the needed paint color.
         (This can be roughly counted as the number of color changes needed if the robot has to paint tiles of multiple colors;
          for simplification assume worst case: one color change at start if robot color is different from the first needed paint color.)
    5. Since the domain allows repainting tiles, color changes can be considered costly, so the heuristic "penalizes" color switches minimally.
       However, to remain efficient, the heuristic accounts only for one color change if needed.
    6. The heuristic returns 0 if the state satisfies all goals.

    Notes:
    - To keep efficiency, the heuristic uses BFS for distances and simple counting strategies.
    - The heuristic is non-admissible but informed enough to reduce needless expansions.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all goal paint assignments: tiles to required colors.
        self.goal_paints = {}  # tile -> color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "yh6q5j8r":
                tile = parts[1]
                color = parts[2]
                self.goal_paints[tile] = color

        # Build an adjacency graph of tiles for robot movement from static facts.
        # The directional edges are given by predicates:
        # xr2t1c1d, oh0smogc, w224no0l, g7ya35is (all from BFFR59SC to BFFR59SC)
        # These define directed edges: predicate(tile_from tile_to)
        self.adj = {}  # tile -> set of adjacent reachable tiles
        relevant_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Initialize adjacency dict for all tiles appearing in static facts.
        tiles = set()
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in relevant_preds:
                src = parts[1]
                dst = parts[2]
                tiles.add(src)
                tiles.add(dst)

        for t in tiles:
            self.adj[t] = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in relevant_preds:
                src = parts[1]
                dst = parts[2]
                self.adj[src].add(dst)

        # Extract all tiles and colors for quick membership tests.
        # Tiles are gathered from static or from goals, union for safety.
        self.all_tiles = set(tiles)
        self.all_tiles.update(self.goal_paints.keys())

        # Extract available colors (used by cdb9op3o predicate in static facts)
        self.colors = {get_parts(f)[1] for f in self.static if get_parts(f)[0] == "cdb9op3o"}

    def __call__(self, node):
        state = node.state

        # Quick goal test: if all the goal paint facts hold, heuristic = 0.
        if self.goal_paints.items() <= {
            (get_parts(f)[1], get_parts(f)[2]) for f in state if get_parts(f)[0] == "yh6q5j8r"
        }.items():
            return 0

        # Identify robot(s) in state: for heuristic just pick one robot.
        # Find robot tile and robot current color.
        robot_tile = None
        robot_color = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "gk9u3vg3" and parts[1].startswith("robot"):
                # robot position fact (gk9u3vg3 robot tile)
                robot_tile = parts[2]
            elif parts[0] == "i4u94rl9" and parts[1].startswith("robot"):
                # robot color fact (i4u94rl9 robot color)
                robot_color = parts[2]
            if robot_tile is not None and robot_color is not None:
                break

        # If robot not found (should not happen), return a large heuristic to avoid expansions.
        if robot_tile is None or robot_color is None:
            return 10000

        # Identify tiles that need painting with a different color.
        # A tile needs painting if:
        # - it is in goal_paints,
        # - current tile's color in state differs or is unpainted,
        # where unpainted = (d7ekk7vb tile) predicate true means empty.
        # We extract the current tile colors from yh6q5j8r predicates in state.

        tile_current_color = {}
        tiles_empty = set()

        # Parse current tile colors and empties.
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yh6q5j8r":
                tile_current_color[parts[1]] = parts[2]
            elif pred == "d7ekk7vb":
                tiles_empty.add(parts[1])

        tiles_to_paint = []

        for tile, goal_color in self.goal_paints.items():
            current_c = tile_current_color.get(tile, None)
            # If tile empty or colored differently than goal -> needs painting
            if (tile in tiles_empty) or (current_c != goal_color):
                tiles_to_paint.append((tile, goal_color))

        # If nothing to paint, heuristic is zero (should have caught above already).
        if not tiles_to_paint:
            return 0

        # Precompute shortest distances from robot tile to all tiles needing painting using BFS on self.adj graph.
        # Compute all distances from robot_tile to reachable tiles.
        dist = {robot_tile: 0}
        queue = [robot_tile]
        while queue:
            current = queue.pop(0)
            current_dist = dist[current]
            for nbr in self.adj.get(current, ()):
                if nbr not in dist:
                    dist[nbr] = current_dist + 1
                    queue.append(nbr)

        # For each tile to paint, get distance from robot position; if unreachable, assign large cost.
        # Sum heuristic as sum(distance to tile + 1 paint action) -- assume optimal path visiting tiles independently.
        # Also, check if robot color differs from the color needed by that tile.
        # Count 1 color change if a differing color occurs anywhere (simplification).

        total_moves = 0
        color_changes_needed = False

        # For heuristic efficiency, visit tiles greedily by distance:
        # Sort tiles by distance ascending (those closer counted first).
        # Accumulate total moves as sum of min moves between tiles (approximate by summing from robot to each tile).
        # This is non-admissible but efficient.

        # To approximate multi-tile visiting cost better,
        # we approximate cost as:
        #   minimal distance to closest tile +
        #   number of tiles to paint - 1 (for moves between tiles heuristically)
        #   + painting cost (one paint action per tile)
        #   + at most one color change if needed

        # Find closest tile distance:
        tile_distances = []
        for tile, color in tiles_to_paint:
            d = dist.get(tile, 1000)  # large if unreachable
            tile_distances.append((d, tile, color))

        tile_distances.sort()

        if tile_distances[0][0] >= 1000:
            # No reachable tile to paint -> large heuristic (unsolvable)
            return 10000

        closest_dist = tile_distances[0][0]

        # Check if any tile requires painting with color different from robot color to add color change cost.
        for _, _, color in tile_distances:
            if color != robot_color:
                color_changes_needed = True
                break

        # Heuristic calculation:
        # Moves:
        #   move from robot tile to first tile needing paint: closest_dist
        #   plus minimal moves between the other tiles roughly: number of tiles_to_paint - 1
        # Painting:
        #   1 paint action per tile needing painting
        # Color changes:
        #   1 color switch action if needed (non-admissible but simple)

        moves_cost = closest_dist + (len(tile_distances) - 1)
        paint_cost = len(tile_distances)
        color_change_cost = 1 if color_changes_needed else 0

        heuristic_value = moves_cost + paint_cost + color_change_cost

        # Return finite heuristic estimate
        return heuristic_value
