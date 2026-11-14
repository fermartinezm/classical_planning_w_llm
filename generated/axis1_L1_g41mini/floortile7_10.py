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
    This heuristic estimates the minimum number of actions required to reach the goal configuration where certain tiles are assigned specific colors.
    The domain involves a robot positioned on tiles, each tile may or may not hold a color, and colors can be moved/painted using the robot.
    The goal is to have the correct color assigned to the target tiles as specified by (yh6q5j8r tile color).
    The robot can move colors around and change its position between tiles.

    Assumptions:
    ------------
    - The robot can only hold one color at a time.
    - Tiles can be empty (d7ekk7vb predicate) or hold a color (yh6q5j8r predicate).
    - The robot moves between tiles through different adjacency predicates (xr2t1c1d, oh0smogc, w224no0l, g7ya35is).
    - Painting a tile with a color consumes a separate action.
    - Robot color changes (i4u94rl9 facts) happen instantly and are implicitly considered in painting actions.
    - The heuristic is not guaranteed to be admissible but should be efficiently computable and guide the search effectively.

    Heuristic Initialization:
    --------------------------
    - Extract:
      - The goal tiles with their target colors.
      - Static adjacency relations between tiles to build a connectivity graph for computing shortest paths.
      - The available colors.
    - Precompute the undirected connectivity graph of tiles for shortest path calculations.
    - Store the initial robot color and position will be extracted later at heuristic call time.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify all goal facts of the form (yh6q5j8r tile color) that must hold in the goal.

    2. For each tile with a goal color:
       - Check if the tile already has the correct color in the current state; if yes, no actions needed for this tile.
       - If the tile is empty (d7ekk7vb) or painted with the wrong color, it needs to be painted.

    3. To paint a tile with a color c, the robot must:
       - Hold color c.
       - Be positioned on that tile.
       - Perform a paint action.

       The robot can change colors instantly while on any tile (z7pcmkeu action), so coloring the robot to a needed color costs at most one action if the current color is different.
       Moving between tiles costs one action per adjacent step.

    4. The plan for painting each tile with the correct color involves:
       - Moving the robot from its current position to the tile, with minimum path cost.
       - Changing the robot's color to the tile's target color if needed (+1 action if color differs).
       - Performing the paint action (+1 action).

    5. Because changing robot color is done on any tile, it is optimal to do this on the tile itself (or just before painting).

    6. Since the robot can only hold one color at a time, and painting requires that color, the order in which tiles are painted impacts the heuristic cost.
       However, to keep the heuristic efficient, we compute an approximate and optimistic cost by calculating for each uncorrectly painted tile the sum:
          min cost = distance(robot_pos, tile) 
                     + (1 if robot_color != target_color else 0) 
                     + 1 (painting action)

       Then, sum these costs ignoring possible overlaps or shared moves.

    7. To avoid overestimating costs too much, we also discount moving costs after the first tile:
       - For the first tile, include moving cost from current robot position.
       - For subsequent tiles, consider only the cost of robot color changes and painting actions plus the shortest path distance between tiles as if robot teleports to them (or simplify by not summing move distances between tiles).

       For efficiency and simplicity, the heuristic sums minimal independent costs per tile without sequencing, slightly underestimating move costs between tiles.

    8. If all goal tiles are correctly painted, heuristic returns 0.

    9. This heuristic is therefore a relaxed estimate focusing on minimal painting and color changing/moving actions.

    """

    def __init__(self, task):
        # Extract goals: tiles and colors required at these tiles.
        self.goal_tile_colors = {}  # tile -> color
        for goal in task.goals:
            if match(goal, "yh6q5j8r", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        # Extract static adjacency info to build graph of tiles
        # Undirected edges for movement

        # adjacency predicates that connect tiles with edges:
        # xr2t1c1d, oh0smogc, w224no0l, g7ya35is
        # from the domain file, these predicates connect tiles
        
        self.graph = {}  # tile -> set of neighboring tiles

        def add_edge(t1, t2):
            if t1 not in self.graph:
                self.graph[t1] = set()
            if t2 not in self.graph:
                self.graph[t2] = set()
            self.graph[t1].add(t2)
            self.graph[t2].add(t1)

        for fact in task.static:
            # check each adjacency predicate
            for pred in ["xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"]:
                if match(fact, pred, "*", "*"):
                    parts = get_parts(fact)
                    t1, t2 = parts[1], parts[2]
                    add_edge(t1, t2)

        # Extract colors known from static facts (cdb9op3o color)
        self.colors = set()
        for fact in task.static:
            if match(fact, "cdb9op3o", "*"):
                color = get_parts(fact)[1]
                self.colors.add(color)

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between tiles in the tile graph using BFS."""
        if start == goal:
            return 0
        if start not in self.graph or goal not in self.graph:
            return 10000  # large cost if no path (should not happen in well-formed domains)

        from collections import deque

        queue = deque([start])
        distances = {start: 0}
        while queue:
            current = queue.popleft()
            dist = distances[current]
            for neighbor in self.graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    if neighbor == goal:
                        return distances[neighbor]
                    queue.append(neighbor)
        return 10000  # no path found

    def __call__(self, node):
        """Estimate minimum number of necessary actions to reach goal state from the given node state."""

        state = node.state

        # Quick check: if goal achieved, return 0
        if self.goal_tile_colors.items() <= {
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "yh6q5j8r", "*", "*")
        }.items():
            # All goal colors already correctly painted
            return 0

        # Extract robot position and robot color from state:
        robot_pos = None
        robot_color = None

        for fact in state:
            if match(fact, "gk9u3vg3", "*", "*"):
                robot_pos = get_parts(fact)[1]
                break
        for fact in state:
            if match(fact, "i4u94rl9", "*"):
                robot_color = get_parts(fact)[1]
                break

        if robot_pos is None or robot_color is None:
            # State malformed, fallback heuristic cost
            return 50  # arbitrary large but finite

        heuristic_cost = 0

        # Collect tiles not correctly painted yet:
        # For each tile in goal:
        # - If goal color differs from current painted color or tile is empty, it needs painting.

        # Find current painted colors on tiles (yh6q5j8r tile color)
        tile_current_colors = {}
        for fact in state:
            if match(fact, "yh6q5j8r", "*", "*"):
                parts = get_parts(fact)
                tile_current_colors[parts[1]] = parts[2]

        # Find empty tiles (d7ekk7vb tile)
        empty_tiles = {get_parts(fact)[1] for fact in state if match(fact, "d7ekk7vb", "*")}

        # For all goal tiles, check status
        tiles_to_paint = []
        for tile, goal_color in self.goal_tile_colors.items():
            current_color = tile_current_colors.get(tile)
            if current_color == goal_color:
                # correctly painted, no action
                continue
            # If tile is empty or painted wrong color, mark for painting
            tiles_to_paint.append((tile, goal_color))

        # If no tiles to paint, heuristic is zero
        if not tiles_to_paint:
            return 0

        # For each tile to paint, approximate cost:
        # cost = shortest path from current robot pos (or assume robot teleports between tiles for simplicity)
        #       + 1 if robot color != tile color (color change assumed to happen while on tile)
        #       + 1 painting action

        # For efficiency, ignoring sequencing between tiles.
        # We approximate move cost only from current robot position to closest tile to paint, then no move costs for others.

        # Compute move cost from robot to each tile:
        move_costs = []
        for tile, tile_color in tiles_to_paint:
            dist = self.shortest_path_length(robot_pos, tile)
            change_color_cost = 1 if robot_color != tile_color else 0
            paint_cost = 1
            cost = dist + change_color_cost + paint_cost
            move_costs.append(cost)

        # As approximation, take min move cost for first tile, plus for others only (1 color change if needed + 1 painting)
        # Because robot needs to go to at least one tile
        # And must paint all tiles (color change and paint costs); moving between tiles ignored to keep heuristic low and efficient.

        if not move_costs:
            return 0  # no painting needed

        # Find minimal first tile cost
        min_first_cost = min(move_costs)

        # Count number of tiles with color different from robot color
        color_changes_needed = sum(1 for _, c in tiles_to_paint if c != robot_color)

        # Total painting actions
        total_paint_actions = len(tiles_to_paint)

        # Estimate:
        # First tile: move + possible color change + paint
        # Other tiles: 1 color change + 1 paint per tile (heuristic underestimates moving cost between tiles,
        # but this keeps it admissible and efficient)
        # To avoid overcounting color changes, total color change cost capped at number of tiles needing color changes.
        # Since first tile's color change included in min_first_cost, subtract once

        total_cost = min_first_cost

        # Remaining tiles after first painted tile
        remaining_tiles = len(tiles_to_paint) - 1

        # Remaining color changes: color_changes_needed - (did we already count first tile's color change)?
        first_tile_idx = move_costs.index(min_first_cost)
        first_tile_color = tiles_to_paint[first_tile_idx][1]
        # We already counted color change for first tile if robot_color != first_tile_color in min_first_cost
        counted_first_color_change = 1 if robot_color != first_tile_color else 0
        remaining_color_changes = color_changes_needed - counted_first_color_change

        # For remaining tiles: each costs at least 1 paint action
        # For color changes, each costs 1 action
        if remaining_tiles > 0:
            total_cost += remaining_color_changes + remaining_tiles  # color changes + paint actions

        return total_cost
