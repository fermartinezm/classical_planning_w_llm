from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state by
    considering the robot(s) moving on the tile grid and painting tiles to their goal colors.
    It accounts for:
    - The minimal movement cost of robot(s) from their positions to tiles that still need painting.
    - The number of paint actions required (one paint per tile).
    - The number of color changes needed by each robot to paint all tiles in the correct colors.

    # Assumptions
    - Moving between adjacent tiles costs one move action.
    - Painting a tile costs exactly one paint action.
    - Changing the robot's held color costs one color change action.
    - Only one robot can paint at one tile at a time but multiple robots paint independently.
    - The grid connections (up/down/left/right) define adjacent tiles.
    - Each robot can have exactly one color at a time.
    - The set of available colors is fixed and known.
    - Clear tiles mean unoccupied (and paintable) tiles.
    - The heuristic is not necessarily admissible but efficiently computable to guide greedy best-first search.

    # Heuristic Initialization
    - Extract:
      * The grid topology from static facts (adjacency relations: up/down/left/right) as a graph for shortest paths.
      * The set of all tiles.
      * The available colors in the domain.
    - Precompute shortest path lengths between all pairs of tiles using the grid adjacency.
    - Extract robots present in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Given a state:
    1. Identify the goal tiles and their goal colors from the goal conditions.
    2. For each tile in the goal, check if it is already painted with the correct color.
       - If yes, no further action needed on that tile.
       - If not, this tile must be painted.
    3. Gather the current positions and colors held by each robot.
    4. For each tile that requires painting, estimate the minimal cost for any robot to reach and paint it:
       - Movement cost: Use precomputed shortest path distances.
       - Painting cost: 1 action.
       - If robot's held color differs from the tile's goal color, add 1 for color change.
    5. Assign each paint task to the robot with minimal total cost for that tile.
    6. Sum over all tiles the minimal costs obtained.
    7. To avoid underestimating the cost when multiple tiles require different paint colors, add an estimate of minimal number of color changes each robot needs to make for tiles assigned to it.
       This can be roughly bounded by the number of different colors assigned to that robot minus one.
    8. The heuristic value is the total sum of:
       - Movement cost to tiles.
       - Paint actions.
       - Necessary color changes per robot.
    9. If no tile needs painting (goal reached), heuristic is 0.

    This approach balances accuracy and efficiency by using precomputed distances and greedy assignment, without exhaustive search.

    """

    def __init__(self, task):
        """Initialize heuristic by processing static info and goals."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "available-color", "*")
        }

        # Extract grid adjacency for tiles from up/down/right/left predicates
        # We'll construct an adjacency dict: tile -> set of adjacent tiles
        self.adj = dict()

        # Extract all tiles first for initialization
        tiles = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] in {"up", "down", "left", "right"}:
                tiles.add(parts[1])
                tiles.add(parts[2])
        for tile in tiles:
            self.adj[tile] = set()

        # Fill adjacency
        for fact in self.static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "up":
                self.adj[parts[1]].add(parts[2])
                self.adj[parts[2]].add(parts[1])
            elif pred == "down":
                self.adj[parts[1]].add(parts[2])
                self.adj[parts[2]].add(parts[1])
            elif pred == "left":
                self.adj[parts[1]].add(parts[2])
                self.adj[parts[2]].add(parts[1])
            elif pred == "right":
                self.adj[parts[1]].add(parts[2])
                self.adj[parts[2]].add(parts[1])

        # Precompute shortest distances between tiles using BFS for efficiency
        # Store distances in dict-of-dicts: distances[tile1][tile2] = minimum steps
        self.distances = dict()
        for tile in tiles:
            self.distances[tile] = self._bfs_distances(tile)

        # Extract list of robots (assuming their names are in facts mentioning (robot-at ?r ?x) or (robot-has ?r ?c))
        # Collect all robots from static + goals + task facts (better to get from domain facts + initial state)
        # In floortile domain robots only appear in init and state, not static facts, so we get robots lazily in __call__

    def _bfs_distances(self, start_tile):
        """Compute shortest path distances from start_tile to all reachable tiles using BFS."""
        distances = {start_tile: 0}
        queue = [start_tile]
        while queue:
            current = queue.pop(0)
            for neighbor in self.adj[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute heuristic estimate of number of actions to complete painting according to goals.

        Steps:
        1) Identify tiles and goal colors from goals.
        2) Check current painted colors.
        3) Find tiles needing painting (unpainted or wrongly painted).
        4) Extract robots, their positions and held colors.
        5) Assign tiles to robots minimizing cost = move + paint + color change.
        6) Sum costs with an adjustment for color changes per robot.
        7) Return 0 if goal is reached.
        """
        state = node.state

        # Extract goal tiles and required paint colors
        goal_tiles_colors = dict()
        for goal_fact in self.goals:
            if match(goal_fact, "painted", "*", "*"):
                parts = get_parts(goal_fact)
                tile, color = parts[1], parts[2]
                goal_tiles_colors[tile] = color

        # If no painting goals, heuristic is 0
        if not goal_tiles_colors:
            return 0

        # Extract currently painted tiles and colors
        painted_tiles_colors = dict()
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                painted_tiles_colors[tile] = color

        # Identify tiles that still need painting or repainting
        tiles_to_paint = []
        for tile, goal_color in goal_tiles_colors.items():
            current_color = painted_tiles_colors.get(tile)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))

        # If all painted correctly, heuristic is 0
        if not tiles_to_paint:
            return 0

        # Extract robots: positions and held colors
        robots = dict()  # robot_name -> {"pos": tile, "color": color}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, pos = parts[1], parts[2]
                robots.setdefault(robot, {})["pos"] = pos
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                robots.setdefault(robot, {})["color"] = color

        # Defensive: ignore robots missing position or color (should not occur)
        robots = {r: info for r, info in robots.items() if "pos" in info and "color" in info}

        # If no robots found, return high cost (cannot paint)
        if not robots:
            return float('inf')

        # For fast access, build list of robots as tuples: (name, pos, color)
        robot_list = [(r, info["pos"], info["color"]) for r, info in robots.items()]

        # Estimate costs for assigning each tile to a robot:
        # cost = movement cost (distance) + paint (1) + color change if needed (1)
        # For movement, distance is from robot's position to tile, or large if unreachable.

        # For efficiency, store for each robot the cost to reach each tile
        # We'll assign tiles greedily to robots minimizing their individual tile cost.

        # Prepare mapping: robot -> list of assigned tiles, and colors assigned
        robot_assigned_tiles = {r: [] for r, _, _ in robot_list}

        # Sum of costs
        total_cost = 0

        # For each tile to paint, find robot with minimal cost
        for tile, tile_color in tiles_to_paint:
            best_cost = float("inf")
            best_robot = None
            for r, pos, rcolor in robot_list:
                dist_map = self.distances.get(pos)
                if dist_map is None:
                    # robot position not in graph? skip this robot
                    continue
                dist = dist_map.get(tile)
                if dist is None:
                    # tile unreachable by this robot in graph? skip
                    continue
                move_cost = dist
                paint_cost = 1
                color_change_cost = 0 if rcolor == tile_color else 1
                cost = move_cost + paint_cost + color_change_cost
                if cost < best_cost:
                    best_cost = cost
                    best_robot = r
            if best_robot is None:
                # no robot can reach tile? heuristic infinite
                return float("inf")
            robot_assigned_tiles[best_robot].append(tile_color)
            total_cost += best_cost

        # Add extra cost for multiple color changes per robot greedily:
        # Minimal additional color changes per robot = (number of distinct colors assigned - 1)
        # since robot starts with one color, needs to change for every new assigned color except the first
        for r, tile_colors in robot_assigned_tiles.items():
            if not tile_colors:
                continue
            distinct_colors = set(tile_colors)
            # robot currently holds some color, deduct one since no color change for first assigned color if matches
            current_color = robots[r]["color"]
            if current_color in distinct_colors:
                color_change_extra = len(distinct_colors) - 1
            else:
                # First change to any assigned color costs 1, so total changes = number colors assigned
                color_change_extra = len(distinct_colors)
            # We've already added one color change cost per tile if color needed changing in the previous sum,
            # so this is an approximate adjustment to account for costs of switching back and forth.
            # To avoid overcount, reduce total cost by color_change_extra (already counted one per tile)
            # but since multiple tiles with same color incur multiple changes (actually one change per color cluster),
            # this heuristic balances by augmenting by the minimal needed color switches.
            # We will add color_change_extra here as an optimistic estimate of extra changes needed.
            total_cost += color_change_extra

        return total_cost
