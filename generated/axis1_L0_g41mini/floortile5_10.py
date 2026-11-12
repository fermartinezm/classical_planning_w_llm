from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    A domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the number of actions needed to reach the goal state
    from the current state by approximating the minimal number of:
      - moves required for the robot(s) to reach all tiles that still need to be painted,
      - paint actions needed to paint these tiles,
      - color changes required if the robot's current paint color does not match the goal tile color.

    Assumptions:
    - There may be one or multiple robots, each holding exactly one color at a time.
    - The robot can only paint adjacent tiles (up/down/left/right direction) relative to its current tile.
    - To paint a tile, the tile must be clear (not painted yet) and adjacent to the robot's position.
    - The robot must have the color matching the tile's goal color to paint it; otherwise, it must change color before painting.
    - All tiles that appear in the goal must be painted with the specified color.
    - The heuristic does not guarantee admissibility but aims to guide the greedy best-first search efficiently.

    Heuristic Initialization:
    - Extract goal tiles with their required colors.
    - Extract adjacency relations between tiles (up, down, left, right) to build a graph for movement.
    - Extract the list of robots and their current positions and held colors from initial static info.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all tiles that must be painted as per the goal and not yet painted accordingly in the current state.
    2. For each robot, note its current tile and currently held color.
    3. Construct an undirected graph representing tile adjacency (using up/down/left/right relations).
    4. Approximate the minimum total cost to paint all needed tiles by:
       - For each unpainted tile:
         a) Determine the shortest distance (in moves) from any robot's current position or painted tile position for the robot to get adjacent to the tile.
         b) If the robot color differs from the tile's required color, add one action (color change).
         c) Add one paint action per tile.
    5. To keep the heuristic computation efficient:
       - We approximate by summing independent minimal painting costs per tile ignoring order and conflicts.
       - Movement costs use shortest path distances in the adjacency graph.
       - For multiple robots, choose the minimal cost per tile among robots.
    6. Return zero if there are no tiles needing painting (goal already reached).

    This heuristic balances accuracy with computational efficiency to reduce expanded nodes in search.
    """

    def __init__(self, task):
        # Store goals mapping tile -> desired color
        self.goal_paint = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paint[tile] = color

        # Parse adjacency relations to build undirected graph {tile: set(neighbors)}
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate in ("up", "down", "left", "right") and len(parts) == 3:
                src, dest = parts[1], parts[2]
                # Add undirected edges to adjacency:
                self.adjacency.setdefault(src, set()).add(dest)
                self.adjacency.setdefault(dest, set()).add(src)
        # Some tiles may not be keys yet, ensure all tiles from goals appear
        for tile in self.goal_paint:
            if tile not in self.adjacency:
                self.adjacency[tile] = set()

    def __call__(self, node):
        state = node.state

        # 1) Identify tiles that must be painted and are not yet painted correctly
        tiles_to_paint = {}
        for tile, color in self.goal_paint.items():
            # Check if tile is already painted with desired color
            fact = f"(painted {tile} {color})"
            if fact not in state:
                tiles_to_paint[tile] = color

        # If no tiles to paint, heuristic is 0 (goal reached)
        if not tiles_to_paint:
            return 0

        # 2) Extract robots' info: location and held color
        robots = {}
        # robot-at and robot-has facts are in state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot = parts[1]
                tile = parts[2]
                if robot not in robots:
                    robots[robot] = {"pos": tile, "color": None}
                else:
                    robots[robot]["pos"] = tile
            elif parts[0] == "robot-has":
                robot = parts[1]
                color = parts[2]
                if robot not in robots:
                    robots[robot] = {"pos": None, "color": color}
                else:
                    robots[robot]["color"] = color

        # If no robot info found, no movement possible - return a large number
        if not robots:
            return float('inf')

        # Precompute for all tiles: tiles adjacent to them (needed because painting acts on adjacent tile)
        # We'll precompute adjacency for all tiles.
        # Since adjacency graph is undirected, adjacency already stored.

        # Precompute a map: for each tile T, tiles adjacent to T (neighbors)
        # This is self.adjacency already.

        # 3) For each tile that must be painted, we estimate cost based on minimal cost for some robot to paint it:
        # Actions needed per tile:
        # - Move robot from current position to a tile adjacent to the tile to paint (distance as number of moves)
        # - If robot color != tile color, 1 change_color action
        # - 1 paint action
        # We ignore ordering and possible blocking to remain efficient.

        # 4) Function to compute shortest distance in adjacency graph from a start tile to any of target tiles.
        def shortest_distance(start, targets):
            if start in targets:
                return 0
            visited = set()
            frontier = [(start, 0)]
            visited.add(start)
            while frontier:
                current, dist = frontier.pop(0)
                neighbors = self.adjacency.get(current, set())
                for neigh in neighbors:
                    if neigh not in visited:
                        if neigh in targets:
                            return dist + 1
                        visited.add(neigh)
                        frontier.append((neigh, dist + 1))
            # If no path, return large
            return float('inf')

        total_cost = 0

        for tile, desired_color in tiles_to_paint.items():
            # Tiles adjacent to this tile where a robot must stand to paint it
            adjacent_tiles = self.adjacency.get(tile, set())
            if not adjacent_tiles:
                # No adjacency: impossible to paint tile, infinite cost
                return float('inf')

            # For each robot, compute:
            # - shortest distance from robot position to some tile adjacent to the tile to paint
            # - color change cost: 1 if robot color != desired_color else 0
            # - paint cost: 1
            min_cost_for_tile = float('inf')
            for robot, info in robots.items():
                robot_pos = info["pos"]
                robot_color = info["color"]
                if robot_pos is None or robot_color is None:
                    # Missing info, ignore this robot
                    continue

                dist = shortest_distance(robot_pos, adjacent_tiles)
                if dist == float('inf'):
                    continue  # robot cannot reach tile adjacency

                color_change_cost = 0 if robot_color == desired_color else 1
                paint_cost = 1
                total_tile_cost = dist + color_change_cost + paint_cost
                if total_tile_cost < min_cost_for_tile:
                    min_cost_for_tile = total_tile_cost

            # If no robot can reach adjacency, infinite cost
            if min_cost_for_tile == float('inf'):
                return float('inf')

            total_cost += min_cost_for_tile

        return int(total_cost)
