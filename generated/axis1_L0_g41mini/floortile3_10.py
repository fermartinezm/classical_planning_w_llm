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
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    Summary
    -------
    This heuristic estimates the minimum number of actions required to reach a goal state
    where all specified tiles in the goal are painted with the correct color by one or more robots.
    It considers the movements of robots to tiles that need painting, possible color changes of robots,
    and the painting actions required.

    Assumptions
    -----------
    - The goal consists of conjunctive painted predicates specifying tile-color pairs.
    - Robots can paint only adjacent tiles from their current location and must hold the correct color.
    - Robots can change color only if they currently hold some color and the target color is available.
    - Tiles that are already painted with the correct color do not require action.
    - The map topology (adjacencies: up, down, left, right) is given as static facts.
    - Robots cannot move onto non-clear tiles.
    - Each painting action consumes one action; moving from one tile to an adjacent tile consumes one action.
    - Changing a robot's paint color consumes one action.

    Heuristic Initialization
    ------------------------
    - Extract the goal tile-color pairs that must be painted.
    - Extract static adjacency relations (up, down, left, right) between tiles and store as a graph (undirected).
    - Extract the set of available colors.
    - Extract the set of robots.
    - Precompute shortest distances between all tiles using the adjacency graph for efficient movement cost estimation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which tiles in the goal are not yet painted correctly in the current state.
    2. From the state, find each robot’s current position and the color it holds.
    3. For each tile that needs painting:
       a. Compute the minimum cost for any robot to paint that tile:
          - If robot holds the correct color:
            - Cost = minimal movement cost from robot position to a tile adjacent to the tile to paint 
              (since painting can only be done from adjacent tile) + 1 (painting action).
          - If robot holds a wrong color:
            - Cost = minimal movement cost to adjacent tile + 1 (color change) + 1 (painting action).
       b. Take the minimal such cost over all robots.
    4. Sum up these minimal costs for all tiles.
    5. The heuristic is the sum of these minimal estimated action sequences.
    6. If no tiles remain to be painted, heuristic value is 0.

    Notes:
    - Movement cost to a tile adjacent to target is computed as shortest path to any of that tile’s neighbors.
    - The heuristic is not admissible but aims at reducing search expansions by well-informed estimation.
    - Color changes are counted once per painting action if needed.
    """

    def __init__(self, task):
        """Initialize by extracting goals, static adjacencies, available colors, robots, and precompute shortest paths."""
        self.goals = task.goals  # goal facts
        self.static = task.static  # static facts

        # Extract goal painted tile-color pairs: dict tile -> color
        self.goal_tiles = {}
        for fact in self.goals:
            # fact format: (painted tile color)
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        # Extract adjacency relations from static facts and build undirected graph: tile -> set(adjacent tiles)
        self.adjacencies = {}
        for fact in self.static:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate in {"up", "down", "left", "right"}:
                src, dest = parts[1], parts[2]
                self.adjacencies.setdefault(src, set()).add(dest)
                self.adjacencies.setdefault(dest, set()).add(src)

        # Gather all tiles from adjacency keys
        self.tiles = set(self.adjacencies.keys())

        # Precompute shortest distances between tiles using BFS for fast lookup during heuristic evaluation
        self._all_pairs_shortest_paths = self._compute_all_pairs_shortest_paths()

        # Extract available colors from static facts
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

        # Extract robot names from task facts (looking at robot-at)
        robots_set = set()
        for fact in self.static | task.initial_state | self.goals:
            if match(fact, "robot-at", "*", "*"):
                robots_set.add(get_parts(fact)[1])
            elif match(fact, "robot-has", "*", "*"):
                robots_set.add(get_parts(fact)[1])
        self.robots = robots_set

    def _compute_all_pairs_shortest_paths(self):
        """Compute shortest paths between all tiles using BFS on adjacency graph."""
        distances = {}

        for start in self.tiles:
            distances[start] = {}
            visited = {start}
            queue = [(start, 0)]
            while queue:
                current, dist = queue.pop(0)
                distances[start][current] = dist
                for neighbor in self.adjacencies.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return distances

    def __call__(self, node):
        """Calculate heuristic estimate for the node's state."""
        state = node.state

        # Helper: get robot positions and colors
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                pred, arg1, arg2 = parts
                if pred == "robot-at":
                    robot_positions[arg1] = arg2
                elif pred == "robot-has":
                    robot_colors[arg1] = arg2

        # If goal reached, heuristic is zero
        if all(match(f"(painted {tile} {color})") in state or f"(painted {tile} {color})" in state
               for tile, color in self.goal_tiles.items()):
            return 0

        # Find all tiles that are not yet painted correctly (need painting)
        tiles_to_paint = []
        for tile, color in self.goal_tiles.items():
            painted_fact = f"(painted {tile} {color})"
            # not painted with correct color yet
            if painted_fact not in state:
                tiles_to_paint.append((tile, color))

        if not tiles_to_paint:
            return 0  # all done

        # For each tile to paint, find the minimal cost for any robot to paint it
        total_cost = 0

        # Precompute adjacent tiles for all tiles needing paint: these are the tiles from which a robot can paint the tile
        # painting a tile requires robot at an adjacent tile (up/down/left/right)
        adjacent_to_tile = {}
        for tile, _ in tiles_to_paint:
            neighbors = self.adjacencies.get(tile, set())
            adjacent_to_tile[tile] = neighbors

        # For each tile needing paint:
        for tile, desired_color in tiles_to_paint:
            minimal_paint_cost_for_tile = None

            for robot in self.robots:
                robot_pos = robot_positions.get(robot, None)
                robot_color = robot_colors.get(robot, None)

                if robot_pos is None or robot_color is None:
                    continue  # incomplete info about robot

                # We must move robot to one of the tile's adjacent tiles (for painting)
                # Compute minimal distance from robot_pos to any adjacent tile of the tile to paint
                adj_tiles = adjacent_to_tile[tile]
                if not adj_tiles:
                    # No adjacency => isolated tile => cannot paint; ignore this tile (should not occur in valid tasks)
                    continue

                # Find shortest distance from robot_pos to any adj_tile
                distances_from_robot = self._all_pairs_shortest_paths.get(robot_pos, {})
                # Filter only adj_tiles reachable from robot_pos
                reachable_adj_tiles = [t for t in adj_tiles if t in distances_from_robot]
                if not reachable_adj_tiles:
                    # No reachable adjacent tiles for this tile from this robot position
                    continue

                min_move_cost = min(distances_from_robot[adj] for adj in reachable_adj_tiles)

                # Calculate cost components
                # Move cost: min_move_cost (moving robot to adjacent tile)
                # Color change cost: 0 if robot_color == desired_color else 1 (one change_color action)
                # Paint cost: always 1 (paint_xxx action)
                color_change_cost = 0 if robot_color == desired_color else 1
                paint_cost = 1

                cost = min_move_cost + color_change_cost + paint_cost

                if (minimal_paint_cost_for_tile is None) or (cost < minimal_paint_cost_for_tile):
                    minimal_paint_cost_for_tile = cost

            # If no robot can paint (should not happen in valid task), assume large cost to guide search
            if minimal_paint_cost_for_tile is None:
                minimal_paint_cost_for_tile = 1000

            total_cost += minimal_paint_cost_for_tile

        return total_cost
