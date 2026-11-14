from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    
    Example: '(gk9u3vg3 robot1 tile_0_1)' -> ['gk9u3vg3', 'robot1', 'tile_0_1']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    
    - `fact`: The complete fact as a string, e.g., "(yh6q5j8r tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (domain w133ormt).
    
    Summary:
    ----------
    Estimates the minimal number of required actions to reach a goal state by focusing on the number of tile-color assignments missing and the robot(s) movements and paint color switches required to satisfy those goal assignments.
    The heuristic is not guaranteed to be admissible but aims to be accurate and efficient for greedy best-first search.
    
    Assumptions:
    --------------
    - There is one or multiple robots, each always located on exactly one tile.
    - Robots carry exactly one paint color at any time from a finite set of colors.
    - A tile can either be "empty" (d7ekk7vb predicate true) or "painted" with a color (yh6q5j8r predicate).
    - The goal requires certain tiles to be painted with a specific color.
    - Painting a tile requires the robot to be on the tile, carrying a paint color, and the tile being empty.
    - Changing paint color can occur only when the robot is on some tile.
    - Moving robots between tiles respects connectivity predicates: (xr2t1c1d), (oh0smogc), (w224no0l), (g7ya35is).
    - All these connectivity predicates define different forms of adjacency usable for movement actions.
    - Robots can only carry one paint color at any time.
    - Each painting action costs roughly one action.
    - Each move action costs roughly one action.
    - Each color-switch action costs roughly one action.
    
    Heuristic Initialization:
    --------------------------
    - Extract goal tile-to-color assignments (predicates yh6q5j8r).
    - Extract static graph connectivity information between tiles using all adjacency predicates.
      These represent undirected edges between tiles to compute movement distances.
    - Extract all colors available in the domain (with cdb9op3o predicate).
    
    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Determine which tile-color goal assignments are not yet satisfied. Missing goals must be painted.
    2. For each robot: get current position (tile) and current paint color carried.
    3. Construct an undirected graph of tile adjacencies from static facts for shortest path computations.
    4. For each missing goal tile-color (tile, target_color):
        - Heuristic estimates minimal cost for some robot to:
          a) Move from current location to the tile.
          b) Switch paint color if needed (if robot does not carry target_color).
          c) Paint the tile (one action).
        
        - For multiple robots, take the minimum of this cost over all robots.
    5. Sum these minimal costs for all missing tile-color goals.
       This assumes paint actions are independent (ignores possible synergy but provides a reasonable estimate).
       
    6. The heuristic value is zero if all goal goals are satisfied.
    
    This heuristic balances simplicity and informedness by:
    - Using shortest path distances between tiles from the connectivity graph,
    - Including paint color switch costs,
    - Counting minimal moves and painting for each required tile-color assignment.
    """

    def __init__(self, task):
        # Store goal tile-color assignments: dict tile -> color
        self.goal_tile_colors = {}
        for goal in task.goals:
            if match(goal, "yh6q5j8r", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        # Extract static adjacency relations between tiles for pathfinding
        # Adjacency predicates that yield edges between tiles:
        adjacency_predicates = (
            "xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"
        )
        self.adjacent = {}  # tile -> set(adjacent tiles)
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in adjacency_predicates:
                t1, t2 = parts[1], parts[2]
                self.adjacent.setdefault(t1, set()).add(t2)
                self.adjacent.setdefault(t2, set()).add(t1)

        # Extract all robots (objects of type ioytrsy9) by scanning facts with gk9u3vg3 predicate
        # These facts are (gk9u3vg3 robot tile)
        self.robots = set()
        # Extract available paint colors (cdb9op3o color)
        self.colors = set()
        for fact in task.static:
            if match(fact, "cdb9op3o", "*"):
                color = get_parts(fact)[1]
                self.colors.add(color)

        # We'll extract robots during heuristic call from state (positions + carried color)
        # because the robot set might be static or dynamic, but positions/colors are state-dependent.

    def __call__(self, node):
        state = node.state

        # Quick goal check: heuristic is 0 if goal reached
        if self.goal_tile_colors.items() <= {
            (get_parts(f)[1], get_parts(f)[2])
            for f in state if match(f, "yh6q5j8r", "*", "*")
        }:
            return 0

        # Extract robot info: for each robot extract:
        # - current tile location from (gk9u3vg3 robot tile)
        # - current carried color from (i4u94rl9 robot color)
        robots_info = {}
        for fact in state:
            if match(fact, "gk9u3vg3", "*", "*"):
                parts = get_parts(fact)
                robot = parts[1]
                tile = parts[2]
                if robot not in robots_info:
                    robots_info[robot] = {"tile": None, "color": None}
                robots_info[robot]["tile"] = tile
            elif match(fact, "i4u94rl9", "*", "*"):
                parts = get_parts(fact)
                robot = parts[1]
                color = parts[2]
                if robot not in robots_info:
                    robots_info[robot] = {"tile": None, "color": None}
                robots_info[robot]["color"] = color

        # Extract current tile-color painted facts: yh6q5j8r tile color
        painted_tiles = {}
        for fact in state:
            if match(fact, "yh6q5j8r", "*", "*"):
                parts = get_parts(fact)
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        # Extract tiles currently empty (d7ekk7vb tile)
        empty_tiles = set(
            parts[1] for fact in state if match(fact, "d7ekk7vb", "*")
        )

        # Calculate shortest distances between tiles using BFS over adjacency graph
        # To avoid costly computations, cache distances between tiles requested.
        distance_cache = {}

        def shortest_distance(start, goal):
            # Return the shortest number of moves between two tiles
            if start == goal:
                return 0
            key = (start, goal)
            if key in distance_cache:
                return distance_cache[key]
            from collections import deque
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                if current == goal:
                    distance_cache[key] = dist
                    return dist
                visited.add(current)
                for neighbor in self.adjacent.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # unreachable implies no path -> assign large cost
            distance_cache[key] = float('inf')
            return float('inf')

        total_cost = 0

        # We estimate cost for each tile-color goal that is unsatisfied:
        for tile, goal_color in self.goal_tile_colors.items():
            # If tile already painted with goal color, no cost for this tile
            if painted_tiles.get(tile, None) == goal_color:
                continue

            # This tile must be painted: estimate minimal cost for any robot to fulfill it
            best_tile_cost = float('inf')

            for robot, info in robots_info.items():
                robot_tile = info["tile"]
                robot_color = info["color"]

                if robot_tile is None or robot_color is None:
                    # Incomplete info about this robot; skip safely
                    continue

                # Movement cost: distance between current robot tile and target tile
                move_cost = shortest_distance(robot_tile, tile)

                # Paint color switch needed if current color != goal color
                color_switch_cost = 1 if robot_color != goal_color else 0

                # 1 action to paint tile (c1qh1p5g or cadwr4wd painting action equivalent)
                paint_cost = 1

                # Check tile is empty; if not empty, must wait for tile clearing or repainting.
                # Since painting actions in domain require tile empty (d7ekk7vb),
                # if tile is currently empty, cost is as above.
                # If tile is not empty, we must at least remove old paint by painting again or estimate extra cost.
                #
                # Here we conservatively add 1 extra paint action to first clear old paint:
                if tile not in empty_tiles and tile not in painted_tiles:
                    # This case unlikely; fallback cost +1 paint action added
                    paint_cost += 1
                elif tile in painted_tiles and painted_tiles[tile] != goal_color:
                    # Tile is painted with wrong color. Painting action in domain overwrites correctly,
                    # but requires tile empty first. The domain operators allow painting only on d7ekk7vb tile.
                    # So an extra step needed to revert tile to empty before repainting.
                    # However, operators f2q9mxxi, cf57hd49, tgataxn3, ylvh432k seem to swap paint between tiles (propagate),
                    # making tile emptying complicated.
                    #
                    # To keep heuristic efficient, we add 1 extra paint/clear step penalty.
                    paint_cost += 1

                # Sum actions
                robot_cost = move_cost + color_switch_cost + paint_cost

                if robot_cost < best_tile_cost:
                    best_tile_cost = robot_cost

            # If no robot info found, assume very large cost (state infeasible?)
            if best_tile_cost == float('inf'):
                best_tile_cost = 10_000  # large number to avoid problematic states

            total_cost += best_tile_cost

        return total_cost
