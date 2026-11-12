from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' allowed.

    - fact: a PDDL fact string, e.g. "(available-color color1 tile2)"
    - args: a sequence of patterns, e.g. "available-color", "*", "tile2"
    Returns True if all parts match.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (color) domain.

    Summary:
    This heuristic estimates the number of required actions to achieve all
    painting goals (i.e. all required tiles painted). The domain involves robots
    of different "color" types moving along tiles, changing their available position
    on tiles (via change_color), moving down, and painting tiles that are "down".
    The goal is to have certain tiles painted.

    Assumptions:
    - Each tile-2 that must be painted is initially down.
    - Painting a tile requires the robot to be present at a color tile that is available,
      and the tile-2 to be available on the robot's tile.
    - Movement of robots along tiles is possible only by moving "up" along tiles (via change_color)
      and "down" (via move_down).
    - The "available-color" facts indicate where a robot or tile currently is available.
    - Robots at certain colors can paint down tiles if free-color predicate holds for the color.
    - Painting a tile requires one paint_up action per tile.
    - Moving a robot to the correct tile and then painting is necessary.
    - Changing color availability moves a robot along tiles in an "up" relation.

    Heuristic Initialization:
    - Extract from static facts all "up" relations among tiles to represent the tile graph.
    - Extract the set of tiles that are down and require painting.
    - Extract all robot and color objects from initial state.
    - Extract goals (painted tile-2 facts).
    - Prepare data structures for quick distance computation and lookup.
    - Precompute adjacency graph for robot movement (change_color) based on "up" relations.
    - Distances on tiles for robot movements are approximated by shortest path on "up" edges.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all tiles that must be painted according to the goal but are not yet painted.
    2. For each tile-2 that must be painted, determine the tile (tile-2) related down tile it corresponds to.
       Note: Painted predicate applies only to tile-2; painting requires certain "available-color"
       on tile and robot-at color predicates.
    3. For each robot currently at some location, compute the minimal tile distance to reach the tile of
       the tile-2 that must be painted.
    4. Estimate movement cost for the robot to get to a suitable tile where it can paint, assuming:
       - One "change_color" action per step along "up" edges.
       - Possibly one "move_down" action needed to reach correct tile.
       - One "paint_up" action to paint the tile.
    5. If multiple robots exist, assign each painting task to the closest robot movement cost (minimize
       heuristic by considering best robot).
    6. Sum costs over all tiles-2 that must be painted and are currently unpainted.
    7. The heuristic ignores effects ordering and concurrency and does not guarantee admissibility,
       but tries to keep estimates efficient and informed by shortest tile moves.

    Implementation notes:
    - The heuristic uses shortest path search (BFS) on "up" tile graph to estimate move costs.
    - The heuristic accounts for the need of available-color predicates (robots and tiles)
      moving along the tile graph.
    - Heuristic returns 0 if goal is already achieved (all tiles painted).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract and store "up" edges: key=tile, value=set of tiles above it.
        self.up_graph = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                parts = get_parts(fact)
                below, above = parts[1], parts[2]
                self.up_graph.setdefault(below, set()).add(above)

        # Reverse adjacency for downward moves (to allow BFS in two directions).
        self.down_graph = {}
        for below, aboves in self.up_graph.items():
            for above in aboves:
                self.down_graph.setdefault(above, set()).add(below)

        # Extract tiles and tile-2 sets:
        self.all_tiles = set()
        self.all_tile_2 = set()
        for fact in self.static:
            parts = get_parts(fact)
            # From type declarations, tile and tile-2 are distinct.
            # But only "up" uses tile? We'll track all tiles seen.
            if parts[0] == 'up':
                self.all_tiles.add(parts[1])
                self.all_tiles.add(parts[2])
            elif parts[0] == 'down':
                self.all_tile_2.add(parts[1])

        # Precompute set of goals that require painted(tile-2)
        self.paint_goals = {
            get_parts(g)[1] for g in self.goals if g.startswith("(painted ")
        }

    def _bfs_distance(self, start_tiles, target_tiles):
        """
        Compute shortest distance from any tile in start_tiles to any tile in target_tiles
        walking up along "up" edges. 

        Returns minimum distance (int steps) or None if no path exists.
        """
        from collections import deque

        visited = set()
        queue = deque([(tile, 0) for tile in start_tiles])

        while queue:
            current, dist = queue.popleft()
            if current in target_tiles:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.up_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """
        Estimate the minimal number of actions needed to paint all target tiles.

        Actions to consider per tile to paint:
        - Moving the robot (available-color) from current tile(s) to the tile required for painting:
          * Estimation uses shortest path on up-edges.
          * Each change_color action moves availability up one tile.
          * move_down action needed if final painting requires robot to move down.
        - One paint_up action per tile-2 to paint.
        - Assume robots can share work; assign tile painting to robot minimizing distance.

        Returns 0 if goal achieved, otherwise sum of estimated actions.
        """
        state = node.state

        # If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Which tiles-2 remain to paint:
        unpainted_tiles_2 = {tile2 for tile2 in self.paint_goals if f"(painted {tile2})" not in state}

        if not unpainted_tiles_2:
            # No unpainted tiles-2 remain, cost 0.
            return 0

        # Extract relevant facts from current state:

        # Find current "available-color" facts for robots and tiles.
        # available-color facts appear to be between a robot/color object and a tile.
        # We'll map robot/color objects to their currently available tiles.
        robot_tiles = {}
        tile_available_positions = {}  # Tiles' available positions (used below)
        for fact in state:
            if match(fact, "available-color", "*", "*"):
                parts = get_parts(fact)
                obj, tile = parts[1], parts[2]
                # We consider obj is a robot or color-2 object (like color1) or even tile?
                # Because the example uses values like (available-color color1 tile4) or (available-color tile1 tile4),
                # we filter those starting with "color" or "color-2" or "tile".
                # Approximate by checking if obj starts with "color" or "tile".
                if obj.startswith("color") or obj.startswith("color1") or obj.startswith("color2") or obj.startswith("color3") or obj.startswith("color4") or obj.startswith("color5") or obj.startswith("color6") or obj.startswith("color7") or obj.startswith("color8") or obj.startswith("color9") or obj.startswith("color10") or obj.startswith("color11"):
                    # Treat as robot/color-2 object
                    robot_tiles.setdefault(obj, set()).add(tile)
                else:
                    # Possibly tiles' available positions - used for completeness
                    tile_available_positions.setdefault(obj, set()).add(tile)

        # Extract robots (color-2) objects from robot_tiles keys
        robots = list(robot_tiles.keys())
        if not robots:
            # No robots found - cannot paint - return large heuristic
            return 10000

        # Extract robot-at facts: (robot-at ?v1 - color-2 ?v3 - color)
        robots_at_colors = {}  # robot -> set of colors where robot is at
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                robots_at_colors.setdefault(robot, set()).add(color)

        # Extract free-color facts: colors that are free
        free_colors = set()
        for fact in state:
            if match(fact, "free-color", "*"):
                parts = get_parts(fact)
                free_colors.add(parts[1])

        # Extract "available-color" facts for colors (for painting preconditions)
        # The paint_up requires (available-color robot tile), (available-color tile-2 tile), (robot-at robot color)
        # We assume paint_up is possible only when the robot is at some color and the tile-2 is available at the tile.

        # For heuristic, approximate distance as shortest move on "up" edges from robot current tiles to the tile where tile-2 is.

        # For each tile-2 to paint, determine related "down" tiles required (down ?v6 - tile-2)
        # Use static "down" facts to link tile-2 to tile.
        tile2_to_down_tiles = {}
        for fact in self.static:
            if match(fact, "down", "*"):
                parts = get_parts(fact)
                tile_2 = parts[1]
                tile2_to_down_tiles.setdefault(tile_2, tile_2)  # For this domain, down relation is from tile-2 to itself?
        # Actually, from domain, down has signature ?v6 - tile-2, so tile-2 is the argument.
        # In the example facts, down tiles are tile1, tile5, tile7, tile14, tile16, etc.
        # We consider "down tileX" means tileX is a tile-2 in down state.
        # From example, painted tiles must be tile-2 to paint.
        # So, tiles that are down tiles to be painted are tile-2.

        # Hence, tiles to paint are the tile-2 objects that are down.

        # Let's define tile_2_down_set from static:
        tile_2_down_set = {get_parts(fact)[1] for fact in self.static if match(fact, "down", "*")}

        # To estimate distances: from robot available-color tiles to the tile-2 that must be painted,
        # we need to find the related tile (tile) to tile-2:

        # The domain design suggests that painting tile-2 corresponds to painting on their tile position.
        # Each tile-2 might correspond to a tile where it can be painted.
        # From available-color(tile-2 tile) facts in example state, e.g. (available-color tile1 tile4)
        # Find such associations to relate tile-2 to tile.

        # Map tile-2 -> their associated tile from available-color facts for tile-2
        tile2_to_tile = {}
        for fact in state:
            if match(fact, "available-color", "*", "*"):
                parts = get_parts(fact)
                obj, tile = parts[1], parts[2]
                # If obj is tile-2 and tile is tile, associate them
                if obj in self.all_tile_2:
                    tile2_to_tile[obj] = tile

        # If some tile-2 has no tile assigned, try fallback to same name (may not happen)
        for tile_2 in unpainted_tiles_2:
            if tile_2 not in tile2_to_tile:
                tile2_to_tile[tile_2] = tile_2  # fallback

        # For each robot, gather all their current available tiles
        # For simplicity, we consider each robot's available-color tiles as start positions for BFS
        # We compute distance from these start tiles to the target tile of tile-2 to paint

        total_estimate = 0

        for tile_2 in unpainted_tiles_2:
            target_tile = tile2_to_tile[tile_2]

            # Estimate minimal distance over all robots to target tile
            best_distance = None
            for robot in robots:
                start_tiles = robot_tiles.get(robot, set())
                if not start_tiles:
                    continue
                dist = self._bfs_distance(start_tiles, {target_tile})
                if dist is not None and (best_distance is None or dist < best_distance):
                    best_distance = dist

            if best_distance is None:
                # If no path found, assign high cost to reflect difficulty
                best_distance = 50

            # Actions estimated:
            # - Each step distance corresponds to one change_color action.
            # - One paint_up action to paint tile-2.
            # - Possibly one move_down action is required to be able to paint.
            # From domain, move_down requires available-color of two tiles and robot at some color.
            # We'll conservatively add +1 for move_down action.

            total_estimate += best_distance + 1 + 1  # change_color steps + move_down + paint_up

        return total_estimate
