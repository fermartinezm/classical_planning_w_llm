from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by stripping parentheses and splitting.

    :param fact: string, e.g. "(available-color color1 tile2)"
    :return: list of strings, e.g. ["available-color", "color1", "tile2"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches the given pattern with wildcards.

    :param fact: string fact
    :param args: patterns with wildcards allowed ('*')
    :return: True if all parts match the pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain "color".

    Summary:
    -----------
    This heuristic estimates the number of actions required to achieve all goals,
    which involve painting specific tile-2 objects. Painting a tile requires a robot
    to be at a specific color, the tile to be "down", the robot to have a free color,
    and the tile to be available to that robot's color on a tile position.
    Robots can move "available-color" along tiles vertically using "change_color" actions
    and can "move_down" and "paint_up". The heuristic tries to estimate how many
    painting actions remain, plus necessary moves/changes required to position robots 
    and colors appropriately.

    Assumptions:
    -----------
    - The goal consists only of "painted" predicates for some tile-2 objects.
    - Each "painted" tile must be painted exactly once.
    - Robots move colors up/down along tiles using "change_color" (up adjacency)
      and "move_down" (complex movement involving robot presence).
    - The heuristic does not need to be admissible, but aims to minimize expanded nodes.
    - Free colors can paint at most one tile before becoming not free.
    - Static info about "up" relations between tiles is fixed.

    Heuristic Initialization:
    -------------------------
    - Extract goal painted tiles.
    - Extract all "up" relations as a directed graph for tile connectivity.
    - Identify all tile-2 objects (paintable tiles).
    - Collect all robots and colors from the initial facts.
    - Prepare fast access structures for adjacency and reachability on tiles.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which tiles need to be painted (from goal), which are painted already (in state).
       Only count unpainted tiles.
    2. For each unpainted tile:
       - Determine which robots/colors can paint it by reaching the tile-2 object or adjacent tiles.
         The painting requires:
           * The tile-2 object to correspond to a tile position (a tile).
           * The robot must have the correct "available-color" positioned on a tile adjacent (via "up") to the tile-2.
           * The robot must be at some color and color must be free.
           * The tile-2 must be "down".
       - Estimate cost to move a robot's "available-color" to the tile (using "change_color" operators).
       - Estimate cost to move the robot itself if necessary (via "move_down").
       - Estimate cost to paint (1 action "paint_up" per tile).
    3. Since detailed per-robot, per-color state is complex, approximate:
       - Count how many tiles need painting (unpainted).
       - Count how many robots have free colors available.
       - Estimate a cost proportional to unpainted tiles.
       - Add estimated number of "change_color" moves to bring robots/colors upward along tile chains.
       - Penalize for robot relocations needed (heuristic tries to estimate minimal moves)
    4. Use static "up" relations to compute minimal number of "change_color" actions needed to move colors upwards stepwise.
       - If color is on tile T1 and target is tile Tn, count number of up moves.
    5. For unreachable or complicated states, fallback on number of unpainted tiles + a multiplicative factor.
    6. Return 0 if already all painted (goal reached).

    The heuristic tries to balance the actual painting actions (one per tile-2 to paint)
    plus an estimated minimal amount of "change_color" and "move_down" actions needed.
    This should guide the search efficiently.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all "up" relations: directed edges: tile -> tile
        self.up_graph = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, from_tile, to_tile = get_parts(fact)
                self.up_graph.setdefault(from_tile, set()).add(to_tile)

        # Precompute all tiles reachable via "up" from each tile (transitive closure)
        self.reachable_up = {}
        for tile in self.up_graph.keys():
            self.reachable_up[tile] = self._compute_reachable_up(tile)

        # Store all tiles that appear in static facts (to handle isolated tiles)
        tiles = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up":
                tiles.add(parts[1])
                tiles.add(parts[2])
        self.all_tiles = tiles

        # Extract all tile-2 objects from goals and static info (tiles to paint)
        self.goal_painted_tiles = set()
        for goal in self.goals:
            if goal.startswith("(painted "):
                tile_2 = get_parts(goal)[1]
                self.goal_painted_tiles.add(tile_2)

    def _compute_reachable_up(self, start_tile):
        """Compute all tiles reachable from start_tile using 'up' links (transitive closure)."""
        reachable = set()
        stack = [start_tile]
        while stack:
            current = stack.pop()
            if current not in reachable:
                reachable.add(current)
                if current in self.up_graph:
                    stack.extend(self.up_graph[current])
        return reachable

    def __call__(self, node):
        state = node.state

        # Quick goal check: if goal painted tiles are all painted, heuristic is 0
        painted_tiles = set()
        for fact in state:
            if match(fact, "painted", "*"):
                painted_tiles.add(get_parts(fact)[1])
        unpainted_tiles = self.goal_painted_tiles - painted_tiles
        if not unpainted_tiles:
            return 0

        # Extract robots, colors, and free colors from state
        robots = set()
        colors = set()
        free_colors = set()
        robot_at = {}  # (robot_color2) -> color (color)
        available_color_positions = {}  # (robot_color2) -> tile or tile-2 where available-color is

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "available-color":
                # available-color ?robot ?tile_or_tile2
                robot = parts[1]
                tileobj = parts[2]
                available_color_positions.setdefault(robot, set()).add(tileobj)
            elif parts[0] == "free-color":
                free_colors.add(parts[1])
            elif parts[0] == "robot-at":
                # robot-at ?robot_color2 ?color
                robot = parts[1]
                color = parts[2]
                robot_at[robot] = color
                robots.add(robot)
                colors.add(color)

        # Count number of free robots ready to paint (free colors + have available-color)
        # We approximate a free robot as one that has at least one free color, and any available-color position
        free_robots = set(robot for robot in robot_at if robot_at[robot] in free_colors and robot in available_color_positions)

        # Heuristic estimate:
        # For each unpainted tile-2, estimate the minimal number of "change_color" steps needed to bring
        # some robot's available-color to a tile adjacent "up" to the tile-2.
        # Also, count 1 paint_up action per tile.

        # We know paint_up requires:
        # - available-color of robot on tile adjacent below tile-2 (down ?tile-2)
        # - free color at robot-at color of robot

        # We approximate by:
        # - For each unpainted tile-2, find number of robots that have available-color at or reachable to some "tile" such that "up tile tile-2" holds.
        # - Estimate minimal cost to move available-color to adjacent tile below tile-2, if not already there.
        # - If no robot currently can, add cost for robot movement and color movement to reach the tile.

        # Build reverse "up" to find tiles directly below each tile-2
        # The domain has mixed tile and tile-2 types; assume tile-2 are paintable tiles on top of tile.
        # Since "paint_up" requires (available-color ?v1 ?v2), (available-color ?v6 ?v2), and (down ?v6),
        # where ?v6 is tile-2, ?v2 is tile.

        # We'll find, for each tile-2 in unpainted_tiles:
        # 1. The tile below: tile s.t. (up tile tile-2) or direct link in static
        # Since "up" predicates only relate tile to tile, and here tile-2 involved, we approximate:
        # Check all "available-color" positions of robots and see if any is equal to tile below tile-2 (from available facts).
        # If not, estimate cost to move available-color from current tile to required tile via up moves.

        # First build map: tile-2 -> tiles below it by scanning static facts for "down tile-2" and "available-color tile tile" facts
        # But domain only defines (up tile tile) relations, so to find tile below tile-2, use (available-color tile-2 tile) in initial state?

        # Approximate: Since problem states "paint_up" requires available-color on tiles adjacent below tile-2,
        # we'll consider "available-color" on the tile below the tile-2 needs to be under the tile-2.

        # Find candidate tiles under tile-2 by scanning available-color facts:
        # We'll find stable mapping of tile-2 to tiles from facts like "(available-color tile-2 tile4)"

        # Build tile2_to_tile mapping from any fact (available-color tile-2 tileX) or from init facts (we approximate):
        tile2_to_tiles = {}
        for fact in state | self.static:
            if match(fact, "available-color", "*", "*"):
                p = get_parts(fact)
                owner = p[1]
                pos = p[2]
                # tile-2 type is ambiguous, but the owner can be tile-2 or robot-color2 or color
                # We guess if owner starts with "tile" and has dash in it, it is tile-2 or tile: no clean easy way, so use length heuristic:
                if owner in unpainted_tiles:  # Means owner is tile-2 (paint target)
                    # pos is tile below tile-2
                    tile2_to_tiles.setdefault(owner, set()).add(pos)

        # To handle missing info for tile2_to_tiles, try "down tile-2" facts (tile-2 marked down)
        # but they don't give tile association, so fallback:
        for tile2 in unpainted_tiles:
            if tile2 not in tile2_to_tiles:
                # No "available-color tile-2 tile" facts found; fallback: assume tiles directly below from "up" facts (approximate)
                candidates = set()
                for up_pred in self.up_graph.get(tile2, set()):
                    candidates.add(up_pred)
                if candidates:
                    tile2_to_tiles[tile2] = candidates
                else:
                    tile2_to_tiles[tile2] = set()  # no info, leave empty

        # Now, for each unpainted tile-2, compute minimal cost estimate:
        total_est_cost = 0

        # Precompute for quick lookup the "available-color" positions per robot
        # Flatten available_color_positions values to set with robot reference
        pos_robotcolor = set()
        for robot, positions in available_color_positions.items():
            for pos in positions:
                pos_robotcolor.add((robot, pos))

        for tile2 in unpainted_tiles:
            # Paint action cost = 1 per tile-2
            est_cost = 1

            tiles_below = tile2_to_tiles.get(tile2, set())
            if not tiles_below:
                # No info about tile below tile-2, add penalty
                est_cost += 5
                total_est_cost += est_cost
                continue

            # For each robot with free color, estimate minimal distance from current available-color tile positions to any tile below tile2
            min_move_cost = None  # count of change_color needed

            for robot in free_robots:
                if robot not in available_color_positions:
                    continue
                # Current positions of available-color for this robot (can be multiple)
                current_positions = available_color_positions[robot]

                # For each current position, compute minimal steps (number of "change_color" up moves) to reach one of tiles_below
                for start_pos in current_positions:
                    # If start_pos == target tile below tile2, cost 0
                    if start_pos in tiles_below:
                        cost = 0
                    else:
                        cost = self._min_up_moves(start_pos, tiles_below)
                        if cost == -1:
                            # no path, cost infinite, ignore this possibility
                            continue

                    # Additionally, robot might not be at needed color or free-color
                    # We already filter free_robots, so no extra cost here.

                    if min_move_cost is None or cost < min_move_cost:
                        min_move_cost = cost

            if min_move_cost is None:
                # No robot can reach required tile below tile2; heavy penalty
                est_cost += 10
            else:
                # Estimate that moving available-color costs one action per "change_color"
                # To be optimistic, add min_move_cost moves + 1 for move_down or robot steps to position
                est_cost += min_move_cost + 1  # +1 for robot "move_down" to position robot properly

            total_est_cost += est_cost

        return total_est_cost

    def _min_up_moves(self, start_tile, target_tiles):
        """
        Compute minimal number of "change_color" (up) actions to get from start_tile to any tile in target_tiles,
        following the "up" graph edges.

        If unreachable, return -1.
        """

        if start_tile == "":
            return -1

        if start_tile in target_tiles:
            return 0

        # BFS on up_graph edges
        visited = set()
        queue = [(start_tile, 0)]

        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current in target_tiles:
                return dist
            for neighbor in self.up_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        return -1
