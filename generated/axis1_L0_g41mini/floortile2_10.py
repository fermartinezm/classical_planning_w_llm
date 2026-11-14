from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract predicate and arguments from a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern parts, with "*" wildcard allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary
    -------
    This heuristic estimates the number of actions required for the robot(s) to paint all required tiles
    with their goal colors. It considers: 
    - Tiles not yet painted with the goal color,
    - Movement cost for the robots to reach tiles adjacent to unpainted tiles,
    - Cost of changing colors in robots when necessary.

    Assumptions
    -----------
    - Multiple robots may be available.
    - Each action (move, paint, change_color) has equal cost 1.
    - Tiles can only be painted if the robot is adjacent (up/down/left/right),
      and the tile is clear (unpainted).
    - Robots can only paint tiles when holding the appropriate paint color.
    - Changing the robot's paint color costs one action.
    - The heuristic is not required to be admissible but should guide search efficiently.
    - Heuristic only counts minimal "distance" estimates ignoring obstacles caused by other robots or paint order.

    Heuristic Initialization
    ------------------------
    From the static facts, this heuristic extracts:
    - The set of tiles.
    - The adjacency graph between tiles (for movement).
    - The set of available colors.
    - The robot identities.
    - For each tile, the set of adjacent tiles via up/down/left/right relations.
    Also, it extracts from goals which tiles need to be painted with which color.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all tiles that must be painted with a specific color according to the goal.
    2. Identify tiles already painted correctly in current state; exclude them from counting.
    3. For each tile still needing painting, estimate the minimal cost to paint it as:
       - The minimal movement cost from any robot's current position to a tile adjacent to it that is clear,
         plus one paint action.
       - If none of the robots currently hold the required color, add one color-change action.
    4. For color changing, if multiple tiles require the same color, the cost of changing color is counted only once per robot if that robot paints any tile of that color.
    5. We sum minimal painting costs per tile to get the heuristic value.
    6. The heuristic is zero if all goal tiles are already painted correctly.
    7. For efficiency, precompute shortest movement distances between tiles using BFS on the adjacency graph.
    8. The heuristic approximates the painting plan by ignoring paint order and interactions.

    This heuristic balances accuracy and efficiency to minimize node expansions during greedy best-first search.
    """

    def __init__(self, task):
        """Extract static info and goal paint requirements and build tile adjacency graph."""
        self.goals = task.goals
        self.static = task.static

        # Extract tiles, robots and colors
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        self.adjacency = dict()  # tile -> set of adjacent tiles (up/down/left/right)

        # Initialize adjacency dict for all tiles found
        # Extract available colors from static
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "available-color":
                self.colors.add(parts[1])
            elif pred in ("up", "down", "left", "right"):
                src, tgt = parts[1], parts[2]
                self.tiles.add(src)
                self.tiles.add(tgt)
                if src not in self.adjacency:
                    self.adjacency[src] = set()
                self.adjacency[src].add(tgt)
            # Also collect tiles from adjacency indirectly:
            elif pred == "clear" or pred == "painted":
                self.tiles.add(parts[1])
            elif pred == "robot-at":
                self.robots.add(parts[1])
                self.tiles.add(parts[2])
            elif pred == "robot-has":
                self.robots.add(parts[1])
                self.colors.add(parts[2])
            # Defensive: initialize adjacency for each tile
        for tile in self.tiles:
            if tile not in self.adjacency:
                self.adjacency[tile] = set()

        # Build a bidirectional adjacency for undirected movement
        # Because moves can be up/down/left/right but effectively robot can move over tiles both ways
        # Static facts define directed edges, but movement is reversible if clear.
        # We build undirected edges so BFS for shortest path works correctly:
        full_adjacency = {tile: set(neigh) for tile, neigh in self.adjacency.items()}
        for tile in self.adjacency:
            for neighbor in self.adjacency[tile]:
                if neighbor not in full_adjacency:
                    full_adjacency[neighbor] = set()
                full_adjacency[neighbor].add(tile)
        self.adjacency = full_adjacency

        # Extract goal painting requirements: tile -> color
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Precompute shortest paths between all pairs of tiles via BFS for efficiency
        # Will be used to estimate minimal movements needed
        self._shortest_paths_cache = {}

    def _shortest_path_length(self, start_tile, goal_tile, clear_tiles):
        """
        Compute the shortest path length between start_tile and goal_tile on the tile adjacency graph,
        considering only clear tiles (the robot moves only over tiles that are clear).

        Returns an integer >= 0 for path length or None if no path exists.
        Uses BFS.

        Memoizes results in self._shortest_paths_cache.
        """
        key = (start_tile, goal_tile, frozenset(clear_tiles))
        if key in self._shortest_paths_cache:
            return self._shortest_paths_cache[key]

        if start_tile == goal_tile:
            self._shortest_paths_cache[key] = 0
            return 0

        from collections import deque

        queue = deque([(start_tile, 0)])
        visited = {start_tile}

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                # Robot can only move onto clear tiles (must be free)
                if neighbor in clear_tiles and neighbor not in visited:
                    if neighbor == goal_tile:
                        self._shortest_paths_cache[key] = dist + 1
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        self._shortest_paths_cache[key] = None
        return None

    def __call__(self, node):
        """
        Compute heuristic estimate of the number of actions needed to paint all required tiles.

        Returns 0 if the state is a goal state.

        For each tile that must be painted in a color different from the current painted color,
        the heuristic estimates:
        - minimal moves for some robot to reach an adjacent tile to that tile,
        - plus 1 paint action,
        - plus 1 color change if the robot doesn't currently hold the required color.

        The final estimate is the sum of these minimal costs for each tile.
        """
        state = node.state

        # If goal reached, heuristic is 0
        if self.goal_painted.items() <= {
            (tile, color) for fact in state if match(fact, "painted", "*", "*")
            for tile, color in [(get_parts(fact)[1], get_parts(fact)[2])]
        }:
            return 0

        # Extract current robot positions: robot -> tile
        robot_positions = dict()
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot_positions[parts[1]] = parts[2]

        # Extract current robot held colors: robot -> color
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot_colors[parts[1]] = parts[2]

        # Extract painted tiles with colors: tile -> color
        painted_tiles = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                painted_tiles[parts[1]] = parts[2]

        # Extract which tiles are currently clear (unpainted)
        clear_tiles = {fact[7:-1] for fact in state if match(fact, "clear", "*")}

        # For each tile that must be painted with a color (goal), if painted correctly, skip
        # Otherwise estimate minimal cost to paint it.
        # The robot must move adjacent to the tile on a clear tile,
        # hold the correct color, paint (1 action), change color if needed (1 action).
        # Painting can only be done to clear tiles adjacent to the robot position.

        total_cost = 0

        # We will sum minimal cost per tile independently.
        # For color change, count the minimal number of color changes needed summed across robots.
        # To keep heuristic simple, we consider color-change cost per tile independently,
        # not interleaving color switching across tiles collectively.

        for tile, required_color in self.goal_painted.items():
            if painted_tiles.get(tile) == required_color:
                # Tile already painted correctly, no cost.
                continue

            # Tiles to paint must be clear (paint actions require clear tiles)
            # If tile is not clear (e.g., painted wrong color), the tile won't be clear in state.
            # Such tiles cannot be painted unless we consider repainting, which domain doesn't allow directly.
            # So heuristic assumes tile is clear or must be painted once.

            # We find all tiles adjacent to the target tile (up/down/right/left)
            adjacent_tiles = set()
            for pred in ("up", "down", "left", "right"):
                for fact in self.static:
                    if match(fact, pred, "*", tile):
                        parts = get_parts(fact)
                        adjacent_tiles.add(parts[1])  # tile adjacent to 'tile'
                    if match(fact, pred, tile, "*"):
                        parts = get_parts(fact)
                        adjacent_tiles.add(parts[2])

            # Among adjacent tiles, only those clear can the robot stand on to paint
            candidate_positions = adjacent_tiles & clear_tiles
            if not candidate_positions:
                # No clear adjacent tile to paint from; heuristic is set high:
                # At least one move needed to clear an adjacent tile. 
                # To keep heuristic finite, assign a large penalty.
                # This signals difficult painting conditions.
                total_cost += 10
                continue

            min_paint_cost_for_tile = None

            # For each robot, compute cost to get to any candidate position + paint + change color cost if needed
            for robot, robot_pos in robot_positions.items():
                # Robot current color
                robot_color = robot_colors.get(robot, None)

                # Color change cost: 0 if holding correct color else 1
                color_change_cost = 0 if robot_color == required_color else 1

                # Movement cost: minimal distance to any candidate adjacent tile to 'tile'
                dist_to_candidates = []
                for adj in candidate_positions:
                    dist = self._shortest_path_length(robot_pos, adj, clear_tiles)
                    if dist is not None:
                        dist_to_candidates.append(dist)
                if not dist_to_candidates:
                    # Robot cannot reach any adjacent tile, skip
                    continue
                move_cost = min(dist_to_candidates)

                # Paint action cost is always 1
                paint_action_cost = 1

                cost = move_cost + color_change_cost + paint_action_cost

                if (min_paint_cost_for_tile is None) or (cost < min_paint_cost_for_tile):
                    min_paint_cost_for_tile = cost

            if min_paint_cost_for_tile is None:
                # No robot can reach or paint this tile reasonably.
                # Assign a large penalty to keep heuristic finite but large.
                total_cost += 10
            else:
                total_cost += min_paint_cost_for_tile

        return total_cost
