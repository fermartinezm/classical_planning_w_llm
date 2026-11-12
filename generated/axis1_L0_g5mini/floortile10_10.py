from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'floortile' domain.

    This heuristic estimates the number of actions required to achieve all
    (painted tile color) goal facts still missing in the current state.
    It pairs each goal tile with the robot that can most cheaply paint it
    (considering movement, an optional color-change action, and the paint action),
    and sums these per-tile estimates. The heuristic is optimistic in that it
    treats each goal tile independently (so it may underestimate costs due to
    shared robot movement being counted multiple times), but it tries to be
    informative by using the actual tile adjacency (static) graph and current
    robot locations and held colors.

    Assumptions
    -----------
    - Painting a tile requires a robot to be on a neighboring tile (up/down/left/right),
      hold the desired color, and for the target tile to be clear at painting time.
    - Changing a robot's color requires one action (change_color) and can be done
      at any robot location as long as the color is available (we assume the
      available-color facts in static make all relevant colors changeable).
    - Movement cost between adjacent tiles is 1 per move and is computed using
      the static adjacency relations (edges from the domain's up/down/left/right facts).
    - If a tile is already painted with a different color than required by the goal,
      the goal for that instance is unreachable (since there is no action that
      removes or changes a painted tile). In that case the heuristic returns
      a large number to indicate infeasibility.
    - The heuristic ignores dynamic occupancy (clear facts) when computing
      movement distances (optimistic assumption) but honors exact painted facts
      already present in the state.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals_by_tile: mapping tile -> desired color for each painted goal.
    - self.move_graph: adjacency mapping tile -> set(neighbor tiles) used for robot moves
      (constructed from static relations; for a static fact like "(up Y X)" there is an
      edge X -> Y because move_up uses (up Y X) with robot at X to move to Y).
    - self.paint_neighbors: mapping tile_to_paint -> set(tiles where a robot can stand to paint it).
      For a static fact "(up Y X)" the tile X is a valid standing tile to paint Y;
      similarly for down/left/right facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals are satisfied in the state, return 0.
    2. Parse current robot locations and the color each robot currently holds.
    3. For each goal (painted TILE COLOR):
       a. If TILE is already painted with the goal COLOR in the state: cost = 0 (skip).
       b. If TILE is painted with a different color in the state: the goal is impossible
          (no operator can remove/change painted facts) -> return a large value.
       c. Otherwise the tile still needs painting:
          - Determine set S of tiles where a robot could stand to paint TILE (paint_neighbors[TILE]).
          - For each robot, compute shortest path distance (in the static move_graph) from
            the robot's current tile to any tile in S using BFS. If unreachable, treat as infinite.
          - For that robot, the cost to paint TILE is:
                move_distance + (1 if robot's current color != desired color else 0) + 1_for_paint
            where move_distance is 0 if robot already stands on a painting neighbor.
          - Take the minimum such cost across robots.
       d. If no robot can reach any painting neighbor for TILE (all infinite), return large value.
    4. Sum the minimal costs for all required goal tiles and return that sum.

    Notes on admissibility and performance
    --------------------------------------
    - The heuristic is not guaranteed to be admissible because it ignores interactions
      between goals (robots may paint several tiles in one tour; we count each tour separately),
      but it is computationally efficient (BFS per robot on the static graph) and
      informative since it uses real adjacency and current robot positions/colors.
    """

    INF_COST = 10 ** 6

    def __init__(self, task):
        # Goals: map tile -> required color for painted goals
        self.goals_by_tile = {}
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goals_by_tile[tile] = color

        # Build movement graph and paint-neighbor mapping from static facts.
        # For a static fact like "(up Y X)" movement edge is X -> Y (robot can move from X to Y).
        self.move_graph = {}  # tile -> set(neighbor tiles robot can move to)
        self.paint_neighbors = {}  # tile_to_paint -> set(tiles where robot can stand to paint it)
        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("up", "down", "left", "right") and len(parts) >= 3:
                a, b = parts[1], parts[2]  # fact is like "(pred a b)" where a is target tile, b is robot-standing tile
                # movement from b to a
                self.move_graph.setdefault(b, set()).add(a)
                # Ensure nodes exist
                self.move_graph.setdefault(a, set())
                # robot standing tiles that can paint 'a' include b
                self.paint_neighbors.setdefault(a, set()).add(b)
                # ensure paint_neighbors entry for b exists (maybe not needed)
                self.paint_neighbors.setdefault(b, set())

        # Ensure all tiles known in static are present in move_graph
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] in ("up", "down", "left", "right") and len(parts) >= 3:
                for t in (parts[1], parts[2]):
                    self.move_graph.setdefault(t, set())
                    self.paint_neighbors.setdefault(t, set())

    def _bfs_distances(self, start):
        """Return dict tile->distance from start using self.move_graph (unweighted)."""
        dist = {}
        q = deque()
        dist[start] = 0
        q.append(start)
        while q:
            u = q.popleft()
            for v in self.move_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0.
        # (This also ensures we return 0 only for goal states as required.)
        if self.goals_by_tile.keys() and all(
            f"(painted {tile} {color})" in state for tile, color in self.goals_by_tile.items()
        ):
            return 0

        # Parse current painted facts to detect conflicts or already-satisfied goals.
        painted_in_state = {}  # tile -> color currently painted
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                painted_in_state[parts[1]] = parts[2]

        # If any goal tile is painted with the wrong color -> unreachable (no unpaint operator)
        for tile, desired_color in self.goals_by_tile.items():
            if tile in painted_in_state and painted_in_state[tile] != desired_color:
                return self.INF_COST

        # Collect current robot locations and colors.
        robot_locations = {}  # robot -> tile
        robot_colors = {}  # robot -> color
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robot_locations[robot] = tile
            elif parts[0] == "robot-has" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        # If there are no robots, cannot reach goals (unless all already satisfied).
        if not robot_locations:
            return self.INF_COST

        # Precompute BFS distances for each robot from its current position.
        robot_dist_maps = {}
        for robot, start_tile in robot_locations.items():
            # If start_tile not in move_graph, BFS will yield only start (we still record it)
            robot_dist_maps[robot] = self._bfs_distances(start_tile)

        total_cost = 0

        # For every goal tile that is not yet satisfied, compute minimal cost across robots.
        for tile, desired_color in self.goals_by_tile.items():
            # skip if already satisfied
            if f"(painted {tile} {desired_color})" in state:
                continue

            # Determine candidate positions from which a robot can paint this tile.
            candidates = self.paint_neighbors.get(tile, set())
            if not candidates:
                # No tile position exists from which to paint this tile -> unreachable
                return self.INF_COST

            best_cost_for_tile = None

            for robot, dist_map in robot_dist_maps.items():
                # color handling
                robot_color = robot_colors.get(robot, None)
                change_cost = 0 if robot_color == desired_color else 1

                # find minimal distance to any candidate tile
                min_dist = None
                for cand in candidates:
                    # if candidate equals robot's current tile, distance is 0
                    if cand in dist_map:
                        d = dist_map[cand]
                        if min_dist is None or d < min_dist:
                            min_dist = d

                if min_dist is None:
                    # robot cannot reach any painting neighbor for this tile
                    continue

                # cost = moves + (maybe change color) + paint action
                cost = min_dist + change_cost + 1  # 1 for paint
                if best_cost_for_tile is None or cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            if best_cost_for_tile is None:
                # No robot can reach this tile's painting positions -> unreachable
                return self.INF_COST

            total_cost += best_cost_for_tile

        return int(total_cost)