from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict
import math


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain "color" problem that estimates how many
    actions are still required to achieve all "painted" goal facts (paint the
    required tile-2 objects). The heuristic explicitly reasons about the
    required preparations for a single paint operation: bringing a tile-2
    object, a robot, and a free color together on the same tile, then doing
    the move_down and paint_up actions. The heuristic computes an optimistic
    (but informed) estimate for each goal instance and combines per-goal
    estimates into a total estimate. It attempts to avoid reusing the same
    free color for multiple goal tiles (colors are consumed when used).

    Assumptions
    -----------
    - A "painted" goal is achieved by a paint_up action that requires:
      * the tile-2 object (with predicate available-color <tile-2> <tile>),
      * a robot that is available on that same tile (available-color <robot> <tile>),
      * a color that is free (free-color <color>) and available on that tile
        (available-color <color> <tile>), and
      * robot-at linking robot to the color (which is produced by move_down).
    - A single paint operation minimally requires:
      * moving the three relevant objects (tile-2, robot, color) along the
        static "up" edges until they meet on the same tile (each step costs one
        change_color action per object moved along one up edge),
      * one move_down action to associate the robot with the color, and
      * one paint_up action to paint the tile-2 object.
    - The static "up" graph is directed; items can be moved along up edges
      (change_color moves objects from a tile to its "up" neighbor).
    - Free colors are consumed by painting (paint_up makes free-color false),
      so the heuristic assigns distinct free colors to different goals when
      possible (greedy assignment). If there are fewer free colors than goals,
      the heuristic will return a large finite estimate to reflect the cost.

    Heuristic Initialization
    ------------------------
    - Save goal facts (self.goals) so we can quickly check goal satisfaction.
    - Extract static 'up' edges from the task.static facts and build a directed
      adjacency list. These edges are used to compute reachable tiles and the
      number of change_color steps required to move between tiles.
    - Precompute the set of tiles appearing in the static graph (nodes) to
      assist the BFS computations at runtime.

    Step-By-Step Thinking for Computing Heuristic (for a state)
    -----------------------------------------------------------
    1. If the state's facts already include all goal facts -> return 0.
    2. Parse the current state to extract:
       - available_positions: mapping object -> tile for facts (available-color OBJ TILE)
         (we take the first occurrence if multiple facts exist).
       - robots: set of robot object names (first argument of any (robot-at R C) fact).
       - free_colors: set of objects for which (free-color C) holds.
    3. For the static "up" graph compute shortest distances (in number of
       change_color steps) between tiles. Distances are only along the
       direction given by "up" edges. The distance from a tile to itself is 0.
    4. For every goal tile-2 object that is not yet painted:
       a. Determine the tile where the tile-2 object currently is
          (looking up available_positions).
       b. For each free color C and for each robot R, compute the minimal
          combined number of change_color actions required to bring
          (tile-2, robot, color) to a common meeting tile T:
             dist(tile-2 -> T) + dist(robot -> T) + dist(color -> T)
          (only consider T that are reachable from each of the three objects).
       c. Add 2 to the minimal such combined movement cost to account for
          move_down and paint_up actions.
       d. The cost for a goal is the minimum over all (color, robot) pairs.
    5. Colors are consumed when painting. To reflect this, build a cost
       matrix (goals x free_colors) where the cost entry is the minimal cost
       to paint that goal using that free color (minimizing over robots).
       Use a greedy assignment that, iteratively, picks the cheapest available
       goal-color pair and marks that color as used. (This is a cheap
       approximation to the assignment problem and keeps the heuristic fast.)
    6. Sum the assigned costs over goals. If any goal cannot be reached using
       available information (no reachable meeting tile or not enough colors),
       return a large finite cost (math.inf is avoided because the planner
       expects numeric heuristics; instead use a large finite integer).
    7. Return the resulting sum as the heuristic estimate.

    Notes
    -----
    - The heuristic is admissible in the sense of producing non-negative
      integers, but it is not required to be admissible; it aims to be
      informative and fast to compute.
    - The BFS distance is computed only on the static "up" graph, which is
      small and therefore efficient to explore.
    """

    def __init__(self, task):
        # Save goals for quick goal-checking.
        self.goals = task.goals
        # Extract 'up' edges from static facts to build directed adjacency.
        self.up_adj = defaultdict(list)  # tile -> list of tiles reachable in one 'up' step
        self.tiles = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.up_adj[a].append(b)
                self.tiles.add(a)
                self.tiles.add(b)
        # We will lazily compute distances per tile when called, but having the
        # static adjacency here is enough.

    def _bfs_distances(self, start):
        """
        Compute distances (number of up-steps) from start tile to every tile
        reachable via repeated up edges. Return dict tile->distance.
        """
        # If start is None or unknown tile, return empty dict
        if start is None:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for w in self.up_adj.get(v, []):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
        return dist

    def __call__(self, node):
        state = node.state

        # 1) If already a goal state, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Parse state facts:
        available_positions = {}  # object -> tile (from available-color facts)
        robots = set()  # robots (first arg of robot-at facts)
        free_colors = set()  # colors with free-color fact

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, tile = parts[1], parts[2]
                # store first seen position for the object (deterministic choice)
                if obj not in available_positions:
                    available_positions[obj] = tile
            elif pred == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                robots.add(robot)
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])

        # Extract goal tile-2 objects that must be painted.
        goal_tiles = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) >= 2:
                goal_tiles.append(parts[1])

        # If there are more goals than available free colors, painting is still
        # possible only if the domain provides extra free colors later; but
        # paint_up removes free-color and there is no operator adding it back in
        # the domain. So if free colors are fewer than remaining goals, it's
        # likely impossible. We still compute a finite large cost in this case.
        if not free_colors:
            # no free colors — no painting possible -> return large heuristic
            return 1000000

        # Precompute BFS distances from tiles that appear in available_positions
        # and from goal tiles. Only tiles for which we have a start matter.
        tile_distance_cache = {}
        def dist_cache(tile):
            if tile not in tile_distance_cache:
                tile_distance_cache[tile] = self._bfs_distances(tile)
            return tile_distance_cache[tile]

        INF = 1000000

        # Build cost matrix: for each goal and each free color, compute minimal
        # cost over robots to paint the goal using that color.
        goals_unpainted = [g for g in goal_tiles if f"(painted {g})" not in state]
        if not goals_unpainted:
            return 0

        # Ensure robots list is non-empty; if empty but domain requires robots,
        # heuristic returns large cost.
        if not robots:
            return 1000000

        # Prepare list of free colors that have an available position (if a color
        # has no available-color tile, it may still be usable if move_down
        # doesn't require the color to be present — but move_down requires
        # available-color <color> <tile>, so we require a tile).
        usable_colors = [c for c in free_colors if c in available_positions]
        if not usable_colors:
            return 1000000

        # For each goal, for each color, compute minimal cost over robots.
        cost_matrix = {g: {} for g in goals_unpainted}
        for g in goals_unpainted:
            tile_g = available_positions.get(g)
            if tile_g is None:
                # cannot find where the tile-2 is; make painting very expensive.
                for c in usable_colors:
                    cost_matrix[g][c] = INF
                continue
            dist_g = dist_cache(tile_g)
            for c in usable_colors:
                tile_c = available_positions.get(c)
                if tile_c is None:
                    cost_matrix[g][c] = INF
                    continue
                dist_c = dist_cache(tile_c)
                best_for_color = INF
                # For a given color c, pick the best robot
                for r in robots:
                    tile_r = available_positions.get(r)
                    if tile_r is None:
                        # If we don't know where robot currently is, skip robot
                        continue
                    dist_r = dist_cache(tile_r)
                    # Meeting tile T must be reachable from all three starts.
                    # We consider all candidate tiles that appear in the union of
                    # reachable tiles from the three starts (including starts).
                    candidate_tiles = set(dist_g.keys()) | set(dist_c.keys()) | set(dist_r.keys())
                    for T in candidate_tiles:
                        dg = dist_g.get(T, INF)
                        dc = dist_c.get(T, INF)
                        dr = dist_r.get(T, INF)
                        if dg >= INF or dc >= INF or dr >= INF:
                            continue
                        move_cost = dg + dc + dr
                        # +2 for move_down and paint_up
                        total_cost = move_cost + 2
                        if total_cost < best_for_color:
                            best_for_color = total_cost
                cost_matrix[g][c] = best_for_color

        # Greedy assignment of distinct colors to goals to account for color consumption.
        # This is a simple fast approximation: repeatedly pick the (goal, color)
        # pair with minimal cost, assign that color to that goal, and remove the
        # goal and color from further consideration.
        remaining_goals = set(goals_unpainted)
        remaining_colors = set(usable_colors)
        total_estimated_cost = 0

        # Precompute all pairs and sort them to drive greedy selection.
        # Use a list of tuples (cost, goal, color).
        pairs = []
        for g in remaining_goals:
            for c in remaining_colors:
                cost = cost_matrix[g].get(c, INF)
                pairs.append((cost, g, c))
        pairs.sort(key=lambda x: x[0])

        # Iterate picking cheapest pair that still has both goal and color available.
        for cost, g, c in pairs:
            if g not in remaining_goals or c not in remaining_colors:
                continue
            if cost >= INF:
                # No feasible assignment for this pair. Try next.
                continue
            total_estimated_cost += cost
            remaining_goals.remove(g)
            remaining_colors.remove(c)
            # If no more goals, break.
            if not remaining_goals:
                break

        # If after greedy assignment some goals remain unassigned, treat as high cost.
        if remaining_goals:
            # if there are still remaining goals but no colors left or no feasible pairs,
            # return a large finite estimate to indicate difficulty/infeasibility.
            return 1000000

        # Return integer heuristic
        return int(total_estimated_cost)