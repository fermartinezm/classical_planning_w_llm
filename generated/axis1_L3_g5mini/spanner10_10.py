from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """Return True if fact matches the pattern args (supports '*' via fnmatch)."""
    parts = get_parts(fact)
    return all(fnmatch(p, a) for p, a in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state
    in the `color` (secret_domain) domain. Goals are of the form (painted tileX).
    The heuristic counts, for each not-yet-painted tile-2 goal, an estimate of the
    minimal number of actions needed to paint it and sums these estimates.
    It is domain-dependent (uses the 'up' static relation and the concrete action
    preconditions/effects) but not necessarily admissible; it is designed to be
    informative and quick to compute for greedy best-first search.

    Assumptions
    -----------
    - Each goal is a single predicate (painted <tile-2>).
    - The fact "(available-color OBJ TILE)" indicates OBJ is currently at TILE.
      This holds for both robot objects (color-2) and other color objects; the
      heuristic treats any object with an available-color as a potential actor
      (robot) for conservative estimation.
    - The static relation "(up A B)" defines directed edges A -> B that can be
      traversed by the change_color action (each such traversal costs 1 action).
    - Painting a tile requires at least one paint_up action. In most typical
      scenarios painting additionally requires one move_down to obtain the
      correct robot-at mapping or some number of change_color actions to bring
      a robot to the tile; the heuristic reflects these costs.
    - The domain's other constraints (e.g., free-color availability) are
      considered where statable from the current state to refine estimates.

    Heuristic Initialization
    ------------------------
    During initialization the heuristic extracts:
    - The set of goal facts (task.goals).
    - The directed "up" graph from task.static (edges A -> B for facts matching "up").
      This graph is used to compute change_color distances (shortest path lengths).
    The static graph is stored as a dict mapping a tile to a list of successor tiles.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If all goal (painted ...) facts are already present in the state, return 0.
    2. Parse the state into useful structures:
       - available_map: object -> tile for every (available-color obj tile).
       - robot_at_map: robot -> color for every (robot-at robot color).
       - free_colors: set of color tokens for which (free-color color) is true.
       - down_set: set of tile-2 objects that currently have (down tile-2).
       - painted_set: set of tile-2 objects that are already painted.
    3. For each goal tile-2 that is not painted:
       a. Determine its current tile (where the tile-2 is available). If that
          information is missing, pessimistically assume a moderate traversal cost.
       b. Check a fast "ready-to-paint" condition:
          - If there exists some object R such that:
            * available_map[R] == tile_of_tile2,
            * robot_at_map.get(R) exists and that color is free (in free_colors),
            * (down tile2) is true,
            then we assume a single paint_up action suffices (cost = 1).
       c. Otherwise estimate the cost as:
          cost = (min number of up-steps required to bring some available object
                  to the tile_of_tile2) + 2
          Explanation of terms:
            - The "up-steps" correspond to change_color actions needed to align
              an object's available-color tile with the tile-of-the-target. Each
              step is one directed up-edge traversal.
            - "+2" accounts for the typical sequence: one move_down (to set
              robot-at appropriately) and one paint_up.
       d. If start/target tiles are not present in the static up graph or no
          path exists, a conservative finite penalty (graph_size + 2) is used to
          keep the heuristic finite while remaining informative.
    4. Sum the per-goal estimates and return the sum as the heuristic value.

    Notes on design trade-offs
    --------------------------
    - The heuristic ignores some global interactions (e.g., reusing a single
      robot for several nearby paints could be cheaper than summing independent
      estimates). This makes the heuristic potentially inadmissible but fast.
    - The heuristic is intentionally conservative but finite: when graph paths
      cannot be found, it uses a bounded penalty rather than infinity.
    """

    def __init__(self, task):
        """Extract goals and the static 'up' graph from the task."""
        self.goals = task.goals
        # Build directed graph from "up" static facts: up A B means A -> B
        self.up_graph = {}
        for fact in task.static:
            if match(fact, "up", "*", "*"):
                _, a, b = get_parts(fact)
                self.up_graph.setdefault(a, []).append(b)

        # Precompute a conservative finite penalty when no path is found:
        # use number of nodes in up_graph plus a small constant.
        self.max_penalty = max(1, len(self.up_graph)) + 3

    def _bfs_distance(self, start, target):
        """
        Compute shortest directed distance from start to target following up_graph.
        If start == target, return 0. If no path/unknown nodes, return self.max_penalty.
        """
        if start is None or target is None:
            return self.max_penalty
        if start == target:
            return 0
        if start not in self.up_graph:
            return self.max_penalty
        visited = set([start])
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for succ in self.up_graph.get(node, ()):
                if succ == target:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, dist + 1))
        return self.max_penalty

    def __call__(self, node):
        """Return an estimate of actions required to reach the goal from node.state."""
        state = node.state

        # Quick goal check: heuristic is 0 for goal states
        if self.goals <= state:
            return 0

        # Parse state
        available_map = {}  # object -> tile for (available-color obj tile)
        robot_at_map = {}  # robot -> color for (robot-at robot color)
        free_colors = set()  # set of colors for (free-color color)
        down_set = set()  # tile-2 objects with (down tile-2)
        painted_set = set()  # tile-2 objects already painted

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, tile = parts[1], parts[2]
                # If an object appears multiple times, keep the last seen location.
                available_map[obj] = tile
            elif pred == "robot-at" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_at_map[robot] = color
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "down" and len(parts) >= 2:
                down_set.add(parts[1])
            elif pred == "painted" and len(parts) >= 2:
                painted_set.add(parts[1])

        total_cost = 0

        # Extract goal tile-2 names from self.goals
        goal_tile2s = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "painted" and len(parts) >= 2:
                goal_tile2s.append(parts[1])

        # For each goal tile-2 that is not yet painted, estimate cost.
        for tile2 in goal_tile2s:
            if tile2 in painted_set:
                continue  # already achieved

            # Determine tile where this tile-2 currently resides (if known)
            tile_location = available_map.get(tile2)

            # If down is required and not present, painting is impossible in current state
            # under normal operator semantics; to keep heuristic finite, we use a penalty.
            if tile2 not in down_set:
                # Penalize but keep finite
                total_cost += self.max_penalty
                continue

            # Check if there exists a ready-to-paint robot:
            # available_map[robot] == tile_location, robot_at(robot) exists,
            # and that color is free. If so, cost = 1 (a single paint_up).
            ready = False
            if tile_location is not None:
                for obj, obj_tile in available_map.items():
                    if obj_tile != tile_location:
                        continue
                    # If object has a robot-at mapping and that color is free -> ready
                    rcolor = robot_at_map.get(obj)
                    if rcolor and rcolor in free_colors:
                        ready = True
                        break
            if ready:
                total_cost += 1
                continue

            # Otherwise estimate: min distance (change_color steps) from any available object's tile
            # to tile_location plus two actions (move_down + paint_up).
            min_dist = self.max_penalty
            # If we have knowledge of where the tile2 is and there are available objects to start from:
            if tile_location is not None and available_map:
                for obj, start_tile in available_map.items():
                    d = self._bfs_distance(start_tile, tile_location)
                    if d < min_dist:
                        min_dist = d
            else:
                # If we do not know tile_location or there are no available objects, use penalty
                min_dist = self.max_penalty

            estimated_cost = min_dist + 2  # +2 for move_down and paint_up
            total_cost += estimated_cost

        # Ensure finite integer return
        return int(total_cost)