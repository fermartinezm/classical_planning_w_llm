from fnmatch import fnmatch
from collections import deque
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a simple pattern with optional wildcards.
    Example: match("(at obj loc)", "at", "*", "loc") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (w133ormt). The domain corresponds to painting
    tiles using one or more robots that can hold a color, change their held color,
    move between tiles, and paint adjacent tiles (according to certain directional
    relations). The goal is to have specific tiles painted with specified colors
    (yh6q5j8r facts).

    The heuristic estimates the number of actions required to reach the goal from
    the current state by, for each not-yet-painted goal tile, computing a relaxed
    cost: minimal robot travel distance (on the static adjacency graph) to any
    tile-position from which that tile can be painted, plus the paint action, and
    plus a color-change action if the chosen robot does not currently hold the
    required color. Costs for different tiles are summed in a fully relaxed manner
    (i.e., robots are allowed to be reused independently; movement obstacles like
    already-painted tiles are ignored). This produces a fast, informative,
    non-admissible heuristic that reduces node expansions in greedy best-first search.

    Assumptions
    -----------
    - Painting a tile always consumes exactly one "paint" action (c1qh1p5g or cadwr4wd).
    - Changing a robot's held color (z7pcmkeu) costs exactly one action.
    - Moving a robot between neighboring tiles costs 1 per step (movement actions).
      The heuristic uses the static adjacency graph (derived from xr2t1c1d, oh0smogc,
      g7ya35is, w224no0l facts) to compute shortest-path distances (ignoring dynamic
      occupancy constraints like d7ekk7vb).
    - Multiple robots may exist. For each goal tile we choose the robot that
      minimizes the relaxed cost to paint that tile (robots are treated independently).
    - If a tile has no valid painting positions in the static information, we treat
      it as unreachable and assign a large penalty.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and preprocess static information:
    - self.adjacency: undirected adjacency mapping between tiles for movement (BFS).
    - self.paint_positions: for each tile T, the set of tiles P such that a robot
      standing on P can paint tile T. This is taken from static facts where a
      predicate of type xr2t1c1d or oh0smogc relates T and P as in the domain's
      painting preconditions.
    - self.available_colors: set of colors declared with cdb9op3o (used for sanity).
    - self.goals_map: dictionary mapping target tile -> required color, extracted
      from the provided task goals (predicate yh6q5j8r).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goal facts are already true in the state, return 0.
    2. Parse robot information from the state:
       - For each robot we locate its current tile (gk9u3vg3 facts).
       - For each robot we read its current held color (i4u94rl9 facts).
    3. Precompute BFS distance maps from each robot's current tile to all other tiles
       using the static adjacency graph.
    4. For each goal tile that is not yet painted in the state:
       a. Retrieve the set of painting positions (adjacent tiles from which the tile
          can be painted). If empty -> unreachable: add a large penalty.
       b. For every robot, compute the minimal BFS distance to any painting position
          for that goal tile. If no robot can reach any painting position -> unreachable.
       c. For the best robot (minimizing travel distance), the relaxed cost to paint
          that tile is:
             move_cost (distance) + paint_cost (1) + color_change_cost (0 or 1)
          where color_change_cost is 0 if the robot already holds the required color,
          else 1.
       d. Sum these minimal costs across all missing goal tiles.
    5. Return the summed cost (integer). This is finite for normally solvable states
       except when a tile has no painting positions in the static graph, in which case
       a large finite penalty is returned to push the search away from such branches.

    Notes on Design Choices
    -----------------------
    - The heuristic ignores dynamic movement constraints (e.g., tiles that become
      non-traversable after being painted). This is acceptable because admissibility
      is not required; the relaxation makes the heuristic cheaper to compute and
      still strongly informative.
    - We consider multiple robots and choose the best robot per goal tile independently,
      which tends to yield better guidance when many robots are available.
    - An unreachable tile (missing painting positions) gives a large finite cost rather
      than infinity to keep the heuristic numeric and the search stable.
    """

    def __init__(self, task):
        """Extract static adjacency, painting positions and goal mapping from task."""
        self.goals = task.goals  # goal facts as strings (e.g., "(yh6q5j8r tile_1_1 white)")

        # Extract goal mapping: tile -> color for yh6q5j8r goals
        self.goals_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goals_map[tile] = color

        # Static facts (unchanging throughout execution)
        static_facts = task.static

        # Build undirected adjacency graph for movement using all four relation types.
        # We'll treat all of them as bidirectional for distance computation.
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("xr2t1c1d", "oh0smogc", "g7ya35is", "w224no0l") and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)

        # Build paint positions: for a target tile T, painting positions are tiles P
        # such that there's a static fact (xr2t1c1d T P) or (oh0smogc T P).
        # The painting operators in the domain require these particular relations.
        self.paint_positions = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("xr2t1c1d", "oh0smogc") and len(parts) >= 3:
                target, painter_pos = parts[1], parts[2]
                self.paint_positions.setdefault(target, set()).add(painter_pos)

        # Colors available in the domain (cdb9op3o facts)
        self.available_colors = {
            get_parts(f)[1] for f in static_facts if match(f, "cdb9op3o", "*")
        }

        # A reasonable large penalty for unreachable tiles to keep heuristic finite.
        self.UNREACHABLE_PENALTY = 1000

    def _bfs_distances(self, start):
        """
        Compute BFS shortest-path distances (unweighted) from start to all tiles
        in the static adjacency graph. Returns a dict tile->distance.
        """
        dist = {}
        if start not in self.adjacency:
            # Start might be isolated or not present in adjacency; still return empty/inf mapping.
            return dist
        dq = deque([start])
        dist[start] = 0
        while dq:
            v = dq.popleft()
            for nb in self.adjacency.get(v, ()):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    dq.append(nb)
        return dist

    def __call__(self, node):
        """Estimate number of actions to reach a goal state from node.state."""
        state = node.state

        # If goal already satisfied, heuristic is 0 (check using goal subset)
        if self.goals <= state:
            return 0

        # Parse robot positions and colors from the state.
        # There may be multiple robots; keep mappings robot->tile and robot->color.
        robot_positions = {}  # robot -> tile
        robot_colors = {}     # robot -> color

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot, tile = parts[1], parts[2]
                robot_positions[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # If no robots in the state, return a large heuristic value.
        if not robot_positions:
            return self.UNREACHABLE_PENALTY

        # Precompute BFS distances from each robot's current tile.
        robot_distance_maps = {}
        for robot, tile in robot_positions.items():
            robot_distance_maps[robot] = self._bfs_distances(tile)

        total_cost = 0

        # For each goal tile required color, if not yet satisfied, compute relaxed cost.
        for tile, required_color in self.goals_map.items():
            goal_fact = f"(yh6q5j8r {tile} {required_color})"
            if goal_fact in state:
                # Already achieved.
                continue

            # Determine painting positions from which this tile can be painted.
            painter_positions = self.paint_positions.get(tile, set())
            if not painter_positions:
                # No static way to paint this tile -> treat as unreachable with a penalty.
                total_cost += self.UNREACHABLE_PENALTY
                continue

            # For each robot, compute minimal distance to any painter position.
            best_cost_for_tile = inf
            for robot, dist_map in robot_distance_maps.items():
                # Minimal movement distance to any painting position for this tile.
                min_move = inf
                for pos in painter_positions:
                    # If pos is same as robot tile, distance might be 0 and present in dist_map.
                    d = dist_map.get(pos)
                    if d is not None:
                        if d < min_move:
                            min_move = d
                # If robot cannot reach any painter position, skip it.
                if min_move is inf:
                    continue

                # Painting action costs 1. Color switch costs 0 if robot already holds color,
                # else 1 (z7pcmkeu).
                robot_color = robot_colors.get(robot)
                color_switch_cost = 0 if robot_color == required_color else 1

                cost = min_move + 1 + color_switch_cost
                if cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            if best_cost_for_tile is inf:
                # No robot can reach painter positions for this tile -> unreachable penalty.
                total_cost += self.UNREACHABLE_PENALTY
            else:
                total_cost += best_cost_for_tile

        # Return integer heuristic estimate.
        return int(total_cost)