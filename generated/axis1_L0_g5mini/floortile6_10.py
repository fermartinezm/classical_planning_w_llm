from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "floortile" domain.

    The heuristic estimates the number of actions required to reach a goal
    painting configuration. It models the cost to paint each goal tile that is
    not yet painted in the required color by:
      - moving some robot to a tile position from which it can paint the goal tile,
      - optionally changing the robot's held color to the goal color (one action),
      - performing the paint action (one action).

    The heuristic returns the sum of the cheapest per-goal estimates (each goal
    considered independently). It is not admissible (it can be optimistic or
    pessimistic), but it is inexpensive to compute and informative for
    guiding greedy best-first search.

    Assumptions
    -----------
    - Painting a tile requires the tile to be clear and a robot to be positioned
      at a specific adjacent tile (as given by the static relations up/down/left/right).
    - A paint action costs 1, a single change_color action costs 1, and each
      robot move between adjacent tiles costs 1.
    - Any robot can change color to any available color in one action (as long
      as the target color is present in the static available-color facts).
    - The heuristic ignores dynamic interactions between tasks:
      it treats each goal tile independently (so a robot may be counted as
      painting multiple tiles without accounting for the extra movements
      between those painting operations). This trade-off keeps the heuristic
      efficient while remaining informative.
    - If a goal tile is already painted with the required color, cost is 0.
    - If a goal tile is not clear and does not already have the required color,
      it is considered impossible to fulfill (large heuristic value returned).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
      - The set of goal painting requirements (mapping tile -> color).
      - The adjacency graph of tiles from static facts (directed edges x -> y
        correspond to an operator that allows a robot at x to move to y).
      - For each tile T, the set of tile positions from which a robot can paint
        T (these are the "paint source" positions x for which static facts
        (up T x), (down T x), (left T x) or (right T x) exist).
      - The set of available colors (from static facts like (available-color white)).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal painted facts are already present in the state, return 0.
    2. Parse current robot positions and current robot colors from the state.
    3. For each goal tile t with required color c:
       a. If "(painted t c)" is already in the state -> cost 0 for this tile.
       b. If "(clear t)" is not in the state and the correct painted fact is missing
          -> the tile cannot be painted anymore (we return a large cost).
       c. Otherwise, compute the minimal cost to paint t:
          - For each robot r:
            i. Compute movement distance from r's current tile to any of the
               paint-source tiles x for t (BFS on the static adjacency graph).
           ii. If no paint-source is reachable by r, ignore that robot.
          - For each reachable robot, the cost is:
                distance_to_source + (1 if robot_color != c else 0) + 1_for_paint
          - The tile's estimated cost is the minimum of these robot costs.
       d. If no robot can reach any paint-source for t, consider the task
          impossible and return a large cost.
    4. Sum the per-goal costs and return the result.

    Notes
    -----
    - The heuristic returns a large integer (self.INF) when it detects an
      impossible-to-achieve goal under current facts (for example, tile already
      non-clear and not painted with the required color, or the required color
      is not available).
    - BFS distances are computed per robot once per heuristic call for
      efficiency.
    """

    INF = 10**6

    def __init__(self, task):
        """Extract goals and static adjacency information from the given task."""
        self.goals = task.goals  # set of goal facts as strings
        static_facts = task.static

        # Parse goal painted facts: mapping tile -> required color
        self.goal_map = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted":
                tile = parts[1]
                color = parts[2]
                self.goal_map[tile] = color

        # Build adjacency graph for robot movement.
        # For any static fact (up Y X) we add an edge X -> Y (robot at X can move to Y).
        # Similarly for down/left/right.
        self.adj = {}  # tile -> set(neighbor tiles reachable in one move)
        # Also build mapping tile -> set of positions from which it can be painted.
        # For a static fact "(up Y X)" robot-at X can paint Y, so X is a paint-source for Y.
        self.paint_sources = {}  # target_tile -> set(source positions)
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                # parts: [pred, Y, X] -> add edge X -> Y and declare X as paint-source for Y.
                if len(parts) >= 3:
                    y, x = parts[1], parts[2]
                    self.adj.setdefault(x, set()).add(y)
                    # Ensure nodes without outgoing edges are present in adj mapping.
                    self.adj.setdefault(y, set())
                    self.paint_sources.setdefault(y, set()).add(x)
            elif pred == "available-color":
                if len(parts) >= 2:
                    self.available_colors.add(parts[1])

        # Ensure all tiles mentioned as paint targets are at least keys in paint_sources
        for t in list(self.goal_map.keys()):
            self.paint_sources.setdefault(t, set())

    def _bfs_distances(self, start):
        """
        BFS on the static adjacency graph starting at tile `start`.
        Returns dict tile -> distance (int). If start is not in graph, returns empty dict.
        """
        if start not in self.adj:
            return {}
        dist = {start: 0}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    def __call__(self, node):
        """Compute and return the heuristic estimate for the given search node."""
        state = node.state  # frozenset of facts (strings)

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current robot positions and colors from the state.
        robot_pos = {}   # robot -> tile
        robot_color = {}  # robot -> color

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robot_pos[robot] = tile
            elif pred == "robot-has" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robot_color[robot] = color

        # Precompute BFS distances for each robot to avoid repeated searches.
        robot_distances = {}
        for r, pos in robot_pos.items():
            robot_distances[r] = self._bfs_distances(pos)

        total_cost = 0

        # Evaluate cost for every goal tile that is not yet achieved.
        for tile, required_color in self.goal_map.items():
            # If already painted with required color, skip.
            painted_fact = f"(painted {tile} {required_color})"
            if painted_fact in state:
                continue

            # If required color is not available in domain static -> impossible.
            if required_color not in self.available_colors:
                return self.INF

            # The tile must be clear to allow painting (paint precondition). If it is not
            # clear and not already correctly painted, the goal is unreachable.
            clear_fact = f"(clear {tile})"
            if clear_fact not in state:
                # Either already painted with wrong color or blocked; assume impossible.
                return self.INF

            # Find the set of positions from which this tile can be painted.
            sources = self.paint_sources.get(tile, set())
            if not sources:
                # No tile position allows painting this target (static connectivity missing).
                return self.INF

            # For each robot compute the minimal cost to paint this tile.
            best_cost_for_tile = None

            for r, dist_map in robot_distances.items():
                # Robot must exist with a known position; ensure we have that.
                if r not in robot_pos:
                    continue

                # For each paint-source position x, check distance from robot to x.
                for src in sources:
                    # If robot is already at src, distance 0; otherwise check dist_map.
                    if src == robot_pos[r]:
                        d = 0
                    else:
                        # If src unreachable from robot start according to static adjacency, skip
                        if src not in dist_map:
                            continue
                        d = dist_map[src]

                    # Cost components:
                    # - movement cost: d moves
                    # - color-change cost: 1 if robot does not currently have required_color
                    # - paint cost: 1
                    has_color = robot_color.get(r, None)
                    color_change_cost = 0 if has_color == required_color else 1
                    cost = d + color_change_cost + 1  # +1 for paint action

                    if best_cost_for_tile is None or cost < best_cost_for_tile:
                        best_cost_for_tile = cost

            if best_cost_for_tile is None:
                # No robot can reach any paint-source -> impossible
                return self.INF

            total_cost += best_cost_for_tile

        return total_cost