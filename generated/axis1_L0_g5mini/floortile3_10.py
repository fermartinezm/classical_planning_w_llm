from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'floortile' domain. The heuristic estimates
    the number of actions (moves, optional color changes, and paint actions) required
    to achieve all goal 'painted' facts from the current state. It is intended to
    guide a greedy best-first search (not guaranteed admissible) and therefore aims
    for a reasonably accurate yet inexpensive estimate.

    Assumptions
    -----------
    - Goals are only of the form (painted tile color).
    - Robots can move along the adjacency graph defined by the static relations
      (up/down/left/right). We treat those relations as undirected edges for
      movement distance estimation.
    - A tile can be painted from any adjacent tile x such that a static relation
      (up tile x), (down tile x), (left tile x) or (right tile x) exists.
    - Changing the robot color (change_color action) costs 1 action; painting
      costs 1 action; each move between adjacent tiles costs 1 action.
    - If a target tile is currently not (clear), we add a small penalty (1 action)
      to account for the effort needed to make it clear. This is a crude but
      inexpensive relaxation.
    - We do not attempt to sequence multiple painting tasks per robot precisely
      (which would be expensive). Instead we sum for each remaining tile the
      minimal single-robot cost to approach a valid painting position, change
      color if necessary, and paint. This is a reasonable domain-aware estimate
      that keeps computation cheap.

    Heuristic Initialization
    ------------------------
    In the constructor, the heuristic:
    - Extracts the goal mapping tile -> required_color from the task goals.
    - Builds an adjacency graph of tiles from the static facts (any up/down/left/right
      relation produces an undirected edge between the two tiles).
    - Builds a mapping paint_positions[tile] = set of tile-positions from which the
      tile can be painted (i.e., all x such that static contains (up tile x), etc).
    - Records which colors are available (not strictly required but parsed for completeness).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (with a state):
    1. If all goal painted facts are already present in the state, return 0.
    2. Parse the state to find:
       - For each robot: its current tile position and current held color (if any).
       - Which tiles are currently (clear ...) in the state.
       - Which tiles are already painted (and in which color).
    3. Determine the set of remaining goal tiles that still need painting
       (either not painted at all or painted with the wrong color).
    4. Precompute shortest-path distances (in number of moves) from each robot's
       current tile to every other tile using BFS on the adjacency graph.
    5. For each remaining tile t:
       a) Identify valid painting positions P = paint_positions[t].
       b) For each robot r and each p in P, compute:
          - move_cost = distance(r.pos, p)  (if unreachable, treat as infinite)
          - change_cost = 0 if r.color == goal_color else 1
            (if robot has no color fact, treat as needing change -> 1)
          - clear_penalty = 0 if (clear t) is in state else 1
          - paint_cost = 1
          Total cost candidate = move_cost + change_cost + paint_cost + clear_penalty
       c) Take the minimal candidate over all robots and painting positions as the
          estimated cost to handle tile t.
    6. Sum these minimal estimates for all remaining tiles. Return the sum.
    Notes:
    - This heuristic treats each tile's painting as (mostly) independent and does not
      attempt optimal sequencing of tasks by the same robot. It therefore can
      overestimate or underestimate in some cases but remains efficient and
      informative for greedy search.
    - If a tile is unreachable from all robots (disconnected graph), a very large
      finite cost is returned so the search will deprioritize that branch.
    """

    def __init__(self, task):
        """Extract goals and static adjacency/painting information from the task."""
        self.goals = task.goals  # set of goal fact strings
        static_facts = task.static

        # Extract goals of the form (painted tile color) -> map tile -> color
        self.goal_paint = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goal_paint[tile] = color

        # Build adjacency graph (undirected) from static relations up/down/left/right
        self.adj = {}  # tile -> set(neighbor_tiles)
        # Build paint_positions: tile -> set of positions (tiles) from which it can be painted
        self.paint_positions = {}

        # Record available colors (for completeness; not strictly necessary)
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in ("up", "down", "left", "right") and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # adjacency undirected
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
                # If relation is (pred tile position) then position is a valid painting position
                # (the predicate form is (pred tile pos) per domain)
                tile = a
                pos = b
                self.paint_positions.setdefault(tile, set()).add(pos)
            elif pred == "available-color" and len(parts) >= 2:
                self.available_colors.add(parts[1])

        # Ensure all tiles from goals at least have an entry in paint_positions and adj
        for tile in self.goal_paint:
            self.paint_positions.setdefault(tile, set())
            self.adj.setdefault(tile, set())

        # A large finite value for unreachable situations
        self.INF = 10 ** 6

    def _bfs_distances(self, start):
        """Return dict tile -> distance from start using BFS on self.adj."""
        dist = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for nb in self.adj.get(v, ()):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    q.append(nb)
        return dist

    def __call__(self, node):
        """Compute the heuristic estimate (sum of per-tile minimal costs)."""
        state = node.state

        # If all goals already satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse robot positions and colors
        robots = {}  # robot -> {"pos": tile, "color": color_or_None}
        clear_tiles = set()
        painted_facts = {}  # tile -> color (last seen)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "robot-at" and len(parts) >= 3:
                robot, tile = parts[1], parts[2]
                robots.setdefault(robot, {})["pos"] = tile
            elif pred == "robot-has" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robots.setdefault(robot, {})["color"] = color
            elif pred == "clear" and len(parts) >= 2:
                clear_tiles.add(parts[1])
            elif pred == "painted" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                painted_facts[tile] = color

        # Build list of robot descriptors
        robot_list = []
        for r, info in robots.items():
            pos = info.get("pos", None)
            color = info.get("color", None)
            if pos is None:
                # robot without specified pos -> cannot be used effectively; skip
                continue
            robot_list.append({"name": r, "pos": pos, "color": color})

        # If no robots found, return very large heuristic (unsolvable under actions)
        if not robot_list:
            return self.INF

        # Determine remaining tiles to paint (tile in goals but not painted with required color)
        remaining = []
        for tile, required_color in self.goal_paint.items():
            if painted_facts.get(tile) != required_color:
                remaining.append((tile, required_color))

        # Precompute BFS distances for each robot
        robot_dists = {}
        for r in robot_list:
            robot_dists[r["name"]] = self._bfs_distances(r["pos"])

        total_cost = 0

        for tile, req_color in remaining:
            # If no paint positions known for this tile, it's impossible or costs extra to reach
            paint_pos_candidates = self.paint_positions.get(tile, set())
            if not paint_pos_candidates:
                # No static adjacency as painting positions: treat as unreachable
                return self.INF

            best_cost_for_tile = self.INF

            for r in robot_list:
                dmap = robot_dists[r["name"]]
                robot_color = r.get("color", None)
                for pos in paint_pos_candidates:
                    move_cost = dmap.get(pos, self.INF)
                    if move_cost >= self.INF:
                        continue
                    change_cost = 0 if robot_color == req_color else 1
                    clear_penalty = 0 if tile in clear_tiles else 1
                    paint_cost = 1
                    candidate = move_cost + change_cost + paint_cost + clear_penalty
                    if candidate < best_cost_for_tile:
                        best_cost_for_tile = candidate

            # If no robot can reach any painting position, treat as unreachable
            if best_cost_for_tile >= self.INF:
                return self.INF

            total_cost += best_cost_for_tile

        # Ensure return value is an integer and finite
        return int(total_cost)