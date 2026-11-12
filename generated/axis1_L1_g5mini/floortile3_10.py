from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Return parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches the given pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (w133ormt) that estimates
    the number of actions required to achieve all goal colorings (yh6q5j8r facts).
    The domain describes robots positioned on tiles, tiles linked by several
    directional relations, robots holding a color (i4u94rl9), and painting
    actions that create yh6q5j8r for a tile if certain conditions hold.
    This heuristic is designed for greedy best-first search (not necessarily
    admissible) and focuses on minimizing node expansions by producing a
    realistic cost estimate (moves + color-switches + painting and small
    preparation overheads).

    Assumptions
    -----------
    - Goals are only of the form (yh6q5j8r TILE COLOR).
    - Movement between tiles is done via directed edges encoded in static facts
      with predicates: xr2t1c1d, oh0smogc, w224no0l, g7ya35is. For a static fact
      "(P T S)" we interpret it as an edge S -> T (this matches the domain
      actions' parameter order).
    - To paint a tile T the robot must be located on some tile S such that
      there exists a static relation "(P T S)" (i.e., S is a valid painting
      position for T), the robot must be holding the desired color, and the
      tile T must have predicate (d7ekk7vb T) at the time of painting.
    - The robot can change held color with a single action (z7pcmkeu).
    - Movement actions each cost 1, painting costs 1, color switch costs 1.
    - The heuristic treats d7ekk7vb dynamics approximately: if the target tile
      currently lacks (d7ekk7vb T) we add a small preparation penalty (1) to
      reflect the expected extra work required to make T paintable. This keeps
      the heuristic informative while remaining cheap to compute.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The list of goal (tile, color) pairs.
    - The directed movement graph (adjacency mapping) from static facts:
      for each static fact "(P target source)" where P is one of the four
      movement predicates, we add an edge source -> target.
    - The set of paintable predicates (colors available) from cdb9op3o facts.
    - For convenience we also build a reverse mapping painting_neighbors:
      painting_neighbors[target] = set of sources S such that S -> target exists.
      These are the tile positions the robot can occupy to paint the target tile.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the node's state already satisfies all goals, return 0.
    2. Parse the given state and collect:
       - Current robot positions: from (gk9u3vg3 ROBOT TILE).
       - Current robot-held colors: from (i4u94rl9 ROBOT COLOR) (may be absent).
       - Set of tiles that already satisfy yh6q5j8r (so we skip them).
       - Set of tiles that currently have d7ekk7vb (we use this to reduce penalty).
    3. For each remaining goal (tile T, required_color C) we must paint T.
       We approximate the minimal cost to paint all remaining tiles using a
       greedy assignment scheme:
       - Precompute BFS distances from every robot's current tile to all tiles
         (using the directed adjacency graph). Each BFS step costs 1.
       - Repeatedly select the (robot, goal_tile) pair that yields the minimal
         marginal cost to perform the painting next. The marginal cost is:
            distance(robot_pos -> some painting_neighbor S of T)
            + (1 if robot_current_color != C else 0)   # color switch
            + 1                                       # paint action
            + (1 if d7ekk7vb is currently missing on T else 0)  # prep penalty
         The distance used is the minimal distance to any valid painting neighbor S.
       - After assigning the chosen robot to paint that tile we "simulate" the
         effects for remaining heuristic computation:
           * robot position becomes the chosen neighbor S (robot remains there
             after painting in the real domain),
           * robot held color becomes C (we assume it switches if needed),
           * mark T as painted (so it's removed from remaining tasks).
       - Repeat until no remaining goal tiles.
    4. If a tile is unreachable by any robot (no path to any painting neighbor),
       the heuristic returns a large finite value to indicate difficulty (INF).
    5. Return the accumulated cost as the heuristic estimate.

    Notes on Design Choices
    -----------------------
    - The greedy assignment approximates a sensible painting order (nearest-first)
      and thus captures travel merging effects better than a naive sum of
      independent costs.
    - We intentionally omit the full exact simulation of d7ekk7vb dynamics
      (which involves coupling movement and resource transfers in a nontrivial
      way). Instead, a small additive penalty is used if the tile lacks
      d7ekk7vb at the current state. This keeps the heuristic inexpensive
      while preserving informative guidance for the search.
    """

    def __init__(self, task):
        # Goals are stored as provided by the planner.
        self.goals = task.goals
        static_facts = task.static

        # Movement predicates used in the domain (directional edges).
        self.move_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Build directed adjacency graph: source -> set(targets)
        self.adjacency = {}  # movement graph
        # painting_neighbors: target -> set(sources) where being at source allows painting target
        self.painting_neighbors = {}

        # Colors available (from cdb9op3o facts)
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in self.move_preds and len(parts) >= 3:
                target, source = parts[1], parts[2]
                # Edge source -> target
                self.adjacency.setdefault(source, set()).add(target)
                # painting neighbor: being at source allows painting target
                self.painting_neighbors.setdefault(target, set()).add(source)
            elif pred == "cdb9op3o" and len(parts) >= 2:
                self.available_colors.add(parts[1])

        # Parse goal yh6q5j8r facts into (tile, color) pairs
        self.goal_tiles = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles.append((tile, color))

    def _bfs_distances(self, start):
        """BFS on directed graph self.adjacency returning dist dict start->node."""
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is 0.
        if all(g in state for g in self.goals):
            return 0

        # Extract current robot positions and colors.
        robots_pos = {}     # robot -> tile
        robots_color = {}   # robot -> color (string) or None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "gk9u3vg3" and len(parts) >= 3:
                robot, tile = parts[1], parts[2]
                robots_pos[robot] = tile
            elif parts[0] == "i4u94rl9" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robots_color[robot] = color

        # If no robots are present, unreachable (return large heuristic).
        if not robots_pos:
            return INF

        # Tiles that already are painted correctly in the current state.
        painted = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "yh6q5j8r" and len(parts) >= 3:
                painted.add((parts[1], parts[2]))

        # Tiles currently having the d7ekk7vb predicate.
        tiles_with_d7 = {get_parts(f)[1] for f in state if get_parts(f)[0] == "d7ekk7vb"}

        # Build the list of remaining painting tasks.
        remaining = []
        for tile, color in self.goal_tiles:
            if (tile, color) not in painted:
                remaining.append((tile, color))

        # Precompute BFS distances for each robot start position.
        robots_dist = {}
        for robot, pos in robots_pos.items():
            robots_dist[robot] = self._bfs_distances(pos)

        total_cost = 0

        # A greedy assignment loop: pick the cheapest next painting (robot, tile).
        while remaining:
            best = None  # (marginal_cost, robot, tile, chosen_neighbor)
            # Evaluate each remaining tile with every robot.
            for (tile, color) in remaining:
                neighbors = self.painting_neighbors.get(tile, set())
                if not neighbors:
                    # No position from which the tile can be painted (static disconnected).
                    # This tile is unreachable -> return a large cost.
                    return INF

                for robot in robots_pos:
                    dist_map = robots_dist.get(robot, {})
                    # minimal distance to any neighbor
                    min_dist = INF
                    best_neighbor = None
                    for neigh in neighbors:
                        d = dist_map.get(neigh, INF)
                        if d < min_dist:
                            min_dist = d
                            best_neighbor = neigh

                    if min_dist >= INF:
                        # robot cannot reach any valid painting neighbor
                        continue

                    # color switch cost
                    rc = robots_color.get(robot, None)
                    color_switch = 0 if rc == color else 1

                    # paint cost
                    paint_cost = 1

                    # small penalty if tile currently lacks d7ekk7vb (approximating resource prep)
                    prep_penalty = 0 if tile in tiles_with_d7 else 1

                    marginal = min_dist + color_switch + paint_cost + prep_penalty

                    # Keep the best (smallest marginal) assignment
                    if best is None or marginal < best[0]:
                        best = (marginal, robot, tile, best_neighbor, color_switch, prep_penalty, min_dist)

            # If no feasible assignment found (all robots cannot reach remaining tiles)
            if best is None:
                return INF

            marginal, robot, tile, neighbor, color_switch, prep_penalty, move_dist = best

            # Apply the hypothetical assignment effects for remaining heuristic computation:
            total_cost += marginal

            # Update robot position to neighbor (robot remains at the painting adjacency tile)
            robots_pos[robot] = neighbor
            # Update robot distance map from new position (recompute BFS)
            robots_dist[robot] = self._bfs_distances(neighbor)
            # Update robot color to the painted color (we assume switch happened if needed)
            # Note: this helps for subsequent assignments where the robot may already hold the needed color.
            # We do not remove color availability from global colors since colors are reusable.
            for fact in self.available_colors:
                pass
            robots_color[robot] = next((c for (t, c) in [(tile, None)] if True), None)  # placeholder to ensure variable exists
            # Set to the actual color: we can retrieve it from the remaining list element
            # but we have 'tile' and 'neighbor'; we need the color used in this iteration:
            # Recompute the color from remaining list (it is known)
            # Find the correct color from remaining
            chosen_color = None
            for (tt, cc) in remaining:
                if tt == tile:
                    chosen_color = cc
                    break
            if chosen_color is None:
                # fallback: set as None
                robots_color[robot] = None
            else:
                robots_color[robot] = chosen_color

            # Mark the tile as painted for the heuristic simulation (so it is not considered again).
            remaining = [pair for pair in remaining if pair[0] != tile]

            # Painting removes d7ekk7vb from that tile in the real domain; reflect that.
            if tile in tiles_with_d7:
                tiles_with_d7.remove(tile)
            # The movement and d7ebb dynamics are complex; we approximate and do not
            # simulate them precisely beyond removing the d7ekk7vb from painted tile.

        return total_cost