from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokenized parts of a PDDL fact string, e.g. "(pred a b)" -> ['pred','a','b']."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Return True if the fact matches the given pattern using shell-style wildcards.
    Example: match("(at obj room)", "at", "*", "room") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (w133ormt) tile-painting domain.

    This heuristic estimates the number of actions required to achieve all goal
    "yh6q5j8r <tile> <color>" facts (paint tiles in specific colors). It computes,
    for each goal tile that is not yet painted with the desired color, a conservative
    but informative estimate of the number of actions needed:
      - move actions to position a robot so it can paint the tile,
      - optionally one color-switch action if the robot's current color differs
        from the goal color,
      - the painting action itself.

    Assumptions
    -----------
    - Goals are of the form "(yh6q5j8r <tile> <color>)".
    - A painting action (c1qh1p5g or cadwr4wd) requires a robot to be at a tile R
      such that a static adjacency relation P(<tile> R) exists (we infer these
      R positions from static facts). We therefore compute shortest paths in the
      tile graph to reach any such R.
    - Movement between tiles is possible along the static adjacency relations
      xr2t1c1d, oh0smogc, w224no0l and g7ya35is. For efficiency we treat these
      relations as undirected edges when computing shortest-path distances.
    - Changing the robot's held color is achieved by a single action z7pcmkeu.
    - Painting a tile requires exactly one paint action (c1qh1p5g or cadwr4wd).
    - We ignore the low-level constraint concerning d7ekk7vb (availability flag)
      in the estimate to keep the heuristic efficient. This may underestimate
      cost in some corner cases, but keeps the heuristic finite and fast to compute.
    - Multiple robots are considered: for each goal tile we choose the robot that
      yields the minimal estimated cost to paint that tile.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals).
    - Parse static facts to:
      * build an undirected adjacency graph of tiles from the relations
        xr2t1c1d, oh0smogc, w224no0l and g7ya35is.
      * build a map from a tile T to the set of robot positions R that allow
        painting T, i.e., for which a static fact (P T R) exists for one of the
        adjacency predicates (these R are valid robot positions to paint T).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    When called on a node/state:
    1. If all goal facts already hold in the state, return 0.
    2. Extract from the state:
       - robot positions: facts matching "(gk9u3vg3 <robot> <tile>)".
       - robot held colors: facts matching "(i4u94rl9 <robot> <color>)".
       - painted tiles: facts of form "(yh6q5j8r <tile> <color>)".
    3. For each goal fact "(yh6q5j8r <tile> <color>)" that is not present:
       - Identify the set of valid robot positions R (from the precomputed map)
         from which a paint action can be executed for <tile>.
       - For each robot, compute the shortest-path distance (in number of move
         actions) from the robot's current tile to any r in R. The path length
         equals the number of move-operators required.
       - For that robot the estimated cost to achieve that single goal is:
           distance_to_valid_R + (1 if robot_color != goal_color else 0) + 1
         where the final +1 is the paint action itself.
       - Take the minimal estimated cost across robots for that tile.
       - If no valid R is known (static information missing), fall back to a
         conservative constant estimate (we use graph_size to keep values finite).
    4. Sum the per-goal minimal costs over all missing goal facts and return
       the sum as the heuristic value.
    """

    def __init__(self, task):
        """Extract goals and static adjacency information for efficient lookup."""
        super().__init__(task) if hasattr(super(), "__init__") else None
        self.goals = set(task.goals)  # goal facts (strings)

        static = task.static
        # adjacency predicates that correspond to movement operators
        self.adj_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Build undirected adjacency graph of tiles
        self.adj = defaultdict(set)  # tile -> set(neighbor_tiles)
        # Map: tile -> set of robot positions R from which tile can be painted
        self.paint_positions = defaultdict(set)
        # Set of all tiles seen in static information
        self.tiles = set()

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred in self.adj_preds and len(parts) >= 3:
                # static facts are of form "(pred tileA tileB)"; we treat them
                # as undirected adjacency for the purpose of distance estimation.
                a, b = parts[1], parts[2]
                self.tiles.add(a)
                self.tiles.add(b)
                # undirected adjacency
                self.adj[a].add(b)
                self.adj[b].add(a)
                # additionally, according to painting action preconditions,
                # a static fact of shape (pred tile target_pos) indicates that
                # a robot at target_pos can perform the painting for "tile".
                # In static facts the first argument is often the tile to be painted
                # and the second the robot position, so record that mapping.
                # (This is aligned with preconditions like (xr2t1c1d ?aepxj ?ajv3b).)
                self.paint_positions[a].add(b)

        # Ensure that if any tiles are only visible via d7ekk7vb or goals, we include them.
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "d7ekk7vb" and len(parts) >= 2:
                self.tiles.add(parts[1])

        # Keep a safe graph size for fallbacks
        self.graph_size = max(len(self.tiles), 1)

    def _bfs_distance_to_any(self, start, targets):
        """
        Return the shortest distance (number of edges / move actions) from start
        to any node in targets using BFS on the adjacency graph. If start is in targets,
        distance is 0. If no path exists, return a conservative large value (graph_size).
        """
        if start in targets:
            return 0
        if not targets:
            return self.graph_size  # no known valid painting positions

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                queue.append((nbr, dist + 1))
        # unreachable -> return conservative finite cost
        return self.graph_size

    def __call__(self, node):
        """Compute heuristic estimate for the node.state."""
        state = node.state

        # 1) If all goals are satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Extract robot positions and colors
        # robot_positions: robot -> tile
        robot_positions = {}
        # robot_colors: robot -> color
        robot_colors = {}
        # Painted tiles present in current state (tile -> color)
        painted = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot, tile = parts[1], parts[2]
                robot_positions[robot] = tile
                self.tiles.add(tile)
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif pred == "yh6q5j8r" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                painted[tile] = color

        # If there are no robots known, return conservative estimate: number of missing goals
        if not robot_positions:
            missing_goals = [g for g in self.goals if g not in state]
            return len(missing_goals) * (1 + self.graph_size)  # paint + unknown movement

        total_cost = 0

        # For each goal fact, if missing, compute minimal cost to achieve it.
        for goal_fact in self.goals:
            if goal_fact in state:
                continue  # already satisfied
            parts = get_parts(goal_fact)
            if parts[0] != "yh6q5j8r" or len(parts) < 3:
                # Skip unexpected goal shapes conservatively (count 1)
                total_cost += 1
                continue
            tile = parts[1]
            needed_color = parts[2]

            # Determine candidate robot positions R from which tile can be painted.
            valid_positions = self.paint_positions.get(tile, set())

            # If there are no valid positions from static info, fall back to all known tiles.
            if not valid_positions:
                # fallback: assume any tile could be used (less informed but finite)
                valid_positions = set(self.tiles)

            # For each robot, compute: distance to any valid position + color switch (0/1) + paint (1)
            best_cost_for_tile = None
            for robot, robot_tile in robot_positions.items():
                dist = self._bfs_distance_to_any(robot_tile, valid_positions)
                color = robot_colors.get(robot, None)
                color_change = 0 if color == needed_color else 1
                cost = dist + color_change + 1  # +1 for the painting action itself
                if best_cost_for_tile is None or cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            # If for some reason no robot costs computed, fallback to conservative estimate
            if best_cost_for_tile is None:
                best_cost_for_tile = self.graph_size + 1

            total_cost += best_cost_for_tile

        # Return the (integer) estimated number of actions
        return int(total_cost)