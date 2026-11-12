from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `floortile` domain. The heuristic estimates
    the number of actions required to achieve all goal `painted` facts that are
    not yet true in the current state. It combines a shortest-path estimate for
    robot movement on the static tile adjacency graph with the minimal number of
    color-change and paint actions needed.

    Assumptions
    -----------
    - Each paint action paints exactly one target tile from a neighboring tile.
      The static relations (up/down/left/right) describe valid painting neighbors:
      a fact "(up Y X)" indicates that a robot at X can perform "paint_up" to paint Y.
    - Each change_color action switches the robot's held color to another available
      color at cost 1, and is always possible if the robot currently has some color
      and the target color is listed as available in the static facts.
    - Movement between adjacent tiles (according to static up/down/left/right facts)
      costs 1 per step (one move action). Paint and change_color each cost 1.
    - For efficiency and robustness the heuristic ignores dynamic blocking due to
      other robots or previously painted tiles (it computes distances on the static
      adjacency graph). This makes the heuristic optimistic in terms of path
      availability but still informative for guiding search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The set of goal painted facts (mapping tile -> required color).
    - The static adjacency graph between tiles (undirected edges from up/down/left/right facts).
    - For each target tile, the set of neighboring tile positions from which it can be painted
      (derived from facts like "(up Y X)" meaning X is a painting-position for Y).
    - The set of available colors (mainly to check change_color feasibility, though the
      domain instances normally list them).
    These static structures are computed once and reused for each heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal painted conditions are already satisfied in the state, return 0.
    2. Parse the state's facts to collect:
       - Which tiles are already painted (and with which color).
       - Each robot's current tile position.
       - Each robot's currently held color (robot-has).
    3. For each goal tile that is not yet painted with the required color:
       a. Identify the painting positions (neighbor tiles) from which painting that tile is possible
          (from the precomputed static paint-neighbors).
       b. For each robot, compute (or reuse) the shortest-path distance from the robot's tile
          to each painting position using BFS on the static adjacency graph.
       c. For that robot, the cost to paint the target tile is:
            distance_to_paint_position + color_change_cost + 1 (for the paint action)
          where color_change_cost is 0 if the robot already holds the desired color, else 1.
       d. The estimated cost to achieve that goal tile is the minimum cost across all robots.
       e. If the tile has no reachable painting position from any robot (disconnected), a
          large finite penalty is used for that tile to keep the heuristic finite.
    4. The heuristic value is the sum of the per-goal-tile minimal costs computed above.
       (This treats painting tasks independently and ignores that a robot can paint several
       tiles in sequence; this underestimates true cost but is cheap to compute and
       provides useful guidance for greedy best-first search.)
    """

    def __init__(self, task):
        """
        Extract static information from the task:
        - goals: set/list of goal strings
        - paint_goals: dict mapping target_tile -> required_color
        - adjacency: dict tile -> set(neighbor_tiles) for movement (undirected)
        - paint_neighbors: dict target_tile -> set(tiles) (robot positions from which target can be painted)
        - available_colors: set of available color names
        """
        self.goals = task.goals  # set of goal fact strings

        # Extract goal painted conditions: map tile -> color
        self.paint_goals = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.paint_goals[tile] = color

        # Build adjacency graph and paint-neighbor mapping from static facts
        self.adjacency = defaultdict(set)        # undirected graph for movement
        self.paint_neighbors = defaultdict(set)  # for each target tile, set of positions robot must occupy to paint it
        self.available_colors = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            # Movement adjacency: treat up/down/left/right as undirected edges
            if pred in ("up", "down", "left", "right") and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # add undirected edge
                self.adjacency[a].add(b)
                self.adjacency[b].add(a)

                # For painting neighbors: a relation like (up Y X) indicates
                # robot at X can paint Y using paint_up. So X is a paint-neighbor for Y.
                # The fact format is (pred target position). So:
                target = parts[1]
                painter_pos = parts[2]
                self.paint_neighbors[target].add(painter_pos)

            elif pred == "available-color" and len(parts) >= 2:
                self.available_colors.add(parts[1])

        # A small finite penalty to assign when a tile appears unreachable by any robot
        # Keep this proportional to the problem size so it stays finite but large enough
        # to discourage selecting unreachable tiles as cheap.
        self.unreachable_penalty = max(50, len(self.paint_goals) * 5)

    def __call__(self, node):
        """
        Compute heuristic value for a given search node.

        Returns an integer estimate of remaining actions to satisfy all painting goals.
        """
        state = node.state

        # If all goals already satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # Parse state facts: current painted tiles, robot positions, robot colors
        painted_now = {}      # tile -> color
        robot_pos = {}        # robot -> tile
        robot_color = {}      # robot -> color (string)
        tiles_set = set(self.adjacency.keys())

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "painted" and len(parts) >= 3:
                painted_now[parts[1]] = parts[2]
            elif pred == "robot-at" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robot_pos[robot] = tile
                tiles_set.add(tile)
            elif pred == "robot-has" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robot_color[robot] = color

        # Precompute BFS distances from each robot to all tiles (static adjacency graph)
        robot_distances = {}
        for robot, start in robot_pos.items():
            dist = {start: 0}
            q = deque([start])
            while q:
                cur = q.popleft()
                for nb in self.adjacency.get(cur, ()):
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            robot_distances[robot] = dist

        total_cost = 0

        # For each painting goal that is not yet satisfied in the current state compute min cost
        for target_tile, required_color in self.paint_goals.items():
            # If already painted with required color, cost 0
            if target_tile in painted_now and painted_now[target_tile] == required_color:
                continue

            # Find possible painting positions (tiles where a robot must stand to paint target)
            painting_positions = self.paint_neighbors.get(target_tile, set())
            if not painting_positions:
                # No static position from which this tile can be painted -> unreachable in domain
                total_cost += self.unreachable_penalty
                continue

            best_cost_for_tile = None

            # Evaluate each robot as potential painter
            for robot, start in robot_pos.items():
                distmap = robot_distances.get(robot, {})
                # minimal distance to any painting position
                min_dist = None
                for pos in painting_positions:
                    if pos in distmap:
                        d = distmap[pos]
                        if min_dist is None or d < min_dist:
                            min_dist = d
                if min_dist is None:
                    # This robot cannot reach any painting position (in static graph)
                    continue

                # Color change cost: 0 if robot already holds required_color, else 1
                # If robot has no color fact, conservatively assume 1.
                has_color = robot_color.get(robot, None)
                color_change_cost = 0 if has_color == required_color else 1

                # +1 for paint action itself
                cost = min_dist + color_change_cost + 1

                if best_cost_for_tile is None or cost < best_cost_for_tile:
                    best_cost_for_tile = cost

            if best_cost_for_tile is None:
                # No robot could reach painting positions -> unreachable in this heuristic's model.
                total_cost += self.unreachable_penalty
            else:
                total_cost += best_cost_for_tile

        # Ensure finite integer return
        return int(total_cost)