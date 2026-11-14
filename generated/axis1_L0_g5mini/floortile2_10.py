from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the floortile domain. The heuristic estimates
    the number of actions needed to satisfy all (painted TILE COLOR) goal
    facts from a given state. It is tuned to guide greedy best-first search
    (not admissible) and is efficiently computable.

    Assumptions
    -----------
    - Movement between adjacent tiles costs 1 per step (as in the domain).
    - Painting a tile costs 1 action (paint_* action).
    - Changing the robot's held color costs 1 action (change_color).
    - Robots can move freely along the static adjacency graph (we IGNORE
      dynamic blocking due to clear/not-clear except for detecting already
      incorrectly-painted tiles). This makes the heuristic inadmissible but
      fast and effective in practice.
    - If a goal tile is painted with a different color in the current state,
      the problem is treated as unsolvable (the domain does not provide an
      action to "clear" a painted tile), so the heuristic returns a large
      value in that case.
    - We estimate the cost of each remaining goal tile independently by
      selecting the robot that can reach any neighbor of the target tile with
      minimal movement cost, adding 1 for painting and 1 if the robot needs
      to change color. We then sum these per-tile estimates. This is a
      simplification (it may overcount movement if a single robot paints many
      tiles sequentially) but is cheap and reduces node expansions.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static adjacency relations from
    the task.static facts to build an undirected graph of tile neighbors.
    It also extracts the set of available colors (not strictly needed for the
    estimate but kept for completeness) and maps goal (painted ...) facts to
    a dictionary goal_tile -> goal_color.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Quick goal check: if all goal facts already hold in the state return 0.
    2. Parse the current state to obtain:
       - robot positions (robot -> tile)
       - robot held colors (robot -> color)
       - current painted tiles (tile -> color)
    3. Sanity check: if any goal tile is already painted with a different
       color than required, return a large value (unsolvable under the domain).
    4. For each goal tile that is not yet painted with the required color:
       a. Consider all neighbor tiles (from static adjacency). A robot must be
          located on one of these neighbor tiles to perform the paint action.
       b. For each robot, compute the shortest path distance (in the static
          adjacency graph) from its current tile to each neighbor tile.
       c. For a robot reaching a neighbor, estimate its cost to achieve the
          painting of the target as:
              move_distance + color_change_cost + 1 (painting)
          where color_change_cost is 0 if robot already has the required color,
          otherwise 1.
       d. Take the minimum cost over robots and neighbors for this goal tile.
       e. If no robot can reach any neighbor (disconnected), treat as
          unsolvable and return a large value.
    5. Sum the per-goal-tile minimal costs and return that sum as the heuristic.
    """

    def __init__(self, task):
        """Extract static adjacency and goal requirements."""
        self.goals = task.goals  # set or frozenset of goal facts as strings
        static = task.static

        # Build undirected adjacency graph between tiles. We consider any of the
        # predicates up/down/left/right as an edge between the two tiles.
        self.neighbors = {}  # tile -> set(neighbor_tiles)
        self.available_colors = set()
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                a, b = parts[1], parts[2]
                self.neighbors.setdefault(a, set()).add(b)
                self.neighbors.setdefault(b, set()).add(a)
            elif pred == "available-color":
                self.available_colors.add(parts[1])

        # Map goal tiles -> required color for quick lookup.
        self.goal_map = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_map[tile] = color

    def _bfs_distances(self, start_tile):
        """BFS on the static adjacency graph returning a dict tile -> distance."""
        dist = {start_tile: 0}
        q = deque([start_tile])
        while q:
            t = q.popleft()
            for nb in self.neighbors.get(t, ()):
                if nb not in dist:
                    dist[nb] = dist[t] + 1
                    q.append(nb)
        return dist

    def __call__(self, node):
        """Compute the heuristic value for the node's state."""
        state = node.state

        # If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse state: robot positions, robot colors, painted tiles.
        robot_pos = {}  # robot -> tile
        robot_color = {}  # robot -> color
        painted = {}  # tile -> color

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "robot-at":
                # (robot-at robot1 tile_3_2)
                robot, tile = parts[1], parts[2]
                robot_pos[robot] = tile
            elif pred == "robot-has":
                # (robot-has robot1 white)
                robot, color = parts[1], parts[2]
                robot_color[robot] = color
            elif pred == "painted":
                # (painted tile_1_1 white)
                tile, color = parts[1], parts[2]
                painted[tile] = color

        # Check for unsolvable goals: a goal tile painted with different color
        for tile, required_color in self.goal_map.items():
            if tile in painted and painted[tile] != required_color:
                return INF

        # Prepare BFS distances for each robot.
        robot_dists = {}
        for robot, pos in robot_pos.items():
            # If robot position not in adjacency graph, BFS will only include start.
            robot_dists[robot] = self._bfs_distances(pos)

        total_cost = 0

        # For each goal tile not yet satisfied, estimate minimal cost to paint it.
        for tile, required_color in self.goal_map.items():
            if tile in painted and painted[tile] == required_color:
                continue  # already satisfied

            # The robot must be on a neighbor tile of 'tile' to paint it.
            neighs = self.neighbors.get(tile, set())
            if not neighs:
                # No neighbor information: cannot paint this tile.
                return INF

            best_for_tile = INF

            # Try all robots and all neighbor tiles; compute minimal cost to paint.
            for robot, dmap in robot_dists.items():
                # robot may not have a recorded color (shouldn't happen), assume change needed.
                r_color = robot_color.get(robot, None)
                color_change_cost = 0 if r_color == required_color else 1

                # For each neighbor compute distance to reach it.
                for n in neighs:
                    # If robot is already on the neighbor, distance is 0.
                    # If neighbor not reachable in static graph, skip.
                    if n not in dmap:
                        continue
                    move_cost = dmap[n]
                    # Painting itself costs 1.
                    cost = move_cost + color_change_cost + 1
                    if cost < best_for_tile:
                        best_for_tile = cost

            if best_for_tile == INF:
                # No robot can reach any neighbor -> unsolvable under static graph.
                return INF

            total_cost += best_for_tile

        # Heuristic is the sum of per-tile minimal estimates.
        return int(total_cost)