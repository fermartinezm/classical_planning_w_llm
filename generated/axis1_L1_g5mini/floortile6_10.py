from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split fact into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'secret_domain' (tile-painting with robots).
    The domain consists of robots that move on a directed grid and paint tiles with colors.
    Goals are of the form (yh6q5j8r <tile> <color>) meaning "<tile> must have <color>".
    Movement between tiles is given by several static predicates (directed edges).
    Painting a tile requires a robot to be on an adjacent tile, the tile to be free,
    and the robot to hold the desired color. Robots can change held color with one
    action (z7pcmkeu). Movement between adjacent tiles costs one action, painting costs
    one action, and changing color costs one action.

    Assumptions
    -----------
    - Movement along any of the static adjacency predicates costs 1 action.
    - Changing a robot's held color costs 1 action (always possible if the color is in static facts).
    - Painting a tile costs 1 action and requires the target tile to be "free" (d7ekk7vb true).
    - If a tile is not free in the current state, we approximate the cost to make it free as 1 action
      (a conservative inexpensive estimate, since freeing may require moving an occupying robot).
    - We ignore dynamic blocking constraints during path search (we compute shortest paths on the static
      directed graph and do not model intermediate occupancy changes). This keeps the heuristic fast.
    - Multiple robots are handled by a greedy assignment: repeatedly assign the currently cheapest
      (robot, tile) pair, simulate that robot painting the tile (update its position and color),
      and continue until all goal tiles are assigned.

    Heuristic Initialization
    ------------------------
    During construction the heuristic extracts and stores:
    - goal_tiles: mapping tile -> desired_color for each yh6q5j8r goal fact.
    - movement_graph: directed adjacency map (tile -> set(neighbor_tiles)) derived from static facts
      for predicates xr2t1c1d, oh0smogc, w224no0l, g7ya35is. A static fact "(pred v u)" means there is
      a directed move from u -> v (as the domain operators use the pattern (pred aepxj ajv3b) with the
      robot at ajv3b to move to aepxj).
    - painting_neighbors: for each tile t, the set of robot positions from which t can be painted. These
      are the second arguments u in static facts "(pred t u)".
    - colors: the set of valid colors present in static (cdb9op3o facts).
    The static extraction is done once for efficiency.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given planner node/state the heuristic works as follows:

    1. Fast goal-check:
       - If all goal facts are already true in the state, return 0.

    2. Parse current state into useful structures:
       - robots_pos: mapping robot -> tile where the robot currently stands (facts gk9u3vg3).
       - robots_color: mapping robot -> held_color (facts i4u94rl9).
       - free_tiles: set of tiles that currently have the fact d7ekk7vb (considered free).
       - painted_facts: set of (yh6q5j8r <tile> <color>) facts present in the state.

    3. Identify the set of goal tiles that still require painting (goal facts not in state).

    4. Precompute BFS distances on the static movement graph from any robot position when needed.
       These distances represent the minimal number of move actions to reach any tile (we ignore dynamic
       occupancy constraints for speed).

    5. Greedy assignment loop:
       - While there are unpainted goal tiles:
         a) For each remaining tile and each robot, compute a candidate cost:
            - Find the neighbor tile n from which the tile can be painted (n in painting_neighbors[tile])
              that minimizes BFS distance from the robot's current position.
            - distance_cost = number of moves to get the robot to n (0 if already there).
            - color_cost = 0 if robot already holds the goal color, else 1 (one color-change action).
            - free_cost = 0 if tile is currently free, else 1 (approximate cost to make it free).
            - paint_cost = 1 (the painting action itself).
            - candidate_cost = distance_cost + color_cost + free_cost + paint_cost
         b) Select the (robot, tile, neighbor) with minimal candidate_cost.
         c) Add that candidate_cost to the accumulated heuristic value.
         d) Simulate the effect of that painting in our heuristic's bookkeeping:
            - Update the chosen robot's position to the neighbor tile (robots do not move as a result of painting).
            - Update the chosen robot's color to the goal color (painting implies the robot used that color).
            - Mark the tile as painted (remove it from the unpainted set) and update free_tiles to reflect
              that painted tiles become non-free (d7ekk7vb removed).
         e) Repeat until all goal tiles are assigned or until no reachable assignment is found.

    6. If at some point a tile is unreachable by any robot (based on static adjacency), return a large finite
       heuristic (which signals difficulty but keeps value finite).

    The heuristic returns the accumulated estimated action count from the greedy assignment. It is
    inadmissible (it is an approximation) but tries to capture movement + color-change + painting + freeing costs
    and to account for multiple robots by simulating assignments greedily.
    """

    def __init__(self, task):
        self.goals = task.goals  # set of goal fact strings
        static_facts = task.static

        # Predicates that encode directed adjacency information in static facts.
        adjacency_preds = {"xr2t1c1d", "oh0smogc", "w224no0l", "g7ya35is"}

        # Build movement graph and painting neighbor mapping.
        # A static fact "(pred v u)" indicates movement from u -> v, and that u is a valid robot
        # position to paint tile v (since painting preconditions use the pattern (pred aepxj ajv3b)).
        self.movement_graph = {}  # tile -> set(neighbor_tiles) (directed edges)
        self.painting_neighbors = {}  # tile -> set(robot_positions)
        self.colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in adjacency_preds and len(parts) >= 3:
                v = parts[1]
                u = parts[2]
                # movement: u -> v
                self.movement_graph.setdefault(u, set()).add(v)
                # ensure nodes exist
                self.movement_graph.setdefault(v, set())
                # painting: to paint v, robot may be at u (record u as a painting neighbor for v)
                self.painting_neighbors.setdefault(v, set()).add(u)
            elif pred == "cdb9op3o" and len(parts) >= 2:
                self.colors.add(parts[1])

        # Some tiles might not appear as keys; ensure default empty sets
        for tile in list(self.painting_neighbors.keys()):
            self.movement_graph.setdefault(tile, set())

        # Extract goal mapping tile -> color
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "yh6q5j8r" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        # Pre-allocated large value for unreachable/very expensive situations
        self.INF = 10 ** 6

        # Cache for BFS results keyed by start tile
        self._bfs_cache = {}

    def _bfs_distances(self, start):
        """
        BFS on the directed static movement graph from start tile.
        Returns a dict tile -> distance (number of moves).
        Caches results to avoid recomputation when a robot revisits the same tile in heuristic simulation.
        """
        if start in self._bfs_cache:
            return self._bfs_cache[start]

        dist = {}
        q = deque()
        dist[start] = 0
        q.append(start)
        while q:
            u = q.popleft()
            for v in self.movement_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._bfs_cache[start] = dist
        return dist

    def __call__(self, node):
        state = node.state

        # Quick goal check: return 0 for goal states only.
        if self.goals <= state:
            return 0

        # Parse state: robots' positions, robots' colors, which tiles are free, which tiles already painted.
        robots_pos = {}  # robot -> tile
        robots_color = {}  # robot -> color
        free_tiles = set()  # tiles with d7ekk7vb
        painted_facts = set()  # current yh6q5j8r facts in state

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "gk9u3vg3" and len(parts) >= 3:
                robot = parts[1]
                tile = parts[2]
                robots_pos[robot] = tile
            elif pred == "i4u94rl9" and len(parts) >= 3:
                robot = parts[1]
                color = parts[2]
                robots_color[robot] = color
            elif pred == "d7ekk7vb" and len(parts) >= 2:
                free_tiles.add(parts[1])
            elif pred == "yh6q5j8r" and len(parts) >= 3:
                painted_facts.add(fact)

        # Determine which goal tiles still need painting.
        unpainted_tiles = []
        for tile, color in self.goal_tiles.items():
            goal_fact = f"(yh6q5j8r {tile} {color})"
            if goal_fact not in state:
                unpainted_tiles.append(tile)

        # If there are no robots, painting is impossible -> return large finite heuristic.
        robot_list = list(robots_pos.keys())
        if not robot_list:
            return self.INF

        # Greedy assignment simulation:
        total_cost = 0

        # Make mutable copies for simulation bookkeeping.
        robots_pos_sim = dict(robots_pos)
        robots_color_sim = dict(robots_color)
        free_tiles_sim = set(free_tiles)
        unpainted = set(unpainted_tiles)

        # If some robots have no color specified in the state, assume they can change to any color by one action.
        # Represent missing color as None.
        for r in robot_list:
            robots_color_sim.setdefault(r, None)

        # Precompute painting_neighbors for tiles that might not be in static -> ensure keys exist
        for t in list(unpainted):
            self.painting_neighbors.setdefault(t, set())
            self.movement_graph.setdefault(t, set())

        # Assignment loop
        while unpainted:
            best = None  # tuple (cost, robot, tile, chosen_neighbor, distance)
            # For each robot, compute BFS distances from its current position
            bfs_cache_for_robot = {}
            for r in robot_list:
                pos = robots_pos_sim.get(r)
                if pos is None:
                    bfs_cache_for_robot[r] = {}
                else:
                    # BFS distances from current robot position (cached)
                    bfs_cache_for_robot[r] = self._bfs_distances(pos)

            # Evaluate candidate costs for each (robot, tile) pair
            for tile in unpainted:
                desired_color = self.goal_tiles[tile]
                # if no painting neighbor exists at all, tile is unreachable
                neighbors = self.painting_neighbors.get(tile, set())
                if not neighbors:
                    continue
                tile_free = tile in free_tiles_sim
                free_cost = 0 if tile_free else 1
                for r in robot_list:
                    pos = robots_pos_sim.get(r)
                    dists = bfs_cache_for_robot[r]
                    # Find best neighbor (closest) for this robot
                    best_neighbor = None
                    best_dist = None
                    for n in neighbors:
                        # Distance from pos to n
                        if pos is None:
                            # If robot has no known position, treat as unreachable in static graph.
                            continue
                        dist = dists.get(n)
                        if dist is None:
                            continue
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_neighbor = n
                    if best_dist is None:
                        # this robot cannot reach any neighbor for this tile
                        continue
                    # color change cost: 0 if already has desired_color, else 1
                    current_color = robots_color_sim.get(r)
                    color_cost = 0 if current_color == desired_color else 1
                    paint_action_cost = 1
                    candidate_cost = best_dist + color_cost + free_cost + paint_action_cost
                    if best is None or candidate_cost < best[0]:
                        best = (candidate_cost, r, tile, best_neighbor, best_dist, color_cost, free_cost)

            if best is None:
                # No reachable assignment for any tile -> return large value
                return self.INF

            # Apply chosen assignment in the simulation bookkeeping
            (cost, robot, tile, neighbor, dist_used, color_cost_used, free_cost_used) = best
            total_cost += cost

            # Update robot position and color after painting:
            # - Robot remains on the neighbor tile after painting.
            robots_pos_sim[robot] = neighbor
            robots_color_sim[robot] = self.goal_tiles[tile]  # robot used that color
            # Update free tiles: painted tile becomes non-free (d7ekk7vb removed).
            if tile in free_tiles_sim:
                free_tiles_sim.remove(tile)
            # Mark the tile as painted (remove from unpainted).
            unpainted.remove(tile)

            # Continue until all painted

        # Return accumulated heuristic estimate (integer)
        return int(total_cost)