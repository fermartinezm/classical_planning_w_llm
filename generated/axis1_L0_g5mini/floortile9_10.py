from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the `floortile` domain that estimates the number of actions
    needed to achieve all (painted tile color) goals from a given state.
    The heuristic models the following costs:
      - Moving a robot one tile to an adjacent tile: cost 1 (per move action).
      - Changing a robot's held color to any other available color: cost 1.
      - Painting a tile (the paint action): cost 1.

    The heuristic is computed by greedily assigning remaining (unfulfilled)
    painting tasks to robots. For each unpainted goal tile, any adjacent tile
    (up/down/left/right) is a valid robot location from which the tile can be
    painted. The heuristic repeatedly picks the robot-to-task assignment that
    has minimum immediate cost (distance to a neighboring tile + optional
    change_color + paint) and simulates that assignment (updates the robot's
    position and color). Distances are computed on the static adjacency graph
    (ignoring dynamic blocking caused by painted tiles). This produces a
    fast, realistic estimate that is not necessarily admissible but tends to
    strongly guide greedy best-first search.

    Assumptions
    -----------
    - Moving between two adjacent tiles costs 1 (each move_* action).
    - Changing color costs 1 (change_color action).
    - Painting a tile costs 1 (paint_* action).
    - Adjacency information is read from static facts (up/down/left/right).
    - We ignore dynamic blocking that painting may introduce (i.e., we allow
      movement over the static adjacency graph even if a tile is not clear in
      the current state). This tends to underestimate true cost when painting
      blocks future movement, but keeps the heuristic fast to compute.
    - If a goal's required painted fact already exists in the state, that goal
      is considered satisfied and contributes 0 to the heuristic.
    - If a goal's tile has the wrong paint in the state, we still plan to add
      the correct paint fact (this follows the domain semantics where painting
      adds painted facts). The heuristic does not attempt to detect impossibility.

    Heuristic Initialization
    ------------------------
    - Extract goal requirements (the set of (painted tile color) goals).
    - Build an undirected adjacency graph of tiles from static facts:
      for each static fact (up a b), (down a b), (left a b), (right a b),
      an undirected edge is created between a and b. This graph is used to
      compute shortest path distances (in number of moves) between tiles.
    - Record available colors if present in static facts (not strictly needed,
      but available for potential extensions).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Parse the current state:
       - Locate robots and their current tile positions (facts "robot-at").
       - Record each robot's current held color (facts "robot-has").
       - Record which tiles are already painted with which colors (facts "painted").

    2. Identify the set of painting tasks:
       - For each goal (painted tileX colorC) in the task's goals, if the
         identical fact already exists in the state, treat the goal as achieved.
         Otherwise the tile is a pending painting task requiring colorC.

    3. For each pending task (tile t, required color c), identify the set of
       robot positions from which t can be painted: these are the static
       neighbors of t (any tile adjacent to t in the adjacency graph).

    4. Greedy assignment loop:
       - While there are pending tasks:
         a) For each robot and each pending task, compute the minimal cost to
            service that task assuming the robot starts at its current simulated
            position and color:
              cost = distance(robot_pos, chosen_neighbor)   # moves (>=0)
                   + (1 if robot_color != required_color else 0)  # change color
                   + 1  # paint action
            where chosen_neighbor is any neighbor of the goal tile; the robot
            must move to that neighbor and then execute (maybe change color)
            and paint.
         b) Select the (robot, task, neighbor) tuple with minimum cost.
         c) Add that cost to the running total heuristic value.
         d) Simulate the effect of that assignment on the robot:
            - robot.position := chosen_neighbor (robot remains there after painting)
            - robot.color := required_color (if we counted a change_color action)
         e) Remove the task from the pending set and continue.

    5. If any task is unreachable (no path to any neighbor in the adjacency
       graph), a large finite penalty is returned (to keep the heuristic finite).
       This keeps the heuristic defined even for disconnected but solvable
       instances; for truly solvable instances with connected graphs the result
       will be finite and meaningful.

    Notes on design choices
    -----------------------
    - The greedy assignment with simulation of robot positions and colors is
      inexpensive and captures important sequencing effects (robots moving and
      changing colors between paints) while remaining fast to compute.
    - The heuristic ignores painted tiles' dynamic blocking; this reduces
      computational overhead and keeps costs low, which usually improves
      guidance of greedy search in this domain.
    """

    def __init__(self, task):
        """Extract goals and static adjacency information from the task."""
        self.goals = set(task.goals)  # goal facts as strings
        static_facts = task.static

        # Map: tile -> set(neighbor_tiles) (undirected adjacency)
        self.adjacency = {}
        self.available_colors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("up", "down", "left", "right"):
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)
            elif pred == "available-color":
                self.available_colors.add(parts[1])

        # Extract goal mapping: tile -> required color for painted goals
        # (there might be non-painted goals, but floortile problems use painted).
        self.goal_paints = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) >= 3:
                tile = parts[1]
                color = parts[2]
                self.goal_paints[tile] = color

    def _bfs_distances(self, start):
        """Return a dict mapping tile -> distance (moves) from start on adjacency graph."""
        # If start not in adjacency, return empty dict (unreachable)
        if start not in self.adjacency:
            return {}
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
        """Compute the heuristic value for the given search node (state)."""
        state = node.state

        # Quick goal check: if all painted goals already in state, return 0.
        if self.goals <= state:
            return 0

        # Parse robot positions and colors from the current state.
        robots = {}  # robot -> {"pos": tile, "color": color}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robots.setdefault(robot, {})["pos"] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robots.setdefault(robot, {})["color"] = color

        # Ensure each robot has a pos and a color; if missing, give defaults.
        for r, info in robots.items():
            info.setdefault("pos", None)
            info.setdefault("color", None)

        # Current painted facts in the state
        painted_now = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) >= 3:
                tile, color = parts[1], parts[2]
                painted_now.setdefault(tile, set()).add(color)

        # Build set of tasks: tiles that need painting with a specific color
        pending = {}
        for tile, req_color in self.goal_paints.items():
            # If the exact painted predicate already true, skip
            if tile in painted_now and req_color in painted_now[tile]:
                continue
            pending[tile] = req_color

        # If no pending tasks, heuristic is 0
        if not pending:
            return 0

        # Precompute BFS distances from every tile that is a robot position
        # or that is needed as neighbor for any pending task.
        # We'll compute BFS from robot positions on demand.
        precomputed_dists = {}

        # Convert adjacency to have empty sets for tiles that appear only in goals but not in static.
        for t in list(pending.keys()):
            self.adjacency.setdefault(t, set())

        # Simulated robots for greedy scheduling: list of dicts with 'pos' and 'color'
        sim_robots = []
        for r, info in robots.items():
            sim_robots.append({"name": r, "pos": info.get("pos"), "color": info.get("color")})

        # If there are no robots, we cannot paint; return a large finite estimate
        if not sim_robots:
            return 9999

        total_cost = 0

        # Greedy assignment loop
        while pending:
            best = None  # (cost, robot_index, task_tile, neighbor_tile, change_color_flag, dist_val)
            for i, robot in enumerate(sim_robots):
                rpos = robot["pos"]
                rcolor = robot["color"]
                # If robot position is None or not in adjacency graph, still attempt BFS (will be empty)
                if rpos not in precomputed_dists:
                    precomputed_dists[rpos] = self._bfs_distances(rpos) if rpos is not None else {}
                dists = precomputed_dists[rpos]

                for tile, req_color in pending.items():
                    # Possible robot positions to paint this tile are neighbors of the tile
                    neighbors = self.adjacency.get(tile, set())
                    if not neighbors:
                        # no neighbor information => unreachable in our static graph
                        continue
                    # For each neighbor, compute distance from robot to neighbor
                    best_dist_for_task = None
                    best_neighbor = None
                    for n in neighbors:
                        # distance from rpos to n
                        dist = dists.get(n)
                        if dist is None:
                            # If not known, try computing bfs from rpos to populate (already done)
                            continue
                        if best_dist_for_task is None or dist < best_dist_for_task:
                            best_dist_for_task = dist
                            best_neighbor = n
                    if best_dist_for_task is None:
                        # robot cannot reach any neighbor of this tile
                        continue
                    change_cost = 0 if (rcolor == req_color) else 1
                    paint_cost = 1
                    cost = best_dist_for_task + change_cost + paint_cost
                    candidate = (cost, i, tile, best_neighbor, change_cost == 1, best_dist_for_task)
                    if best is None or candidate[0] < best[0]:
                        best = candidate

            if best is None:
                # No reachable assignments for remaining tasks in the static graph.
                # Return a large but finite estimate.
                return 9999

            cost, r_idx, tile_chosen, neighbor_chosen, change_flag, dist_val = best

            # Apply chosen assignment to simulation
            total_cost += cost
            sim_robot = sim_robots[r_idx]
            # Robot ends up at the neighbor tile after moving and painting
            sim_robot["pos"] = neighbor_chosen
            # Robot color updates if we counted a change
            if change_flag:
                sim_robot["color"] = pending[tile_chosen]
            # Remove the task
            del pending[tile_chosen]

            # Optionally precompute distances from the updated robot position if not done
            if sim_robot["pos"] not in precomputed_dists:
                precomputed_dists[sim_robot["pos"]] = self._bfs_distances(sim_robot["pos"])

        # Return total estimated number of actions (int)
        return int(total_cost)