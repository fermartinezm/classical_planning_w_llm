from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into components."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a fact matches a pattern (supports '*' wildcards)."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the "color" (secret_domain) planning domain. The domain's tasks
    require painting specific tile-2 objects with a paint_up action. To apply
    paint_up for a goal tile G the following must hold:
      - down G
      - there exists a tile position P such that:
          - (available-color G P)
          - (available-color R P)   for some robot-like object R (first paint param)
          - (available-color C P)   for some color C (second paint param for move_down)
        and the robot-at fact (robot-at R C) must hold or be created via move_down.
      - (free-color C) must hold for the color C used by the robot at paint time.
    The heuristic estimates the number of actions required to make these preconditions
    true and apply paint_up. It accounts for:
      - the number of change_color actions needed to move available-color facts along
        the static `up` graph to align the required objects at the same tile P,
      - a move_down action if robot-at (R,C) must be created,
      - the paint_up action itself.
    The heuristic is not admissible but aims to be informative and cheaply computable.

    Assumptions
    -----------
    - change_color moves an available-color fact forward by exactly one `up` edge.
      Thus the minimal number of change_color actions required to move an object's
      available-color from tile A to tile B is the forward distance along `up`.
    - There is no action that creates an available-color out of thin air; it must
      exist initially for an object and can only be moved forward along `up`.
    - free-color facts are consumed by paint_up and cannot be recreated; therefore,
      the number of paints that can be executed is bounded by the number of free-color
      facts present in the current state. The heuristic respects this by returning a
      large value when there are fewer free colors than unpainted goals (these states
      are unsolvable unless future actions produce free-color, which the domain does
      not provide).
    - The heuristic treats goals independently (optimistic decomposition). It does
      not attempt to exploit shared movement costs between different goals beyond
      the implicit availability of free-color resources.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - The set of goal tile-2 object names to be painted.
      - The static `up` graph (mapping tile -> tile' for every (up tile tile')).
      - The set of all tiles appearing in the static facts (used as BFS nodes).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node (state), the heuristic proceeds as follows:

    1. Parse the state into useful structures:
       - available_pos[obj] -> tile where (available-color obj tile) holds
       - free_colors -> set of objects that currently have (free-color X)
       - robot_at_pairs -> set of (R, C) tuples where (robot-at R C) holds
       - painted -> set of tile-2 objects already painted

    2. Quick checks:
       - If all goal tiles are already painted: heuristic = 0
       - If the number of remaining goals is greater than the number of free colors:
         return a large finite value (heuristic believes the state is unsolvable
         because paint_up consumes free-color and there is no action to add them).

    3. Precompute forward reachable distances along the `up` graph:
       For every object O that has an available-color at tile S, compute distances
       from S to every tile reachable by repeatedly following `up` edges.
       Distance of 0 at the starting tile S (meaning no change_color needed).

    4. For each unpainted goal tile G:
       - If G has no available-color anywhere in the current state, the goal is
         unreachable -> assign a large cost for that goal.
       - Otherwise, let dist_G[t] be the forward distance for the available-color
         of G to reach tile t. We consider candidate meeting tiles t (reachable
         by G).
       - For each candidate tile t, we estimate two possibilities:
         A) Use an existing robot-at (R,C) where C is free:
            - Need to move available-color for R to t and also move G's available-color
              to t. Cost = dist_R[t] + dist_G[t] + 1 (one paint_up action).
            - No move_down required.
         B) Create a robot-at via move_down using a free color C and robot R:
            - Need available-color for R and C at t and for G at t.
            - Cost = dist_R[t] + dist_C[t] + dist_G[t] + 1 (move_down) + 1 (paint_up).
       - Take the minimal cost across all choices of R, C (where C must be in free_colors)
         and tiles t.

    5. Sum the per-goal minimal costs and return this sum as the heuristic value.

    Notes on completeness and safety:
    - The heuristic always returns 0 for goal states.
    - If it detects obvious unsolvability due to lack of free colors or lack of
      required available-color facts, it returns a large finite number rather than
      infinity, to remain usable for greedy search algorithms.
    """

    def __init__(self, task):
        """Extract static information and goals from the given task."""
        self.goals = set()  # set of goal facts as strings
        for g in task.goals:
            # store goal facts as strings (the planner represents them as such)
            self.goals.add(g)

        # Build up-graph from static facts:
        self.up_graph = defaultdict(list)  # tile -> list of tiles reachable in one up step
        self.tiles = set()
        for fact in task.static:
            if match(fact, "up", "*", "*"):
                a, b = get_parts(fact)[1], get_parts(fact)[2]
                self.up_graph[a].append(b)
                self.tiles.add(a)
                self.tiles.add(b)

    def _bfs_distances_from(self, start_tile):
        """
        BFS forward along up edges from start_tile returning a dict
        tile -> distance (0 at start_tile). If start_tile is None or
        not in tiles, returns empty dict.
        """
        d = {}
        if start_tile is None:
            return d
        q = deque()
        d[start_tile] = 0
        q.append(start_tile)
        while q:
            u = q.popleft()
            for v in self.up_graph.get(u, []):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def __call__(self, node):
        """Compute and return the heuristic value for the given search node."""
        state = node.state

        # Parse the state
        available_pos = {}  # object -> tile where its available-color currently is
        free_colors = set()  # objects that have free-color
        robot_at_pairs = set()  # (R, C) pairs for robot-at facts
        painted = set()  # painted tile-2 objects

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj = parts[1]
                tile = parts[2]
                available_pos[obj] = tile
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "robot-at" and len(parts) >= 3:
                robot_at_pairs.add((parts[1], parts[2]))
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])

        # Extract goal tile names from the stored goal facts (they are strings like "(painted tile1)")
        goal_tiles = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 2:
                goal_tiles.append(parts[1])

        # If all goals satisfied -> heuristic 0
        unpainted_goals = [g for g in goal_tiles if g not in painted]
        if not unpainted_goals:
            return 0

        # If there are fewer free colors than unpainted goals, the planner cannot
        # execute that many paints (paint_up consumes free-color and no action
        # re-adds it). Return a large value to indicate likely unsolvability.
        if len(free_colors) < len(unpainted_goals):
            return 1000  # large finite heuristic

        # Precompute forward distances for every object that has an available-colour
        forward_dist = {}  # object -> dict(tile -> distance)
        for obj, tile in available_pos.items():
            # BFS along the up graph starting at tile:
            forward_dist[obj] = self._bfs_distances_from(tile)
            # also include the starting tile even if it has no outgoing up edges:
            if tile not in forward_dist[obj]:
                forward_dist[obj][tile] = 0

        total_cost = 0
        INF_COST = 500  # used for unreachable single-goal cases

        # For each unpainted goal, estimate minimal cost to paint it
        for g in unpainted_goals:
            # g must have an available-color somewhere to be painted
            if g not in available_pos:
                # cannot move an available-color for g -> unreachable under domain ops
                total_cost += INF_COST
                continue

            dist_g = forward_dist.get(g, {})
            if not dist_g:
                total_cost += INF_COST
                continue

            best_for_goal = INF_COST

            # Candidate meeting tiles are tiles reachable by g's available-color
            candidate_tiles = dist_g.keys()

            # Consider scenario A: reuse an existing robot-at (R,C) where C is free
            for (R, C) in robot_at_pairs:
                if C not in free_colors:
                    continue  # cannot use this robot_at because color not free
                if R not in forward_dist:
                    continue  # R has no available-color to satisfy paint_up precond
                dist_R = forward_dist[R]
                for t in candidate_tiles:
                    if t in dist_R and t in dist_g:
                        cost = dist_R[t] + dist_g[t] + 1  # 1 paint_up
                        if cost < best_for_goal:
                            best_for_goal = cost

            # Consider scenario B: create robot-at via move_down using a free color C and robot R
            # This requires available-color for R, C and g to meet at tile t:
            if free_colors:
                # iterate possible R (objects that have available-pos)
                for R, dist_R in forward_dist.items():
                    for C in free_colors:
                        if C not in forward_dist:
                            continue  # C must be movable (have an available-color) to reach tile t
                        dist_C = forward_dist[C]
                        # iterate candidate tiles t where g, R and C can reach
                        for t in candidate_tiles:
                            if t in dist_R and t in dist_C and t in dist_g:
                                cost = dist_R[t] + dist_C[t] + dist_g[t] + 1 + 1
                                # +1 move_down , +1 paint_up
                                if cost < best_for_goal:
                                    best_for_goal = cost

            # If no combination was found best_for_goal may still be INF_COST
            total_cost += best_for_goal

        # Return a finite heuristic (cap it to avoid extremely large numbers)
        return min(total_cost, 10000)