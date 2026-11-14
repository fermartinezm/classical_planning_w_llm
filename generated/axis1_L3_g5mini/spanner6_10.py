from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the space-separated parts of a PDDL fact string, removing surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (color) domain. The heuristic
    estimates the number of actions required to achieve all "painted" goals by
    reasoning about where paintable tile-2 objects are located (available-color
    facts), where robot color-2 objects and color objects are located, and how many
    change_color / move_down operations are needed to bring the required facts
    together for a paint_up action.

    Assumptions
    -----------
    - Each goal is a literal of the form (painted <tile-2>).
    - To paint a target tile-2 object X, exactly one paint_up action is required
      (its preconditions must be satisfied first).
    - change_color moves an available-color object forward along one "up" edge.
      Thus the number of change_color actions equals the number of up-edges traversed.
    - move_down creates a robot-at relation for a color-2 object at a tile if both
      an available-color (color-2) and an available-color (color) are located at
      that same tile; it costs 1 action.
    - paint_up itself costs 1 action.
    - free-color predicates are required by paint_up. We treat colors marked
      free in the state as usable and do not model freeing of colors (no action
      re-adds free-color), so only free colors are considered usable for painting.
    - For efficiency the heuristic computes a local, additive estimate:
      for each unpainted goal tile it computes a minimal local cost to make a
      paint_up possible at the tile's current available position(s). Costs for
      different goals are summed. (This is not admissible in general but gives
      a useful, quickly-computed estimate.)

    Heuristic Initialization
    ------------------------
    - Extract static "up" facts from task.static and build a forward adjacency map
      representing allowed single-step change_color movements.
    - Precompute no distances eagerly (computed on demand); the up graph is kept
      for quick BFS distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node:
    1. If the node's state already satisfies all goals, return 0.
    2. Parse the state's relevant predicates:
       - available-color <obj> <tile>: record available positions for every object.
       - robot-at <color-2> <color>: record existing robot-at pairs.
       - free-color <color>: collect currently free colors.
       - painted <tile-2>: which goals are already satisfied.
    3. For every goal tile G that is not yet painted:
       a. Determine the set of tile-positions P where (available-color G P) holds.
          If empty, we conservatively assign a large finite cost for this goal (the
          planner may not be able to paint it from the current state).
       b. For each candidate position p in P, compute a minimal local cost to
          enable paint_up at p. The paint_up preconditions require:
             - (available-color G p)   -- already true by choice of p
             - (available-color R p)   -- some color-2 object R available at p
             - (robot-at R C) and (free-color C)
             - (down G)                -- assumed true for unpainted goals
          We consider several ways to achieve this:
             - If some robot-at R C (with C free) already exists and R is already
               available at p, cost = 1 (one paint_up).
             - If robot-at R C exists and C is free, but R is at some other tile q,
               we can move R forward along "up" edges until p. Each up-edge is one
               change_color action. Cost = distance(q -> p) + 1 (paint_up).
             - If no suitable robot-at exists, we can create one at p by ensuring
               both a color-2 R and a free color C are available at p and then
               execute move_down (cost 1) followed by paint_up (cost 1). If R
               or C are not at p, we add change_color costs to bring them to p.
       c. Take the minimal cost among all choices (R, C, p) for goal G.
    4. Sum the minimal per-goal costs to obtain the heuristic estimate.
    5. If for some goal no feasible (finite) option exists given the up-graph and
       current available-color positions, the heuristic returns a large finite
       value (cap) to keep the value finite while indicating difficulty.
    """

    def __init__(self, task):
        # store goals (set of fact strings)
        self.goals = task.goals
        # static up-edges: build adjacency map for forward movement (v7 -> v8 when (up v7 v8))
        self.up_adj = {}  # tile -> set(of tiles reachable in one change_color)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.up_adj.setdefault(src, set()).add(dst)
        # cache for BFS distances between tiles (from a start tile to any reachable tile)
        self._dist_cache = {}  # (start_tile, target_tile) -> distance (int) or None if unreachable
        # heuristic cap: returned when a goal appears unreachable by our simplified checks
        self.CAP = 1000

    def _forward_distance(self, start, target):
        """
        Compute minimal number of change_color (single up-edge moves) needed
        to move an available-color object from tile `start` to tile `target`.
        Returns an integer >= 0, or None if unreachable.
        Uses BFS and caches results.
        """
        if start == target:
            return 0
        key = (start, target)
        if key in self._dist_cache:
            return self._dist_cache[key]
        # BFS forward along up edges
        visited = set([start])
        q = deque([(start, 0)])
        found = None
        while q:
            node, d = q.popleft()
            for nbr in self.up_adj.get(node, ()):
                if nbr in visited:
                    continue
                if nbr == target:
                    found = d + 1
                    q.clear()
                    break
                visited.add(nbr)
                q.append((nbr, d + 1))
        self._dist_cache[key] = found
        return found

    def _min_distance_from_any(self, starts, target):
        """
        Given a collection of start tiles (iterable), return the minimal forward distance
        to `target` by change_color moves. If any start equals target, distance is 0.
        If none can reach, returns None.
        """
        best = None
        for s in starts:
            dist = self._forward_distance(s, target)
            if dist is None:
                continue
            if best is None or dist < best:
                best = dist
                if best == 0:
                    break
        return best

    def __call__(self, node):
        state = node.state

        # 1) If goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse state facts relevant for reasoning.
        available_positions = {}  # obj -> set of tiles where available-color obj tile holds
        robot_at = set()  # set of tuples (color2_obj, color_obj)
        free_colors = set()  # set of color names that are free
        painted = set()  # set of painted tile-2 names

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, tile = parts[1], parts[2]
                available_positions.setdefault(obj, set()).add(tile)
            elif pred == "robot-at" and len(parts) >= 3:
                color2, color = parts[1], parts[2]
                robot_at.add((color2, color))
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])

        # Determine which goal-painted tiles remain
        goal_tiles = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted":
                tile = parts[1]
                if tile not in painted:
                    goal_tiles.append(tile)

        total_cost = 0

        # All candidate color-2 objects we might use are those that appear as keys in available_positions
        # (they can be moved) and any color-2 already present in robot_at relations.
        candidate_Rs = set(available_positions.keys()) | {r for (r, _) in robot_at}
        # Candidate color objects that can be used must both be free and appear as available somewhere
        candidate_Cs = {c for c in free_colors if c in available_positions}

        # If there are no candidate_Cs, painting may be impossible by our model -> return CAP
        if not candidate_Cs and not any((r_c in free_colors) for (_, r_c) in robot_at):
            # No free color available anywhere; indicate very costly/unachievable
            return self.CAP

        # For each goal tile compute minimal local cost
        for goal_tile in goal_tiles:
            best_for_goal = None

            # positions where the tile-2 is available (where it can be painted)
            target_positions = available_positions.get(goal_tile, set())
            if not target_positions:
                # if the tile is not available anywhere (unexpected), we set a large finite cost
                total_cost += self.CAP
                continue

            # Try each candidate painting position
            for p in target_positions:
                # For each candidate R (color-2 object)
                for R in candidate_Rs:
                    R_positions = available_positions.get(R, set())
                    # If R has no available positions at all, skip
                    if not R_positions:
                        continue

                    # Case 1: there exists an existing robot-at (R, C) where C is free.
                    for (rR, rC) in robot_at:
                        if rR != R:
                            continue
                        if rC not in free_colors:
                            continue  # robot color not free -> can't paint with it
                        # we can use (rR, rC) if R can be brought to p
                        dist_R = 0 if p in R_positions else self._min_distance_from_any(R_positions, p)
                        if dist_R is None:
                            continue
                        cost = dist_R + 1  # change_color moves + paint_up
                        if best_for_goal is None or cost < best_for_goal:
                            best_for_goal = cost

                    # Case 2: create a robot-at at p by move_down with some free color C at p.
                    # We need available-color R p and available-color C p (may require moving both to p).
                    for C in candidate_Cs:
                        C_positions = available_positions.get(C, set())
                        if not C_positions:
                            continue
                        # distances to bring R and C to p
                        dist_R = 0 if p in R_positions else self._min_distance_from_any(R_positions, p)
                        dist_C = 0 if p in C_positions else self._min_distance_from_any(C_positions, p)
                        if dist_R is None or dist_C is None:
                            continue
                        # move R to p (dist_R), move C to p (dist_C), move_down (1), paint_up (1)
                        cost = dist_R + dist_C + 1 + 1
                        if best_for_goal is None or cost < best_for_goal:
                            best_for_goal = cost

                # Also consider the situation where robot_at exists for some other R' and R' can be used
                # (we already covered that by iterating robot_at entries above)

            # If we could not find any feasible option (best_for_goal None) mark with CAP
            if best_for_goal is None:
                total_cost += self.CAP
            else:
                total_cost += best_for_goal

        # Finally, return a finite estimate
        return int(total_cost)