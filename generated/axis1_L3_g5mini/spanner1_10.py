from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (color) domain.

    The domain's goal atoms are of the form (painted TILE-2). To produce
    (painted TILE-2) the single action `paint_up` must be applied. The
    `paint_up` action requires:
      - (down TILE-2) -- the tile to be painted must be in the 'down' state
      - (available-color R P) for a color-2 object R at some tile position P
      - (available-color TILE-2 P) the tile-2 object's availability must be
        at the same tile position P
      - (robot-at R C) for the same color-2 R and some color C
      - (free-color C) that color C is currently free

    The domain provides actions:
      - change_color: moves (available-color R X) -> (available-color R Y) when (up X Y).
        This moves the availability of a color-2 object forward along the static
        up graph (one up-edge per action).
      - move_down: when (available-color R P) and (available-color C P) holds,
        it creates (robot-at R C) (consuming the available-color of C at P).
      - paint_up: consumes (down TILE-2), (free-color C) and the needed
        available-color alignments, and produces (painted TILE-2).

    Assumptions
    -----------
    - The tile-2 object's available position (available-color TILE-2 P) is
      static for the purpose of planning painting of that tile (it cannot be
      moved by available domain actions in this domain).
    - The only way to move a color-2 object's availability along tile positions
      is via repeated `change_color` actions, following the `up` static edges.
    - A color C with (free-color C) is required at the tile position P to
      produce (robot-at R C) (via move_down). The heuristic is optimistic and
      assumes resource consumption does not prevent later paintings (i.e.,
      it does not conservatively model consumption across multiple goal tiles).
    - If there is already (robot-at R C) and (free-color C), no move_down is
      needed for that R. Otherwise, move_down is needed (and requires a free
      color C that is available at the target position P).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
      - the set of goal painted tiles (self.goal_tiles)
      - the directed adjacency graph of tile positions defined by static (up X Y)
        relations (self.up_graph)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. If the goal is already reached (all painted facts present) return 0.

    2. Parse the current state to obtain:
       - available_map: for each object O, the set of tile positions P for which
         (available-color O P) holds.
       - robot_at_map: mapping R -> set of colors C for which (robot-at R C).
       - free_colors: the set of colors C with (free-color C).

    3. For each goal tile T (tile-2 object) that is not painted:
       a) Find its fixed tile position P_T from available_map (i.e., the P
          such that (available-color T P) is true). If none exists, treat T as
          unreachable (large cost).
       b) Check whether there exists at least one free color C with
          (available-color C P_T) (needed for move_down). If no such free
          color is available at P_T, painting T is considered unreachable.
       c) Consider all candidate color-2 objects R (inferred from the state:
          objects appearing as the first argument in robot-at facts or in the
          available_map but not being a tile-2 or a color). For each candidate
          R compute the shortest forward distance d along the static up graph
          from any position P_R in available_map[R] to P_T. This distance is
          the number of `change_color` actions needed to align R's availability
          with the tile-2's availability at P_T. If P_R == P_T then d = 0.
       d) For a candidate R that can reach P_T, compute a candidate cost:
            cost = d                      # number of change_color actions
                 + need_move_down         # 1 if there is no (robot-at R C) with
                                          # C in free_colors, else 0
                 + 1                      # paint_up action itself
          The heuristic picks the minimum candidate cost among all R.
       e) If no candidate R can reach P_T (or prerequisites fail), T is
          considered unreachable and a large finite penalty is used.

    4. The heuristic value is the sum of the per-goal minimal costs (optimistic,
       i.e., ignores inter-goal resource conflicts). The heuristic is finite
       for states where each unpainted goal tile has at least one feasible R
       and a free color available at the corresponding tile position.

    Note: This heuristic is intentionally optimistic (not admissible): it
    underestimates interactions (color availability consumption) but provides
    a focused, efficiently computable estimate that guides greedy best-first
    search towards aligning available-color facts and producing the
    robot-at/free-color preconditions needed for painting.
    """

    def __init__(self, task):
        """Extract static up-edges and goal painted tiles from the task."""
        self.goals = task.goals  # set of goal fact strings
        # Build goal tile set (tile-2 objects that must be painted).
        self.goal_tiles = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 2:
                self.goal_tiles.add(parts[1])

        # Build adjacency list for up edges from static facts
        self.up_graph = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.up_graph[src].append(dst)

    def _bfs_distance(self, start_nodes, target):
        """
        Compute shortest forward distance (in number of up-edges) from any of the
        start_nodes to target following self.up_graph. Returns None if unreachable.
        """
        if target in start_nodes:
            return 0
        visited = set(start_nodes)
        queue = deque()
        for s in start_nodes:
            queue.append((s, 0))
        while queue:
            node, dist = queue.popleft()
            for succ in self.up_graph.get(node, ()):
                if succ == target:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # If goal already reached -> h = 0 (requirement: heuristic is 0 only for goal states)
        if self.goals <= state:
            return 0

        # Parse state facts
        available_map = defaultdict(set)  # obj -> set of tile positions P where (available-color obj P)
        robot_at_map = defaultdict(set)  # R (color-2) -> set of colors C where (robot-at R C)
        free_colors = set()  # set of colors C where (free-color C) holds
        painted = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, pos = parts[1], parts[2]
                available_map[obj].add(pos)
            elif pred == "robot-at" and len(parts) >= 3:
                r, c = parts[1], parts[2]
                robot_at_map[r].add(c)
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])

        total_cost = 0
        LARGE = 10000  # large finite penalty for unreachable subgoals

        # Determine candidate color-2 objects:
        # We'll treat as color-2 any object that appears as a key in robot_at_map
        # or appears as a subject in available_map but is not a goal tile and not a free color.
        candidate_Rs = set(robot_at_map.keys())
        for obj in available_map.keys():
            if obj not in self.goal_tiles and obj not in free_colors:
                candidate_Rs.add(obj)

        # For each goal tile that is not yet painted, estimate minimal cost to paint it.
        for t in self.goal_tiles:
            if t in painted:
                continue  # already satisfied

            # Find tile position P_T where (available-color t P_T) holds.
            positions_for_t = available_map.get(t, set())
            if not positions_for_t:
                # No known place where tile T is available -> unreachable (penalize)
                total_cost += LARGE
                continue
            # Domain semantics indicate tile-2 typically has a single position; pick all (conservative)
            # but we'll try all positions and pick minimal.
            best_cost_for_t = None

            for p_t in positions_for_t:
                # Is there at least one free color C available at p_t? Required for move_down.
                free_colors_at_p_t = {c for c in free_colors if p_t in available_map.get(c, set())}
                if not free_colors_at_p_t:
                    # Can't create robot-at at this position because no free color available here.
                    continue

                # For each candidate color-2 R find distance from any of its positions to p_t.
                best_candidate_cost = None
                for R in candidate_Rs:
                    positions_for_R = available_map.get(R, set())
                    if not positions_for_R:
                        # R has no availability position in current state; skip
                        continue
                    dist = self._bfs_distance(positions_for_R, p_t)
                    if dist is None:
                        continue  # this R cannot be moved (via change_color) to align at p_t

                    # Does R already have a robot-at mapping to some free color?
                    already_ok = False
                    for c in robot_at_map.get(R, set()):
                        if c in free_colors:
                            already_ok = True
                            break

                    need_move_down = 0 if already_ok else 1
                    # Move_down requires a free color available at p_t; we already checked existence

                    candidate_cost = dist + need_move_down + 1  # +1 for paint_up
                    if best_candidate_cost is None or candidate_cost < best_candidate_cost:
                        best_candidate_cost = candidate_cost

                if best_candidate_cost is not None:
                    if best_cost_for_t is None or best_candidate_cost < best_cost_for_t:
                        best_cost_for_t = best_candidate_cost

            if best_cost_for_t is None:
                # No viable R/position combination found -> unreachable
                total_cost += LARGE
            else:
                total_cost += best_cost_for_t

        return total_cost