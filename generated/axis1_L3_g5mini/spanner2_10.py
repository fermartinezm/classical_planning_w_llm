from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string by stripping surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'color' (secret_domain) PDDL domain.
    The domain objective is to produce (painted TILE-2) facts. Painting a TILE-2
    requires a sequence of operations that align:
      - the TILE-2 to some physical tile position (via an available-color fact),
      - a color-2 object to the same tile position (also via available-color),
      - a color that is free (free-color) and (if necessary) paired with the color-2
        by creating a robot-at pair (via move_down).
    The core action that achieves a goal (a painted tile) is paint_up. Each paint_up
    needs the preconditions satisfied; this heuristic estimates the number of
    atomic actions (change_color, move_down, paint_up) required to satisfy those
    preconditions and execute paint_up for each remaining goal.

    Assumptions
    -----------
    - The planner represents facts as strings like "(predicate arg1 arg2)".
    - Every TILE-2 that must be painted has an "available-color TILE-2 TILE" fact
      in the state (if it does not, the heuristic returns a large finite cost).
    - The only action that moves available-color facts along tiles is change_color,
      which follows the directed static relation "up A B" (you can move from A to B).
    - change_color moves an available-color binding along one "up" edge per application.
    - move_down is required to create a robot-at pairing between a color-2 object and a color
      when they are available at the same tile; paint_up then consumes that pairing
      together with a free-color and the TILE-2 available at the tile.
    - free-color facts are consumable by paint_up (paint_up removes free-color),
      so a color with free-color may only be reliably used once (this heuristic
      will not attempt a full simulation of resource consumptions across goals;
      it sums per-goal minimal costs for efficiency).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - The set of goal facts (task.goals).
    - The static directed adjacency relation given by "up X Y" facts and store it
      as an adjacency dictionary for fast shortest-path (change_color steps) queries.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1) If the state's facts already satisfy all goal facts, return 0.
    2) Parse the current state and extract:
       - available_map: mapping object -> tile for all (available-color OBJ TILE) facts.
         This includes color-2 objects, colors, and TILE-2 objects (the domain uses the
         same predicate for several types).
       - free_colors: set of objects that appear in (free-color X).
       - robot_at: set of (color2, color) pairs from (robot-at A B).
       - painted: set of objects already painted.
    3) Build the directed graph of tile adjacency from static "up A B" facts.
       The number of change_color actions needed to move an available-color binding
       from tile P to tile T is the length of the shortest directed path P -> ... -> T.
       If no path exists, regard the distance as infinite.
    4) For each goal (painted TILE-2) that is not yet achieved:
       - Determine the tile where the TILE-2 is currently available (pos_g).
         If no such available entry exists, treat the tile as unreachable (large cost).
       - Consider all candidate color-2 objects (any object that has an available entry)
         and all candidate colors (objects in free_colors). For each pairing (c2, c)
         compute a conservative estimate of the number of required actions:
           * If robot_at already contains (c2, c) and c is free:
               - Only need to move c2 (if it is not already at pos_g) to pos_g
                 using change_color steps (distance d_c2), then do paint_up (1).
               - Cost = d_c2 + 1
           * Otherwise (no existing robot_at for the pair):
               - Need to move both c2 and c to pos_g (d_c2 + d_c), perform move_down (1),
                 and perform paint_up (1).
               - Cost = d_c2 + d_c + 2
         - Take the minimum finite cost across all valid pairs (c2, c). If none are valid,
           assign a large finite cost for that goal.
    5) The heuristic value is the sum of the per-goal minimal costs. This is efficient
       to compute and captures the principal costs: change_color movements plus the
       necessary move_down and paint_up actions. It is not strictly admissible but
       usually effective at guiding greedy best-first search toward reducing the number
       of remaining paint operations and aligning resources.

    Note: This heuristic intentionally avoids expensive global resource re-simulation
    (e.g., exact consumption/transfer accounting across multiple goals) for speed.
    It is designed to be informative (accounts for movement distances on the 'up' graph)
    while staying cheap to compute.
    """

    def __init__(self, task):
        # Goals: set of strings like "(painted tile1)"
        self.goals = task.goals
        # Build directed adjacency from static "up" facts for change_color reachability.
        self.up_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.up_graph.setdefault(src, set()).add(dst)

        # Precompute list of all nodes in up graph for BFS, though BFS will tolerate missing nodes.

    def _shortest_up_distance(self, src, dst):
        """
        Shortest directed distance from tile src to tile dst using the 'up' edges.
        Returns non-negative integer distance (0 if src == dst), or a large integer
        if dst is unreachable from src.
        """
        if src == dst:
            return 0
        if src not in self.up_graph:
            return float("inf")
        # BFS
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            # expand neighbors
            for succ in self.up_graph.get(node, ()):
                if succ == dst:
                    return d + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, d + 1))
        return float("inf")

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse state
        available_map = {}  # object -> tile (from available-color facts)
        free_colors = set()  # set of objects that are free-color
        robot_at = set()  # set of (a,b) pairs from robot-at facts
        painted = set()
        down_set = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) >= 3:
                obj, tile = parts[1], parts[2]
                available_map[obj] = tile
            elif pred == "free-color" and len(parts) >= 2:
                free_colors.add(parts[1])
            elif pred == "robot-at" and len(parts) >= 3:
                robot_at.add((parts[1], parts[2]))
            elif pred == "painted" and len(parts) >= 2:
                painted.add(parts[1])
            elif pred == "down" and len(parts) >= 2:
                down_set.add(parts[1])

        # Extract goal objects that are painted in the problem specification
        goal_painted_objs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "painted" and len(parts) >= 2:
                goal_painted_objs.append(parts[1])

        total_cost = 0
        LARGE = 1000  # finite large cost for unreachable cases

        # Candidate lists
        candidate_c2s = [obj for obj in available_map.keys()]  # possible color-2 objects (conservative)
        candidate_colors = list(free_colors)  # only colors that are free are useful

        for goal_obj in goal_painted_objs:
            if goal_obj in painted:
                continue  # already satisfied

            # Determine tile where the tile-2 object is available
            pos_g = available_map.get(goal_obj)
            if pos_g is None:
                # If the TILE-2 has no available-color entry, painting is unlikely; assign large cost.
                total_cost += LARGE
                continue

            best_cost = float("inf")

            # For each candidate color-2 (provider) and free color, compute an estimated cost.
            for c2 in candidate_c2s:
                pos_c2 = available_map.get(c2)
                if pos_c2 is None:
                    continue
                d_c2 = self._shortest_up_distance(pos_c2, pos_g)
                if d_c2 == float("inf"):
                    # cannot bring this c2 to required tile
                    continue

                # If there are no free colors, we cannot perform paint_up (needs free-color). Skip.
                if not candidate_colors:
                    continue

                for c in candidate_colors:
                    # c must be free (we restrict to free_colors)
                    # pos_c may be None (color might have no available mapping), in which case
                    # it may be impossible to move it — treat as infinite distance.
                    pos_c = available_map.get(c)
                    d_c = self._shortest_up_distance(pos_c, pos_g) if pos_c is not None else float("inf")

                    if (c2, c) in robot_at:
                        # robot-at pairing already exists; only need to move c2 to pos_g (if necessary) and paint.
                        # paint_up requires free-color c (we ensured c in free_colors)
                        cost = d_c2 + 1
                        if cost < best_cost:
                            best_cost = cost
                    else:
                        # Need both at pos_g, then move_down and paint_up
                        if d_c == float("inf"):
                            continue
                        cost = d_c2 + d_c + 2  # move both (change_color steps) + move_down + paint_up
                        if cost < best_cost:
                            best_cost = cost

            if best_cost == float("inf"):
                total_cost += LARGE
            else:
                # best_cost is the minimal number of actions (change_color steps + move_down + paint_up)
                total_cost += int(best_cost)

        # Ensure heuristic is finite
        if total_cost >= LARGE:
            # return a large finite value rather than infinity
            return LARGE
        return total_cost