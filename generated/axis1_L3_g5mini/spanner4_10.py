from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string like '(pred a b)' -> ['pred', 'a', 'b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to achieve all
    (painted <tile-2>) goal facts in the secret_domain (color) domain.
    The domain's painting sequence for a single tile-2 typically requires:
      - moving (with change_color) the two "available-color" providers (one
        provider that corresponds to the tile-2 itself or another movable
        object, and one color object which must also be available at the same
        tile) along the static "up" chain so that both providers are available
        at the same tile L, then
      - executing move_down at L to create the required robot-at fact, and
      - executing paint_up to produce the final painted fact.
    The heuristic is domain-dependent (exploits the structure of available-color,
    up relations, free-color, move_down and paint_up) and is intentionally
    optimistic (not admissible) to guide greedy best-first search.

    Assumptions
    -----------
    - Facts are strings of the form '(predicate arg1 arg2)'.
    - Static "up" facts define a directed graph of tiles where change_color
      moves an available-color entry from one tile to the next along an up edge.
    - Any object that currently has an '(available-color X S)' fact can be
      moved along up edges by repeated change_color actions (the domain's
      typing is inconsistent but the available-color/change_color pair is used
      for moving providers in practice).
    - A painting of a tile-2 T is possible only at tile locations L for which
      '(available-color T L)' holds in the current state (the tile-2 provides
      one of the required available-color facts for the same L).
    - A color usable for painting must both be free (a '(free-color C)' fact)
      and be available at L (or be movable to L with change_color).
    - The heuristic ignores resource contention and ordering interactions
      between multiple goal tiles (it sums optimistic costs per goal).

    Heuristic Initialization
    ------------------------
    - self.goals: store goal facts (task.goals).
    - parse static up facts from task.static and build:
        - self.up_adj: adjacency mapping a -> [b] for each (up a b).
        - self.rev_up_adj: reverse adjacency b -> [a] to allow quick BFS from
          a target tile back to possible source tiles (to compute minimal
          number of change_color hops).
    - A large finite penalty (self.INF_COST) is used when a goal looks
      unreachable under our optimistic model.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given planner node (node.state):
    1. If all goals are already satisfied by node.state return 0.
    2. Parse the state's facts:
       - Build available_at: mapping object -> set(tiles) where '(available-color obj tile)'.
       - Build available_at_tile: mapping tile -> set(objects available there).
       - Collect free_colors: set of objects X for which '(free-color X)' is true.
    3. For each goal painted tile T that is not yet true in the state:
       a. Identify all tile locations L where '(available-color T L)' holds.
          If none, painting T is considered unreachable (we add a large cost).
       b. For each such location L, the paint_up preconditions require:
          - some provider V (any object with available-color) to be available at L
            (we may have to move it along up edges),
          - some color C that is free and available at L (again we may move it).
          Both V and C may already be at L (cost 0) or require a number of
          change_color hops to bring them to L.
       c. Compute minimal number of change_color hops needed to bring any
          provider V to L and any free color C to L. We compute distances with
          BFS on the reverse-up graph from L to any source tiles where the
          provider/color currently is available. If either provider or color
          cannot reach L, that (V,C) combination is invalid.
       d. For a valid pair, the optimistic cost to paint T via L is:
            d_provider + d_color + 1 (move_down) + 1 (paint_up).
          Take the minimum across all (V, C) pairs and across all L.
       e. If no valid (V,C) pair exists for all L, treat T as unreachable and
          add the large penalty.
    4. Sum the optimistic minimal costs for all unpainted goal tiles and
       return the sum. This is finite for solvable states; unreachable goals
       result in a large but finite value.

    Notes on Admissibility and Practicality
    --------------------------------------
    - The heuristic is not admissible: it ignores negative interactions
      (competition for the same available-color facts and free_colors) and
      ignores the need to restore resources between multiple paintings.
    - It is carefully constructed to remain efficiently computable: the
      reverse-BFS from target tiles uses the static up graph only and the
      dynamic per-state data is restricted to available-color/free-color facts.
    """

    def __init__(self, task):
        # Goals (set of strings)
        self.goals = task.goals

        # Build adjacency for static up facts (directed a -> b for (up a b))
        self.up_adj = {}  # forward adjacency (not required for BFS but kept)
        self.rev_up_adj = {}  # reverse adjacency: b -> set(a) so we can BFS from target
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "up" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.up_adj.setdefault(a, set()).add(b)
                self.rev_up_adj.setdefault(b, set()).add(a)
        # Large finite penalty for "unreachable" under optimistic model
        self.INF_COST = 1000

    def _min_distance_to_target(self, starts, target):
        """
        Compute minimal number of forward 'up' hops needed to move an available-color
        item from any tile in 'starts' to 'target'. We BFS on the reverse graph
        starting from target and return the distance to the nearest start tile.
        If starts is empty or unreachable, return None.
        """
        if not starts:
            return None
        if target in starts:
            return 0
        # BFS on reverse graph
        visited = {target}
        queue = deque([(target, 0)])  # (node, dist_from_target)
        while queue:
            node, dist = queue.popleft()
            # Check predecessors (tiles that can move along up -> node)
            for prev in self.rev_up_adj.get(node, ()):
                if prev in visited:
                    continue
                # If prev is one of the start tiles, then the forward distance
                # from prev to target is dist + 1 (because prev -> ... -> target).
                if prev in starts:
                    return dist + 1
                visited.add(prev)
                queue.append((prev, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # 1) If goals satisfied return 0
        if self.goals <= state:
            return 0

        # 2) Parse dynamic facts from the state
        available_at = {}  # object -> set(tiles) where (available-color obj tile)
        available_at_tile = {}  # tile -> set(objects)
        free_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "available-color" and len(parts) == 3:
                obj, tile = parts[1], parts[2]
                available_at.setdefault(obj, set()).add(tile)
                available_at_tile.setdefault(tile, set()).add(obj)
            elif pred == "free-color" and len(parts) == 2:
                free_colors.add(parts[1])

        # 3) Extract goal tile-2 names that must be painted
        goal_tiles = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 2:
                goal_tiles.append(parts[1])

        total_cost = 0

        # All objects that can act as a provider (anything that currently has an available_at entry)
        provider_objects = set(available_at.keys())

        # For each goal tile not yet painted, compute optimistic minimal cost
        for tile in goal_tiles:
            painted_fact = f"(painted {tile})"
            if painted_fact in state:
                continue  # already done

            # Candidate locations L where the tile-2 is available to be painted
            candidate_locations = available_at.get(tile, set())
            if not candidate_locations:
                # No place to paint this tile under the current available-color facts.
                total_cost += self.INF_COST
                continue

            best_for_tile = None

            # For each candidate location L, compute minimal cost to bring some provider V
            # and some free color C to L (each possibly requiring change_color hops),
            # then perform move_down + paint_up (2 actions).
            for L in candidate_locations:
                # Determine free colors that can reach L (possibly distance 0)
                viable_colors = []
                for c in free_colors:
                    d_color = self._min_distance_to_target(available_at.get(c, set()), L)
                    if d_color is not None:
                        viable_colors.append((c, d_color))
                if not viable_colors:
                    # No free color can be made available at L -> skip this L
                    continue

                # Determine providers that can reach L
                viable_providers = []
                for v in provider_objects:
                    d_provider = self._min_distance_to_target(available_at.get(v, set()), L)
                    if d_provider is not None:
                        viable_providers.append((v, d_provider))
                if not viable_providers:
                    # No provider can be brought to L -> skip this L
                    continue

                # Combine best provider and best color (independent optimistic choice)
                # minimal combined distance among all pairs
                min_pair = None
                for _, d_p in viable_providers:
                    for _, d_c in viable_colors:
                        combined = d_p + d_c
                        if min_pair is None or combined < min_pair:
                            min_pair = combined
                if min_pair is None:
                    continue

                # Add the cost for move_down + paint_up
                candidate_cost = min_pair + 2
                if best_for_tile is None or candidate_cost < best_for_tile:
                    best_for_tile = candidate_cost

            if best_for_tile is None:
                # No valid way found to paint this tile under our optimistic model
                total_cost += self.INF_COST
            else:
                total_cost += best_for_tile

        return int(total_cost)