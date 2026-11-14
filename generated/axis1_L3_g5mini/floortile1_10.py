from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string '(pred x y ...)' as a list."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'spanner' domain (called 'secret_domain' here).
    The domain's goals are of the form (tightened-2 <locatable> <location>).
    Informally, achieving a goal (tightened-2 L Loc) requires:
      - having some nut that is tightened at a nearby locatable,
      - optionally moving that nut's loose-2 record to the goal location,
      - and finally performing a single operation (walk / pickup_spanner style)
        that produces (tightened-2 L Loc).
    This heuristic estimates the number of actions remaining by, for each missing
    tightened-2 goal, finding the cheapest (optimistic) sequence of actions to
    produce the goal using any available nut. It sums these per-goal estimates.

    Assumptions
    -----------
    - We are intentionally optimistic (heuristic need not be admissible):
      * We assume the required 'link' fluent for the final operation can be arranged
        without extra cost beyond the counted actions.
      * We allow the same nut to be (optimistically) reused for multiple goals.
    - The static loose/at relations define a directed graph along which a 'tightened'
      label can be propagated. If (loose U V) is static, then a nut tightened at V
      can (in one action) be moved to U (by tighten_nut-2 / walk-2 / pickup_spanner-2).
    - Moving a nut's loose-2 location (the loose-2 predicate) can be done in one
      action (tighten_nut), because tighten_nut changes loose-2 from its current
      location to any at-2 location in one step.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - self.goals: set of required goal fact strings (task.goals).
      - static relations:
          * adjacency graph "propagation_adj" where an edge v -> u exists if
            (loose u v) is in static facts. This models that a tightened nut at v
            can be moved to u in one action.
          * static_loose_pairs set with (u, v) for every (loose u v) in static.
          * static_at_pairs set with (u, v) for every (at u v) in static. The
            final operation that yields tightened-2 may use either a loose or an
            at relation between the goal locatable and the locatable where the
            nut is tightened.
      - We keep the static at-2 facts set (locations) although it is only
        implicitly used (moving loose-2 is always possible to any at-2).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If the entire goal set is already satisfied in the state, return 0.
    2. Parse the state to extract:
         - tightened_map: mapping nut -> locatable where (tightened nut loc) holds.
         - nut_location_map: mapping nut -> location where (loose-2 nut location) holds.
           If a nut has no loose-2 fact in the state we treat its loose-2 location as unknown
           and assume a single move (cost 1) will make it available at the desired location.
    3. For each goal (tightened-2 L Loc):
         - If the goal fact is already true in the state, cost = 0.
         - Otherwise, for every nut that currently has a tightened position M0, run a
           BFS over the static propagation graph from M0 to compute distances (number of
           propagation moves) to other locatables.
         - A candidate final configuration to create (tightened-2 L Loc) exists if we can
           reach a locatable M such that static has either (loose L M) or (at L M).
           * The propagation cost is dist(M0 -> M) actions (each action moves tightened from
             one locatable to the next).
           * The final action (walk / pickup_spanner) that creates tightened-2 L Loc costs 1.
           * If the nut's current loose-2 location != Loc, we add 1 for the tighten_nut
             action to move the nut's loose-2 to Loc (tighten_nut can move it in one action).
         - The total candidate cost for that nut is: dist + 1 (final) + (0 or 1 to move loose-2).
         - We take the minimum candidate cost among all nuts. If no nut can reach any M
           that satisfies the required adjacency relation, we fall back to a conservative
           small penalty (we choose 3) to keep the heuristic finite and informative.
    4. Sum the per-goal minimal costs to obtain the heuristic value for the state.
    5. Return the sum (an integer).

    Rationale & Expected Behavior
    -----------------------------
    - The heuristic is optimistic but domain-informed: it respects the static adjacency
      (where tightened labels can be propagated) and the fact that loose-2 relocations
      take one action. Therefore it usually estimates a small integer number of actions
      for each missing tightened-2 goal and helps prioritize states where tightened
      labels are already nearby.
    - The heuristic returns 0 iff all goal facts are already true in the state.
    - The heuristic is finite for solvable states because it either finds a path
      in the static propagation graph or falls back to a bounded penalty per goal.
    """

    def __init__(self, task):
        # Goals (set of fact strings)
        self.goals = task.goals

        # Static facts
        static_facts = task.static

        # Build propagation adjacency:
        # If static has (loose U V) then a tightened nut at V can be moved to U in one action.
        # We model edges V -> U
        self.propagation_adj = defaultdict(list)
        self.static_loose_pairs = set()
        self.static_at_pairs = set()
        self.at2_locations = set()  # (at-2 location) facts

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "loose" and len(parts) >= 3:
                u, v = parts[1], parts[2]
                # edge v -> u
                self.propagation_adj[v].append(u)
                self.static_loose_pairs.add((u, v))
            elif pred == "at" and len(parts) >= 3:
                u, v = parts[1], parts[2]
                self.static_at_pairs.add((u, v))
            elif pred == "at-2" and len(parts) >= 2:
                self.at2_locations.add(parts[1])

    def _bfs_distances(self, start):
        """Return dictionary of minimal distances (number of propagation moves) from start to each reachable node."""
        dist = {start: 0}
        q = deque([start])
        while q:
            node = q.popleft()
            for nbr in self.propagation_adj.get(node, ()):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    q.append(nbr)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal already reached -> heuristic 0
        if self.goals <= state:
            return 0

        # Parse tightened facts in current state: tightened nut locatable
        tightened_map = {}  # nut -> locatable
        # Parse loose-2 facts: nut -> location
        nut_location_map = {}  # nut -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "tightened" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                tightened_map[nut] = loc
            elif pred == "loose-2" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                nut_location_map[nut] = loc

        # Collect all goals of the relevant predicate (tightened-2)
        missing_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            if parts[0] == "tightened-2" and len(parts) >= 3:
                locatable, location = parts[1], parts[2]
                if goal not in state:
                    missing_goals.append((locatable, location))

        total_cost = 0

        # Precompute BFS distances for each distinct tightened starting locatable
        bfs_cache = {}

        # If there are no tightened nuts at all, be finite: assume each missing goal needs at least 3 actions
        if not tightened_map:
            return len(missing_goals) * 3

        for (goal_locatable, goal_location) in missing_goals:
            best_cost = None

            # For each nut available, compute candidate cost
            for nut, start_loc in tightened_map.items():
                # distances from start_loc (where nut is currently tightened)
                if start_loc not in bfs_cache:
                    bfs_cache[start_loc] = self._bfs_distances(start_loc)
                distances = bfs_cache[start_loc]

                # The nut's current loose-2 location (might be missing)
                current_loose2_loc = nut_location_map.get(nut, None)

                # We need to find a locatable M such that static has either (loose goal_locatable M) or (at goal_locatable M)
                # and M is reachable from start_loc. For any such M:
                #   cost = dist(start_loc -> M) [propagate tightened] + 1 [final walk/pickup] + (1 if loose-2 != goal_location else 0)
                # If current_loose2_loc is None, assume we need one action to set it (conservative).
                additional_loose2_cost = 0 if current_loose2_loc == goal_location else 1

                # Iterate reachable nodes M
                candidate_for_nut = None
                # Quick check: maybe start_loc itself satisfies the relation
                if (goal_locatable, start_loc) in self.static_loose_pairs or (goal_locatable, start_loc) in self.static_at_pairs:
                    # dist = 0
                    candidate_for_nut = 0 + 1 + additional_loose2_cost
                else:
                    for M, d in distances.items():
                        if (goal_locatable, M) in self.static_loose_pairs or (goal_locatable, M) in self.static_at_pairs:
                            c = d + 1 + additional_loose2_cost
                            if candidate_for_nut is None or c < candidate_for_nut:
                                candidate_for_nut = c

                if candidate_for_nut is not None:
                    if best_cost is None or candidate_for_nut < best_cost:
                        best_cost = candidate_for_nut

            # If no nut could reach any appropriate M, fallback to a small finite penalty
            if best_cost is None:
                best_cost = 3

            total_cost += best_cost

        # Return an integer heuristic value
        return int(total_cost)