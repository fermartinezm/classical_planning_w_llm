from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokens of a PDDL fact string '(pred a b ...)' as a list."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `spanner` domain (called `secret_domain` in the task).
    The goals in this domain are of the form (tightened-2 <locatable> <location>).
    The heuristic estimates how many actions are required to produce the missing
    tightened-2 facts. It reasons about:
      - where each nut is currently tightened (tightened ?nut ?locatable),
      - where each nut's loose-2 token is (loose-2 ?nut ?location),
      - the static "loose" graph between locatables (loose ?a ?b),
      - and which locatables can be the direct predecessor for a desired tightened-2 fact
        (an op can produce tightened-2 for X only if there exists a locatable Y with
         (loose X Y) and the nut is tightened at Y at the moment of the final action).

    Assumptions
    -----------
    - We assume that the static relation "loose" (loose <loc1> <loc2>) encodes the adjacency
      required to move a tightened nut from locatable U to V in a single tighten-move
      action when loose V U holds. Concretely, an operator exists that moves a tightened
      nut from U to W if loose W U is true.
    - The loose-2 facts (nut at a location) can be moved to a desired location in one
      tighten_nut action (if needed), so we cost 1 action if the nut's loose-2 is not yet
      at the desired location.
    - We ignore the detailed token behavior of the predicate "link" (which in the domain
      toggles between locatables). For heuristic purposes we optimistically assume
      a destination link is available when needed (this makes the heuristic optimistic
      and efficiently computable).
    - A single nut can be used (optimistically) for any number of goal tightened-2 facts
      (we do not enforce exclusive resource consumption). This choice favors an informative
      but optimistic estimate and keeps the heuristic inexpensive.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - the goal facts (self.goals),
      - static loose relations (used to build a directed graph for moving tightened
        along locatables),
      - the set of all locatables found in static facts (for safety),
      - the set of `at-2` (location) facts available statically (not directly used
        except to validate reachable locations).
    The directed graph for tightened-moves has an edge u -> w if there is a static (or
    state) fact (loose w u). This models that a tightened nut at u can be moved to w
    in one tighten_nut-2 / walk-2 style operator.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact (tightened-2 X target_location):
      1. If the goal is already true in the current state, cost is 0.
      2. Otherwise, try to find a nut that can produce this tightened-2 with minimal actions:
         - The final action producing (tightened-2 X target_location) requires:
             * Some nut N that is tightened at a locatable Y
             * loose X Y (X is a direct neighbour in the sense of the loose relation)
             * loose-2 N target_location (the nut N must have its loose-2 token at the goal location)
           If any of these is not true, estimate the cost to make them true:
             a) If Y is not already the locatable where the nut is tightened, we may need
                several tighten-move actions to move the tightened nut from its current
                locatable L to Y. We compute the shortest number of tighten-moves using
                BFS on the directed graph (edges u->w if loose w u).
             b) If the nut's loose-2 token is not at the goal location, we add 1 action
                (tighten_nut) to move the loose-2 to the target location (this is a
                conservative single-action estimate).
             c) Add 1 for the final action that produces tightened-2 (walk/pickup_spanner).
         - The cost candidate for that nut is: moves_to_reach_neighbor + (0 or 1 to move loose-2) + 1 (final).
      3. The heuristic for this goal is the minimum candidate cost across all nuts. If no nut
         can reach any neighbor of X (disconnected), we use a bounded large cost to avoid
         infinite returns but still reflect difficulty.
      4. Sum the costs of all individual goals to yield the final heuristic value.
    The heuristic is optimistic (it may underestimate because it ignores link-token contention
    and other ordering constraints), but it is informed by the graph distance of tightened
    movement and whether the nut's loose-2 token already sits at the target location.

    Notes
    -----
    - The heuristic returns 0 exactly for states where all goal facts are present.
    - For unreachable goals (w.r.t. the static loose graph), a large finite cost is used.
    """

    def __init__(self, task):
        # Goal facts (strings)
        self.goals = task.goals
        # Static facts available in every state
        static = task.static

        # Collect static loose relations (loose A B) -> used to build directed edges:
        # if (loose w u) then tightened can be moved u -> w in one action.
        self.loose_edges = set()
        # Also collect all locatables and static at-2 locations (for validation).
        self.locatables = set()
        self.at2_locations = set()

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "loose" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # record directed edge b -> a (tightened can move from b to a)
                self.loose_edges.add((b, a))
                self.locatables.add(a)
                self.locatables.add(b)
            elif pred == "at" and len(parts) >= 3:
                # at locatable1 locatable2 is static adjacency information
                self.locatables.add(parts[1])
                self.locatables.add(parts[2])
            elif pred == "at-2" and len(parts) >= 2:
                self.at2_locations.add(parts[1])

        # Build adjacency dict for BFS: from node u, neighbors are nodes w where (u -> w) exists.
        self.adj = {}
        for u, w in self.loose_edges:
            self.adj.setdefault(u, []).append(w)

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals already satisfied return 0
        if self.goals <= state:
            return 0

        # Parse current tightened positions: tightened ?nut ?locatable
        tightened_by_nut = {}  # nut -> locatable
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "tightened" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                tightened_by_nut[nut] = loc

        # Parse current loose-2 positions: loose-2 ?nut ?location
        loose2_by_nut = {}  # nut -> location
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "loose-2" and len(parts) >= 3:
                nut, loc = parts[1], parts[2]
                loose2_by_nut[nut] = loc

        # Build combined loose adjacency (include any loose facts that might appear in the state)
        # but prefer static edges already in self.adj; extend with dynamic ones found in state
        adj = {k: list(v) for k, v in self.adj.items()}
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "loose" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # directed edge b -> a
                adj.setdefault(b, []).append(a)

        # Helper: BFS that finds shortest distance from `start` to any node in `targets`
        def shortest_distance_to_any(start, targets):
            if start in targets:
                return 0
            visited = {start}
            q = deque([(start, 0)])
            while q:
                cur, dist = q.popleft()
                for nbr in adj.get(cur, []):
                    if nbr in visited:
                        continue
                    if nbr in targets:
                        return dist + 1
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
            return None  # unreachable

        # Gather all nuts we can consider. Use union of keys from tightened_by_nut and loose2_by_nut.
        nuts = set(tightened_by_nut.keys()) | set(loose2_by_nut.keys())
        if not nuts:
            # If no nuts in state (weird), return a conservative finite estimate
            return len(self.goals) * 5

        total_estimate = 0
        # Large finite penalty for unreachable goals (keeps heuristic finite)
        LARGE_PENALTY = max(50, len(self.goals) * 10)

        # Precompute for each goal the neighbor set: {Y | loose X Y}
        # Use both static and dynamic loose facts (state may contain loose)
        loose_pairs = set()
        # include static loose edges as (a,b) pairs
        for u, vs in adj.items():
            for v in vs:
                # adj built as b->a previously; but we want raw loose X Y facts where
                # we treat loose X Y meaning X is first arg and Y second arg.
                # adj was b->a because static loose (a b) -> adj[b] contains a
                # To reconstruct loose pairs we reverse that mapping:
                loose_pairs.add((v, u))
        # Also include any explicit (loose X Y) found in state or static (already covered)
        # Build neighbor set: for goal X, neighbors Y are those with (loose X Y)
        neighbors_of = {}
        for (x, y) in loose_pairs:
            neighbors_of.setdefault(x, set()).add(y)

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "tightened-2" or len(parts) < 3:
                # This heuristic only reasons about tightened-2 goals; if other goals exist,
                # provide a small default cost contribution.
                total_estimate += 1
                continue
            target_locatable = parts[1]
            target_location = parts[2]

            # If goal already in state, cost 0 for this goal
            if goal in state:
                continue

            # The final action requires there to be some Y with loose target_locatable Y,
            # and the nut being tightened at that Y. So gather candidate Ys.
            candidate_Ys = neighbors_of.get(target_locatable, set())
            if not candidate_Ys:
                # No loose X Y relation -> cannot produce tightened-2 for this X under any nut
                total_estimate += LARGE_PENALTY
                continue

            best_cost_for_goal = None
            # Evaluate each nut as a candidate producer of the goal
            for nut in nuts:
                if nut not in tightened_by_nut:
                    # if the nut is not currently tightened anywhere, it cannot be moved along the
                    # tightened-chain by the tighten_nut-2 / walk-2 operators that require a tightened nut.
                    # skip this nut for now.
                    continue
                start_loc = tightened_by_nut[nut]
                # compute minimal number of tighten-move actions to reach any Y in candidate_Ys
                d = shortest_distance_to_any(start_loc, candidate_Ys)
                if d is None:
                    # this nut cannot reach any required neighbor Y
                    continue
                # cost to move loose-2 of nut to required target_location, if needed
                move_loose2_cost = 0
                if loose2_by_nut.get(nut) != target_location:
                    # optimistic single action to move loose-2 to target_location
                    move_loose2_cost = 1

                # one action for the final creation of tightened-2 (walk/pickup_spanner type)
                final_action_cost = 1

                candidate_cost = d + move_loose2_cost + final_action_cost

                if best_cost_for_goal is None or candidate_cost < best_cost_for_goal:
                    best_cost_for_goal = candidate_cost

            if best_cost_for_goal is None:
                # No nut can produce this goal (disconnected), penalize heavily but finite.
                total_estimate += LARGE_PENALTY
            else:
                total_estimate += best_cost_for_goal

        return int(total_estimate)