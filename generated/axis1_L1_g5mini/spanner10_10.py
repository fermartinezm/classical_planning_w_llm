from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split the fact into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (e2zu8y93). The domain
    models workers (ilfxqhcc) that can pick up spanners (e2zu8y93) and use them
    to loosen nuts (rupjajag). The operators available are:
      - n9sbfmrs: move a worker along directed adjacency edges (kxokb5hx).
      - moi6wnfu: pick up a spanner by a worker when both are co-located.
      - kjsqfni8: use a (good) spanner carried by a worker to loosen a nut.

    The heuristic estimates the number of actions required to achieve all goal
    predicates of the form (g59iwyxf <nut>) — i.e. to loosen the goal nuts.

    Assumptions
    -----------
    - Moving a worker along one adjacency edge (kxokb5hx) costs 1.
    - Picking up a spanner (moi6wnfu) costs 1.
    - Using a spanner to loosen a nut (kjsqfni8) costs 1.
    - A spanner can be used at most once: its g0brvc06 fact is required for
      usage and is removed when kjsqfni8 is applied. We therefore treat each
      good spanner as a single-use resource.
    - Workers can carry at most any number of spanners syntactically; in the
      domain a worker picks up a spanner using moi6wnfu and then can move while
      carrying it (there is no restriction preventing moving while carrying).
    - The heuristic is not required to be admissible. It aims to be informative
      and cheaply computable.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Extracts the set of goal nuts from task.goals (predicates g59iwyxf).
    - Builds a directed adjacency graph of locations from static facts of the
      form (kxokb5hx locA locB). This graph is used to compute shortest-path
      distances (BFS) between locations on demand.
    - Keeps the full set of goals for quick goal-checking.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic proceeds as follows:

    1. Quick goal check:
       - If all goal facts are already present in the state return 0.

    2. Extract dynamic world information:
       - Worker locations from (u4srx1nb <worker> <loc>).
       - Spanner availability: spanners that have (g0brvc06 <spanner>) are
         considered "good" and usable. For each good spanner we determine its
         current location: either it is carried by a worker (yud7t3vb <worker> <spanner>),
         in which case its location equals that worker's location, or it is located
         at a place via (u4srx1nb <spanner> <loc>).
       - Nut locations from (u4srx1nb <nut> <loc>) and whether a nut is already
         loosened via (g59iwyxf <nut>) or still intact via (bu1om0m1 <nut>).

    3. Distance computation:
       - Distances between two locations are computed with BFS over the directed
         adjacency graph built from static facts. Distances are cached per source
         to avoid repeated BFS computations.

    4. Greedy sequential simulation:
       - The heuristic treats the (likely single) worker(s) as agents that will
         sequentially go and loosen nuts. For each not-yet-loosened goal nut,
         the heuristic computes the minimal cost to loosen that nut given the
         current simulated worker positions and remaining good spanners.
       - For a specific worker w and spanner s, with nut at location nloc:
         * If w already carries s -> cost = distance(w_loc, nloc) + 1 (use).
         * If s is at the same location as w (not carried) -> cost = 1 (pick) +
           distance(w_loc, nloc) + 1 (use).
         * If s is at a different location -> cost = distance(w_loc, s_loc)
           + 1 (pick) + distance(s_loc, nloc) + 1 (use).
       - Among all (worker, spanner) pairs the heuristic selects the combination
         producing the smallest cost for some nut, applies that action-sequence
         in the simulation (i.e., updates simulated worker location to the nut's
         location and removes the spanner from the set of available good spanners),
         and adds the cost to the running heuristic total.
       - Repeat until all goal nuts are simulated as loosened or until no usable
         spanners / paths remain. If a nut is unreachable given the static graph
         we add a large finite penalty to keep the heuristic finite.

    Notes on soundness and efficiency
    ---------------------------------
    - The heuristic is finite for solvable states. If something is unreachable
      the heuristic returns a large finite value (a tunable penalty) rather than
      infinity.
    - The greedy sequential simulation captures resource consumption (each
      good spanner can be used once) and worker movement effects, producing a
      much more informed estimate than summing independent per-nut optimistic
      costs.
    - BFS distance caching keeps the computation efficient: each distinct source
      location's BFS is computed at most once per heuristic call.
    """

    def __init__(self, task):
        # Keep goal facts (strings) for quick checking.
        self.goals = task.goals

        # Extract goal nuts (their names) from goals like "(g59iwyxf nut1)".
        self.goal_nuts = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "g59iwyxf" and len(parts) >= 2:
                self.goal_nuts.append(parts[1])

        # Build directed adjacency graph from static facts: (kxokb5hx A B) => A -> B
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "kxokb5hx" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adjacency.setdefault(a, set()).add(b)

    # BFS distance cache per-call (populated in __call__)
    def _bfs_distances(self, source, cache):
        """
        Return a dict mapping location -> shortest distance (#moves) from source
        using directed edges. Uses cache to avoid recomputation.
        """
        if source in cache:
            return cache[source]

        dist = {source: 0}
        q = deque([source])
        while q:
            cur = q.popleft()
            for nb in self.adjacency.get(cur, ()):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        cache[source] = dist
        return dist

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper: find facts in state
        def facts_with_pred(pred):
            return [f for f in state if get_parts(f)[0] == pred]

        # Extract worker locations: (u4srx1nb <entity> <loc>) where entity may be worker or other objects.
        worker_locs = {}  # worker -> loc
        for fact in facts_with_pred("u4srx1nb"):
            parts = get_parts(fact)
            if len(parts) >= 3:
                ent, loc = parts[1], parts[2]
                # Identify workers by their usage in yud7t3vb (they must be ilfxqhcc type),
                # but we conservatively treat any entity that also appears in yud7t3vb as worker,
                # and also collect entities that are not spanners/nuts by name patterns.
                # Simpler: gather all entities that appear as first arg in any yud7t3vb or that
                # appear to be ilfxqhcc in the domain examples (no typing info here),
                # but to be robust we include any entity that has at least one u4srx1nb and also appears
                # as first arg in a yud7t3vb or in none (we still keep them - harmless).
                # We'll treat all such entities as potential workers.
                # For this domain examples, actual workers are the ilfxqhcc objects like 'bob'.
                worker_locs[ent] = loc

        # Extract which worker carries which spanner: yud7t3vb <worker> <spanner>
        carrier_of_spanner = {}  # spanner -> worker
        for fact in facts_with_pred("yud7t3vb"):
            parts = get_parts(fact)
            if len(parts) >= 3:
                worker, spanner = parts[1], parts[2]
                carrier_of_spanner[spanner] = worker
                # Make sure worker has a known location; if not, leave it absent (will be handled).
                if worker not in worker_locs:
                    # there might be cases where worker location is not in u4srx1nb because worker was never
                    # explicitly listed; we won't add a fake location, we simply won't consider such worker.
                    pass

        # Identify all good spanners: facts (g0brvc06 <spanner>)
        good_spanners = set()
        for fact in facts_with_pred("g0brvc06"):
            parts = get_parts(fact)
            if len(parts) >= 2:
                good_spanners.add(parts[1])

        # For good spanners, determine their current location (either carried -> worker location,
        # or located via u4srx1nb).
        spanner_loc = {}  # spanner -> location (if known)
        # First, spanners explicitly at locations.
        for fact in facts_with_pred("u4srx1nb"):
            parts = get_parts(fact)
            if len(parts) >= 3:
                ent, loc = parts[1], parts[2]
                if ent in good_spanners:
                    spanner_loc[ent] = loc
        # Then update from carried spanners (overwrites if both facts present; carried implies that).
        for s, w in carrier_of_spanner.items():
            if s in good_spanners and w in worker_locs:
                spanner_loc[s] = worker_locs[w]

        # Nuts: their locations and whether they are already loosened.
        nut_loc = {}     # nut -> location
        loosened = set() # nuts already having g59iwyxf
        for fact in facts_with_pred("u4srx1nb"):
            parts = get_parts(fact)
            if len(parts) >= 3:
                ent, loc = parts[1], parts[2]
                # If ent is a nut (it will appear in goal list), store loc.
                # We'll only record those nuts that are relevant goals; others are ignored.
                if ent in self.goal_nuts:
                    nut_loc[ent] = loc

        for fact in facts_with_pred("g59iwyxf"):
            parts = get_parts(fact)
            if len(parts) >= 2:
                loosened.add(parts[1])

        # Build list of pending nuts (goals not yet satisfied).
        pending_nuts = [n for n in self.goal_nuts if n not in loosened]

        # If no workers known, heuristic cannot simulate movements; return a coarse estimate:
        if not worker_locs:
            # Count number of pending nuts and assume each needs pick+use at minimum (2 actions),
            # plus a movement penalty if we cannot determine distances. Use a moderate finite value.
            return max(1, len(pending_nuts) * 3)

        # BFS distance cache
        dist_cache = {}

        # Helper to obtain distance between two locations (directed), returns None if unreachable.
        def distance(a, b):
            if a == b:
                return 0
            dmap = self._bfs_distances(a, dist_cache)
            return dmap.get(b, None)

        # Prepare simulated state for greedy sequential assignment:
        simulated_worker_pos = dict(worker_locs)  # worker -> current loc
        # Available good spanners (we will remove when used)
        available_spanners = set(s for s in good_spanners if s in spanner_loc)
        # For spanners carried by a worker, track that as well (spanner is available at that worker's current loc)
        # The spanner location will be derived from simulated_worker_pos if needed.
        # Nut locations are static for this domain (nuts do not move).
        total_cost = 0
        PENALTY = 1000  # large finite penalty for unreachable combos

        # If any pending nut has no recorded location, it's problematic; add penalty.
        for n in pending_nuts:
            if n not in nut_loc:
                total_cost += PENALTY

        # Filter pending_nuts to only those with known locations for the main loop.
        pending = [n for n in pending_nuts if n in nut_loc]

        # Greedy loop: pick the next nut with minimal simulated cost and update simulation.
        while pending:
            best = None  # tuple (cost, nut, worker, spanner, path_info)
            for nut in pending:
                nloc = nut_loc[nut]
                # For each worker and each available spanner compute cost.
                for worker, wloc in simulated_worker_pos.items():
                    # If worker currently carries some spanner(s), those spanners might be available even if not in available_spanners
                    # But domain uses yud7t3vb only for spanners that workers picked up; we ensured carried spanners in spanner_loc
                    # So iterate over available_spanners only.
                    for s in list(available_spanners):
                        # Determine spanner location in current simulation:
                        # If spanner was carried originally by some worker, and that worker exists in simulated_worker_pos,
                        # then its location equals that worker's current simulated pos. Otherwise use spanner_loc stored location.
                        s_loc = spanner_loc.get(s)
                        # If spanner was carried by some worker cw originally, update s_loc to simulated position of that worker
                        if s in carrier_of_spanner:
                            cw = carrier_of_spanner[s]
                            if cw in simulated_worker_pos:
                                s_loc = simulated_worker_pos[cw]
                        # If for some reason s_loc is still None, skip this spanner.
                        if s_loc is None:
                            continue

                        # Now compute cost depending on relationships
                        # Case 1: worker is already carrying s (yud7t3vb worker s present in state)
                        carrying = (carrier_of_spanner.get(s) == worker)
                        if carrying:
                            # cost = distance(worker_loc, nut_loc) + 1 (use)
                            dist = distance(simulated_worker_pos[worker], nloc)
                            if dist is None:
                                c = PENALTY
                            else:
                                c = dist + 1
                        else:
                            # Worker must either go to spanner then pick then go to nut then use.
                            # If spanner is at same location as worker initially, pick immediately (1), then go to nut, then use (1).
                            if simulated_worker_pos[worker] == s_loc:
                                dist_w_to_n = distance(simulated_worker_pos[worker], nloc)
                                if dist_w_to_n is None:
                                    c = PENALTY
                                else:
                                    c = 1 + dist_w_to_n + 1
                            else:
                                # Worker -> spanner -> nut
                                dist_w_to_s = distance(simulated_worker_pos[worker], s_loc)
                                dist_s_to_n = distance(s_loc, nloc)
                                if dist_w_to_s is None or dist_s_to_n is None:
                                    c = PENALTY
                                else:
                                    c = dist_w_to_s + 1 + dist_s_to_n + 1

                        # Select best option
                        if best is None or c < best[0]:
                            best = (c, nut, worker, s)

            # If we couldn't find any feasible assignment (best is None), break with penalty
            if best is None:
                total_cost += PENALTY * len(pending)
                break

            cost, nut_chosen, worker_chosen, spanner_chosen = best
            total_cost += cost

            # Update simulation:
            # Worker ends up at nut location (after using spanner).
            simulated_worker_pos[worker_chosen] = nut_loc[nut_chosen]
            # Remove spanner from available set (used up)
            if spanner_chosen in available_spanners:
                available_spanners.remove(spanner_chosen)
            # Also remove from carrier_of_spanner to avoid using it again
            if spanner_chosen in carrier_of_spanner:
                del carrier_of_spanner[spanner_chosen]
            # Remove nut from pending list
            pending.remove(nut_chosen)

            # Continue loop until all pending nuts processed.

        return int(total_cost)