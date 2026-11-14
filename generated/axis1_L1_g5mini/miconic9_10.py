from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its predicate and arguments."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain domain (internal name ud3uf11z).
    The domain models an agent (represented by the fluent `omggbn9z <loc>`) that can
    move over locations connected by `acovodrf` edges, can perform a `diwlmgg9` action
    at a location where `(kut2ksnk <p> <loc>)` to produce `(f2xo7hb6 <p>)`, and then
    perform a `y7nik9q2` action at a location where `(klw11kjf <p> <loc>)` and
    `(f2xo7hb6 <p>)` holds to produce the goal `(a7sb56om <p>)`.

    This heuristic estimates the number of actions required to achieve all goal
    predicates `a7sb56om` for the objects of type `nxbd1hc3`. It combines movement
    costs (each move between adjacent locations costs 1) with the action costs:
      - diwlmgg9 (produce f2xo7hb6) costs 1 action
      - y7nik9q2 (produce a7sb56om) costs 1 action

    Assumptions
    -----------
    - Movement between two locations costs 1 per edge; movement is possible along an
      edge (u,v) if the static relation `acovodrf` contains (u v) or (v u). We
      therefore treat acovodrf facts as providing (practically) undirected connectivity
      between the mentioned locations (an edge allows travel in the corresponding direction).
    - The predicates `klw11kjf` (the target location(s) where the final action is applied)
      are static (they do not change), but we collect them both from static facts and any
      initial state facts for robustness.
    - The predicates `kut2ksnk` (the locations where we can produce `f2xo7hb6`) are taken
      from the current state (they can be consumed by diwlmgg9) and thus are looked up in
      the dynamic state when computing the heuristic.
    - There is at least one `omggbn9z` fact in solvable states (the agent's current location).
      If none exists, the heuristic returns a large finite number (INF).

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - self.goals: the goal facts to be achieved (task.goals).
    - klw_map: mapping from an object p (nxbd1hc3) -> set of locations l such that
               (klw11kjf p l) is present in static or in the initial state.
    - adjacency: a dictionary mapping each location -> set(neighbor_locations) derived
                 from the static acovodrf facts. For each static fact (acovodrf u v)
                 both u->v and v->u are added (movement is supported in both directions
                 when either direction is present).
    """

    def __init__(self, task):
        # store goal facts (set of strings)
        self.goals = task.goals

        # build adjacency graph for locations based on acovodrf static facts
        self.adjacency = {}
        static = task.static if task.static is not None else frozenset()

        def add_edge(u, v):
            if u not in self.adjacency:
                self.adjacency[u] = set()
            self.adjacency[u].add(v)

        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) == 3:
                u, v = parts[1], parts[2]
                # allow movement in both directions if acovodrf fact present:
                add_edge(u, v)
                add_edge(v, u)

        # collect klw11kjf facts (target locations for the final action).
        # klw11kjf might be in static or in init - include both for robustness.
        self.klw_map = {}  # p -> set(loc)
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf" and len(parts) == 3:
                p, loc = parts[1], parts[2]
                self.klw_map.setdefault(p, set()).add(loc)

        # also include any klw11kjf in the task's initial state (if present there)
        if hasattr(task, "initial_state"):
            for fact in task.initial_state:
                parts = get_parts(fact)
                if parts[0] == "klw11kjf" and len(parts) == 3:
                    p, loc = parts[1], parts[2]
                    self.klw_map.setdefault(p, set()).add(loc)

    # --- helper methods -------------------------------------------------
    def _bfs_distances_from(self, source):
        """
        BFS on the adjacency graph to compute shortest distances (in moves)
        from `source` to all reachable locations. Returns a dict loc->distance.
        """
        dist = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        """
        Step-By-Step Thinking for Computing Heuristic
        --------------------------------------------
        1. If all goal facts are already present in the current state, return 0.
        2. Extract the current agent positions from `(omggbn9z <loc>)` facts in the state.
           If there are no such facts, return a large finite number (INF).
        3. For every goal object p (where goal is `(a7sb56om p)`), if p is already true in
           the state, skip it. Otherwise, estimate the cost to achieve it as follows:
             - If `(f2xo7hb6 p)` is already true, we only need to move the agent to any
               location `l_klw` with `(klw11kjf p l_klw)` and perform `y7nik9q2` (cost
               = movement distance + 1).
             - Otherwise, we must first produce `f2xo7hb6 p` by executing `diwlmgg9` at
               some location `l_kut` where `(kut2ksnk p l_kut)` holds in the current state.
               Then travel from `l_kut` to a `l_klw` (where `klw11kjf p l_klw` holds) and
               perform `y7nik9q2`. Cost = dist(current_agent_loc, l_kut) + 1 (diwlmgg9) +
                      dist(l_kut, l_klw) + 1 (y7nik9q2).
           - If there are multiple `l_kut` or `l_klw`, pick the pairing that minimizes the cost.
           - If either a required `l_kut` or `l_klw` does not exist or is unreachable,
             treat the cost as INF.
        4. Because multiple goal objects must be achieved sequentially with a single agent,
           we simulate a greedy sequence: starting from the agent's current location, at each
           step pick the remaining goal whose incremental cost from the current agent location
           is minimal, add that cost, update the agent location to the `l_klw` chosen for that
           goal, and continue. This produces a compact and informative estimate without
           solving the full routing problem (cheap and effective).
        5. Return the accumulated cost (or INF if some required facts/locations are missing).
        """
        state = node.state

        # 1) Goals already achieved?
        if self.goals <= state:
            return 0

        # 2) Get current agent locations (omggbn9z facts)
        agent_locs = [get_parts(f)[1] for f in state if get_parts(f)[0] == "omggbn9z"]
        if not agent_locs:
            # No agent location available - return large finite number
            return INF

        # For simplicity pick one starting agent location (the domain uses a single agent).
        cur_loc = agent_locs[0]

        # 3) Collect dynamic facts needed:
        # - which p already has f2xo7hb6
        f2_set = {get_parts(f)[1] for f in state if get_parts(f)[0] == "f2xo7hb6"}
        # - which (kut2ksnk p loc) facts exist in the current state
        kut_map = {}  # p -> set(loc)
        for f in state:
            parts = get_parts(f)
            if parts[0] == "kut2ksnk" and len(parts) == 3:
                p, loc = parts[1], parts[2]
                kut_map.setdefault(p, set()).add(loc)

        # - goals: extract p for each (a7sb56om p) in self.goals
        goal_ps = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "a7sb56om":
                goal_ps.append(parts[1])

        # Build a set of locations that we might need distances for:
        # include all adjacency nodes, all klw targets, all kut locations and current loc
        all_locs = set(self.adjacency.keys())
        for vals in self.klw_map.values():
            all_locs.update(vals)
        for vals in kut_map.values():
            all_locs.update(vals)
        all_locs.add(cur_loc)

        # Ensure adjacency has entries for locations encountered but possibly isolated.
        for l in list(all_locs):
            self.adjacency.setdefault(l, set())

        # Precompute BFS distances on demand and cache them
        dist_cache = {}

        def dist(u, v):
            """Return shortest distance (in moves) from u to v or INF if unreachable."""
            if u == v:
                return 0
            if u not in dist_cache:
                dist_cache[u] = self._bfs_distances_from(u)
            return dist_cache[u].get(v, INF)

        total_cost = 0
        remaining = [p for p in goal_ps if f"(a7sb56om {p})" not in state]

        # If there are no remaining goals, return 0
        if not remaining:
            return 0

        # Greedy sequencing: at each step pick the goal p with minimal incremental cost
        while remaining:
            best = None
            best_cost = INF
            best_choice = None  # store (p, chosen_l_kut, chosen_l_klw, cost)

            for p in remaining:
                # find all possible klw targets for p
                klw_locs = self.klw_map.get(p, set())
                if not klw_locs:
                    # no target location known -> cannot satisfy this goal
                    continue

                # If p already has f2xo7hb6, we only need to move to a klw location and apply y7nik9q2
                if p in f2_set:
                    # minimal cost is min over klw_locs of dist(cur_loc, klw) + 1
                    for klw in klw_locs:
                        d = dist(cur_loc, klw)
                        if d >= INF:
                            continue
                        c = d + 1  # y7nik9q2
                        if c < best_cost:
                            best_cost = c
                            best_choice = (p, None, klw, c)
                else:
                    # must first perform diwlmgg9 at some kut location (from current state)
                    kut_locs = kut_map.get(p, set())
                    if not kut_locs:
                        # cannot produce f2xo7hb6 for p -> skip (unsolvable)
                        continue
                    # try all (kut, klw) pairings
                    for kut in kut_locs:
                        d1 = dist(cur_loc, kut)
                        if d1 >= INF:
                            continue
                        for klw in klw_locs:
                            d2 = dist(kut, klw)
                            if d2 >= INF:
                                continue
                            c = d1 + 1 + d2 + 1  # move to kut + diwlmgg9 + move kut->klw + y7nik9q2
                            if c < best_cost:
                                best_cost = c
                                best_choice = (p, kut, klw, c)

            # If we could not find any feasible next goal (missing facts/locations/unreachable),
            # then the state is (for our heuristic) effectively unsolvable -> return INF.
            if best_choice is None:
                return INF

            # Apply the chosen step in the heuristic simulation:
            p_chosen, chosen_kut, chosen_klw, cost_chosen = best_choice
            total_cost += cost_chosen
            # After completing the goal, agent will be at chosen_klw
            cur_loc = chosen_klw
            # Mark the goal as achieved in our simulation
            remaining.remove(p_chosen)
            # Also mark f2xo7hb6 as consumed/removed (y7nik9q2 deletes it), but that only matters
            # if other goals could depend on the same f2xo7hb6 (they don't - f2xo7hb6 is per p).
            # We also remove any kut facts for p_chosen because diwlmgg9 deletes kut2ksnk (not necessary here).

        return total_cost if total_cost < INF else INF