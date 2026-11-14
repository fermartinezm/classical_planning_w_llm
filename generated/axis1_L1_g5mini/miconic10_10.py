from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a fact string like '(pred a b)' -> ['pred','a','b']."""
    return fact.strip()[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z). The domain
    contains a single "active location" token (omggbn9z <loc>) that can be
    moved along directed connectivity edges (acovodrf <from> <to>).
    For each object p of type nxbd1hc3 the goal is to obtain (a7sb56om p).
    To achieve (a7sb56om p) the following sequence is required:
      1. Have p carry the intermediate flag (f2xo7hb6 p) by executing
         diwlmgg9 at a location where (kut2ksnk p <loc_kut>) holds while the
         active marker is at <loc_kut>.
      2. Then move the active marker to the location <loc_klw> where
         (klw11kjf p <loc_klw>) holds and execute y7nik9q2 to obtain
         (a7sb56om p).
    Movement of the active marker is achieved by the operators qs8pdrqj and
    i0veo5kv that require and add/remove the omggbn9z fact along edges given
    by acovodrf.

    Assumptions
    -----------
    - Moving the active marker along a single acovodrf edge costs 1 (one action).
    - The two domain actions that affect a p's progression to the goal,
      diwlmgg9 and y7nik9q2, each cost 1.
    - klw11kjf facts are not deleted by any operator in the domain (they are
      static or remain present), but kut2ksnk facts can be removed by
      diwlmgg9. Therefore klw locations are extracted once from the task
      specification, while kut locations are read from the current state.
    - If required connectivity or required facts are missing, the heuristic
      returns a large finite number to indicate difficulty/unsolvability.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal facts (a7sb56om ...).
    - Store the static facts.
    - Build the directed adjacency graph of locations from all acovodrf facts
      found in the task static facts (and initial facts just in case).
    - Precompute shortest-path distances between all locations in the graph
      using BFS from every node (sufficient since the graph is small and BFS
      is efficient).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (state), the heuristic computes an estimate of the
    number of actions required to reach a goal state as follows:

    1. If all goals are already satisfied in the state, return 0.

    2. Parse the current state to locate:
       - current active marker location (the unique fact '(omggbn9z <loc>)'),
       - which objects already have the intermediate flag (f2xo7hb6 p),
       - the current kut locations (kut2ksnk p <loc>) available in the current
         state (these may be absent if they were already consumed).

       Klw locations (klw11kjf p <loc>) are taken from the task's static/initial
       facts that were extracted during initialization since they are never
       removed by domain actions.

    3. For every object p whose goal (a7sb56om p) is not yet achieved,
       compute the cost to handle p when the active marker is at some
       current location L:
         - If p already has f2xo7hb6: cost = dist(L, loc_klw) + 1 (one action to
           apply y7nik9q2 at loc_klw).
         - Otherwise: cost = dist(L, loc_kut) + 1 (diwlmgg9) + dist(loc_kut, loc_klw) + 1 (y7nik9q2)
       Distances are shortest-path lengths in the acovodrf graph. If a
       required distance is not defined (no path) or a required location is
       missing, we treat the cost as a large finite number.

    4. Schedule objects in a greedy order to reflect that the active marker
       remains where the last y7nik9q2 was applied:
       - Start from the current active marker position.
       - Repeatedly pick the unsatisfied object p with minimal incremental
         cost from the current active position, add that cost to the total,
         and update the current active position to p's klw location (since
         after finishing p the active marker will be there).
       This greedy ordering keeps the heuristic informative while being
       efficient (O(n^2) for n goal objects).

    5. Return the accumulated cost as the heuristic estimate. The value is 0
       exactly at goal states and finite for solvable states (or a large
       finite number if connectivity/facts prevent achievement).
    """

    def __init__(self, task):
        # Goals: set of strings like '(a7sb56om p1)'
        self.goals = set(task.goals)

        # Save static facts for later use (they persist in every state)
        self.static_facts = set(task.static)

        # For klw11kjf we extract locations from task.static U task.initial_state
        # because these facts are not deleted by domain actions.
        known_facts = set(task.static) | set(task.initial_state)
        self.klw_map = {}
        for fact in known_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "klw11kjf" and len(parts) >= 3:
                p = parts[1]
                loc = parts[2]
                self.klw_map[p] = loc

        # Build directed adjacency from all acovodrf facts present in static or
        # initial facts (connectivity is static in the provided instances).
        self.adj = {}
        nodes = set()
        for fact in known_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                src = parts[1]
                dst = parts[2]
                nodes.add(src)
                nodes.add(dst)
                self.adj.setdefault(src, set()).add(dst)

        # Also ensure nodes include any klw locations we discovered
        for loc in self.klw_map.values():
            nodes.add(loc)
            self.adj.setdefault(loc, set())

        # Precompute all-pairs shortest distances using BFS per node
        self.distances = {n: {} for n in nodes}
        for src in nodes:
            # BFS from src
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.distances[src] = dist

        # A conservative large cost for unreachable/unrepresentable situations.
        self.large_cost = 10 ** 6

    def __call__(self, node):
        state = set(node.state)

        # If goals already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Current active marker location: '(omggbn9z <loc>)'
        active_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "omggbn9z" and len(parts) >= 2:
                active_loc = parts[1]
                break
        # If no active marker, treat as unreachable (large cost)
        if active_loc is None:
            return self.large_cost

        # Current intermediate flag holders
        f2_holders = {get_parts(f)[1] for f in state if get_parts(f)[0] == "f2xo7hb6"}

        # Current kut locations (may be consumed during planning), look in state U static
        union_current = state | self.static_facts
        kut_map = {}
        for fact in union_current:
            parts = get_parts(fact)
            if parts and parts[0] == "kut2ksnk" and len(parts) >= 3:
                p = parts[1]
                loc = parts[2]
                kut_map[p] = loc

        # Set of goal objects (those that must have a7sb56om)
        goal_objs = [get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "a7sb56om"]

        # Unsatisfied objects: ones whose a7sb56om goal not in current state
        unsatisfied = [p for p in goal_objs if f"(a7sb56om {p})" not in state]

        total_cost = 0
        current_pos = active_loc

        # Greedy schedule: always pick the next object that is cheapest from current_pos
        while unsatisfied:
            best_p = None
            best_cost = None
            best_klw = None
            best_kut = None

            for p in unsatisfied:
                # klw location must be known (from initialization)
                klw_loc = self.klw_map.get(p)
                if klw_loc is None:
                    # cannot ever apply y7nik9q2 for this p if klw missing
                    cost = self.large_cost
                    kut_loc = None
                else:
                    if p in f2_holders:
                        # Need to move from current_pos to klw_loc and apply y7nik9q2
                        if current_pos in self.distances and klw_loc in self.distances[current_pos]:
                            cost = self.distances[current_pos][klw_loc] + 1
                        else:
                            cost = self.large_cost
                        kut_loc = None
                    else:
                        # Need to visit kut then klw
                        kut_loc = kut_map.get(p)
                        if kut_loc is None:
                            # If kut location is missing in the union of current & static, unreachable
                            cost = self.large_cost
                        else:
                            d1 = self.distances.get(current_pos, {}).get(kut_loc)
                            d2 = self.distances.get(kut_loc, {}).get(klw_loc)
                            if d1 is None or d2 is None:
                                cost = self.large_cost
                            else:
                                cost = d1 + 1 + d2 + 1
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_p = p
                    best_klw = klw_loc
                    best_kut = kut_loc

            # Add best_cost to total and update structures
            if best_cost is None:
                return self.large_cost
            # If unreachable (large cost), we can short-circuit to return large cost
            if best_cost >= self.large_cost:
                return self.large_cost

            total_cost += best_cost
            # After completing best_p, the active marker will be at its klw location
            current_pos = best_klw if best_klw is not None else current_pos

            # Simulate that best_p is now satisfied (we don't mutate the real state)
            unsatisfied.remove(best_p)
            # Also simulate that its f2 flag is consumed/created as relevant:
            # After the full sequence, a7sb56om is true for best_p, so remove from f2_holders if present
            f2_holders.discard(best_p)

        return int(total_cost)