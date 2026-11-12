from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the list of tokens of a PDDL fact string like '(pred a b)' -> ['pred', 'a', 'b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z).

    The domain models a simple two-stage processing pipeline for objects of
    type nxbd1hc3 (I'll call them "items" or p-objects) using facilities of
    type t4rdxh32 (I'll call them "facilities" or f-objects). The planner
    controls a single movable marker represented by the fluent (omggbn9z f)
    which indicates where the "processing agent" currently is. There are
    two processing operators for items:
      - diwlmgg9 (first stage): requires the agent at a facility f where
        (kut2ksnk p f) holds; it sets (f2xo7hb6 p) and removes (kut2ksnk p f).
      - y7nik9q2 (second stage): requires the agent at a facility f where
        (klw11kjf p f) holds and (f2xo7hb6 p) holds; it produces (a7sb56om p).

    Facilities are connected by static relations (acovodrf f1 f2). Two
    separate movement operators allow moving the agent along these edges;
    practically this yields undirected travel along an edge listed in the
    static facts. Each travel operation (moving the agent from one facility
    to an adjacent facility) costs one action. Each processing operator also
    costs one action.

    The heuristic estimates the number of actions (moves + processing ops)
    required to reach the goals (facts of the form (a7sb56om p)). The estimate
    is not required to be admissible; it is designed to be informative while
    remaining cheap to compute.

    Assumptions
    -----------
    - The static facts include connectivity (acovodrf ...) and the location
      predicates for the second-stage processing (klw11kjf ...). The first-stage
      availability (kut2ksnk p f) is a dynamic fact (present in the current
      state) and must be read from the state.
    - There is at least one (omggbn9z f) fluent active in the state. If none
      exist the heuristic returns a large finite value (the state is likely
      unsolvable).
    - Movement along an acovodrf pair is treated as undirected for planning
      purposes: if acovodrf u v exists then the agent can move between u and v
      in both directions using the available operators.
    - Each diwlmgg9 and y7nik9q2 application costs 1 action; each move along an
      adjacency costs 1 action.
    - When multiple items remain, the heuristic simulates a greedy sequence:
      it repeatedly selects the item whose remaining (simulated) completion
      cost is smallest from the current agent location, applies that simulated
      completion (updating the agent location), and accumulates cost. This
      captures sharing of travel costs and gives a stronger estimate than a
      naive independent-sum heuristic, while remaining efficient.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and stores:
    - The set of goal item identifiers (from task.goals).
    - The static adjacency graph of facilities based on acovodrf facts.
      The adjacency is stored undirected: a static fact (acovodrf x y)
      yields edges x<->y.
    - The mapping from each item to the set of facilities where the second-stage
      action y7nik9q2 is applicable (klw11kjf p f). This mapping is static.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node (which provides the current state as frozenset of facts)
    the heuristic proceeds:

    1. Quick goal check:
       - If all goal facts are already true in the state, return 0.

    2. Extract dynamic facts from the current state:
       - The set of items already completed (a7sb56om p).
       - The set of items that have finished the first stage (f2xo7hb6 p).
       - The dynamic mapping of items to facilities where the first-stage
         operation is currently available (from kut2ksnk p f facts).
       - The current positions of the agent (the facilities f with omggbn9z f).
         If multiple are present we pick one to simulate from (we pick an
         arbitrary one but the greedy sequencing mitigates the effect).

    3. Precompute shortest path distances between all facilities that appear
       in the problem (facilities mentioned in static connectivity, in klw
       mappings, and in the current state's kut/omgg facts). Distances are
       computed with BFS on the undirected adjacency graph. Unreachable
       distances are treated as large finite costs.

    4. Greedy simulated completion:
       - Maintain a set of items that still need (a7sb56om).
       - Simulate finishing items one by one. In each simulation step:
         a) For every remaining item, compute the minimal cost to finish it
            when the agent is currently at `current_pos`:
               - If the item has f2xo7hb6: only the second-stage action is
                 needed. Cost = distance(current_pos, any klw-facility) + 1
                 (apply y7nik9q2). The chosen endpoint becomes that klw facility.
               - Otherwise (first-stage still required): we need to
                 (move to some kut-facility) + diwlmgg9 + (move to some
                 klw-facility) + y7nik9q2. Cost = dist(current_pos, kut_f)
                 + 1 + dist(kut_f, klw_f) + 1, minimized over kut_f and klw_f.
         b) Choose the item with minimal simulated completion cost, add that
            cost to the heuristic total, update current_pos to the endpoint
            facility after completion, and remove the item from the remaining set.
       - Repeat until all remaining items are simulated finished.

    5. Return the accumulated simulated cost as the heuristic estimate.

    Notes
    -----
    - The greedy ordering attempts to exploit shared travel between processing
      different items; it is a fast approximation and typically yields fewer
      expanded nodes than an independent-sum heuristic.
    - The heuristic always returns 0 for states that already satisfy all
      goal facts. For unreachable items (no available kut- or klw-facilities
      or disconnected graph), it returns a large finite value so the search
      can still proceed but will de-prioritize such states.
    """

    def __init__(self, task):
        """Extract static information: goals, klw mapping, and the facility graph."""
        # Raw goal fact strings
        self.goal_facts = set(task.goals)

        # Parse goal facts to extract item identifiers for goals of form (a7sb56om p)
        self.goal_items = []
        for gf in self.goal_facts:
            parts = get_parts(gf)
            if parts and parts[0] == "a7sb56om" and len(parts) >= 2:
                self.goal_items.append(parts[1])

        # Static facts: adjacency (acovodrf) and klw11kjf locations.
        static = set(task.static)

        # Facility adjacency graph (undirected): if acovodrf x y exists, add both x->y and y->x
        self.adj = {}
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # klw_map: item -> set of facilities where second-stage op is applicable (static)
        self.klw_map = {}
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "klw11kjf" and len(parts) >= 3:
                p, f = parts[1], parts[2]
                self.klw_map.setdefault(p, set()).add(f)

        # If some facilities mentioned as nodes have no adjacency list entry yet,
        # ensure they appear (so BFS can handle isolated nodes).
        # Also ensure facilities mentioned in klw_map are present.
        for facilities in self.klw_map.values():
            for f in facilities:
                self.adj.setdefault(f, set())

    def __call__(self, node):
        """Compute the heuristic estimate for the node's state."""
        state = node.state

        # If goal already reached, heuristic is 0 (check using stored goal facts).
        if self.goal_facts <= state:
            return 0

        # Extract dynamic facts from the state
        # Items already completed by second-stage (a7sb56om)
        completed = set()
        # Items that finished first stage (f2xo7hb6)
        first_done = set()
        # Dynamic mapping for kut2ksnk facts: item -> set of facilities
        kut_map = {}
        # Current agent locations (facilities with omggbn9z)
        omgg_positions = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "a7sb56om" and len(parts) >= 2:
                completed.add(parts[1])
            elif pred == "f2xo7hb6" and len(parts) >= 2:
                first_done.add(parts[1])
            elif pred == "kut2ksnk" and len(parts) >= 3:
                p, f = parts[1], parts[2]
                kut_map.setdefault(p, set()).add(f)
                # ensure facility exists in adj as a node
                self.adj.setdefault(f, set())
            elif pred == "omggbn9z" and len(parts) >= 2:
                omgg_positions.add(parts[1])
                self.adj.setdefault(parts[1], set())

        # Set of items that still need to become a7sb56om
        remaining_items = [p for p in self.goal_items if p not in completed]
        if not remaining_items:
            return 0

        # Build the full set of facility nodes to consider for BFS:
        facilities = set(self.adj.keys())
        # Also include any facilities mentioned in kut_map or klw_map or omgg_positions
        for s in kut_map.values():
            facilities.update(s)
        for s in self.klw_map.values():
            facilities.update(s)
        facilities.update(omgg_positions)

        # If we have no known facility nodes, return a large finite estimate
        if not facilities:
            return 1000

        # Precompute BFS distances from every facility node in facilities
        # Use undirected adjacency self.adj which may have been extended above.
        # Distances stored as dict: distances[src][dst] = integer steps (moves)
        distances = {}
        for src in facilities:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, ()):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            distances[src] = dist

        # Helper to get distance with a finite large penalty for unreachable nodes.
        INF = 10**6

        def dist_between(a, b):
            if a in distances and b in distances[a]:
                return distances[a][b]
            return INF

        # Choose an initial current position for the agent.
        # If multiple omgg positions exist pick the one that yields smallest sum
        # of minimal distances to required facilities (heuristic pick).
        if not omgg_positions:
            # No agent location in the state; unreachable in normal domain usage but
            # return large finite heuristic.
            return 1000

        # Pick the best start among omgg_positions heuristically:
        # For each candidate start, compute sum of minimal distances to finish each item
        # (cheap heuristic) and pick the start with minimum sum.
        best_start = None
        best_score = None
        for start in omgg_positions:
            score = 0
            for p in remaining_items:
                # if p already first_done then need nearest klw facility
                if p in first_done:
                    klw_fs = self.klw_map.get(p, set())
                    if not klw_fs:
                        score += INF
                    else:
                        score += min(dist_between(start, f) for f in klw_fs)
                else:
                    kut_fs = kut_map.get(p, set())
                    klw_fs = self.klw_map.get(p, set())
                    if not kut_fs or not klw_fs:
                        score += INF
                    else:
                        # minimal cost to do both stages starting at start (ignoring op costs)
                        best_pair = INF
                        for k in kut_fs:
                            dk = dist_between(start, k)
                            for l in klw_fs:
                                dk_l = dist_between(k, l)
                                best_pair = min(best_pair, dk + dk_l)
                        score += best_pair
            if best_score is None or score < best_score:
                best_score = score
                best_start = start
        current_pos = best_start

        total_cost = 0

        # Greedy simulated completion: repeatedly pick the next item with minimal
        # simulated completion cost starting from current_pos.
        remaining = set(remaining_items)
        while remaining:
            best_item = None
            best_item_cost = None
            best_end_pos = None

            for p in remaining:
                klw_fs = self.klw_map.get(p, set())
                if not klw_fs:
                    # no place to perform second stage -> very costly/unreachable
                    item_cost = INF
                    end_pos = None
                elif p in first_done:
                    # only second-stage required
                    # cost = moves to nearest klw facility + 1 (y7nik9q2)
                    d = min(dist_between(current_pos, f) for f in klw_fs)
                    item_cost = d + 1
                    # choose endpoint facility yielding minimal d
                    for f in klw_fs:
                        if dist_between(current_pos, f) == d:
                            end_pos = f
                            break
                else:
                    # need first-stage (kut) then second-stage (klw)
                    kut_fs = kut_map.get(p, set())
                    if not kut_fs:
                        item_cost = INF
                        end_pos = None
                    else:
                        best_pair_cost = INF
                        best_pair_end = None
                        for k in kut_fs:
                            d1 = dist_between(current_pos, k)
                            for l in klw_fs:
                                d2 = dist_between(k, l)
                                # cost includes the two processing actions (+1 each)
                                cost = d1 + 1 + d2 + 1
                                if cost < best_pair_cost:
                                    best_pair_cost = cost
                                    best_pair_end = l
                        item_cost = best_pair_cost
                        end_pos = best_pair_end

                if best_item_cost is None or item_cost < best_item_cost:
                    best_item_cost = item_cost
                    best_item = p
                    best_end_pos = end_pos

            # If the best item is unreachable (cost INF), return a large finite value.
            if best_item_cost is None or best_item_cost >= INF:
                return 1000

            # Accumulate cost, update simulated agent position, and remove item.
            total_cost += best_item_cost
            if best_end_pos is not None:
                current_pos = best_end_pos
            # Mark item as done in the simulation (so it won't be processed again)
            remaining.remove(best_item)

        return int(total_cost)