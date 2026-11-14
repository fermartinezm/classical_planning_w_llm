from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (domain ud3uf11z).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal predicates `(a7sb56om ?p)` 
    for all relevant objects `?p` of type `nxbd1hc3`. The goal requires that these objects hold the predicate 
    `a7sb56om`. The heuristic evaluates for each such object whether this predicate already holds or how many 
    steps are needed to achieve it, based on the domain rules and current facts.

    Assumptions
    -----------
    - The goal predicates require `(a7sb56om ?praz5)`.
    - Objects of type nxbd1hc3 (denoted ?p) need to transition from an initial unknown state to having `(a7sb56om ?p)`.
    - To achieve `(a7sb56om ?p)`, the object must previously have `(f2xo7hb6 ?p)`.
    - To get `(f2xo7hb6 ?p)`, the object requires to have a `(kut2ksnk ?p ?w)` fact for some `?w` with `(omggbn9z ?w)` true, then apply the action `diwlmgg9`.
    - `(omggbn9z ?w)` predicates represent "current location" or reachable nodes.
    - Movement on `t4rdxh32` objects is possible based on `(acovodrf ?from ?to)` connectivity and actions `qs8pdrqj`/`i0veo5kv`.
    - The goal is achieved when `(a7sb56om ?p)` is true for all goal objects.
    - The heuristic is not necessarily admissible but must efficiently estimate costs to minimize expanded nodes.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal objects of type `nxbd1hc3` for which `(a7sb56om ?p)` is required.
    - Extract static facts:
        * The connectivity graph implied by `(acovodrf ?from ?to)` for `t4rdxh32` objects, which are movement edges.
        * The static predicates `(klw11kjf ?p ?w)` for objects `?p` and locations `?w`.
    - Precompute the static directed graph of locations for efficient shortest path computations.
    - Store the `t4rdxh32` objects' set for validation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state (all goal `(a7sb56om ?p)` predicates hold), heuristic returns 0.

    2. For each goal object `p` (nxbd1hc3), if `(a7sb56om p)` already holds, no cost is added.

    3. Otherwise, estimate cost to reach `(a7sb56om p)`:
       a) To get `(a7sb56om p)`, `(f2xo7hb6 p)` must be true and `(klw11kjf p w)` must hold at some `w`.
       b) To get `(f2xo7hb6 p)`, `(kut2ksnk p w')` and `(omggbn9z w')` must hold for some `w'`,
          followed by action `diwlmgg9` to add `(f2xo7hb6 p)` and remove `(kut2ksnk p w')`.
       c) Then, with `(f2xo7hb6 p)` and `(klw11kjf p w)`, and `(omggbn9z w)`, apply `y7nik9q2` to achieve `(a7sb56om p)`.

    4. The main difficulty comes from `(omggbn9z ?w)` facts representing the current positions on `t4rdxh32` objects.
       The planner actions `qs8pdrqj` and `i0veo5kv` allow moving `(omggbn9z ?w)` along edges defined by `(acovodrf ?from ?to)`.

    5. From the current state, there is exactly one `(omggbn9z ?w_curr)` true (a single current location).

    6. The heuristic estimates the minimal number of moves to move `(omggbn9z ?w)` from current location to locations needed for `kut2ksnk` and `klw11kjf` predicates.

    7. For each object `p`:
       - Identify all `w_kut` with `(kut2ksnk p w_kut)` true in current or static (since `kut2ksnk` is not static, check current).
       - Identify all `w_klw` with `(klw11kjf p w_klw)` from static.
       - The object must:
         i)    Move `(omggbn9z)` from current location to one of the `w_kut` locations where `(kut2ksnk p w_kut)` holds to apply `diwlmgg9`.
         ii)   Apply `diwlmgg9` once: cost 1.
         iii)  Move `(omggbn9z)` to one of the `w_klw` locations to apply `y7nik9q2`.
         iv)   Apply `y7nik9q2` once: cost 1.

    8. Moving `(omggbn9z ?w)` between locations is done by `qs8pdrqj` or `i0veo5kv` (cost 1 action per edge).

    9. To minimize cost, compute shortest path distances between current `(omggbn9z)` location and all possible `w_kut` and between each `w_kut` and possible `w_klw`.

    10. The heuristic cost per object is minimum over all pairs `(w_kut, w_klw)` of:
          distance(current_loc, w_kut) + 1 (diwlmgg9) + distance(w_kut, w_klw) + 1 (y7nik9q2)

    11. Sum this cost over all objects lacking `(a7sb56om p)`.

    12. Return the sum as heuristic value.

    Efficiency considerations
    -------------------------
    - Precompute shortest paths using BFS on the static directed graph of `t4rdxh32` locations.
    - Use memoization to avoid repeated BFS per call.
    - Return finite heuristic for solvable states.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal objects of type nxbd1hc3 that require (a7sb56om p)
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om":
                self.goal_objects.add(parts[1])

        # Static facts extraction:
        # Build directed graph of locations (t4rdxh32) from (acovodrf ?from ?to)
        self.locations = set()
        self.location_graph = dict()  # adjacency list: loc -> set of neighbors

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf":
                src, dst = parts[1], parts[2]
                self.locations.add(src)
                self.locations.add(dst)
                self.location_graph.setdefault(src, set()).add(dst)

        # Extract all klw11kjf static facts for all objects p and locations w
        # These define possible target locations for y7nik9q2
        self.klw_map = dict()  # p -> set of w with (klw11kjf p w)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf":
                p, w = parts[1], parts[2]
                self.klw_map.setdefault(p, set()).add(w)

    def _bfs_distances(self, start):
        """
        Compute shortest path distances in the location graph from start to all reachable locations.
        Returns a dict: location -> distance
        """
        from collections import deque

        dist = {start: 0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self.location_graph.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] = dist[u] + 1 if v != u else 0
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Find the current omggbn9z location (exactly one should be true)
        current_locations = [get_parts(fact)[1] for fact in state if fact.startswith("(omggbn9z ")]
        if not current_locations:
            # No current location, no way to move, return large cost
            return float('inf')
        current_location = current_locations[0]

        # Build maps of current state for quick lookup
        # kut2ksnk predicates in current state: for all p and locations w
        kut_map = dict()  # p -> set of w where (kut2ksnk p w) holds
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "kut2ksnk":
                p, w = parts[1], parts[2]
                kut_map.setdefault(p, set()).add(w)

        # f2xo7hb6 predicates in current state (objects p having predicate)
        f2xo7hb6_set = {get_parts(fact)[1] for fact in state if fact.startswith("(f2xo7hb6 ")}

        # a7sb56om predicates in current state (objects p having predicate)
        a7sb56om_set = {get_parts(fact)[1] for fact in state if fact.startswith("(a7sb56om ")}

        # Precompute shortest distances from current location to all others via BFS
        # If current location is not in graph (isolated), distances empty
        if current_location in self.location_graph or current_location in self.locations:
            distances_from_current = self._bfs_distances(current_location)
        else:
            distances_from_current = {}

        total_cost = 0

        for p in self.goal_objects:
            if p in a7sb56om_set:
                # Already satisfied for this goal object: 0 cost
                continue

            # Cost estimation for object p:

            # We must apply action y7nik9q2:
            # preconditions: (omggbn9z w_klw) && (klw11kjf p w_klw) && (f2xo7hb6 p)
            # To have f2xo7hb6 p: apply diwlmgg9:
            # preconditions: (omggbn9z w_kut) && (kut2ksnk p w_kut)

            # Check if object p already has (f2xo7hb6)
            # If yes, only need to move OM to some w_klw in klw11kjf[p] and apply y7nik9q2

            # Otherwise, need also to do diwlmgg9 first.

            # Get possible kut and klw locations
            kut_ws = kut_map.get(p, set())
            klw_ws = self.klw_map.get(p, set())

            if not klw_ws:
                # No known klw location, hard to estimate; add a conservative penalty
                klw_ws = set()

            if p in f2xo7hb6_set:
                # Already has f2xo7hb6: only need y7nik9q2 step
                # Cost: minimal distance from current_location to any w_klw + 1 (action)

                min_dist_to_klw = min([distances_from_current.get(w, float('inf')) for w in klw_ws], default=float('inf'))
                if min_dist_to_klw == float('inf'):
                    # unreachable, penalize heavily
                    total_cost += 1000
                    continue
                cost = min_dist_to_klw + 1  # 1 for y7nik9q2 action
                total_cost += cost

            else:
                # Must do diwlmgg9 and y7nik9q2

                # If no kut_ws locations available for p (cannot apply diwlmgg9), heuristic cannot progress
                if not kut_ws or not klw_ws:
                    total_cost += 1000
                    continue

                # For efficiency, precompute distances from each kut_w to all klw_ws
                # We do BFS from each kut_w to find shortest paths to klw_ws

                min_cost_for_p = float('inf')

                for w_kut in kut_ws:
                    dist_to_w_kut = distances_from_current.get(w_kut, float('inf'))
                    if dist_to_w_kut == float('inf'):
                        continue
                    # BFS distances from w_kut
                    dist_from_w_kut = self._bfs_distances(w_kut)
                    dist_to_klw = min([dist_from_w_kut.get(w_klw, float('inf')) for w_klw in klw_ws], default=float('inf'))
                    if dist_to_klw == float('inf'):
                        continue

                    # Total cost:
                    # move OM from current_location to w_kut: dist_to_w_kut
                    # diwlmgg9 action: 1
                    # move OM from w_kut to w_klw: dist_to_klw
                    # y7nik9q2 action: 1
                    cost = dist_to_w_kut + 1 + dist_to_klw + 1
                    if cost < min_cost_for_p:
                        min_cost_for_p = cost

                if min_cost_for_p == float('inf'):
                    # unreachable
                    total_cost += 1000
                else:
                    total_cost += min_cost_for_p

        return total_cost
