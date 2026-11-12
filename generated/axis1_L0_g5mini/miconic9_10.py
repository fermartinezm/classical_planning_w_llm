from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math
from collections import deque


def get_parts(fact):
    """Extract the components of a PDDL fact by removing the surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a PDDL fact matches a pattern of components.
    Wildcard '*' is supported via fnmatch.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the miconic elevator domain. The heuristic estimates
    the number of actions still required to serve all passengers (i.e., make all
    (served p) facts true). It accounts for:
      - boarding actions (one per passenger that is not yet boarded),
      - departing actions (one per passenger that is not yet served),
      - an estimate of elevator travel actions required to visit all relevant floors
        (origins of waiting passengers and destinations of unserved passengers),
        starting from the elevator's current floor.

    Assumptions
    -----------
    - Boarding a passenger costs 1 action (the 'board' operator).
    - Departing (serving) a passenger costs 1 action (the 'depart' operator).
    - Moving the elevator between floors uses the 'up' and 'down' operators.
      We compute exact minimal move counts between floors by BFS on the directed
      graph induced by the static 'above' facts (an edge exists for an 'up' move
      when (above f1 f2) is present, and for a 'down' move when (above f2 f1)
      is present).
    - Multiple passengers can be boarded at once (no capacity limit). Thus the
      elevator can pick up several passengers along a route and drop them off
      at their destination floors.
    - The heuristic is allowed to be inadmissible (it is intended for greedy
      best-first search). It returns 0 exactly when the goal is reached.

    Heuristic Initialization
    ------------------------
    In the constructor we extract the following static information from the task:
      - destin_for_passenger: mapping passenger -> destination floor (from (destin p f) facts).
      - floors: set of all floor names (collected from 'above' facts and destin facts).
      - adjacency list for elevator moves derived from 'above' static facts:
          * an up-edge f1 -> f2 exists whenever (above f1 f2) is present.
          * a down-edge f1 -> f2 exists whenever (above f2 f1) is present.
      - We precompute nothing else since distances can be computed on demand cheaply
        (the number of floors is typically small).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the state's facts already include all goal facts, return 0.
    2. Identify the current elevator floor from the fact '(lift-at F)' in the state.
    3. Identify which passengers are already served (skip them).
    4. For each unserved passenger p:
       - If '(boarded p)' is true in the state, the passenger is onboard and we
         only need to visit their destination floor to apply 'depart' (1 action).
         Add the passenger's destination floor to the set of floors_to_visit.
         There is no board action cost for this passenger.
       - If the passenger is not boarded, then they currently have an '(origin p F_o)'
         fact in the state. We must visit F_o to board (1 action) and F_d (their
         destination) to depart (1 action). Add both floors to floors_to_visit.
         If origin equals destination, the floor appears once in the set but
         both actions (board+depart) are still counted in action costs.
    5. Count board_costs = number of unserved passengers that are not boarded.
       Count depart_costs = number of unserved passengers (every unserved passenger
       requires one depart action).
    6. Compute an estimate for the minimal number of elevator move actions needed
       to visit all floors in floors_to_visit starting from the current elevator
       floor. We compute this lower-bound estimate as:
         move_estimate = min_distance_from_lift_to_some_target
                         + MST_weight_over_targets
       where
         - min_distance_from_lift_to_some_target is the minimal BFS distance
           from the current lift floor to any floor in floors_to_visit (0 if empty).
         - MST_weight_over_targets is the weight of a minimum spanning tree (Prim)
           computed on the set floors_to_visit using pairwise shortest-path
           distances between floors as edge weights. The MST weight provides a
           lower bound on the number of moves required to connect all target floors.
       This combination gives an informed but cheaply computable estimate of
       travel effort required to visit all necessary floors.
    7. The final heuristic value is:
         heuristic = board_costs + depart_costs + move_estimate
    Notes:
    - The heuristic is finite for solvable states because the BFS distance
      computations produce finite distances if floors are mutually reachable in
      the static 'above' graph (which must be true in solvable instances).
    - This heuristic tends to prefer states that reduce the number of remaining
      board/depart actions and that place the elevator nearer to remaining
      pickup/dropoff floors. It exploits the fact that boarding and departing
      costs are per-passenger and that elevator moves can be amortized by
      serving multiple passengers on a single route.
    """

    def __init__(self, task):
        """Extract static information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Destination floor for every passenger: from (destin p f).
        self.destin_for_passenger = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_for_passenger[passenger] = floor

        # Build set of floors from 'above' facts and destin facts.
        floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                floors.add(parts[1])
                floors.add(parts[2])
        for floor in self.destin_for_passenger.values():
            floors.add(floor)
        self.floors = sorted(floors)  # sorted only to have deterministic ordering

        # Map floor -> index for arrays
        self.floor_index = {f: i for i, f in enumerate(self.floors)}

        # Build adjacency for elevator move graph (directed), using static 'above'.
        self.adj = {f: set() for f in self.floors}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                # 'up' action allows move f1 -> f2 when (above f1 f2) is true
                if f1 in self.adj:
                    self.adj[f1].add(f2)
                # 'down' action will be represented implicitly by the reverse above fact
                # (if (above f2 f1) is present it adds f2 -> f1). We don't add reverse here.

    def _shortest_distances_from(self, start_floor):
        """
        Run BFS from start_floor to compute minimal number of elevator moves
        to every floor. Returns dict floor->distance (int). Floors unreachable
        will not appear in the returned dict.
        """
        if start_floor not in self.adj:
            return {}
        dist = {start_floor: 0}
        q = deque([start_floor])
        while q:
            f = q.popleft()
            d = dist[f]
            for nbr in self.adj[f]:
                if nbr not in dist:
                    dist[nbr] = d + 1
                    q.append(nbr)
            # Also allow moving in the 'down' direction when the reverse 'above' exists:
            # check all floors that have an adjacency to f (i.e., edges x -> f)
            # Since the graph is small, we can scan adjacency to find predecessors.
            for pred, nbrs in self.adj.items():
                if f in nbrs and pred not in dist:
                    dist[pred] = d + 1
                    q.append(pred)
        return dist

    def _all_pairs_shortest(self):
        """
        Compute all-pairs shortest path distances between floors using repeated BFS.
        Returns a dict of dicts: dist[a][b] = minimal moves from a to b (int).
        """
        distances = {}
        for f in self.floors:
            distances[f] = self._shortest_distances_from(f)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the node's state."""
        state = node.state

        # 1) If goal is reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Find current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, f = get_parts(fact)
                lift_floor = f
                break

        # If for any reason lift floor not found, pessimistically assume 0 move start.
        if lift_floor is None:
            lift_floor = next(iter(self.floors)) if self.floors else None

        # 3) Determine served, boarded and origins
        served = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origins = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, p, f = get_parts(fact)
                origins[p] = f

        # 4) Build floors_to_visit and count action costs for board/depart
        floors_to_visit = set()
        board_costs = 0
        depart_costs = 0

        # Iterate over all passengers known via destin mapping (those are the goal passengers)
        for p, dest in self.destin_for_passenger.items():
            if p in served:
                continue  # no cost for already served passengers

            # This passenger is unserved -> will require a depart action
            depart_costs += 1

            if p in boarded:
                # passenger already onboard -> only need to visit destination
                floors_to_visit.add(dest)
            else:
                # passenger waiting -> need to visit origin and destination, and board once
                board_costs += 1
                orig = origins.get(p)
                if orig is not None:
                    floors_to_visit.add(orig)
                # Even if origin missing (unusual), include destination as a fallback required floor
                floors_to_visit.add(dest)

        # If no floors to visit, movement cost is 0
        if not floors_to_visit:
            return board_costs + depart_costs

        # 5) Compute pairwise shortest distances between floors needed (and from lift)
        # We'll compute all-pairs distances for the small floor graph
        distances = self._all_pairs_shortest()

        # Helper to get distance with fallback to large number
        INF = 10 ** 6

        def dist(a, b):
            if a == b:
                return 0
            if a in distances and b in distances[a]:
                return distances[a][b]
            # try reverse path since edges are treated directed but we allowed predecessors in BFS;
            # BFS implementation already considered reverse edges, but keep fallback
            if b in distances and a in distances[b]:
                return distances[b][a]
            return INF

        # 6) min distance from lift to any target floor
        min_from_lift = INF
        for f in floors_to_visit:
            d = dist(lift_floor, f)
            if d < min_from_lift:
                min_from_lift = d
        if min_from_lift == INF:
            # unreachable: return a large heuristic but finite
            min_from_lift = len(self.floors) + 10

        # 7) Compute MST weight over floors_to_visit using pairwise distances as metric
        # Prim's algorithm
        targets = list(floors_to_visit)
        if not targets:
            mst_weight = 0
        else:
            used = {targets[0]}
            remaining = set(targets[1:])
            mst_weight = 0
            # Precompute pairwise distances between targets
            # (small sets, so O(n^2) is fine)
            while remaining:
                best_edge = None
                best_cost = INF
                for u in used:
                    for v in remaining:
                        c = dist(u, v)
                        if c < best_cost:
                            best_cost = c
                            best_edge = (u, v)
                if best_cost >= INF:
                    # disconnection: assign large penalty and break
                    mst_weight += len(remaining) + 10
                    break
                mst_weight += best_cost
                used.add(best_edge[1])
                remaining.remove(best_edge[1])

        move_estimate = min_from_lift + mst_weight

        # 8) Sum all components
        heuristic_value = board_costs + depart_costs + move_estimate

        # Return integer heuristic
        return int(heuristic_value)