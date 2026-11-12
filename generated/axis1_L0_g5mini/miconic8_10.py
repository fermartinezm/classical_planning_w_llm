from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque


def get_parts(fact):
    """Extract components of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the miconic (elevator) domain.

    The heuristic estimates the number of actions still required to serve all
    passengers (i.e., make (served pX) true for each goal passenger). It combines
    an estimate of the travel (lift move) actions required to visit necessary
    floors with the number of board/depart actions that must still be executed.

    Assumptions
    -----------
    - The domain uses the predicates:
        (origin ?p ?f)    - passenger waiting at origin floor
        (destin ?p ?f)    - passenger's destination floor (static)
        (boarded ?p)      - passenger is currently on the lift
        (served ?p)       - passenger already served (goal)
        (lift-at ?f)      - current lift floor
        (above ?f1 ?f2)   - static relation that allows movement between floors
    - For every static fact (above a b) we assume an up action from a->b and
      a down action from b->a are both possible (so we add undirected edges).
    - Boarding and departing are each counted as 1 action.
    - Moving the lift between two floors requires as many move actions as the
      shortest path length in the floor graph derived from the above facts.
    - The heuristic is not required to be admissible; it is designed to be
      informative and efficiently computable to guide greedy search.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - destin_map: mapping passenger -> destination floor (from static facts).
    - floor_graph: adjacency mapping of floors derived from (above ...) facts,
      treating each (above a b) as an undirected edge a <-> b for movement.
    - goal_passengers: set of passengers that must be served (derived from task.goals)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node (state), the heuristic proceeds as follows:

    1. If all goal-served facts are already true in the state, return 0.

    2. Identify the current lift floor (the unique fact '(lift-at fX)').

    3. Determine passenger status:
       - Served passengers: ignore (they contribute 0).
       - Boarded passengers: they must still be departed at their destination:
         * Add the passenger's destination floor to the set of required floors.
         * Account for 1 action (depart).
       - Not-boarded, not-served passengers: they must be
         transported from their origin to their destination:
         * Add both origin and destination floors to the set of required floors.
         * Account for 2 actions (board + depart).

    4. Compute floor-travel estimation:
       - Build the set R of required floors (destinations and origins as above).
       - Compute shortest-path distances between all floors in R and also from
         the lift floor to floors in R using BFS on the floor graph.
       - If some required floor is unreachable (no path), return a large finite
         heuristic (problem likely unsolvable).
       - Approximate the minimal number of move actions needed to visit all
         required floors starting from the current lift floor by:
         a) computing the minimum distance from the lift floor to any floor in R,
            and
         b) computing the Minimum Spanning Tree (MST) length over the complete
            graph on nodes R with edge weights equal to the shortest-path distances.
         The travel estimate is: min_dist_lift_to_R + MST_length.
         This provides a compact approximation of an optimal tour that visits
         the necessary floors (it captures sharing of travel between multiple stops).

    5. The final heuristic is the sum of:
       - travel estimate (number of move actions),
       - number of board/depart actions still required (1 per boarded depart, 2
         per not-boarded passenger).

    This heuristic is efficient to compute in practice because the number of
    required floors is at most twice the number of unserved passengers, and BFS
    / Prim's algorithm on these small graphs is fast.
    """

    def __init__(self, task):
        # Goals: a set of strings like "(served p1)".
        self.goals = task.goals

        # Extract destin facts from static info: '(destin p f)'
        self.destin_map = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin_map[passenger] = floor

        # Build floor graph from (above f1 f2) facts. For each (above a b),
        # allow movement both ways (up from a->b and down from b->a).
        self.floor_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                a, b = parts[1], parts[2]
                # add undirected edge a<->b (both directions are useful for BFS)
                self.floor_graph.setdefault(a, set()).add(b)
                self.floor_graph.setdefault(b, set()).add(a)

        # Ensure floors appearing in destin_map also exist as nodes (even if isolated)
        for f in self.destin_map.values():
            self.floor_graph.setdefault(f, set())

        # Store the set of passengers that must be served (from goal facts).
        self.goal_passengers = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 2:
                self.goal_passengers.add(parts[1])

    def bfs_distances(self, start):
        """
        Compute shortest-path distances (in number of move actions) from start
        to all reachable floors using the floor_graph. Returns a dict floor->dist.
        """
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.floor_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def prim_mst_length(self, nodes, dist_matrix):
        """
        Compute MST length for a complete graph over 'nodes' where the weight
        between u and v is dist_matrix[u][v]. If any required pair is
        unreachable (distance missing), return None.
        Uses a simple Prim algorithm (O(n^2) for small n).
        """
        if not nodes:
            return 0
        nodes = list(nodes)
        n = len(nodes)
        in_mst = [False] * n
        # key values for Prim
        key = [float("inf")] * n
        key[0] = 0
        total = 0
        for _ in range(n):
            # pick min key vertex not yet in MST
            u = -1
            min_key = float("inf")
            for i in range(n):
                if not in_mst[i] and key[i] < min_key:
                    min_key = key[i]
                    u = i
            if u == -1 or min_key == float("inf"):
                return None  # disconnected
            in_mst[u] = True
            total += key[u]
            # update keys
            for v in range(n):
                if not in_mst[v]:
                    duv = dist_matrix[nodes[u]].get(nodes[v], None)
                    if duv is None:
                        # unreachable edge: treat as infinite
                        candidate = float("inf")
                    else:
                        candidate = duv
                    if candidate < key[v]:
                        key[v] = candidate
        return total

    def __call__(self, node):
        state = node.state

        # 1) If all goals reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Identify current lift floor.
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at" and len(parts) == 2:
                lift_floor = parts[1]
                break
        # If lift position unknown, heuristic uses no initial connection (treat as isolated).
        if lift_floor is None:
            # fall back: pick arbitrary floor from floor_graph if available
            lift_floor = next(iter(self.floor_graph), None)

        # 3) Classify passengers and prepare required floors and action counts.
        required_floors = set()
        board_depart_actions = 0  # count of board+depart actions left (board=1, depart=1)
        # For boarded passengers we only need depart (1).
        # For not-boarded passengers we need board+depart (2).

        # Determine boarded and served sets from state
        boarded = {get_parts(f)[1] for f in state if get_parts(f)[0] == "boarded" and len(get_parts(f)) == 2}
        served = {get_parts(f)[1] for f in state if get_parts(f)[0] == "served" and len(get_parts(f)) == 2}

        # For each passenger required by goals, plan remaining work
        for passenger in self.goal_passengers:
            if passenger in served:
                continue  # already done
            destin = self.destin_map.get(passenger)
            if destin is None:
                # If we don't know the destination for the passenger, assume one
                # action for depart and be conservative with movement: we'll still
                # include no floor (cannot do better). Count actions so heuristic stays finite.
                if passenger in boarded:
                    board_depart_actions += 1
                else:
                    board_depart_actions += 2
                continue

            if passenger in boarded:
                # Need to go to destination and depart
                required_floors.add(destin)
                board_depart_actions += 1  # depart
            else:
                # Not boarded: must visit origin then destination
                # Origin should be present in state as (origin p f)
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                        origin_floor = parts[2]
                        break
                if origin_floor is None:
                    # This should not normally happen (origin removed on board),
                    # but if it does assume we must still perform board+depart (count actions)
                    required_floors.add(destin)
                    board_depart_actions += 2
                else:
                    required_floors.add(origin_floor)
                    required_floors.add(destin)
                    board_depart_actions += 2  # board + depart

        # If there are no required floors left (but some passengers not explicitly in 'served' set),
        # rely on board_depart_actions (shouldn't normally happen), still compute minimal moves = 0.
        if not required_floors:
            return board_depart_actions

        # 4) Compute distances from each required floor (and lift floor) to all others via BFS.
        # Make sure every floor appears in floor_graph (isolated nodes possible).
        for f in list(required_floors) + ([lift_floor] if lift_floor is not None else []):
            self.floor_graph.setdefault(f, set())

        # Precompute BFS distances from every required floor and from lift floor.
        dist_from = {}
        all_nodes_for_bfs = set(required_floors)
        if lift_floor is not None:
            all_nodes_for_bfs.add(lift_floor)
        for node_floor in all_nodes_for_bfs:
            dist_from[node_floor] = self.bfs_distances(node_floor)

        # Check reachability: every required floor should be reachable from every other
        # in the subgraph implied by floor_graph. For MST and lift-to-R we need distances.
        # If any required floor is unreachable from some other required floor, we still
        # can attempt to compute MST; if any needed distance missing, return large finite.
        INF = 10 ** 6

        # 5) Compute min distance from lift to any required floor
        if lift_floor is None or lift_floor not in dist_from:
            min_dist_lift_to_R = INF
        else:
            dists = dist_from[lift_floor]
            min_dist_lift_to_R = min((dists.get(f, INF) for f in required_floors), default=INF)

        if min_dist_lift_to_R >= INF:
            # unreachable from lift
            return INF

        # 6) Build a "distance matrix" mapping floor->(floor->distance) for required nodes
        dist_matrix = {}
        for f in required_floors:
            dist_matrix[f] = {}
            df = dist_from.get(f, {})
            for g in required_floors:
                if f == g:
                    dist_matrix[f][g] = 0
                else:
                    # if unreachable, we leave it absent (Prim treats as missing -> disconnected)
                    if g in df:
                        dist_matrix[f][g] = df[g]

        # 7) Compute MST length over required_floors using Prim (on the complete graph with shortest path weights)
        mst_len = self.prim_mst_length(required_floors, dist_matrix)
        if mst_len is None:
            # disconnected among required floors => unsolvable; return large finite
            return INF

        travel_estimate = min_dist_lift_to_R + mst_len

        # 8) Final heuristic: travel moves + board/depart actions
        heuristic_value = travel_estimate + board_depart_actions

        # Ensure finite integer
        if heuristic_value >= INF:
            return INF
        return int(heuristic_value)