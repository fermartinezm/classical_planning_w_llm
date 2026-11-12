from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the miconic (elevator) domain.

    The heuristic estimates the number of actions required to reach a goal
    state (all passengers served). It reasons about:
    - travel actions required to move the elevator between floors (up/down),
    - the board actions required to pick up passengers at their origin floors,
    - the depart actions required to drop off (serve) passengers at their destination floors.

    The heuristic is not required to be admissible; it aims to be informative
    and cheaply computable so as to reduce node expansions in greedy best-first search.

    Assumptions
    -----------
    - The elevator has effectively unlimited capacity (no capacity constraints in domain).
    - Travel between two floors is possible iff the static facts include
      an "above" relationship that connects the floors (we treat the above
      relations as undirected edges for reachability/travel cost).
    - Each "up" or "down" action has uniform cost 1; similarly, "board" and
      "depart" actions each cost 1. The heuristic counts actions, not time.
    - Static "destin" facts are available (provided as part of task.static).
    - Unreachable floors (disconnected graph) are treated as very expensive
      (large finite cost) so that solvable states still yield finite heuristic values.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - mapping passenger -> destination floor from static facts (destin),
    - the set of floors (collected from static and task facts),
    - an undirected adjacency graph of floors based on "above" static facts.
      We add an undirected edge between f1 and f2 if (above f1 f2) or
      (above f2 f1) appears in static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state, the heuristic computes an estimate as follows:

    1. Identify passengers that still need to be served:
       - A passenger is "served" if "(served p)" is present in the state.
       - Otherwise it is unserved and must be accounted for.

    2. For each unserved passenger p:
       - If p is already boarded (state contains "(boarded p)"):
         - The passenger only requires a depart action when the elevator
           reaches p's destination floor. So we add one depart action for p
           and we include p's destination floor in the set of floors that must
           be visited.
       - If p is not boarded (state contains an "(origin p f_orig)"):
         - The passenger requires a board action at origin and a depart
           action at destination. We add both actions (board + depart) and
           include both origin and destination floors in the set of floors
           that must be visited (origin must be visited before destination,
           but for heuristic aggregation we only require both to be visited).

    3. Travel cost approximation:
       - Let S be the set of distinct floors that must be visited (origins and/or destinations).
       - Let s be the current elevator floor (from "(lift-at f)" in the state).
       - Compute shortest-path distances between floors (unweighted graph built
         from "above" edges). Distances are the number of up/down actions.
       - We approximate the minimal travel actions required to visit all floors
         in S starting from s by:
           travel_cost ≈ MST_cost(S) + min_distance(s, S)
         where MST_cost(S) is the weight of the minimum spanning tree over S
         using shortest-path distances as edge weights, and min_distance(s, S)
         is the minimal shortest-path distance from the start floor s to any
         floor in S.
       - This is a standard cheap approximation for the "cover all nodes"
         routing problem. It is optimistic relative to a naive sum of
         independent passenger travel costs while allowing shared travel to be
         exploited.

    4. Total heuristic:
       - total_cost = travel_cost + total_board_actions + total_depart_actions
       - If there are no required floors (S empty) but there are unserved
         passengers (should not happen under normal states because unserved
         implies at least their destination is required), the heuristic still
         returns a finite count (sum of board/departs).

    Notes on correctness and performance
    -----------------------------------
    - The heuristic returns 0 exactly when the goal is reached (all passengers served).
    - Distances are computed with BFS on the undirected floor graph; for small
      numbers of floors this remains efficient.
    - The MST computation uses Prim's algorithm on the complete graph induced
      by S with precomputed pairwise shortest-path distances.
    - If some floors in S are unreachable from each other in the static graph,
      the distances are treated as a large finite number (len(floors)+1000) so
      that the heuristic still returns a finite value while heavily penalizing
      unreachable situations.
    """

    def __init__(self, task):
        """Extract static information required by the heuristic."""
        self.goals = task.goals
        static_facts = task.static

        # Map passenger -> destination floor from static facts (destin).
        self.destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor

        # Build undirected adjacency graph of floors from above relations.
        # We add an undirected edge between f1 and f2 if (above f1 f2) or (above f2 f1).
        self.floor_neighbors = {}
        floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f1, f2 = parts[1], parts[2]
                floors.add(f1)
                floors.add(f2)
                self.floor_neighbors.setdefault(f1, set()).add(f2)
                self.floor_neighbors.setdefault(f2, set()).add(f1)

        # Ensure floors that might appear only in other facts are present (some tasks
        # may not put all floors into above but use them in destin/origin/lift-at).
        # We'll lazily add floors seen later in a given state if needed.
        self.all_static_floors = set(floors)

        # A large finite penalty for unreachable connections to keep heuristic finite.
        self.INF_COST = 10000

    def _bfs_distances_from(self, start, nodes_of_interest=None):
        """
        Compute shortest-path distances (unweighted) from start to all floors
        reachable via floor_neighbors. If nodes_of_interest is provided, we
        stop BFS early when all of them are discovered.
        """
        visited = {start}
        dist = {start: 0}
        q = deque([start])
        found = set()
        if nodes_of_interest and start in nodes_of_interest:
            found.add(start)
            if found == set(nodes_of_interest):
                return dist
        while q:
            u = q.popleft()
            for v in self.floor_neighbors.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    q.append(v)
                    if nodes_of_interest and v in nodes_of_interest:
                        found.add(v)
                        if found == set(nodes_of_interest):
                            return dist
        return dist

    def _pairwise_distances(self, nodes):
        """
        Compute pairwise shortest-path distances between nodes (floors).
        Return dict-of-dict distances[a][b] = distance or INF_COST if unreachable.
        """
        distances = {}
        for a in nodes:
            dmap = self._bfs_distances_from(a, nodes_of_interest=set(nodes))
            # Fill distances for all nodes; unreachable -> INF_COST
            distances[a] = {b: dmap.get(b, self.INF_COST) for b in nodes}
        return distances

    def _mst_cost(self, nodes, dist_matrix):
        """
        Compute the cost of a minimum spanning tree over 'nodes' using Prim's
        algorithm and the provided complete distance matrix dist_matrix.
        dist_matrix[a][b] gives the weight between nodes a and b.
        If nodes is empty or a single node, return 0.
        """
        if not nodes:
            return 0
        nodes = list(nodes)
        if len(nodes) == 1:
            return 0
        in_mst = {nodes[0]}
        not_in_mst = set(nodes[1:])
        total = 0
        # For each node not yet in MST, track cheapest connection to in_mst
        best = {n: dist_matrix[nodes[0]][n] for n in not_in_mst}
        while not_in_mst:
            # pick node with minimal best connection
            pick = min(not_in_mst, key=lambda x: best[x])
            cost = best[pick]
            total += cost
            in_mst.add(pick)
            not_in_mst.remove(pick)
            # update best values
            for n in not_in_mst:
                if dist_matrix[pick][n] < best[n]:
                    best[n] = dist_matrix[pick][n]
        return total

    def __call__(self, node):
        """
        Estimate the number of actions remaining to serve all passengers.

        Returns an integer heuristic value (0 iff all goals are reached).
        """
        state = node.state

        # Quick goal check: return 0 if all goals are satisfied.
        if self.goals <= state:
            return 0

        # Extract current lift floor.
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                lift_floor = parts[1]
                break

        # Collect dynamic floors that might not be in static facts.
        # Also gather passenger info from the state.
        boarded = set()
        origins = {}  # passenger -> origin floor (only present if not boarded)
        served = set()
        passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "boarded":
                passenger = parts[1]
                boarded.add(passenger)
                passengers.add(passenger)
            elif pred == "origin":
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor
                passengers.add(passenger)
                # remember this floor exists
                self.all_static_floors.add(floor)
            elif pred == "served":
                passenger = parts[1]
                served.add(passenger)
                passengers.add(passenger)
            elif pred == "lift-at":
                if lift_floor is None:
                    lift_floor = parts[1]
                    self.all_static_floors.add(lift_floor)

        # Ensure destin mapping includes all passengers (some tasks might put destin in static)
        # If a passenger has no known destin, we cannot compute; we simply treat as INF.
        # Build set of unserved passengers
        unserved = [p for p in passengers if p not in served]

        # If there are no unserved passengers (should have been caught by goals check),
        # return 0
        if not unserved:
            return 0

        # Build the set S of floors that must be visited:
        required_floors = set()
        total_board_actions = 0
        total_depart_actions = 0

        for p in unserved:
            dest = self.destin.get(p)
            if dest is None:
                # If destination is unknown in static, we cannot estimate properly.
                # Penalize heavily but keep finite.
                total_depart_actions += 1
                continue
            # every unserved passenger will need a depart()
            total_depart_actions += 1
            if p in boarded:
                # already boarded: only need to visit destination and depart
                required_floors.add(dest)
            else:
                # not boarded: we must visit origin to board, then destination to depart
                total_board_actions += 1
                origin = origins.get(p)
                if origin is None:
                    # origin not found in state (should not happen), penalize by assuming it's at dest
                    required_floors.add(dest)
                else:
                    required_floors.add(origin)
                    required_floors.add(dest)

        # If lift_floor is unknown for some reason, pick an arbitrary floor from required_floors
        if lift_floor is None:
            if required_floors:
                # pick one to avoid crashes; heuristic will be valid though possibly weak
                lift_floor = next(iter(required_floors))
            else:
                # No required floors and no lift floor -> just return sum of actions
                return total_board_actions + total_depart_actions

        # Ensure all floors appear in floor_neighbors mapping (so BFS works).
        for f in list(required_floors) + [lift_floor]:
            if f not in self.floor_neighbors:
                # If a floor has no neighbors recorded, add an isolated entry so BFS works.
                self.floor_neighbors.setdefault(f, set())

        # If there are no floors to visit (should not normally happen) return sum of actions.
        if not required_floors:
            return total_board_actions + total_depart_actions

        # Compute pairwise distances between required floors (and include lift_floor)
        floors_for_dist = set(required_floors)
        floors_for_dist.add(lift_floor)
        # Ensure that static-known floors are also present in neighbor map (to avoid KeyError)
        for f in floors_for_dist:
            self.floor_neighbors.setdefault(f, set())

        # Precompute pairwise distances for nodes in required_floors plus lift_floor.
        dist_map = self._pairwise_distances(floors_for_dist)

        # Compute min distance from lift to any required floor
        min_dist_from_lift = min(dist_map[lift_floor].get(f, self.INF_COST) for f in required_floors)

        # Compute MST cost over the required floors only (not including the lift)
        # For MST we need distances between required floors; extract a matrix
        required_list = list(required_floors)
        required_dist_matrix = {
            a: {b: dist_map[a].get(b, self.INF_COST) for b in required_floors} for a in required_floors
        }
        mst_cost = self._mst_cost(required_list, required_dist_matrix)

        travel_cost = mst_cost + min_dist_from_lift

        total_cost = travel_cost + total_board_actions + total_depart_actions

        # As a sanity clamp (keep finite and integer)
        if total_cost >= self.INF_COST:
            total_cost = self.INF_COST

        return int(total_cost)