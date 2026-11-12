from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Miconic (elevator) domain.

    The heuristic estimates the number of actions required to serve all
    remaining passengers (i.e., make all (served pX) facts true). It
    combines:
      - an estimate of the number of lift-move actions required to visit
        the relevant floors, computed as the cost to connect all floors
        that must be visited (using a Minimum Spanning Tree lower bound
        on the graph of floors with shortest-path distances) plus the
        cost to reach that set from the current lift floor, and
      - the count of required board/depart actions (per passenger).

    Assumptions
    -----------
    - The domain contains facts:
        (origin p f), (destin p f), (boarded p), (served p), (lift-at f),
        and static (above f1 f2) facts that define directed edges for lift moves.
    - Multiple passengers can be boarded simultaneously (no capacity limits).
    - Every (destin p f) is available in static facts (this is how instance
      files are provided in the benchmark).
    - The static "above" relation is used to build the (directed) floor-graph
      for computing shortest-path distances. If movement is possible only
      in one direction according to static facts, distances are computed
      respecting that direction.
    - The heuristic is not required to be admissible. It tries to give a
      guiding (and generally informative) estimate while being efficient
      to compute.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
      - stores the goal facts,
      - extracts destin mappings (passenger -> destination floor) from
        static facts (these are usually given as static in Miconic),
      - builds the directed adjacency graph between floors based on
        static (above f1 f2) facts,
      - prepares data structures for computing shortest-path distances
        between floors on demand (BFS + memoization).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic computes:

    1. Quick goal check:
       - If all goal (served p) facts are already true in the state,
         return 0.

    2. Identify which passengers still need to be served:
       - For each passenger that is not yet served:
         - If they are already boarded, they only require a depart action
           executed at their destination floor.
         - Otherwise they require a board action at their origin floor
           and a depart action at their destination floor.

    3. Collect the set of floors that the lift must visit:
       - For each unserved passenger:
         - If boarded: add the passenger's destination floor to the set.
         - If not boarded: add both origin and destination floors to the set.
       - If the set is empty, but the goal is not yet reached (shouldn't
         happen in normal encodings), a small finite cost is returned.

    4. Estimate lift-move actions:
       - Compute shortest-path distances (number of up/down actions) between
         floors using BFS over the directed adjacency given by (above ...).
       - Compute an MST (Prim algorithm) cost over the set of required
         floors using those pairwise shortest-path distances. The MST cost
         is a lower bound on the number of moves needed to connect all
         required floors.
       - Add the minimum distance from the current lift floor to the
         required-floor set (the entry cost).
       - The sum (entry cost + MST cost) is used as the estimated number
         of lift-move actions. Distances that are unreachable are treated
         as large but finite (so the heuristic returns a finite number,
         but such instances are usually unsolvable).

    5. Count passenger-handling actions:
       - Each unserved passenger contributes:
           1 if already boarded (only depart needed),
           2 if not boarded (board + depart).
       - Sum these to obtain board/depart action costs.

    6. Return the total estimate:
       - total = estimated_lift_moves + total_board_depart_actions

    This heuristic is designed to be informative (accounts for which floors
    must be visited and the necessary board/depart actions) while remaining
    efficiently computable for typical Miconic instances.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Extract destination mapping passenger -> destination floor
        self.destin = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) >= 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

        # Build directed adjacency graph for floors from (above f1 f2)
        self.adj = {}
        floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) >= 3:
                f1, f2 = parts[1], parts[2]
                floors.add(f1)
                floors.add(f2)
                self.adj.setdefault(f1, set()).add(f2)
                # Note: do NOT automatically add the reverse edge; above facts
                # determine allowed directed lift moves.

        # Ensure every floor seen in destin or above appears in adjacency map
        for f in list(floors) + list(self.destin.values()):
            self.adj.setdefault(f, set())

        # Memoization for BFS shortest paths between floors
        self._dist_cache = {}  # (src, dst) -> distance (int) or None if unreachable

    def _shortest_distance(self, src, dst):
        """Return shortest directed distance (number of move actions) from src to dst.

        Uses BFS on the directed adjacency graph. Returns a large finite number
        if dst is unreachable (so heuristic remains finite).
        """
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # BFS
        visited = {src}
        queue = deque([(src, 0)])
        found = None
        while queue:
            node, d = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == dst:
                    found = d + 1
                    queue.clear()
                    break
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, d + 1))
        # If not found, treat as large finite distance
        if found is None:
            # Use a large finite cost to keep heuristic finite on (possibly)
            # unsolvable states. This value is chosen to be clearly larger
            # than realistic elevator traversals in typical instances.
            found = 100000
        self._dist_cache[key] = found
        return found

    def __call__(self, node):
        state = node.state

        # If already at goal, return 0
        if self.goals <= state:
            return 0

        # Determine current lift floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at" and len(parts) >= 2:
                lift_floor = parts[1]
                break
        if lift_floor is None:
            # If the state's lift position is missing (shouldn't happen),
            # assume an arbitrary floor from adjacency to keep finite.
            lift_floor = next(iter(self.adj.keys()))

        # Identify passengers that still need serving
        unserved_passengers = []
        boarded_passengers = set()
        served_passengers = set()
        origins = {}  # passenger -> origin if present in state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) >= 2:
                served_passengers.add(parts[1])
            elif parts[0] == "boarded" and len(parts) >= 2:
                boarded_passengers.add(parts[1])
            elif parts[0] == "origin" and len(parts) >= 3:
                origins[parts[1]] = parts[2]

        # Build list of passengers to consider (those that appear in destin mapping)
        for p in self.destin.keys():
            if p not in served_passengers:
                unserved_passengers.append(p)

        if not unserved_passengers:
            # No unserved passengers but goal not satisfied -> return 0 as fallback
            return 0

        # Build set of floors that must be visited by the lift
        required_floors = set()
        for p in unserved_passengers:
            if p in boarded_passengers:
                # Only need to visit destination
                required_floors.add(self.destin.get(p))
            else:
                # Need to visit origin (to board) and destination (to depart)
                origin_floor = origins.get(p)
                if origin_floor is not None:
                    required_floors.add(origin_floor)
                # Destination should exist in destin mapping
                required_floors.add(self.destin.get(p))

        # Ensure all these floors exist in adjacency map (add with empty neighbors)
        for f in list(required_floors):
            if f is not None:
                self.adj.setdefault(f, set())

        # If somehow all required floors are None, return finite fallback
        if not required_floors or all(f is None for f in required_floors):
            # Count board/depart actions only
            total_board_depart = sum(1 if p in boarded_passengers else 2 for p in unserved_passengers)
            return total_board_depart

        # Convert to list and compute pairwise shortest distances
        floors_list = [f for f in required_floors if f is not None]
        n = len(floors_list)

        # Precompute distances between floors used in MST and also from lift floor
        dist_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    dist_matrix[i][j] = self._shortest_distance(floors_list[i], floors_list[j])

        # Entry cost: minimal distance from current lift floor to any required floor
        entry_cost = min(self._shortest_distance(lift_floor, f) for f in floors_list)

        # Compute MST cost over the required floors using Prim's algorithm on the complete graph
        # with weights = shortest-path distances
        mst_cost = 0
        visited = [False] * n
        # Start from node 0
        visited[0] = True
        edges = [dist_matrix[0][j] for j in range(n)]
        for _ in range(n - 1):
            # find smallest edge connecting visited set to unvisited set
            min_cost = None
            min_idx = -1
            for j in range(n):
                if not visited[j]:
                    cost = edges[j]
                    if min_cost is None or cost < min_cost:
                        min_cost = cost
                        min_idx = j
            if min_idx == -1:
                # disconnected; give large penalty
                mst_cost += 100000
                break
            visited[min_idx] = True
            mst_cost += min_cost if min_cost is not None else 100000
            # update edges
            for j in range(n):
                if not visited[j]:
                    if dist_matrix[min_idx][j] < edges[j]:
                        edges[j] = dist_matrix[min_idx][j]

        # Count board/depart actions
        total_board_depart = 0
        for p in unserved_passengers:
            if p in boarded_passengers:
                total_board_depart += 1  # depart
            else:
                total_board_depart += 2  # board + depart

        total_estimate = entry_cost + mst_cost + total_board_depart
        # Return integer heuristic
        return int(total_estimate)