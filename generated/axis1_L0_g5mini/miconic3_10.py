from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque


def get_parts(fact):
    """Return the token list of a PDDL fact string, e.g. "(at p1 f1)" -> ["at","p1","f1"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Miconic domain (elevator / passengers).
    The heuristic estimates the number of actions required to serve all remaining
    (not-yet-served) passengers. It combines:
      - a count of required boarding/departing actions (per passenger),
      - an estimate of the number of elevator moves (up/down) needed to visit
        all necessary floors. The elevator movement estimate is computed by
        constructing a graph of floors (from static `above` facts), computing
        shortest path distances between relevant floors, and using a minimum
        spanning tree (MST) lower-bound plus the distance from the current
        elevator floor to that tree as a compact approximation of the travel
        cost to visit all required floors.

    Assumptions
    -----------
    - Static facts include `above` relations that link floors. We assume the
      `above` relation connects floors so that treating each `above a b` as an
      undirected edge produces a connected graph of floors in solvable tasks.
    - For every passenger there is a static `(destin passenger floor)` fact.
    - Dynamic facts in the state include `(origin passenger floor)`, `(boarded passenger)`,
      `(served passenger)`, and `(lift-at floor)` as in the domain file.
    - Each board and depart action costs 1; each elevator up/down action costs 1.
    - The heuristic is not necessarily admissible (it may overestimate); it is
      designed to be informative and computationally efficient.

    Heuristic Initialization
    ------------------------
    - Extract static `above` facts to build an undirected adjacency map between floors.
    - Extract static `(destin passenger floor)` facts into a mapping passenger -> destination floor.
    - Store the set of all floors observed in the static knowledge (from `above` facts
      and from destin facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a planner node/state:
      1. If all goals (served facts) are already true in the state, return 0.
      2. Find the elevator's current floor by locating the single `(lift-at ...)` fact.
      3. For each passenger (derived from static `destin` facts),
         - If `(served passenger)` is true in the state, ignore the passenger.
         - Else, if `(boarded passenger)` is true, we must still perform a `depart` at
           the passenger's destination: add 1 to the boarding/departing action count
           and include the passenger's destination floor as a required floor to visit.
         - Else (passenger not boarded): we must visit the passenger's origin to `board`
           (1 action), and later visit the destination to `depart` (1 action). Include
           both floors (origin and destination) in the set of required floors.
      4. Compute shortest-path distances (number of elevator moves) between all required
         floors and from the current lift floor to each required floor using BFS on
         the floor adjacency graph (constructed from `above`).
      5. Estimate the elevator travel cost as:
         - If there are no required floors, travel cost is 0.
         - Otherwise, compute the total weight of an MST over the required-floor nodes
           using the pairwise shortest-path distances (Prim's algorithm, O(n^2) for small n).
           Then add the distance from the current elevator floor to the closest required floor.
         This gives a compact lower-bound style estimate for the number of moves needed
         to visit all required floors starting from the current lift location.
      6. Return heuristic = (boarding + departing action count) + (estimated elevator move cost).
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Build undirected adjacency between floors using `above` static facts.
        # We interpret each `(above A B)` as an undirected connection between A and B
        # for the purposes of elevator move reachability. This is consistent with the
        # domain where moves connect floors via above relations.
        self.adjacency = defaultdict(set)
        self.floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":
                f1, f2 = parts[1], parts[2]
                self.adjacency[f1].add(f2)
                self.adjacency[f2].add(f1)
                self.floors.add(f1)
                self.floors.add(f2)

        # Extract static destin facts: passenger -> destination floor
        self.destin = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                self.floors.add(floor)

    def _bfs_distances(self, start):
        """
        BFS from start floor over the adjacency graph to compute shortest path
        distances (# elevator moves) from start to all floors known in self.floors.
        Returns dict floor->distance (unreachable floors absent).
        """
        distances = {}
        if start not in self.adjacency and start not in self.floors:
            # Isolated floor with no adjacency; still mark distance 0 to itself.
            distances[start] = 0
            return distances

        q = deque()
        q.append(start)
        distances[start] = 0
        while q:
            u = q.popleft()
            for v in self.adjacency.get(u, ()):
                if v not in distances:
                    distances[v] = distances[u] + 1
                    q.append(v)
        return distances

    def __call__(self, node):
        state = node.state

        # 1) Goal check: heuristic must be 0 for goal states.
        if self.goals <= state:
            return 0

        # 2) find current lift floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break
        # if lift position not known, treat as isolated/inaccessible - but still proceed
        if lift_floor is None:
            # fallback: choose an arbitrary floor from static floors if available
            lift_floor = next(iter(self.floors), None)

        # 3) Determine required floors to visit and count board/depart actions
        required_floors = set()
        board_depart_count = 0

        # Use destin mapping to iterate over passengers; passengers are keys there.
        # If destin mapping is empty (unlikely), infer passengers from state facts.
        passengers = set(self.destin.keys())
        if not passengers:
            # attempt to collect passengers from state facts
            for fact in state:
                parts = get_parts(fact)
                pred = parts[0]
                if pred in ("origin", "destin", "boarded", "served"):
                    passengers.add(parts[1])

        for p in passengers:
            served_fact = f"(served {p})"
            if served_fact in state:
                continue  # already served

            boarded_fact = f"(boarded {p})"
            if boarded_fact in state:
                # Passenger is already boarded: need a depart at their destination
                board_depart_count += 1  # depart
                dest = self.destin.get(p)
                if dest:
                    required_floors.add(dest)
            else:
                # Not boarded: need a board and a depart -> include origin and destination floors
                board_depart_count += 2  # board + depart
                # origin is dynamic in the state
                origin = None
                origin_fact_prefix = f"(origin {p} "
                # find origin fact for this passenger in state
                for fact in state:
                    if fact.startswith(origin_fact_prefix):
                        # parse it
                        origin = get_parts(fact)[2]
                        break
                if origin:
                    required_floors.add(origin)
                # destination from static mapping (should exist)
                dest = self.destin.get(p)
                if dest:
                    required_floors.add(dest)

        # If there are no required floors (no remaining tasks), return board/depart count
        # (usually 0); but cover case where board_depart_count > 0 but no floors found: keep count.
        if not required_floors:
            return board_depart_count

        # 4) compute shortest-path distances between required floors and from lift_floor
        # We'll compute BFS distances from each node in required_floors and from lift_floor.
        nodes = list(required_floors)
        # Ensure lift_floor is included for computing distance to set
        bfs_sources = set(nodes)
        bfs_sources.add(lift_floor)

        distances_from = {}
        for src in bfs_sources:
            distances_from[src] = self._bfs_distances(src)

        # 5) Check reachability: if any required floor unreachable from any other required floor,
        # the instance is likely unsolvable; produce a large but finite estimate.
        # For practical use, we treat unreachable distances as a large penalty.
        LARGE_PENALTY = 100000

        # Build pairwise distance matrix for required nodes
        n = len(nodes)
        pairwise = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    pairwise[i][j] = 0
                else:
                    a = nodes[i]
                    b = nodes[j]
                    dist = distances_from.get(a, {}).get(b)
                    if dist is None:
                        # try symmetric BFS (should be same when adjacency undirected)
                        dist = distances_from.get(b, {}).get(a)
                    if dist is None:
                        pairwise[i][j] = LARGE_PENALTY
                    else:
                        pairwise[i][j] = dist

        # Distance from lift to closest required floor
        lift_to_nodes = []
        for i, a in enumerate(nodes):
            d = distances_from.get(lift_floor, {}).get(a)
            if d is None:
                lift_to_nodes.append(LARGE_PENALTY)
            else:
                lift_to_nodes.append(d)

        # 6) Estimate travel cost as MST over required nodes + distance from lift to closest required node
        # MST via Prim's algorithm (O(n^2) is fine for small n)
        mst_weight = 0
        visited = [False] * n
        # Start from node 0
        visited[0] = True
        num_visited = 1
        # min_edge_cost[j] = minimum cost to connect node j to the growing MST
        min_edge_cost = [pairwise[0][j] for j in range(n)]
        min_edge_cost[0] = 0

        while num_visited < n:
            # find unvisited node with minimal min_edge_cost
            best = None
            best_cost = None
            for j in range(n):
                if not visited[j]:
                    cost = min_edge_cost[j]
                    if best is None or cost < best_cost:
                        best = j
                        best_cost = cost
            if best is None:
                # disconnected: add large penalty and break
                mst_weight += LARGE_PENALTY
                break
            # add edge
            mst_weight += best_cost
            visited[best] = True
            num_visited += 1
            # update min_edge_cost
            for j in range(n):
                if not visited[j] and pairwise[best][j] < min_edge_cost[j]:
                    min_edge_cost[j] = pairwise[best][j]

        # distance from lift to closest node
        min_lift_dist = min(lift_to_nodes) if lift_to_nodes else 0

        if min_lift_dist >= LARGE_PENALTY or any(pairwise[i][j] >= LARGE_PENALTY for i in range(n) for j in range(n) if i != j):
            # If unreachable distances detected, return a large but finite number:
            return board_depart_count + mst_weight + min_lift_dist + LARGE_PENALTY

        travel_cost = mst_weight + min_lift_dist

        return board_depart_count + travel_cost