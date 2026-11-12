from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding
    parentheses and splitting the content.

    Examples:
      "(lift-at f2)" -> ["lift-at", "f2"]
      "(origin p1 f3)" -> ["origin", "p1", "f3"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Miconic elevator domain. The heuristic
    estimates the number of actions (moves, board, depart) required to serve
    all passengers that are not yet served in the current state. It is designed
    to be informative (to reduce node expansions) while remaining efficiently
    computable.

    Assumptions
    -----------
    - The elevator can carry an arbitrary number of passengers (no capacity limit
      in the provided domain).
    - Boarding and departing are separate actions:
      * board: requires elevator at the passenger's origin and performs one action
      * depart: requires elevator at passenger's destination and performs one action
    - Moving the elevator between floors uses the static "above" facts to define
      connectivity. An (above f1 f2) fact is interpreted as creating a (unit-cost)
      connection between f1 and f2 (we treat this as an undirected connection
      for movement, because the domain provides both up and down actions).
    - The static "destin" facts are taken as immutable destination mappings.
    - Origin facts for unboarded passengers are read from the current state.
    - If a passenger is already 'boarded' in the state, there is no origin fact
      for them; they only need to be transported to their destin floor and departed.

    Heuristic Initialization
    ------------------------
    In the constructor the following static information is precomputed from the
    task.static facts:
    - destin_map: passenger -> destination floor (from "(destin p f)" facts).
    - adjacency: undirected adjacency list of floors derived from "(above f1 f2)"
      facts. Each above pair introduces edges between f1 and f2 with unit cost.
    This enables fast shortest-path queries between floors during heuristic
    computation.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state the heuristic computation follows these steps:

    1. Identify unserved passengers:
       - A passenger is "unserved" if the goal fact "(served p)" is not present
         in the current state.

    2. For each unserved passenger determine:
       - If they are already boarded (state contains "(boarded p)"):
         * They require exactly one depart action at their destination.
         * The destination floor must be visited by the elevator.
       - If they are not boarded:
         * They require one board action at their origin and one depart action at
           their destination.
         * Both the origin and destination floors must be visited.

    3. Count the exact number of board and depart actions still required:
       - boards = number of unserved passengers that are currently not boarded.
       - departs = number of unserved passengers (each needs one depart).

    4. Movement estimate:
       - Let F be the set of distinct floors that must be visited (origins of
         unboarded passengers and destinations of unserved passengers).
       - Include the current elevator floor (from "(lift-at f)") as the starting
         point.
       - Compute shortest-path distances (in number of elevator move actions)
         between floors using BFS on the static adjacency graph.
       - Use the Minimum Spanning Tree (MST) weight (Prim's algorithm) on the
         metric induced by pairwise shortest-path distances between the nodes in
         F U {current_floor} as a lower-bound estimate of the number of move
         actions required to visit all required floors starting from the current
         elevator floor. The MST is an admissible lower bound for a route that
         visits all nodes and is cheap to compute.

    5. Final heuristic value:
       heuristic = MST_weight + boards + departs

    Notes
    -----
    - The heuristic returns 0 exactly when there are no unserved passengers (i.e.,
      when all goals are satisfied).
    - If the elevator/floors are disconnected (which would make the instance
      unsolvable), unreachable distances are treated as very large to reflect
      the difficulty; in normal Miconic instances the floor graph is connected.
    - The heuristic is not guaranteed to be admissible, but aims to be informative
      and cheap to compute (polynomial in the number of relevant floors).
    """

    def __init__(self, task):
        # Store goal facts (though we only need to check served facts in state).
        self.goals = task.goals
        static_facts = task.static

        # destin_map: passenger -> destination floor (from static facts).
        self.destin_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                _, passenger, floor = parts
                self.destin_map[passenger] = floor

        # Build an undirected adjacency graph of floors from "above" static facts.
        # We treat each above relation (above f1 f2) as an undirected unit-cost
        # connection between f1 and f2 to allow both up and down moves.
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                _, f1, f2 = parts
                self.adjacency.setdefault(f1, set()).add(f2)
                self.adjacency.setdefault(f2, set()).add(f1)

        # Precompute list of known floors from adjacency (fallbacks will be added).
        self.known_floors = set(self.adjacency.keys())

        # A large cost to represent unreachable floors (shouldn't happen in valid instances).
        self.INF = 10 ** 6

    def _bfs_distance(self, start, goal):
        """
        Compute shortest number of elevator move actions between two floors
        using BFS on the adjacency graph produced from "above" facts.
        If either floor is unknown or goal unreachable, returns self.INF.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return self.INF
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in self.adjacency.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return self.INF

    def __call__(self, node):
        state = node.state

        # Utility to check membership of a fact pattern in the state.
        def fact_in_state(predicate, *args):
            """
            Return True if a fact with given predicate and exact args exists in the state.
            Example: fact_in_state("lift-at", "f2") checks "(lift-at f2)" in the state.
            """
            fact = "(" + " ".join((predicate,) + args) + ")"
            return fact in state

        # Determine current elevator floor (must exist).
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                lift_floor = parts[1]
                break

        # If no lift-at fact, we cannot reason about movement; return large value.
        if lift_floor is None:
            return self.INF

        # Collect set of passengers to consider: use destin_map keys as passenger list.
        passengers = list(self.destin_map.keys())

        # Identify which passengers are already served in the current state.
        unserved = []
        for p in passengers:
            served_fact = "(served " + p + ")"
            if served_fact not in state:
                unserved.append(p)

        # If no unserved passengers, we are at the goal.
        if not unserved:
            return 0

        # Count required boards and departs, and collect required floors to visit.
        boards_needed = 0
        departs_needed = 0  # equals number of unserved passengers
        required_floors = set()

        for p in unserved:
            departs_needed += 1
            boarded_fact = "(boarded " + p + ")"
            if boarded_fact in state:
                # Passenger is already on the elevator: only destination floor needed.
                dest = self.destin_map.get(p)
                if dest is not None:
                    required_floors.add(dest)
            else:
                # Passenger not boarded: need to visit origin (from state) and destination.
                boards_needed += 1
                # Find origin in the current state: "(origin p f)".
                origin_floor = None
                origin_fact_prefix = "(origin " + p + " "
                for fact in state:
                    if fact.startswith(origin_fact_prefix) and fact.endswith(")"):
                        parts = get_parts(fact)
                        if len(parts) == 3:
                            origin_floor = parts[2]
                        break
                # If origin not present (unexpected), ignore; but prefer to include dest.
                if origin_floor is not None:
                    required_floors.add(origin_floor)
                    # Ensure origin is known in adjacency (add isolated node if needed).
                    if origin_floor not in self.adjacency:
                        self.adjacency.setdefault(origin_floor, set())
                dest = self.destin_map.get(p)
                if dest is not None:
                    required_floors.add(dest)
                    if dest not in self.adjacency:
                        self.adjacency.setdefault(dest, set())

        # Ensure lift floor is known in adjacency (so distances computed from it).
        if lift_floor not in self.adjacency:
            self.adjacency.setdefault(lift_floor, set())

        # If there are no floors to visit (shouldn't happen because unserved nonempty),
        # then movement cost is 0 and heuristic is simply the action counts.
        if not required_floors:
            return boards_needed + departs_needed

        # Build list of nodes for which to compute pairwise distances: required floors
        # plus the current lift floor (if not already in the set).
        nodes = list(required_floors)
        if lift_floor not in required_floors:
            nodes.append(lift_floor)

        # Compute pairwise shortest distances between these nodes.
        # For efficiency we BFS from each node in nodes once.
        dist_matrix = {n: {} for n in nodes}
        for n in nodes:
            for m in nodes:
                if n == m:
                    dist_matrix[n][m] = 0
                else:
                    dist_matrix[n][m] = self._bfs_distance(n, m)

        # If any required floor is unreachable from the lift, return a large value
        # rather than infinity (keeps heuristic finite).
        for floor in required_floors:
            if dist_matrix[lift_floor].get(floor, self.INF) >= self.INF:
                return self.INF // 2

        # Compute MST weight over the nodes using Prim's algorithm.
        # Start at the lift floor to bias the tree toward the start.
        mst_weight = 0
        visited = set([lift_floor])
        remaining = set(nodes)
        remaining.remove(lift_floor)

        # Precompute helper to get distance with safety fallback.
        def d(a, b):
            return dist_matrix.get(a, {}).get(b, self.INF)

        # Prim's algorithm
        while remaining:
            best_edge = None
            best_cost = self.INF
            for u in visited:
                for v in remaining:
                    cost = d(u, v)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = v
            if best_edge is None or best_cost >= self.INF:
                # Graph is disconnected; return large (but finite) heuristic
                return self.INF // 2
            mst_weight += best_cost
            visited.add(best_edge)
            remaining.remove(best_edge)

        # Final estimated cost: movement (mst_weight) + required board/depart actions.
        heuristic_value = mst_weight + boards_needed + departs_needed
        # Ensure heuristic is integer and finite.
        if heuristic_value >= self.INF:
            return self.INF // 2
        return int(heuristic_value)