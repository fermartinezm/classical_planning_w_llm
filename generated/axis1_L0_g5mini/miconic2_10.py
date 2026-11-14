from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "miconic" elevator domain.

    The heuristic estimates the number of actions required to reach a goal
    (all passengers served). It combines:
      - Per-passenger manipulation costs (board and depart actions),
      - An estimate of the number of elevator move actions required to visit
        the relevant floors (origins and destinations) starting from the
        current elevator position.

    Assumptions
    -----------
    - The static facts include (destin P F) for each passenger and (above F1 F2)
      facts describing connectivity between floors.
    - The elevator has unbounded capacity (consistent with the provided domain).
    - The planner represents facts as strings like "(predicate arg1 arg2)".
    - If a passenger is already (served P) they require no further actions.
    - If a passenger has (boarded P) in the current state, only a (depart P)
      action is required (plus movement to their destination).
    - Origins for passengers not boarded are expected to appear in the current
      state as (origin P F). (This is consistent with the miconic domain.)
    - The above relation is treated as an undirected adjacency relation for
      computing shortest paths (each (above a b) implies a direct move
      between a and b in either direction for the purpose of movement cost).

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - self.goals: the goal facts (used to return 0 on goal states)
      - self.destin_by_passenger: mapping passenger -> destination floor from
        static facts (destin facts are static in the domain).
      - self.floor_adj: adjacency mapping of floors (derived from above facts).
      - self.floors: the set of known floor names (collected from static facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the goals are already satisfied return 0.
    2. Identify the elevator current floor from a fact "(lift-at F)".
    3. For every passenger in self.destin_by_passenger:
       - If "(served P)" is true in the current state skip this passenger.
       - If "(boarded P)" is true, the passenger requires:
           * 1 depart action + visiting their destination floor.
       - Otherwise (not boarded and not served), find "(origin P Forigin)"
         in the state and treat the passenger as requiring:
           * 1 board action at Forigin, 1 depart action at Fdest.
         The origin and destination floors are both required to be visited.
    4. Build the set S of floors that must be visited: includes current lift floor
       and all origins/destinations needed by unserved passengers.
    5. Compute shortest-path distances between floors using BFS on self.floor_adj.
       If the domain graph is disconnected in an unusual instance, unreachable
       distances are replaced by a finite large value to keep the heuristic finite.
    6. Estimate the elevator movement cost as the total weight of a minimum
       spanning tree (MST) over the nodes in S using the computed distances.
       The MST gives a compact connectivity-based estimate of how many move
       actions are needed to connect all required floors starting from the
       current elevator position. (This is efficiently computable and
       informative; it may be inadmissible but guides greedy search well.)
    7. The final heuristic value is:
           movement_estimate (MST length)
           + sum_{passengers not served} (1 if boarded else 2)
       where 1 corresponds to depart and 2 corresponds to board+depart.

    Rationale
    ---------
    - Counting board/depart per passenger captures unavoidable manipulation
      actions.
    - The MST-based movement estimate captures the minimal connections among
      required floors and the elevator start; it is a cheap, informative metric
      for movement cost without solving a full routing problem.
    """

    def __init__(self, task):
        # Goal facts (set of strings)
        self.goals = task.goals

        # Extract destin facts from static information: passenger -> destination floor
        self.destin_by_passenger = {}
        self.floor_adj = {}  # adjacency list of floors (undirected)
        self.floors = set()

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "destin":
                # fact format: (destin pX fY)
                passenger, floor = parts[1], parts[2]
                self.destin_by_passenger[passenger] = floor
                self.floors.add(floor)
            elif pred == "above":
                # fact format: (above fA fB)
                f1, f2 = parts[1], parts[2]
                self.floors.add(f1)
                self.floors.add(f2)
                # treat above as undirected adjacency for movement estimates
                self.floor_adj.setdefault(f1, set()).add(f2)
                self.floor_adj.setdefault(f2, set()).add(f1)

        # Ensure every known floor has an entry in adjacency dict
        for f in list(self.floors):
            self.floor_adj.setdefault(f, set())

    def bfs_distances_from(self, start):
        """
        Compute shortest distances (number of moves) from start floor to all
        floors reachable in self.floor_adj using BFS.
        Returns a dict floor->distance. Floors not reachable are absent.
        """
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in self.floor_adj.get(cur, ()):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return dist

    def mst_length_on_metric(self, nodes, distance_func):
        """
        Compute MST length (sum of edge weights) on the complete graph induced by
        'nodes' using distances provided by distance_func(u, v).
        Uses Prim's algorithm. If a distance between nodes is missing (unreachable),
        uses a finite large fallback to keep the heuristic finite.
        """
        if not nodes:
            return 0
        nodes = list(nodes)
        # Fallback large distance for unreachable pairs
        LARGE = max(1, len(self.floors) + 5)

        in_tree = {nodes[0]}
        # min distance from tree to each outside node
        min_edge = {}
        for v in nodes[1:]:
            d = distance_func(nodes[0], v)
            min_edge[v] = d if d is not None else LARGE

        total = 0
        while len(in_tree) < len(nodes):
            # pick outside node with smallest connecting edge
            v_pick = min(min_edge, key=min_edge.get)
            w = min_edge[v_pick]
            total += w
            in_tree.add(v_pick)
            del min_edge[v_pick]
            # update distances
            for v in list(min_edge.keys()):
                d = distance_func(v_pick, v)
                d = d if d is not None else LARGE
                if d < min_edge[v]:
                    min_edge[v] = d
        return total

    def __call__(self, node):
        state = node.state

        # 1) If already goal, return 0
        if self.goals <= state:
            return 0

        # 2) find current lift floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                lift_floor = parts[1]
                break

        # If lift position not found (unexpected), put a fallback to any known floor
        if lift_floor is None:
            lift_floor = next(iter(self.floors)) if self.floors else None

        # 3) For each passenger, determine status and required floors and manipulation costs
        required_floors = set()
        if lift_floor is not None:
            required_floors.add(lift_floor)

        manipulation_cost = 0  # sum of board/depart actions needed

        for passenger, dest_floor in self.destin_by_passenger.items():
            # Skip already served passengers
            if f"(served {passenger})" in state:
                continue

            # If boarded, only depart is needed (1 action) and destination must be visited
            if f"(boarded {passenger})" in state:
                manipulation_cost += 1  # depart
                required_floors.add(dest_floor)
                continue

            # Not boarded and not served: need to board at origin and depart at destination
            # Find origin in current state
            origin_floor = None
            # origin facts appear as "(origin p f)"
            origin_fact_prefix = f"(origin {passenger} "
            for fact in state:
                if fact.startswith(origin_fact_prefix):
                    # parse fact to get floor
                    parts = get_parts(fact)
                    if len(parts) >= 3:
                        origin_floor = parts[2]
                    break

            # If origin not found (unexpected), be conservative: include destination only
            if origin_floor is None:
                # We still count board+depart (2), but we don't know the origin to add
                manipulation_cost += 2
                required_floors.add(dest_floor)
            else:
                manipulation_cost += 2  # board + depart
                required_floors.add(origin_floor)
                required_floors.add(dest_floor)

        # If no floors required (e.g., all served), return manipulation cost (should be zero)
        if not required_floors:
            return manipulation_cost

        # 4) Prepare a cached distance function between floors using BFS from nodes in required_floors
        # Precompute BFS distances from each required floor
        bfs_cache = {}
        for f in required_floors:
            if f in self.floor_adj:
                bfs_cache[f] = self.bfs_distances_from(f)
            else:
                # floor unknown in adjacency: no reachable nodes
                bfs_cache[f] = {}

        def distance(u, v):
            if u == v:
                return 0
            dmap = bfs_cache.get(u)
            if dmap is None:
                return None
            return dmap.get(v, None)

        # 5) Compute MST length on required_floors using distances
        movement_estimate = self.mst_length_on_metric(required_floors, distance)

        # Final heuristic: movement_estimate + manipulation_cost
        return movement_estimate + manipulation_cost