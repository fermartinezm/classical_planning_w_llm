from collections import defaultdict, deque
from math import inf
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the tokens of a fact string '(pred a b ...)' as a list ['pred', 'a', 'b', ...]."""
    return fact.strip()[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Miconic elevator domain.

    The heuristic estimates the number of actions required to serve all passengers
    (i.e., to make (served p) true for every passenger that is not yet served).
    It combines:
      - an estimate of the minimal number of elevator moves required to visit
        the relevant floors (origins and destinations of unserved passengers,
        and destinations of already-boarded but unserved passengers), and
      - the unavoidable boarding and departing actions for each unserved passenger.

    Assumptions
    -----------
    - There is no capacity limit for the elevator (the domain allows arbitrarily many
      boarded passengers).
    - The static "above" relation defines which floor-to-floor moves are allowed
      in one action (we treat it as undirected connectivity because both `up` and
      `down` operators allow moving along the "above" relation in either direction).
    - The static "destin" facts provide each passenger's destination floor.
    - "origin" facts are dynamic and are read from the current state for passengers
      not yet boarded.
    - If a passenger is already `boarded`, we only need to reach its destination
      and perform a `depart` action; if not boarded, we need both `board` and
      `depart` actions plus elevator movements to reach origin and destination floors.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
      - the destination floor for every passenger from static facts "(destin p f)",
      - the undirected adjacency graph of floors from static "(above f1 f2)" facts,
      - the list of all known floors (keys of the adjacency map).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the state already satisfies all goals (all passengers served), return 0.
    2. Identify the current lift floor by searching for "(lift-at ...)" in the state.
    3. For every passenger:
       - If already served: ignore.
       - Else if boarded: we must visit their destination floor and perform one depart action.
       - Else (not boarded): we must visit their origin floor (to board) and their destination floor (to depart),
         and perform one board and one depart action.
    4. Collect the set S of distinct floors that the elevator must visit (origins and/or destinations as above).
    5. Compute a lower bound on the number of elevator move actions to visit all floors in S starting
       from the current lift floor:
       - Build shortest-path distances between floors using BFS on the adjacency graph.
       - Compute an MST (minimum spanning tree) cost over the nodes in S using those distances;
         visiting all nodes requires at least the MST cost (connecting S internally).
       - Additionally, if the current lift floor is not in S, add the shortest distance from the lift floor
         to any floor in S (we must get the elevator into the set S at least once).
       - The sum (MST_cost + connection_from_start) is a cheap and informative lower bound on
         the number of move actions required to visit all required floors.
    6. Count the unavoidable primitive actions:
       - board actions = number of passengers that are not boarded and not served.
       - depart actions = number of unserved passengers (every unserved passenger needs one depart).
    7. The heuristic value is:
         moves_lower_bound + board_actions + depart_actions
       This value is finite (we handle unreachable distances by using a large finite penalty),
       equals 0 exactly for goal states, and is efficiently computable.

    Notes on admissibility:
    - The heuristic is not guaranteed admissible (it may overestimate), but it aims to be
      informative and cheap to compute to guide greedy best-first search.
    """

    def __init__(self, task):
        """
        Extract necessary static information from the task:
          - destin mapping passenger -> floor
          - adjacency graph of floors from 'above' facts (treated undirected)
          - set/list of floors known from static facts
        """
        self.goals = task.goals
        self.static = task.static

        # Map passenger -> destination floor from static 'destin' facts
        self.destin = {}
        # Build undirected adjacency between floors from 'above' facts
        self.adj = defaultdict(set)
        self.floors = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "destin" and len(parts) >= 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                self.floors.add(floor)
            elif pred == "above" and len(parts) >= 3:
                f1, f2 = parts[1], parts[2]
                # treat as undirected edge because both up and down allow moving along above
                self.adj[f1].add(f2)
                self.adj[f2].add(f1)
                self.floors.add(f1)
                self.floors.add(f2)

        # If adjacency is empty but floors exist in destin, ensure all floors appear as keys
        for f in list(self.floors):
            self.adj.setdefault(f, set())

    def _bfs_distances_from(self, start):
        """
        BFS to compute shortest distance (in elevator move actions) from start to every floor.
        Returns a dict floor -> distance (int). If a floor unreachable, distance is inf.
        """
        dist = {f: inf for f in self.floors}
        if start not in self.floors:
            # If start floor is unknown (should not happen), return all infinities
            return dist
        q = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if dist[v] == inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _all_pairs_distances(self, nodes):
        """
        Compute pairwise shortest-path distances for the subset 'nodes' (subset of floors)
        by running BFS from each node. Returns dict-of-dicts distances[a][b] = int or inf.
        """
        distances = {}
        for a in nodes:
            distances[a] = self._bfs_distances_from(a)
        return distances

    def _mst_cost(self, nodes, distances):
        """
        Compute the cost of a minimum spanning tree (Prim) over 'nodes' where edge weights
        are given by distances[u][v]. If any required distance is inf, treat it as a large cost.
        Returns a non-negative integer.
        """
        if not nodes:
            return 0
        nodes = list(nodes)
        visited = {nodes[0]}
        remaining = set(nodes[1:])
        total = 0
        LARGE = max(1, len(self.floors)) + 100  # fallback large penalty for unreachable links
        while remaining:
            best = None
            best_cost = inf
            for u in visited:
                for v in remaining:
                    d = distances.get(u, {}).get(v, inf)
                    if d == inf:
                        d = LARGE
                    if d < best_cost:
                        best_cost = d
                        best = v
            if best is None:
                # disconnected - add penalty and break
                total += LARGE
                break
            total += best_cost
            visited.add(best)
            remaining.remove(best)
        return int(total)

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Returns an integer estimate of the remaining number of actions to reach the goal.
        """
        state = node.state

        # 1) If all goals reached, heuristic is 0
        if self.goals <= state:
            return 0

        # 2) Find current lift floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) >= 2:
                lift_floor = parts[1]
                break

        # 3) Collect boarded, served, and origin facts from current state
        boarded = set()
        served = set()
        origins = {}  # passenger -> floor (dynamic origin facts)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "boarded" and len(parts) >= 2:
                boarded.add(parts[1])
            elif pred == "served" and len(parts) >= 2:
                served.add(parts[1])
            elif pred == "origin" and len(parts) >= 3:
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor

        # 4) Determine set S of floors that must be visited by the elevator
        required_floors = set()
        unserved_passengers = set()
        not_boarded_passengers = set()

        # Build list of all passengers we know about: union of destin keys and origin keys and boarded/served
        passengers = set(self.destin.keys()) | set(origins.keys()) | boarded | served

        for p in passengers:
            if p in served:
                continue  # already done
            unserved_passengers.add(p)
            if p in boarded:
                # Only destination needed to depart
                dest = self.destin.get(p)
                if dest:
                    required_floors.add(dest)
            else:
                # Need to board at origin and later depart at destination
                not_boarded_passengers.add(p)
                origin = origins.get(p)
                dest = self.destin.get(p)
                if origin:
                    required_floors.add(origin)
                if dest:
                    required_floors.add(dest)

        # If no floors required (but some passengers unserved), we still need to count their primitive actions.
        # However if required_floors empty and no knowledge of their floors in static or state, use a small fallback.
        if not required_floors:
            # Only primitive actions (board+depart or depart) remain but moves unknown -> assume 0 moves
            board_actions = len(not_boarded_passengers)
            depart_actions = len(unserved_passengers)
            return board_actions + depart_actions

        # 5) Compute pairwise distances among required floors and from lift floor
        # Ensure the adjacency has entries for all floors encountered (could include destin floors not in static above)
        for f in list(required_floors):
            if f not in self.adj:
                # isolated floor (no above facts); include it with empty adjacency to keep BFS robust
                self.adj.setdefault(f, set())
                self.floors.add(f)

        # distances between required floors
        distances = self._all_pairs_distances(required_floors)

        # MST cost connecting all required floors
        mst_cost = self._mst_cost(required_floors, distances)

        # cost to connect start (lift_floor) to the required-floor component
        if lift_floor is None:
            # unknown lift position: assume we must teleport into the component with a penalty of 1
            start_conn = 1
        elif lift_floor in required_floors:
            start_conn = 0
        else:
            # compute distances from lift_floor (might not be in self.floors yet)
            start_dists = self._bfs_distances_from(lift_floor) if lift_floor in self.floors else {f: inf for f in required_floors}
            # minimal distance from lift to any required floor
            start_conn = min((start_dists.get(f, inf) for f in required_floors))
            if start_conn == inf:
                # unreachable according to static 'above' graph; apply finite penalty
                start_conn = max(1, len(self.floors) // 2)

        moves_lower_bound = int(mst_cost + start_conn)

        # 6) Count primitive actions
        board_actions = len(not_boarded_passengers)  # one board per not-yet-boarded passenger
        depart_actions = len(unserved_passengers)    # one depart per unserved passenger

        total_estimate = moves_lower_bound + board_actions + depart_actions

        # Ensure we return a finite non-negative integer
        if total_estimate < 0 or total_estimate == inf:
            total_estimate = len(self.floors) + board_actions + depart_actions

        return int(total_estimate)