from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into its parts."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact matches a pattern with wildcards.
    Example: match("(origin p1 f2)", "origin", "*", "f2") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Miconic elevator domain.

    The heuristic estimates how many actions are still required to serve all
    passengers (i.e., make (served p) true for every passenger in the goal).
    It accounts for:
      - boarding and departing actions for each unserved passenger,
      - the elevator movement required to visit floors where passengers need
        to be picked up or dropped off.

    Assumptions
    -----------
    - There is no explicit capacity limit of the elevator in the provided
      domain file; therefore the elevator can carry arbitrarily many passengers
      simultaneously. Boarding and departing are counted per passenger (1 action
      each).
    - Floor adjacency is described by static (above ...) facts. We treat the
      adjacency as undirected for movement cost estimation: if the static facts
      contain (above a b) we consider a and b connected (the planner has both
      up and down actions, possibly via other above facts).
    - destins for passengers are static facts in the task (the examples show
      destin in static info); origins are dynamic (part of states) and disappear
      when passengers are boarded.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
      - extracts the set of passengers that must be served from the task goals
        (goals of the form "(served p)"),
      - extracts the mapping from passenger -> destination floor from the
        static facts (facts of the form "(destin p floor)"),
      - builds an undirected adjacency graph of floors from the static
        (above a b) facts (treated as bidirectional edges),
      - precomputes all-pairs shortest-path distances between all floors
        known from the static info to allow fast distance queries during
        heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a current state, the heuristic computes an estimate as follows:

    1. If all goals already hold in the state, return 0.

    2. Identify:
       - the elevator's current floor (the fact "(lift-at fX)"),
       - which passengers are already served (skip those),
       - which passengers are currently boarded (they need only a depart),
       - which passengers are unboarded (they need a board, then travel to
         their destination, then depart).

    3. Count mandatory per-passenger actions (cheap, exact part of the cost):
       - For every unserved passenger who is not boarded: one board action.
       - For every unserved passenger (whether boarded or not): one depart action.
       This yields `boarding_cost + depart_cost`.

    4. Movement cost (estimated):
       - Construct the set of floors that the elevator must visit (a lower-cost
         approximation): include the elevator's current floor, every origin
         floor for unboarded passengers (these are floors where a board must
         be executed), and every destination floor for passengers that are
         boarded or unboarded (these are floors where depart must be executed).
       - Use the precomputed floor-to-floor shortest-path distances (in
         movement actions) to compute a minimum spanning tree (MST) over this
         set of floors. The MST length is used as an optimistic yet informative
         estimate of how many move actions are required to connect these
         floors. The MST is fast to compute and captures grouping benefits:
         visiting floors in an order that serves multiple passengers is cheaper
         than visiting each floor independently.
       - The final heuristic is the sum:
            heuristic = boarding_cost + depart_cost + mst_move_cost

    5. Return the computed heuristic (an integer). The heuristic is finite for
       solvable states because the floor graph is finite; if some floor is
       unreachable according to static adjacency, a large finite distance value
       is used so the heuristic remains finite.

    Notes on design choices
    -----------------------
    - The boarding and departing costs are counted exactly (one action per
      passenger event), while movement is estimated by an MST on floors to
      visit. The MST provides a reasonably informed but inexpensive estimate
      of needed movement and rewards grouping pickups/drops in nearby floors.
    - Treating (above ...) as undirected edges yields conservative and simple
      connectivity assumptions for the elevator movement; it also keeps the
      distance computation robust for many domain encodings (as in the
      provided instances).
    """

    def __init__(self, task):
        # goals as strings, static as frozenset of strings
        self.goals = task.goals
        static_facts = task.static

        # Extract the set of passengers we must serve (from goal facts).
        self.goal_passengers = {
            get_parts(g)[1] for g in self.goals if get_parts(g)[0] == "served"
        }

        # Extract destination mapping for passengers from static facts (destin).
        # Example fact: "(destin p1 f10)"
        self.destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) >= 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

        # Build undirected adjacency between floors from (above f1 f2) facts.
        # We treat above as enabling movement between f1 and f2 in either direction
        # for the heuristic's distance computation.
        self.neighbors = {}
        floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                floors.add(a)
                floors.add(b)
                self.neighbors.setdefault(a, set()).add(b)
                self.neighbors.setdefault(b, set()).add(a)

        # Ensure every observed floor has an entry in neighbors (even if isolated).
        for f in list(floors):
            self.neighbors.setdefault(f, set())

        # Precompute all-pairs shortest path distances between floors known
        # from static info. If a floor appears in destin but not in above facts,
        # ensure it's included.
        for p_floor in self.destin.values():
            self.neighbors.setdefault(p_floor, set())

        # Compute BFS distances from every known floor
        self.all_pairs_dist = {}
        for floor in self.neighbors:
            self.all_pairs_dist[floor] = self._bfs_distances_from(floor)

        # A fallback large distance for unreachable pairs (finite).
        self.large_distance = max(1, len(self.neighbors)) + 10

    def _bfs_distances_from(self, start):
        """Return a dict mapping every floor to its shortest distance (in moves) from start."""
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nbr in self.neighbors.get(cur, ()):
                if nbr not in dist:
                    dist[nbr] = dist[cur] + 1
                    q.append(nbr)
        return dist

    def _floor_distance(self, a, b):
        """Return the shortest move distance between floors a and b using precomputed distances."""
        if a == b:
            return 0
        if a in self.all_pairs_dist and b in self.all_pairs_dist[a]:
            return self.all_pairs_dist[a][b]
        # If we don't know one of the floors or it's unreachable, return a finite large value
        return self.large_distance

    def __call__(self, node):
        state = node.state

        # Fast goal check
        if self.goals <= state:
            return 0

        # Find elevator current floor: fact "(lift-at fX)"
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            # Unusual state, but return a finite conservative estimate.
            lift_floor = next(iter(self.neighbors), None)
            if lift_floor is None:
                return 0  # no floor info anywhere

        # Classify passengers: served / boarded / unboarded (and unserved)
        served = {get_parts(f)[1] for f in state if match(f, "served", "*")}
        boarded = {get_parts(f)[1] for f in state if match(f, "boarded", "*")}

        # Passengers we care about (from goals) but not yet served
        unserved_passengers = [p for p in self.goal_passengers if p not in served]

        # Count exact per-passenger action needs:
        boarding_cost = 0  # board actions required now (passenger not boarded yet)
        depart_cost = 0  # depart actions required for each unserved passenger

        required_floors = set()
        required_floors.add(lift_floor)

        for p in unserved_passengers:
            depart_cost += 1  # every unserved passenger will need a depart action

            if p in boarded:
                # Passenger already boarded: only needs to get to destination and depart.
                dest = self.destin.get(p)
                if dest:
                    required_floors.add(dest)
            else:
                # Not boarded: need a board at origin (origin fact should be in state)
                boarding_cost += 1
                # origin fact is dynamic; find it in the current state
                origin_floor = None
                for f in state:
                    if match(f, "origin", p, "*"):
                        origin_floor = get_parts(f)[2]
                        break
                if origin_floor:
                    required_floors.add(origin_floor)
                dest = self.destin.get(p)
                if dest:
                    required_floors.add(dest)

        # If there are no floors to visit beyond the lift floor, movement cost is zero
        if len(required_floors) <= 1:
            move_cost = 0
        else:
            # Build a complete graph among required_floors with weights = shortest distances
            nodes = list(required_floors)
            # Compute MST using Prim's algorithm (dense but small set)
            # Initialize
            in_mst = {nodes[0]}
            not_in_mst = set(nodes[1:])
            # Precompute distance lookup
            def d(x, y):
                return self._floor_distance(x, y)

            move_cost = 0
            # Greedily add closest node until all are in MST
            while not_in_mst:
                best_edge = None
                best_cost = None
                for u in in_mst:
                    for v in not_in_mst:
                        cost = d(u, v)
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_edge = v
                # Add best_edge to MST
                if best_edge is None:
                    # Disconnected; treat remaining nodes as costing large finite steps
                    move_cost += len(not_in_mst) * self.large_distance
                    break
                move_cost += best_cost
                in_mst.add(best_edge)
                not_in_mst.remove(best_edge)

        total_cost = boarding_cost + depart_cost + move_cost
        # Ensure returned value is integer
        return int(total_cost)