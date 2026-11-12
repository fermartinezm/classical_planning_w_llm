from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its components."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent, efficiently computable heuristic for the (transport) domain.
    It estimates the number of actions (drive / pick-up / drop) required to move
    all packages from their current locations to their goal locations.

    Assumptions
    -----------
    - Goals are only of the form (at <package> <location>).
    - Driving along a single 'road' fact counts as one drive action.
    - Picking up a package always costs 1 action; dropping a package costs 1 action.
    - Vehicles have discrete capacity levels encoded by capacity facts and
      capacity-predecessor facts. We compute a numeric "available slots" value
      (rank) for each capacity symbol: minimal capacity (with no predecessor) has rank 0,
      the next has rank 1, etc. The rank corresponds to how many packages the vehicle
      can still pick up.
    - For efficiency the heuristic treats each package independently and sums
      per-package estimated costs (i.e., it ignores opportunities to consolidate
      multiple packages into the same trip). This makes the heuristic inadmissible
      in general but keeps computation cheap and informative.
    - If there is no path between two locations according to the static 'road'
      graph, a large finite penalty is used instead of infinity.

    Heuristic Initialization
    ------------------------
    - Extracts the directed road graph from static facts and precomputes BFS-based
      shortest-path distances between all locations present in the static graph.
    - Extracts capacity-predecessor relations and constructs a rank map for capacity
      symbols so that we can compute how many "slots" a capacity symbol represents.
    - Extracts the goal location for each package from task.goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Parse the current state to obtain:
       - vehicle capacities (capacity ?v ?s)
       - vehicle current locations (at ?v ?loc)
       - packages that are inside vehicles (in ?p ?v)
       - packages that are on the ground (at ?p ?loc)
    3. For each package that is not yet at its goal:
       a) If the package is inside a vehicle v:
          - Estimated cost = shortest-path distance(vehicle_location, goal_location)
            + 1 (drop action).
       b) If the package is on the ground at location L:
          - For each vehicle v, estimate the cost to use v to move the package:
            cost_v = dist(vehicle_loc, L) + 1 (pick) + dist(L, goal_loc) + 1 (drop).
            If v currently has zero available slots (rank 0), add a small penalty
            to reflect that freeing a slot will require additional actions; this
            keeps the heuristic finite and still informative.
          - The package's estimated cost is the minimum cost_v across vehicles.
          - If no vehicles exist in the state, a large finite cost is used to indicate
            the package cannot be transported with current resources.
    4. Sum the per-package costs to obtain the final heuristic value.
    """

    LARGE_DISTANCE = 1000  # Penalty for unreachable paths or absent vehicles
    SLOT_PENALTY = 2       # Penalty when vehicle currently has zero free slots

    def __init__(self, task):
        """Extract static information and precompute data structures.
        - self.goals: set of goal facts (strings)
        - self.road_graph: dict mapping location -> list of successor locations
        - self.distances: dict mapping location -> dict(location -> distance) computed by BFS
        - self.capacity_pred: map s2 -> s1 for capacity-predecessor (s1 predecessor of s2)
        - self.capacity_rank: map capacity_symbol -> integer rank (0 = minimal)
        """
        self.goals = task.goals
        static_facts = task.static

        # Build road graph (directed) from static 'road' facts.
        self.road_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                _, a, b = parts
                self.road_graph.setdefault(a, []).append(b)
                # ensure nodes exist
                self.road_graph.setdefault(b, [])

        # Precompute BFS distances for all locations in the road graph.
        self.distances = {}
        for src in self.road_graph:
            # BFS from src
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.road_graph.get(u, []):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self.distances[src] = dist

        # Parse capacity-predecessor relations to compute capacity ranks.
        # capacity-predecessor s1 s2  => s1 is predecessor of s2 (s2 has rank one greater than s1)
        self.capacity_pred = {}
        all_capacity_symbols = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "capacity-predecessor":
                _, s1, s2 = parts
                self.capacity_pred[s2] = s1
                all_capacity_symbols.add(s1)
                all_capacity_symbols.add(s2)

        # Compute rank for every capacity symbol appearing in predecessor facts.
        # Rank is number of steps from minimal symbol (which has no predecessor).
        self.capacity_rank = {}

        def compute_rank(s):
            # Follow predecessors until none
            if s in self.capacity_rank:
                return self.capacity_rank[s]
            rank = 0
            cur = s
            visited = set()
            while cur in self.capacity_pred and cur not in visited:
                visited.add(cur)
                cur = self.capacity_pred[cur]
                rank += 1
            # If predecessor chain had a cycle (unexpected), cap rank to avoid infinite loop.
            self.capacity_rank[s] = rank
            return rank

        for s in list(all_capacity_symbols):
            compute_rank(s)

        # Extract goal mapping for packages: package -> goal_location
        self.goal_for_package = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) >= 3:
                _, obj, loc = parts
                self.goal_for_package[obj] = loc

    def _distance(self, a, b):
        """Return shortest-path distance (number of drives) between locations a and b.
        If unknown or unreachable according to the static road graph, return LARGE_DISTANCE.
        """
        if a == b:
            return 0
        if a in self.distances and b in self.distances[a]:
            return self.distances[a][b]
        # If a not in precomputed graph but b exists, try BFS from a on the fly using static graph topology.
        # Build neighbors for BFS using self.road_graph (will treat unknown nodes as isolated).
        # This on-the-fly BFS is cheap for occasional missing nodes.
        visited = {a}
        q = deque([(a, 0)])
        while q:
            node, d = q.popleft()
            if node == b:
                return d
            for succ in self.road_graph.get(node, []):
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, d + 1))
        return self.LARGE_DISTANCE

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node (state)."""
        state = node.state

        # If all goals are already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current state to extract capacities, vehicle locations, package locations and 'in' relations.
        vehicle_capacity_symbol = {}  # v -> capacity_sym
        vehicle_slots = {}            # v -> integer available slots (rank)
        vehicle_location = {}         # v -> location
        package_in_vehicle = {}       # p -> v
        package_on_ground = {}       # p -> loc

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "capacity" and len(parts) >= 3:
                _, v, s = parts
                vehicle_capacity_symbol[v] = s
            elif pred == "at" and len(parts) >= 3:
                _, obj, loc = parts
                # If object has capacity facts somewhere, treat as vehicle; otherwise as package/location fact.
                # We'll classify vehicles using vehicle_capacity_symbol (collected above and possibly below),
                # but initial ordering may mean capacity facts appear later. So collect 'at' facts regardless:
                # temporarily store all at facts; we'll decide after collecting capacities.
                # To keep code simple, store all at facts and filter below.
                # Use package_on_ground for now; revise after reading all facts.
                # We'll just store them now; if obj later determined to be vehicle, we'll move it.
                package_on_ground[obj] = loc
            elif pred == "in" and len(parts) >= 3:
                _, p, v = parts
                package_in_vehicle[p] = v

        # Now separate vehicles from packages using known capacity facts:
        # Vehicle objects are keys in vehicle_capacity_symbol. But it's possible that capacity facts
        # are not present for some vehicles (unlikely). For safety, treat any object that appears
        # in package_in_vehicle values as vehicle as well.
        # Build vehicle_location map using package_on_ground entries where object in vehicles set.
        vehicles = set(vehicle_capacity_symbol.keys()) | set(package_in_vehicle.values())
        for obj, loc in list(package_on_ground.items()):
            if obj in vehicles:
                vehicle_location[obj] = loc
                # remove from package_on_ground since it's a vehicle location fact
                del package_on_ground[obj]

        # There may be vehicles with capacity facts but no explicit 'at' fact in the state.
        # Those vehicles are considered to have unknown location; we treat that as large distance.
        for v in vehicles:
            if v not in vehicle_location:
                vehicle_location[v] = None

        # Compute numeric slots for vehicles using capacity_rank mapping.
        for v, cap_sym in vehicle_capacity_symbol.items():
            vehicle_slots[v] = self.capacity_rank.get(cap_sym, 0)

        # Helper to get a location for an object (package or vehicle). If a package is in a vehicle,
        # its location is the vehicle's current location. If a vehicle location is None (unknown),
        # distance computations will use LARGE_DISTANCE.
        def loc_of_package(p):
            if p in package_in_vehicle:
                v = package_in_vehicle[p]
                return vehicle_location.get(v, None), True, v
            return package_on_ground.get(p, None), False, None

        total_cost = 0

        # For every package with a goal, if not already at the goal compute an estimated cost.
        for p, goal_loc in self.goal_for_package.items():
            # Check if package already at goal in the current state.
            goal_fact = f"(at {p} {goal_loc})"
            if goal_fact in state:
                continue  # cost 0 for this package

            cur_loc, is_in_vehicle, containing_vehicle = loc_of_package(p)

            # If we cannot determine the package location at all, assign a large cost.
            if cur_loc is None:
                total_cost += self.LARGE_DISTANCE
                continue

            if is_in_vehicle:
                # Package is already inside a vehicle v. Need to drive v to goal and drop.
                v = containing_vehicle
                v_loc = vehicle_location.get(v, None)
                if v_loc is None:
                    # unknown vehicle location -> large cost
                    total_cost += self.LARGE_DISTANCE
                else:
                    d = self._distance(v_loc, goal_loc)
                    total_cost += d + 1  # drive(s) + drop
            else:
                # Package on ground at cur_loc. Choose best vehicle to perform pick & drop.
                best_cost = self.LARGE_DISTANCE
                if vehicles:
                    for v in vehicles:
                        v_loc = vehicle_location.get(v, None)
                        # If vehicle has unknown location, treat as far away.
                        if v_loc is None:
                            d1 = self.LARGE_DISTANCE
                        else:
                            d1 = self._distance(v_loc, cur_loc)
                        d2 = self._distance(cur_loc, goal_loc)
                        base = d1 + 1 + d2 + 1  # drive to package + pick + drive to goal + drop
                        # If vehicle currently has zero free slots, add a small penalty.
                        slots = vehicle_slots.get(v, 0)
                        if slots <= 0:
                            base += self.SLOT_PENALTY
                        if base < best_cost:
                            best_cost = base
                else:
                    # No vehicles available in state -> very large cost.
                    best_cost = self.LARGE_DISTANCE

                total_cost += best_cost

        # Return integer heuristic value.
        return int(total_cost)