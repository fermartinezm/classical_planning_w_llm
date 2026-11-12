from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hrpmxpjg) domain.

    This heuristic estimates the number of actions required to achieve all
    goal facts of the form (sjuejzxf <object> <location>) by combining:
      - shortest-path distances in the x6ibpitl-location graph (miud1izi edges)
        which measure how many local move actions (arsslvyt) are needed when
        an object is transported directly while on the ground, and
      - simple vehicle-assisted estimates that model bringing a vehicle to an
        object, loading it (szsevy61), driving the vehicle while carrying it
        (arsslvyt on the vehicle), and unloading (tmoxe3x5).

    The heuristic sums per-goal estimates (one estimate per goal object).
    It deliberately ignores complex interactions (e.g., one vehicle moving
    multiple objects in the same trip) and higher-level hub movements
    (hab0vli4 + do9fmoyo effects). The goal is a fast, informative estimate
    for greedy best-first search rather than an admissible lower-bound.

    Assumptions
    -----------
    - Facts are represented as strings like "(predicate arg1 arg2)".
    - Locations of type x6ibpitl are connected by miud1izi facts (interpreted as
      undirected edges).
    - Vehicles (type hctyyy8k) appear in do9fmoyo facts (mapping them to a
      kyvrixb5). Vehicles' x6ibpitl positions are given by sjuejzxf facts
      (same predicate used for other objects). An entity that appears in
      do9fmoyo is treated as a vehicle.
    - Loading (szsevy61) and unloading (tmoxe3x5) both cost 1 action each in
      this estimate. Each local movement along a miud1izi edge (arsslvyt)
      costs 1 action.
    - The heuristic ignores hub-level movement effects (hab0vli4 / do9fmoyo
      adjacency moves) when estimating costs, except that vehicles can be
      identified by the presence of do9fmoyo facts.

    Heuristic Initialization
    ------------------------
    In the constructor this heuristic:
    - Extracts the miud1izi edges from the static facts and builds an
      undirected adjacency graph of x6ibpitl locations.
    - Prepares a small cache for shortest-path queries between location pairs.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If all goal facts are already satisfied in the state, return 0.
    2. Parse the state to extract:
       - Ground-location map: which entities (objects and vehicles) are at
         which x6ibpitl (from sjuejzxf facts).
       - Onboard map: which du66z5od objects are currently onboard which
         vehicle (from un606moq facts).
       - Vehicle set and their current x6ibpitl positions (vehicles are the
         subjects appearing in do9fmoyo facts; their x6ibpitl position is
         obtained from the sjuejzxf facts).
    3. For every goal of the form (sjuejzxf OBJ TARGET_LOC) that is not yet
       satisfied compute a per-object estimate:
       a) If OBJ is on the ground at CURRENT_LOC:
          - Ground-walk cost = shortest-path length (in miud1izi graph)
            between CURRENT_LOC and TARGET_LOC. Each step models one
            arsslvyt action.
          - Vehicle-assisted cost (if vehicles exist):
            For each vehicle V (with known x position VEH_LOC):
              cost_v = shortest_distance(VEH_LOC, CURRENT_LOC)     # bring vehicle
                       + 1                                         # load (szsevy61)
                       + shortest_distance(CURRENT_LOC, TARGET_LOC) # carry by vehicle
                       + 1                                         # unload (tmoxe3x5)
            Take the minimum over vehicles (if none known, skip).
          - Use min(ground-walk cost, best vehicle-assisted cost).
       b) If OBJ is already onboard vehicle V:
          - cost = shortest_distance(vehicle_x_loc, TARGET_LOC) + 1
            (drive vehicle carrying the object using arsslvyt steps, then unload).
       c) If an entity/locations are missing from the state or no path exists,
          a conservative finite penalty is used (bounded by number of known
          locations + a small constant) to keep values finite.
    4. Sum the per-goal estimates to form the final heuristic value.
    """

    def __init__(self, task):
        """Extract static miud1izi adjacency information and prepare caches."""
        self.goals = task.goals  # set of goal fact strings
        static_facts = task.static  # static facts (frozenset of strings)

        # Build undirected adjacency graph between x6ibpitl locations using miud1izi facts.
        self.loc_graph = {}  # loc -> set(neighbor_locs)
        self._locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self._locations.add(a)
                self._locations.add(b)
                self.loc_graph.setdefault(a, set()).add(b)
                self.loc_graph.setdefault(b, set()).add(a)

        # If no miud1izi facts present, ensure structures are defined.
        for loc in list(self._locations):
            self.loc_graph.setdefault(loc, set())

        # Cache for shortest path distances between location pairs.
        # Key: (a,b) tuple (strings) -> integer distance
        self._dist_cache = {}

        # A conservative large distance to use when no path is found.
        self._large_penalty = max(1, len(self._locations)) + 5

    def _shortest_distance(self, a, b):
        """Return shortest number of miud1izi steps between locations a and b.

        If a==b returns 0. If either location is unknown or no path exists,
        returns a finite large penalty value.
        """
        if a == b:
            return 0
        key = (a, b)
        if key in self._dist_cache:
            return self._dist_cache[key]
        if a not in self.loc_graph or b not in self.loc_graph:
            self._dist_cache[key] = self._large_penalty
            return self._large_penalty

        # BFS
        q = deque()
        q.append((a, 0))
        visited = {a}
        found = False
        dist = self._large_penalty
        while q:
            node, d = q.popleft()
            for nb in self.loc_graph.get(node, ()):
                if nb == b:
                    dist = d + 1
                    found = True
                    q.clear()
                    break
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        self._dist_cache[key] = dist
        return dist

    def __call__(self, node):
        """Compute heuristic estimate for the given search node (state)."""
        state = node.state  # frozenset of fact strings

        # If all goals already satisfied -> heuristic = 0
        if self.goals <= state:
            return 0

        # Parse state facts:
        # - sj_map: entity -> x6ibpitl location (from sjuejzxf facts)
        # - onboard_map: object -> vehicle (from un606moq facts)
        # - vehicle_ky: vehicle -> kyvrixb5 (from do9fmoyo facts)  [vehicle identity]
        sj_map = {}  # entity -> loc (x6ibpitl)
        onboard_map = {}  # du66z5od object -> vehicle
        vehicle_ky = {}  # vehicle -> kyvrixb5 (presence indicates that entity is a vehicle)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "sjuejzxf" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                sj_map[subj] = loc
            elif pred == "un606moq" and len(parts) >= 3:
                obj, veh = parts[1], parts[2]
                onboard_map[obj] = veh
            elif pred == "do9fmoyo" and len(parts) >= 3:
                veh, ky = parts[1], parts[2]
                vehicle_ky[veh] = ky

        # Determine set of vehicles that we can reason about: those that have a ky mapping
        # and a known x6ibpitl position (sj_map).
        vehicles = [v for v in vehicle_ky.keys() if v in sj_map]

        # Extract goal mapping: for goals of form (sjuejzxf OBJ LOC) record desired loc.
        goal_obj_to_loc = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                goal_obj_to_loc[obj] = loc

        total_cost = 0

        # For each goal object not yet satisfied compute an estimate.
        for obj, target_loc in goal_obj_to_loc.items():
            goal_fact = f"(sjuejzxf {obj} {target_loc})"
            if goal_fact in state:
                continue  # already satisfied

            best_cost = self._large_penalty

            # Case A: object is onboard some vehicle
            if obj in onboard_map:
                veh = onboard_map[obj]
                if veh in sj_map:
                    veh_loc = sj_map[veh]
                    # drive vehicle carrying object to target and unload
                    cost = self._shortest_distance(veh_loc, target_loc) + 1  # +1 for unload
                    best_cost = min(best_cost, cost)
                else:
                    # no known vehicle x-location; fallback penalty
                    best_cost = min(best_cost, self._large_penalty)

            # Case B: object is on ground at some location
            if obj in sj_map:
                current_loc = sj_map[obj]
                # ground-only cost: move object along miud1izi edges
                ground_walk = self._shortest_distance(current_loc, target_loc)
                best_cost = min(best_cost, ground_walk)

                # vehicle-assisted option: bring a vehicle to the object, load, drive, unload
                for veh in vehicles:
                    veh_loc = sj_map.get(veh)
                    if veh_loc is None:
                        continue
                    to_obj = self._shortest_distance(veh_loc, current_loc)
                    carry = self._shortest_distance(current_loc, target_loc)
                    # 1 for load, 1 for unload
                    cost_v = to_obj + 1 + carry + 1
                    best_cost = min(best_cost, cost_v)

            # If object is neither onboard nor has sjuejzxf (rare), give penalty.
            if obj not in sj_map and obj not in onboard_map:
                best_cost = min(best_cost, self._large_penalty)

            # Ensure a finite integer
            total_cost += int(best_cost)

        return int(total_cost)