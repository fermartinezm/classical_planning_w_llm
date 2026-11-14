from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `secret_domain` (hrpmxpjg).
    The domain models items (type `du66z5od`) and vehicles (type `hctyyy8k`) placed at
    locations (`x6ibpitl`). Locations are connected by the static predicate `miud1izi`.
    Vehicles also have a higher-level location/region (type `kyvrixb5`) via `do9fmoyo`,
    and there are region adjacencies `hab0vli4`. Objects can be moved by being
    picked up onto a vehicle (`un606moq`) using pickup/drop actions which also
    change the vehicle's region, while vehicles can move between locations using
    `arsslvyt` (which uses `miud1izi`).

    This heuristic estimates the number of actions required to achieve the goal
    sjuejzxf facts (objects at target locations). It is not guaranteed to be
    admissible, but it tries to produce informative estimates that are cheap
    to compute.

    Assumptions
    -----------
    - Goals are of the form (sjuejzxf <object> <location>), i.e., objects must be
      located at particular x6ibpitl locations.
    - Objects can only change their sjuejzxf (on-ground location) either by
      being picked up onto a vehicle (un606moq) and dropped again, or by being
      moved while on a vehicle as the vehicle moves between locations.
    - Vehicles are recognized by the presence of a (do9fmoyo <vehicle> <region>)
      fact in the state; their current location is the sjuejzxf fact for that vehicle.
    - There is no explicit capacity limit for vehicles (they can carry many objects).
    - The location connectivity given by static miud1izi facts can be treated
      as an undirected graph for the purpose of computing shortest location distances.
    - Pickup and drop actions each cost 1; moving a vehicle between adjacent
      locations (arsslvyt) costs 1 per hop. The heuristic counts these uniformly.

    Heuristic Initialization
    ------------------------
    - The constructor saves the goal facts (task.goals).
    - From the static facts (task.static) it extracts the location adjacency graph
      induced by `miud1izi` (as an undirected graph) to allow fast BFS distance
      queries between any two x6ibpitl locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given node/state:
    1. If all goal facts already hold in `state`, return 0.
    2. Parse the state to collect:
       - For each goal object, whether it is currently:
         a) on the ground at some location L (a sjuejzxf fact with the object),
         b) on a vehicle V (an un606moq <object> <vehicle> fact), or
         c) not present (treated conservatively).
       - Vehicle locations: for each vehicle V (recognized from do9fmoyo facts),
         find its current x6ibpitl location from sjuejzxf facts.
    3. Precompute (or on-demand compute) shortest-path distances between
       locations using the adjacency graph from static `miud1izi`.
       If two locations are equal, distance=0. If there is no path,
       treat the distance as a large finite number (INF).
    4. For each goal (object -> goal_location) compute a minimal estimated
       action cost independently:
       - If the object is already at goal_location (sjuejzxf holds), cost = 0.
       - If the object is currently on a vehicle V located at loc_v:
           cost ≈ distance(loc_v, goal_location) + 1  (one drop action).
       - If the object is on the ground at loc_o:
           Consider all vehicles V (with known location loc_v) and take the min:
             cost_option(V) = distance(loc_v, loc_o)   # vehicle moves to object
                              + 1                      # pickup (szsevy61)
                              + distance(loc_o, goal_location)  # vehicle carries object
                              + 1                      # drop (tmoxe3x5)
           The heuristic for that object is the minimum cost_option(V) over vehicles.
         If no vehicles are found in the state, fall back to:
           cost = distance(loc_o, goal_location) + 2 (conservative: pick+drop implicit)
           (This keeps the heuristic finite even in degenerate states.)
    5. Sum the per-object estimates to obtain the final heuristic value. Because
       vehicles can actually carry multiple objects simultaneously, summing can
       over-estimate interactions; this is acceptable for a greedy best-first
       heuristic and tends to produce informative guidance.
    """

    def __init__(self, task):
        # Save goal facts for quick goal checks.
        self.goals = task.goals
        # Extract miud1izi location adjacency from static facts.
        static_facts = task.static or frozenset()
        self.adj = {}  # adjacency dict: loc -> set(neighbor_locs)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)  # treat as undirected for distances

        # Precompute nothing more here; state-specific info is gathered in __call__.

    def _distance(self, start, goal):
        """
        BFS distance in the static location graph between start and goal.
        If either start or goal is unknown to the static adjacency, treat them
        as isolated nodes (distance infinite). Return an integer >= 0 or a large INF.
        """
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            # If either node absent from adjacency, we still try to consider isolated nodes:
            # if start==goal above handled, otherwise assume unreachable => large.
            return 1000
        # BFS
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, d = dq.popleft()
            for nb in self.adj.get(node, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    dq.append((nb, d + 1))
        # unreachable
        return 1000

    def __call__(self, node):
        state = node.state

        # Quick goal-check: heuristic must be 0 in goal states.
        if self.goals <= state:
            return 0

        # Parse state:
        # - object_ground_loc: mapping object -> location from sjuejzxf when object is on ground
        # - on_vehicle: mapping object -> vehicle from un606moq
        # - vehicles: set of vehicle names known from do9fmoyo facts
        # - vehicle_loc: mapping vehicle -> x6ibpitl location from sjuejzxf facts
        object_ground_loc = {}
        on_vehicle = {}
        vehicles = set()
        vehicle_loc = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "sjuejzxf" and len(parts) >= 3:
                ent, loc = parts[1], parts[2]
                # sjuejzxf is used for both objects and vehicles; record location for now.
                # For objects that are on ground, this will be their location unless they are on vehicle.
                # We will override or use on_vehicle when present.
                object_ground_loc[ent] = loc
            elif pred == "un606moq" and len(parts) >= 3:
                # object on vehicle
                obj, veh = parts[1], parts[2]
                on_vehicle[obj] = veh
            elif pred == "do9fmoyo" and len(parts) >= 3:
                # vehicle's region information; mark the first arg as a vehicle identifier.
                veh = parts[1]
                vehicles.add(veh)

        # Now build vehicle_loc for vehicles we can locate via sjuejzxf facts
        for veh in list(vehicles):
            if veh in object_ground_loc:
                vehicle_loc[veh] = object_ground_loc[veh]

        # Identify goal objects and their goal locations (goals are facts strings).
        goal_pairs = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, goal_loc = parts[1], parts[2]
                goal_pairs.append((obj, goal_loc))
            else:
                # For other goal predicates (not expected here), we ignore them in this heuristic.
                pass

        total_cost = 0
        INF = 1000

        # For each goal object compute a minimal cost estimate to reach the goal location.
        for obj, goal_loc in goal_pairs:
            # If goal already holds in state, cost 0 (checked above globally, but do per-object check too).
            if f"(sjuejzxf {obj} {goal_loc})" in state:
                continue

            # If the object is on some vehicle:
            if obj in on_vehicle:
                veh = on_vehicle[obj]
                loc_v = vehicle_loc.get(veh)
                if loc_v is None:
                    # Vehicle location unknown: be conservative but finite
                    # Assume we need at least one drop action.
                    cost = 1 + INF // 2
                else:
                    # Need to move vehicle from its current location to goal location (vehicle hops),
                    # then drop the object.
                    dist = self._distance(loc_v, goal_loc)
                    cost = dist + 1  # drop
                total_cost += cost
                continue

            # Otherwise the object is on the ground somewhere:
            loc_o = object_ground_loc.get(obj)
            if loc_o is None:
                # Object not present in state facts (conservative fallback)
                total_cost += INF // 2
                continue

            # If there is a vehicle already in the same location, this is cheapest:
            best_cost = INF
            for veh in vehicles:
                loc_v = vehicle_loc.get(veh)
                if loc_v is None:
                    continue
                # vehicle must move to object (loc_v -> loc_o), pick, then carry (loc_o -> goal_loc), drop
                cost = self._distance(loc_v, loc_o) + 1 + self._distance(loc_o, goal_loc) + 1
                if cost < best_cost:
                    best_cost = cost

            if best_cost == INF:
                # No vehicle located in the state. Fall back to using location-only distance + pick/drop surrogate.
                best_cost = self._distance(loc_o, goal_loc) + 2

            total_cost += best_cost

        return int(total_cost)