from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Split a PDDL fact string like '(pred a b)' into ['pred', 'a', 'b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the transport domain. Estimates the number
    of actions required to achieve all (at package location) goal facts by
    summing per-package delivery estimates. For each package the heuristic
    estimates the minimal sequence of operations required to get it to its
    goal location:
      - If the package is already at its goal: 0.
      - If the package is inside a vehicle: drive the vehicle to the goal
        (one drive per road step) and drop the package (1 action).
      - If the package is on the ground at location L: choose the vehicle
        that minimizes (drive vehicle to L) + pick-up (1) + (drive to goal)
        + drop (1). Driving cost is approximated by shortest-path distance
        on the road graph (each road corresponds to one drive action).

    Assumptions
    -----------
    - Capacity and size information is ignored for the heuristic estimate.
      The domain provides capacity facts, but package sizes are not explicitly
      available in the state, so we estimate assuming any vehicle can carry
      a package. This makes the heuristic inadmissible in some cases but
      keeps it efficient.
    - Road facts in the static information define a directed graph of
      locations where each road edge costs 1 drive action.
    - Vehicles are identified via 'capacity' facts in the state (vehicles
      have a capacity fact), and their current locations are taken from
      'at' facts.
    - Only packages that appear in the goal (i.e., goals of the form
      (at <package> <location>)) are considered by the heuristic.

    Heuristic Initialization
    ------------------------
    - Extract goal locations for objects from the task goals (only 'at' goals).
    - Build the directed adjacency list of locations from static 'road' facts.
    - Collect the set of locations known from road facts and goal facts.
    - Initialize a cache for shortest-path distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Parse the state's facts to find:
       - where each package is: either "at <loc>" or "in <vehicle>"
       - where each vehicle is: via "at <vehicle> <loc>"
    3. For each package mentioned in the goals:
       - If the package is at the goal location -> contribution 0.
       - Else if the package is inside vehicle V:
         - Get V's location (must come from an 'at' fact for V).
         - Cost = shortest_path_distance(vehicle_location, goal_location)
                  + 1 (drop action).
       - Else (package on ground at location L):
         - For every vehicle V compute:
             cost_v = shortest_path_distance(vehicle_location, L)  # drive V->L
                      + 1  # pick-up
                      + shortest_path_distance(L, goal_location)  # drive L->goal
                      + 1  # drop
           Take the minimum cost over all vehicles as the package estimate.
       - If no path exists for any required drive, a moderate large penalty is
         added for that package (keeps the value finite while indicating
         difficulty). The penalty is proportional to the number of known
         locations.
    4. Sum the per-package estimates and return that sum as the heuristic value.

    The heuristic is cheap to compute: it uses BFS on the (static) road graph,
    caching shortest-path distances per source location so repeated calls are
    efficient. It is not guaranteed to be admissible, but it tends to
    prioritize states that reduce driving distances and that get packages
    either inside vehicles or nearer to their goals.
    """

    def __init__(self, task):
        # Goals: extract mappings package -> goal_location for 'at' goals only.
        self.goals = task.goals
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) >= 3:
                _, obj, loc = parts[:3]
                self.goal_locations[obj] = loc

        # Build adjacency (directed) list from static 'road' facts.
        self.adj = {}
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) >= 3:
                _, a, b = parts[:3]
                self.locations.add(a)
                self.locations.add(b)
                self.adj.setdefault(a, set()).add(b)

        # Make sure goal locations are included in known locations.
        for loc in self.goal_locations.values():
            self.locations.add(loc)
            self.adj.setdefault(loc, set())

        # Cache for distances: distances[src][dst] = integer distance or None.
        self.dist_cache = {}

        # A reasonable large penalty to use when a path is missing.
        # Proportional to number of known locations to keep values finite.
        self.missing_path_penalty = max(50, len(self.locations) + 10)

    def _ensure_distances_from(self, src):
        """Compute and cache shortest-path distances from src to all locations via BFS."""
        if src in self.dist_cache:
            return
        dist = {}
        # If src is unknown (not in graph), distances remain empty.
        if src not in self.adj:
            # Mark all known locations unreachable
            for loc in self.locations:
                dist[loc] = None
            self.dist_cache[src] = dist
            return

        # BFS
        q = deque()
        q.append(src)
        dist[src] = 0
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # For locations not reached, set None
        for loc in self.locations:
            dist.setdefault(loc, None)
        self.dist_cache[src] = dist

    def _distance(self, src, dst):
        """
        Return shortest-path distance (number of road edges) from src to dst.
        If src == dst returns 0. If no path exists returns None.
        """
        if src == dst:
            return 0
        self._ensure_distances_from(src)
        return self.dist_cache[src].get(dst)

    def __call__(self, node):
        state = node.state

        # If goal already satisfied return 0 (check requirement 2).
        if self.goals <= state:
            return 0

        # Parse state facts to extract 'at' and 'in' relations and vehicles.
        at_map = {}  # maps object -> location for 'at'
        in_map = {}  # maps package -> vehicle for 'in'
        vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                _, obj, loc = parts[:3]
                at_map[obj] = loc
            elif pred == "in" and len(parts) >= 3:
                _, pkg, veh = parts[:3]
                in_map[pkg] = veh
            elif pred == "capacity" and len(parts) >= 2:
                # capacity facts identify vehicles
                _, veh = parts[:2]
                vehicles.add(veh)

        # Ensure vehicle locations exist for vehicles seen via capacity facts.
        # If a vehicle is missing an 'at' fact, we treat it as unknown (no distances).
        vehicle_locations = {}
        for v in vehicles:
            if v in at_map:
                vehicle_locations[v] = at_map[v]
            else:
                vehicle_locations[v] = None

        total_cost = 0

        # For each package that appears in the goals compute estimated cost.
        for pkg, goal_loc in self.goal_locations.items():
            # Current status of the package:
            if pkg in in_map:
                # package is inside a vehicle
                veh = in_map[pkg]
                veh_loc = at_map.get(veh)
                if veh_loc is None:
                    # vehicle location unknown -> penalize
                    total_cost += self.missing_path_penalty
                    continue
                d = self._distance(veh_loc, goal_loc)
                if d is None:
                    total_cost += self.missing_path_penalty
                else:
                    # drive steps + drop
                    total_cost += d + 1
            else:
                # package on the ground at some location
                pkg_loc = at_map.get(pkg)
                if pkg_loc is None:
                    # package location unknown -> penalize
                    total_cost += self.missing_path_penalty
                    continue
                # Already at goal?
                if pkg_loc == goal_loc:
                    continue
                # Try every vehicle and pick minimal cost to deliver this package.
                best = None
                for veh, veh_loc in vehicle_locations.items():
                    if veh_loc is None:
                        # cannot reason about this vehicle
                        continue
                    # drive vehicle to package location
                    d1 = self._distance(veh_loc, pkg_loc)
                    # drive package (in vehicle) from pkg_loc to goal_loc
                    d2 = self._distance(pkg_loc, goal_loc)
                    if d1 is None or d2 is None:
                        # this vehicle cannot perform the required movement
                        continue
                    cost = d1 + 1 + d2 + 1  # drive_to_pkg + pick + drive_to_goal + drop
                    if best is None or cost < best:
                        best = cost
                if best is None:
                    # no vehicle could reach or no vehicles available: penalize
                    total_cost += self.missing_path_penalty
                else:
                    total_cost += best

        return total_cost