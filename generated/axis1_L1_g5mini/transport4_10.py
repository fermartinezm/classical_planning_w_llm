from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing the surrounding
    parentheses and splitting on whitespace.

    Example:
      "(sjuejzxf p1 l2)" -> ["sjuejzxf", "p1", "l2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hrpmxpjg) domain.
    The heuristic estimates the number of actions needed to achieve all
    goal `sjuejzxf` facts for du66z5od objects (packages). It is a
    simple, fast approximation based on distances in the location graph
    (miud1izi) and the observation that packages can only be moved
    by vehicles (hctyyy8k) using load/unload actions (szsevy61/tmoxe3x5)
    and vehicles themselves move between locations using arsslvyt.

    Assumptions
    -----------
    - Packages that appear in the goals are the primary subgoals we care about.
    - Movement of vehicles between x6ibpitl locations is captured by miud1izi edges.
      We model each such movement as costing 1 (one arsslvyt application).
    - Loading a package onto a vehicle (szsevy61) costs 1 action.
    - Unloading a package from a vehicle (tmoxe3x5) costs 1 action.
    - A package not currently attached to any vehicle must be (1) met by some
      vehicle, (2) loaded, (3) transported (vehicle moves) to the package goal
      location, and (4) unloaded. The heuristic chooses the vehicle that
      minimizes this sequence in the current state.
    - If a package is already attached to a vehicle (present through un606moq),
      the heuristic accounts for the vehicle movement to the goal and a single
      unload action.
    - If a required path (in the miud1izi graph) is not present, the heuristic
      assigns a large finite penalty instead of infinity (keeps heuristic finite).

    Heuristic Initialization
    ------------------------
    - Extract goal facts (task.goals) and remember the target location for each
      package whose goal is an sjuejzxf fact.
    - From task.static, extract all directed miud1izi edges between locations
      and build an adjacency map for vehicle movement (vehicles move using
      arsslvyt along miud1izi).
    - Prepares an internal cache for shortest-path computations between locations.

    Step-By-Step Thinking for Computing Heuristic (for a given node/state)
    ---------------------------------------------------------------------
    1. If the node's state already satisfies all goals, return 0.
    2. Parse the current state to discover:
         - Locations of all objects having sjuejzxf facts (both vehicles and packages).
         - Which packages (du66z5od) are currently attached to which vehicles via un606moq.
         - Which objects are vehicles: those that appear in do9fmoyo facts (vehicle->hub).
    3. For each package that still needs to reach its goal (i.e., goal sjuejzxf not true in state):
         a) If the package is attached to a vehicle v:
            - Let v_loc be the current location of v (from sjuejzxf).
            - Cost estimate = shortest-path distance from v_loc to goal_loc (vehicle moves)
                              + 1 (unload).
         b) Else (package at some location p_loc):
            - For each vehicle v available in the state:
                * Compute distance from v_loc -> p_loc (to reach package).
                * Compute distance from p_loc -> goal_loc (to move the package while attached).
                * Total for this vehicle = dist(v_loc, p_loc) + 1 (load) + dist(p_loc, goal_loc) + 1 (unload).
            - Take the minimum over vehicles. If no vehicle yields a finite route, fall back to
              a conservative estimate equal to a large penalty based on number of locations.
    4. Sum the per-package estimates to get the returned heuristic value. This treats
       package deliveries independently (thus may overestimate the true optimal plan,
       but is fast and effective in guiding greedy best-first search).
    """

    def __init__(self, task):
        """
        task: instance of Task providing .goals and .static (facts true in every state)
        """
        # Store goal target locations for packages (facts of the form (sjuejzxf package location)).
        self.goal_locations = {}
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # We assume these are the package goals we must satisfy.
                self.goal_locations[obj] = loc

        # Build adjacency graph for x6ibpitl locations from static miud1izi facts.
        # miud1izi ?from ?to   -> directed edge from -> to
        self.miud_graph = {}
        self.all_locations = set()
        for f in task.static:
            parts = get_parts(f)
            if parts and parts[0] == "miud1izi" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.all_locations.add(src)
                self.all_locations.add(dst)
                self.miud_graph.setdefault(src, set()).add(dst)

        # Provide default for locations without outgoing edges
        for loc in list(self.all_locations):
            self.miud_graph.setdefault(loc, set())

        # Cache for BFS shortest paths from a source location to all reachable locations
        self._dist_cache = {}

        # A finite large penalty used when a route is not found in the static graph.
        # Tuning note: this should remain finite. It is set relative to the number of locations.
        self.large_penalty = max(50, len(self.all_locations) * 5 + 20)

    def _bfs_from(self, source):
        """
        Compute BFS distances from source to all other locations in the miud_graph.
        Results are cached in self._dist_cache[source].
        """
        if source in self._dist_cache:
            return self._dist_cache[source]

        dist = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.miud_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[source] = dist
        return dist

    def _distance(self, src, dst):
        """
        Return shortest-path distance (in number of arsslvyt moves) from src to dst.
        If src == dst return 0. If src or dst is unknown in the adjacency map,
        return None to indicate unreachable.
        """
        if src == dst:
            return 0
        if src not in self.miud_graph:
            return None
        dist_map = self._bfs_from(src)
        return dist_map.get(dst)

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        node.state is a frozenset of PDDL fact strings.
        Return a non-negative integer heuristic estimate (smaller is better).
        """
        state = node.state

        # Quick check: if the goal is reached, heuristic is zero.
        # (Ensure heuristic is 0 only when goals are satisfied.)
        # task.goals are not directly available here; use presence of each goal sjuejzxf fact.
        goal_satisfied = True
        for pkg, goal_loc in self.goal_locations.items():
            fact = f"(sjuejzxf {pkg} {goal_loc})"
            if fact not in state:
                goal_satisfied = False
                break
        if goal_satisfied:
            return 0

        # Parse current state to extract:
        # - loc_of: mapping object -> location for all sjuejzxf facts
        # - attached: mapping package -> vehicle for un606moq facts
        # - vehicles: set of objects that appear as first arg in do9fmoyo facts (vehicle->hub)
        loc_of = {}
        attached = {}
        vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                loc_of[obj] = loc
            elif pred == "un606moq" and len(parts) >= 3:
                package, vehicle = parts[1], parts[2]
                attached[package] = vehicle
            elif pred == "do9fmoyo" and len(parts) >= 3:
                vehicle, hub = parts[1], parts[2]
                vehicles.add(vehicle)
                # we do not use hub directly in movement cost calculation, but do record vehicle presence

        # If there are no vehicles recorded in state, use any object that has sjuejzxf and also appears
        # in attached mapping as a vehicle. (This is defensive; normally vehicles appear in do9fmoyo.)
        if not vehicles:
            for v in attached.values():
                vehicles.add(v)

        total_estimate = 0

        for pkg, goal_loc in self.goal_locations.items():
            # If package already at goal, no cost.
            if f"(sjuejzxf {pkg} {goal_loc})" in state:
                continue

            # If package is currently attached to a vehicle, only account for vehicle travel + unload.
            if pkg in attached:
                vehicle = attached[pkg]
                v_loc = loc_of.get(vehicle)
                if v_loc is None:
                    # Unknown vehicle location -> conservative large penalty
                    total_estimate += self.large_penalty
                    continue
                dist = self._distance(v_loc, goal_loc)
                if dist is None:
                    total_estimate += self.large_penalty
                else:
                    # dist moves + 1 unload
                    total_estimate += dist + 1
                continue

            # Package is not attached -> it must be picked up from its current location.
            p_loc = loc_of.get(pkg)
            if p_loc is None:
                # Unknown package location -> conservative penalty
                total_estimate += self.large_penalty
                continue

            # If no vehicles exist in the state, fallback to a heuristic based on path length
            # in the location graph from package to goal (even though domain does not allow
            # package move alone). This keeps heuristic finite.
            if not vehicles:
                d_direct = self._distance(p_loc, goal_loc)
                if d_direct is None:
                    total_estimate += self.large_penalty
                else:
                    # Even though a vehicle is required, estimate as path length + 2 for load/unload.
                    total_estimate += d_direct + 2
                continue

            # Evaluate best vehicle to perform pickup and delivery.
            best_cost = None
            for v in vehicles:
                v_loc = loc_of.get(v)
                if v_loc is None:
                    continue
                d_to_pkg = self._distance(v_loc, p_loc)
                if d_to_pkg is None:
                    continue
                d_pkg_to_goal = self._distance(p_loc, goal_loc)
                if d_pkg_to_goal is None:
                    continue
                # moves to reach package + load + moves to goal + unload
                cost = d_to_pkg + 1 + d_pkg_to_goal + 1
                if best_cost is None or cost < best_cost:
                    best_cost = cost

            if best_cost is None:
                # No vehicle provided a finite route; give conservative penalty.
                total_estimate += self.large_penalty
            else:
                total_estimate += best_cost

        # Ensure we always return an int and a finite value.
        return int(total_estimate)