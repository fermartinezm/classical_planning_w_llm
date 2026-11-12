from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a fact string like '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hrpmxpjg) problem.
    The domain contains:
      - local locations (predicate miud1izi defines adjacency between x6ibpitl locations),
      - movable agents/vehicles (hctyyy8k objects),
      - items/packages (du66z5od objects),
      - predicates sjuejzxf(obj loc) for objects (vehicles or packages) being at a local location,
      - un606moq(pkg veh) for a package being inside a vehicle,
      - do9fmoyo(veh city) and hab0vli4(city1 city2) for vehicle-level (city-to-city) moves that happen
        as a side effect of load/unload actions.

    The heuristic estimates the number of actions required to bring every goal package
    to its goal local location (sjuejzxf package goal_loc). It is intended to guide a
    greedy best-first search (so admissibility is not required), but it aims to be
    informative while remaining efficient.

    Assumptions
    -----------
    - Packages can only change their local location by being loaded into a vehicle
      (szsevy61) and later unloaded (tmoxe3x5). Those load/unload actions also move the
      vehicle between city nodes (do9fmoyo/hab0vli4), but they do not change the
      vehicle's local location. Local movements of vehicles (arsslvyt) move vehicles
      along miud1izi edges (x6ibpitl adjacency).
    - There is always at least one vehicle in the problem (if none are present, the
      heuristic falls back to a finite large cost).
    - We ignore city-level constraints for whether load/unload is currently possible
      (hab0vli4 / do9fmoyo) to keep the heuristic efficient and optimistic. This may
      make the heuristic inadmissible but typically helps search guidance.
    - Vehicles have unlimited capacity (the domain file does not restrict number of
      packages per vehicle). The heuristic does not attempt to reason about optimal
      packing or combined deliveries — it sums per-package estimated efforts.

    Heuristic Initialization
    ------------------------
    - Extract and store goal package -> goal_local_location mappings from task.goals.
    - Build an undirected adjacency graph of local locations from static miud1izi facts.
      (We add both directions for robustness; domain static often contains both.)
    - Keep a cache for shortest-path distances between local locations (BFS) to avoid
      repeated BFS computations during heuristic calls.
    - Store any other static facts (e.g., hab0vli4) for possible future extensions (not
      required for this estimate).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each package that has a location goal (sjuejzxf pkg goal_loc):
      1. If the goal is already satisfied in the current state, cost contribution is 0.
      2. Determine the package's current status:
         - If the package is on a local location L (there is a fact "(sjuejzxf pkg L)"),
           then it must be (a) picked up by some vehicle (some vehicle must be moved to L),
           (b) loaded (one action), (c) the vehicle moved (locally) from L to the goal
           local location G, and (d) the package unloaded (one action).
           Estimated cost (per package): min_over_vehicles(dist(vehicle_loc, L)) + 1 (load)
                                         + dist(L, G) + 1 (unload)
         - If the package is already inside a vehicle V (there is a fact "(un606moq pkg V)"),
           then we only need to move that vehicle to the goal local location G and unload:
           cost = dist(vehicle_loc, G) + 1 (unload)
      3. Distances dist(a,b) are shortest-path lengths on the local-location graph built
         from miud1izi facts. If no path exists (disconnected graph) or no vehicle location
         is known, the heuristic uses a large finite penalty to keep the value finite.
      4. The heuristic sums these estimated costs for all goal packages.

    Notes on Complexity and Practicality
    ------------------------------------
    - BFS-based distances are cached per source local location so that repeated distance
      queries are efficient across heuristic evaluations.
    - This heuristic intentionally ignores some city-level feasibility checks (hab0vli4/do9fmoyo)
      and vehicle capacity interactions to remain fast and to produce a useful (though not
      necessarily admissible) guidance for greedy best-first search.
    """

    def __init__(self, task):
        # Save goal facts as provided by the planner
        self.goals = set(task.goals)  # set of fact strings

        # Map of package -> goal local location for goals of form (sjuejzxf pkg loc)
        self.goal_locations = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.goal_locations[obj] = loc

        # Build adjacency (undirected) for local locations based on static miud1izi facts
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)  # make undirected for robustness

        # Cache of BFS distance maps: src_loc -> {loc: distance}
        self._dist_cache = {}

        # Keep static hab0vli4 as well in case needed (not used in current heuristic)
        self.city_edges = [get_parts(f) for f in task.static if get_parts(f)[0] == "hab0vli4"]

    def _bfs_distances_from(self, src):
        """Return a dict of shortest distances from src to all reachable local locations."""
        if src in self._dist_cache:
            return self._dist_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[src] = dist
        return dist

    def _distance(self, a, b):
        """Return shortest-path distance between local locations a and b (int), or None if unreachable."""
        if a == b:
            return 0
        dist_map = self._bfs_distances_from(a)
        return dist_map.get(b, None)

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If goals already satisfied -> heuristic = 0 (checklist item 2)
        if self.goals <= state:
            return 0

        # Identify vehicles by looking at do9fmoyo facts: do9fmoyo vehicle city
        vehicles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "do9fmoyo" and len(parts) >= 3:
                vehicles.add(parts[1])

        # Map vehicles -> their local locations using sjuejzxf facts
        vehicle_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                if subj in vehicles:
                    vehicle_loc[subj] = loc

        # For packages: determine whether they are at a loc or in a vehicle
        package_loc = {}     # pkg -> loc (if at some location)
        package_in_vehicle = {}  # pkg -> vehicle (if contained)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "sjuejzxf" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                # If subj is a package that is referenced in goals, record its loc
                if subj in self.goal_locations:
                    package_loc[subj] = loc
            elif parts[0] == "un606moq" and len(parts) >= 3:
                pkg, veh = parts[1], parts[2]
                if pkg in self.goal_locations:
                    package_in_vehicle[pkg] = veh

        total_cost = 0
        LARGE_PENALTY = 1000  # finite but large cost used when distances unknown/unreachable

        # Precompute list of known vehicle local locations
        vehicle_local_locs = list(vehicle_loc.values())

        for pkg, goal_loc in self.goal_locations.items():
            # If package already at goal, contribution 0
            if f"(sjuejzxf {pkg} {goal_loc})" in state:
                continue

            # Case: package is currently inside a vehicle
            if pkg in package_in_vehicle:
                veh = package_in_vehicle[pkg]
                veh_loc = vehicle_loc.get(veh, None)
                if veh_loc is None:
                    # Unknown vehicle local location -> penalty
                    total_cost += LARGE_PENALTY
                    continue
                d = self._distance(veh_loc, goal_loc)
                if d is None:
                    total_cost += LARGE_PENALTY
                else:
                    # Move vehicle to goal local location then unload (1)
                    total_cost += d + 1
                continue

            # Case: package is at a local location L and not in a vehicle
            if pkg in package_loc:
                pkg_loc = package_loc[pkg]

                # Distance from package location to goal location
                d_pkg_goal = self._distance(pkg_loc, goal_loc)
                if d_pkg_goal is None:
                    # unreachable local path -> penalty
                    total_cost += LARGE_PENALTY
                    continue

                # Find nearest vehicle (by local location distance) to the package
                if not vehicle_local_locs:
                    # No known vehicle local positions -> large penalty
                    total_cost += LARGE_PENALTY
                    continue

                best_to_pkg = None
                # Rather than running BFS from every vehicle, run BFS from package once and use it
                # to measure distance to each vehicle location (more efficient if many vehicles).
                dist_from_pkg = self._bfs_distances_from(pkg_loc)
                for vloc in vehicle_local_locs:
                    d_v_to_pkg = dist_from_pkg.get(vloc, None)
                    if d_v_to_pkg is None:
                        continue
                    if best_to_pkg is None or d_v_to_pkg < best_to_pkg:
                        best_to_pkg = d_v_to_pkg

                if best_to_pkg is None:
                    # No vehicle can reach the package locally -> penalty
                    total_cost += LARGE_PENALTY
                    continue

                # Cost: move vehicle to package + load (1) + move to goal + unload (1)
                total_cost += best_to_pkg + 1 + d_pkg_goal + 1
                continue

            # If we don't know where the package is (neither at loc nor in vehicle), be conservative
            total_cost += LARGE_PENALTY

        return total_cost