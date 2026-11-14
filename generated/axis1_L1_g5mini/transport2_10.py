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

    This heuristic estimates the number of actions required to satisfy all
    goal facts of the form (sjuejzxf <package> <goal-location>), i.e.,
    to bring each package (du66z5od objects) to its target location.
    The estimate is computed by treating locations (x6ibpitl) as nodes in a
    graph (edges from miud1izi facts), and combining:
      - the vehicle-to-package rendezvous cost (distance from nearest vehicle
        to the package location),
      - a loading cost (approx. 1 action),
      - the travel cost from the package location to its goal (number of
        location hops),
      - an unloading cost (approx. 1 action).
    If the package is already carried by a vehicle (un606moq), the rendezvous
    cost is replaced by the distance from that vehicle to the package goal,
    and only an unload action is charged in addition to travel.

    Assumptions
    -----------
    - Locations are the nodes in a (possibly undirected) graph defined by
      miud1izi facts in the static facts. We compute shortest-path lengths
      (in number of hops) between locations using BFS.
    - Vehicles (hctyyy8k objects) are identified by the presence of
      do9fmoyo facts in the state (vehicles appear as the first argument of
      do9fmoyo in the provided problems).
    - Each loading and unloading action is approximated as costing 1 action.
      Movement along one miud1izi edge is approximated as 1 action (using
      arsslvyt).
    - If there are multiple vehicles, we assume the nearest vehicle performs
      the pickup (hence we take the minimum rendezvous distance).
    - The heuristic is deliberately optimistic about some domain intricacies
      (e.g., city-changing combined load/unload actions), but balances that
      with location distances to provide good guidance for greedy best-first
      search.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal package -> goal-location mappings from the
      task.goals. We only consider goals whose predicate is `sjuejzxf`.
    - Build an adjacency mapping for locations from static miud1izi facts:
      location -> set(neighbor-locations).
    - Keep the static facts available for connectivity; additional locations
      discovered in states are added dynamically to the adjacency structure
      when computing the heuristic.

    Step-By-Step Thinking for Computing Heuristic (for a given state node)
    ---------------------------------------------------------------------
    1. If the state's facts already satisfy all goals, return 0.
    2. Parse the state to find:
       - package locations: facts of the form (sjuejzxf <pkg> <loc>).
       - carried packages: facts of the form (un606moq <pkg> <veh>).
       - vehicle locations: facts of the form (sjuejzxf <veh> <loc>) for
         vehicles discovered via do9fmoyo facts present in the initial task.
    3. For each goal (sjuejzxf <pkg> <goal_loc>):
       a) If (sjuejzxf <pkg> <goal_loc>) already in the state, cost 0.
       b) Else determine package's current location:
          - If the package is carried (un606moq <pkg> <veh>), use the
            vehicle's current location as the package location and mark it
            as "in-vehicle".
          - Else if (sjuejzxf <pkg> <loc>) exists, use that loc.
          - Otherwise, treat as unknown and assign a conservative large cost.
       c) Compute shortest-path distance between relevant locations:
          - If in-vehicle: distance = BFS(vehicle_loc -> goal_loc).
            total_estimate = distance + unload_cost (1).
          - If not in-vehicle: find nearest vehicle v (min BFS(vehicle_loc -> pkg_loc))
            rendezvous = distance_vehicle_to_pkg
            travel_to_goal = BFS(pkg_loc -> goal_loc)
            total_estimate = rendezvous + load_cost (1) + travel_to_goal + unload_cost (1)
       d) If any necessary distance is unreachable in the location graph,
          use a large penalty value for that package (but finite).
    4. Sum the estimates for all packages required by the goals and return the sum.

    Notes
    -----
    - The heuristic is not admissible; it aims at providing an informative and
      quickly computable guidance for greedy best-first search.
    - BFS is used to compute shortest path lengths in the location graph.
    """

    def __init__(self, task):
        """
        Initialize the heuristic:
        - store goals,
        - build location adjacency from static miud1izi facts,
        - store the set of vehicle names (extracted from static or goal/initial info)
          by observing do9fmoyo facts in the static facts (vehicles are the
          first argument in those facts).
        """
        self.goals = task.goals
        self.static = task.static

        # Build adjacency graph for locations using miud1izi facts in static facts.
        self.loc_neighbors = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.loc_neighbors.setdefault(a, set()).add(b)
                # don't assume symmetry in the domain definition; add reverse if it appears
                # later when parsing other static facts

        # Extract goal mapping for packages: (sjuejzxf <pkg> <goal_loc>)
        self.package_goals = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "sjuejzxf" and len(parts) >= 3:
                pkg, loc = parts[1], parts[2]
                self.package_goals[pkg] = loc

        # Identify vehicles by looking at do9fmoyo facts in static facts if present.
        # (In many problems vehicles are given in the initial state; we'll also
        # detect vehicle names dynamically from states when needed.)
        vehicles = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "do9fmoyo" and len(parts) >= 3:
                vehicles.add(parts[1])
        self.static_vehicles = vehicles

    def bfs_distance(self, start, goal, neighbors):
        """
        BFS shortest path length (in number of hops) between start and goal
        in the graph defined by neighbors: dict node -> set(neighbor nodes).
        Returns an integer >= 0 if reachable, otherwise returns None.
        """
        if start == goal:
            return 0
        if start not in neighbors and start != goal:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in neighbors.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Returns an integer estimate of remaining actions to satisfy all goal
        sjuejzxf facts (packages at target locations).
        """
        state = node.state

        # 1) If goals already satisfied, return 0.
        # task.goal is a set of facts; node.state is the state's fact frozenset.
        if self.goals <= state:
            return 0

        # Build dynamic adjacency: start from static and augment with any miud1izi present in the current state.
        neighbors = {k: set(v) for k, v in self.loc_neighbors.items()}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                neighbors.setdefault(a, set()).add(b)

        # Parse state to find:
        # - package locations: sjuejzxf <obj> <loc>
        # - carried packages: un606moq <pkg> <veh>
        # - vehicle locations: sjuejzxf <veh> <loc> for vehicles deduced from do9fmoyo facts or present in state
        pkg_loc = {}
        carried_by = {}  # pkg -> vehicle
        vehicle_loc = {}

        # First collect vehicles known from the state via do9fmoyo facts (vehicles are first arg)
        vehicles_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "do9fmoyo" and len(parts) >= 3:
                vehicles_in_state.add(parts[1])
        # Merge with static-detected vehicles
        all_known_vehicles = set(self.static_vehicles) | vehicles_in_state

        # Now parse sjuejzxf facts to collect package and vehicle locations.
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # If obj is a known vehicle, record vehicle location; otherwise assume it's a package/location-bearing object.
                if obj in all_known_vehicles:
                    vehicle_loc[obj] = loc
                else:
                    pkg_loc[obj] = loc

        # Parse carried relation un606moq <pkg> <veh>
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "un606moq" and len(parts) >= 3:
                pkg, veh = parts[1], parts[2]
                carried_by[pkg] = veh

        # If a package is carried, but we don't have the vehicle's location recorded yet,
        # try to find vehicle location via sjuejzxf facts in the state.
        # (vehicle_loc might already contain it; if not, leave it unknown)
        # The BFS routines will handle missing information by returning None.

        total_cost = 0
        LARGE = 1000  # penalty for unreachable or unknown situations

        # Precompute list of vehicle locations for rendezvous searches
        available_vehicle_locations = list(vehicle_loc.items())  # list of (vehicle, loc)

        # For each package goal, estimate cost
        for pkg, goal_loc in self.package_goals.items():
            # If pkg already at goal in the current state, cost 0 for this package.
            if f"(sjuejzxf {pkg} {goal_loc})" in state:
                continue

            # Determine current package situation.
            if pkg in carried_by:
                # Package is in a vehicle
                veh = carried_by[pkg]
                vloc = vehicle_loc.get(veh)
                if vloc is None:
                    # vehicle location unknown: try to find it via do9fmoyo + sjuejzxf facts in state
                    # but if not available, assign penalty
                    est = LARGE
                else:
                    # distance from vehicle current location to goal location
                    dist = self.bfs_distance(vloc, goal_loc, neighbors)
                    if dist is None:
                        est = LARGE
                    else:
                        # Moving while carrying requires dist hops; plus unload cost (1)
                        est = dist + 1
            elif pkg in pkg_loc:
                # Package is at some location (not carried)
                pkg_location = pkg_loc[pkg]

                # Distance package -> goal
                pkg_to_goal = self.bfs_distance(pkg_location, goal_loc, neighbors)
                if pkg_to_goal is None:
                    # unreachable within location graph
                    est = LARGE
                else:
                    # Find nearest vehicle to pkg_location
                    best_rendezvous = None
                    for veh, vloc in available_vehicle_locations:
                        d = self.bfs_distance(vloc, pkg_location, neighbors)
                        if d is not None:
                            if best_rendezvous is None or d < best_rendezvous:
                                best_rendezvous = d
                    # If we have no recorded vehicles, we can still try vehicles discovered via static_vehicles but without locations;
                    # in that case we treat rendezvous as unknown (penalize).
                    if best_rendezvous is None:
                        # No vehicle location known that can reach package; assign a conservative cost:
                        # allow for a large penalty but still finite
                        est = pkg_to_goal + 1 + LARGE // 2
                    else:
                        # rendezvous (vehicle->package) + load (1) + travel (pkg->goal) + unload (1)
                        est = best_rendezvous + 1 + pkg_to_goal + 1
            else:
                # Package not observed in either carried_by or sjuejzxf: unknown location -> penalize
                est = LARGE

            total_cost += est

        # Return a finite heuristic estimate (cap to avoid extremely large numbers).
        # Using int for compatibility with planners expecting integer costs.
        return int(min(total_cost, LARGE * len(self.package_goals) + 1))