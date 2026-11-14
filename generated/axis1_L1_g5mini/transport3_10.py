from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (transport-like) domain.

    The domain contains:
    - locations of type x6ibpitl connected by the binary relation miud1izi (movement adjacency).
    - vehicles of type hctyyy8k whose current location is encoded by (sjuejzxf <vehicle> <loc>).
    - packages of type du66z5od whose location can be either (sjuejzxf <pkg> <loc>) or
      (un606moq <pkg> <vehicle>) when loaded.
    - goal facts ask for certain packages to be at particular locations: (sjuejzxf <pkg> <loc>).

    This heuristic estimates the number of actions required to place each package at its
    goal location. It uses the location-adjacency graph (miud1izi) and the current
    vehicle positions to estimate necessary moves and loading/unloading actions.

    Assumptions
    -----------
    - Moving a vehicle along one miud1izi edge costs 1 action (arsslvyt).
    - Loading a package into a vehicle costs 1 action (szsevy61).
    - Unloading a package from a vehicle costs 1 action (tmoxe3x5).
    - A vehicle can carry an arbitrary number of packages (the domain does not limit this).
    - We ignore the more subtle constraints involving hab0vli4 and do9fmoyo in counting
      extra actions beyond move/load/unload. Those constraints might force slightly
      different action sequences in the real domain, but this heuristic captures the
      dominant transport costs and is cheap to compute.
    - If a required path between two locations does not exist in the static adjacency
      graph, a large finite penalty is used for that package (heuristic remains finite).

    Heuristic Initialization
    ------------------------
    - Store the set of goal package->goal_location mappings from task.goals.
    - Build the static adjacency graph of locations using miud1izi facts from task.static.
      The adjacency graph is treated as undirected if reciprocal edges are present in static,
      but we add both directions for safety.
    - Keep the static facts available (not otherwise used here).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For each package that appears in the goals:
    1. If the package is already at its goal location in the current state, cost = 0.
    2. Parse the current state to discover:
       - vehicle identifiers (those that appear as the first argument of do9fmoyo in state).
       - vehicle positions: (sjuejzxf <vehicle> <loc>).
       - package positions: (sjuejzxf <pkg> <loc>).
       - package-in-vehicle relations: (un606moq <pkg> <vehicle>).
    3. For a package that is currently inside a vehicle V:
       - Let posV be the vehicle's current location.
       - Compute the shortest path distance d (number of edges) from posV to the package's goal location.
       - Estimated cost = d (vehicle moves) + 1 (unload).
         (If the vehicle is already at the goal, this yields cost = 1.)
    4. For a package that is at some location LP (not loaded):
       - If LP equals the goal location, cost = 0.
       - Otherwise, if there is a vehicle already at LP, the best plan is:
         load (1) + d (moves from LP to goal) + unload (1) => cost = d + 2.
       - If no vehicle is currently at LP, pick the nearest vehicle V (by shortest path
         from V's position to LP). Then estimated cost is:
         d1 (vehicle moves to LP) + 1 (load) + d2 (moves from LP to goal) + 1 (unload)
         => cost = d1 + d2 + 2.
       - If no vehicle exists at all in the state, we conservatively return a large finite
         penalty for that package.
    5. Sum the per-package estimated costs for all goal packages. This sum is returned
       as the heuristic value.

    Notes on admissibility and practicality
    --------------------------------------
    - This heuristic is not guaranteed to be admissible, but it is informative and cheap
      to compute. It tends to guide greedy search towards solutions by penalizing
      long travel distances and accounting for load/unload operations.
    - It is careful to return 0 exactly when the task.goal_reached(node.state) is true.
    """

    def __init__(self, task):
        # Goal facts (a set of fact strings like "(sjuejzxf p1 l2)")
        self.goals = task.goals
        # Static facts (miud1izi edges and possibly hab0vli4)
        static_facts = task.static

        # Build adjacency graph for locations from miud1izi facts.
        # The graph maps location -> set(neighbor_locations).
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Extract goal mapping: package -> goal_location for sjuejzxf goals.
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.goal_map[obj] = loc

        # A large finite penalty for unreachable situations (keeps heuristic finite).
        self.large_penalty = 1000

    def shortest_distance(self, start, goal):
        """
        Breadth-first search on the static adjacency graph to compute the minimum
        number of miud1izi edges between start and goal. Returns an integer distance
        or None if no path exists (or start/goal not known in graph).
        """
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for neigh in self.adj.get(node, ()):
                if neigh == goal:
                    return dist + 1
                if neigh not in visited:
                    visited.add(neigh)
                    q.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Returns the sum of estimated actions required to bring all goal packages
        to their desired locations according to the strategy described above.
        """
        state = node.state

        # If goals are already satisfied, heuristic is 0.
        # (This ensures requirement 2 from the checklist.)
        # Use the task's goal set membership test if available on node/state.
        try:
            # Many planners provide a method on Task to check goals; ensure correctness here:
            from heuristics.heuristic_base import Heuristic as _H  # ensure import present
            # The boolean check below simply verifies every goal fact is in current state
            if self.goals <= state:
                return 0
        except Exception:
            # If any unexpected error, still proceed to explicit check below
            if self.goals <= state:
                return 0

        # Parse current state facts
        # Identify vehicles via do9fmoyo facts: (do9fmoyo <vehicle> <kyv>)
        vehicles = set()
        vehicle_kyv = {}  # vehicle -> kyv node (not used for movement, kept for completeness)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "do9fmoyo" and len(parts) >= 3:
                veh, kyv = parts[1], parts[2]
                vehicles.add(veh)
                vehicle_kyv[veh] = kyv

        # Vehicle positions: from (sjuejzxf <vehicle> <loc>) where first arg is a vehicle
        vehicle_pos = {}
        # Package positions: (sjuejzxf <pkg> <loc>)
        package_pos = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                    vehicle_pos[obj] = loc
                else:
                    # Could be a package or some other object; only record packages that
                    # appear in the goal_map
                    if obj in self.goal_map:
                        package_pos[obj] = loc

        # Packages loaded into vehicles: (un606moq <pkg> <vehicle>)
        package_in_vehicle = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "un606moq" and len(parts) >= 3:
                pkg, veh = parts[1], parts[2]
                if pkg in self.goal_map:
                    package_in_vehicle[pkg] = veh

        total_cost = 0

        # For each package that appears in the goals, estimate cost
        for pkg, goal_loc in self.goal_map.items():
            # Case: already at goal
            if ("(sjuejzxf %s %s)" % (pkg, goal_loc)) in state:
                continue  # cost 0 for this package

            # Case: package is in a vehicle
            if pkg in package_in_vehicle:
                veh = package_in_vehicle[pkg]
                if veh in vehicle_pos:
                    start_loc = vehicle_pos[veh]
                    d = self.shortest_distance(start_loc, goal_loc)
                    if d is None:
                        total_cost += self.large_penalty
                    else:
                        # d moves to drive vehicle to goal location + 1 unload
                        total_cost += d + 1
                else:
                    # Vehicle location unknown -> conservative large penalty
                    total_cost += self.large_penalty
                continue

            # Case: package is at some location LP (and not in a vehicle)
            if pkg in package_pos:
                lp = package_pos[pkg]
                # If LP equals goal (already checked above), we'd have continued.
                # Find if a vehicle is at LP
                vehicles_at_lp = [v for v, pos in vehicle_pos.items() if pos == lp]
                # Distance LP -> goal
                d_lp_goal = self.shortest_distance(lp, goal_loc)
                if d_lp_goal is None:
                    # If no path from package location to goal, try to see if any vehicle can
                    # reach the goal better; otherwise penalty.
                    # Conservative choice: large penalty
                    total_cost += self.large_penalty
                    continue

                if vehicles_at_lp:
                    # There is at least one vehicle at the package location: load (1) + d + unload (1)
                    total_cost += d_lp_goal + 2
                else:
                    # No vehicle at LP: choose nearest vehicle and sum distances
                    best_cost = None
                    if not vehicle_pos:
                        # No vehicles at all -> large penalty
                        total_cost = total_cost + self.large_penalty
                        continue
                    for v, vpos in vehicle_pos.items():
                        d_v_to_lp = self.shortest_distance(vpos, lp)
                        if d_v_to_lp is None:
                            continue
                        # total moves: move vehicle to LP, load, move to goal, unload
                        c = d_v_to_lp + d_lp_goal + 2
                        if best_cost is None or c < best_cost:
                            best_cost = c
                    if best_cost is None:
                        total_cost += self.large_penalty
                    else:
                        total_cost += best_cost
                continue

            # If the package is neither at a known location nor known to be in a vehicle,
            # be conservative and add a large finite penalty.
            total_cost += self.large_penalty

        return total_cost