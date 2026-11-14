from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the space-separated parts of a fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hrpmxpjg). The domain
    models "vehicles" (objects of type hctyyy8k) that can move between
    fine-grained locations (predicate `sjuejzxf <vehicle> <loc>`) via the
    `arsslvyt` operator using adjacencies `miud1izi` on x6ibpitl locations.
    Packages (du66z5od) are at locations via `sjuejzxf` or inside vehicles via
    `un606moq`. Loading/unloading (operators `szsevy61` and `tmoxe3x5`)
    change the package's on-ground status and also move the vehicle in a
    different (kyvrixb5) graph (`do9fmoyo` and `hab0vli4`). This heuristic
    deliberately simplifies the kyvrixb5 movement and focuses on movement in
    the x6ibpitl location graph plus the required load/unload actions.

    Assumptions
    -----------
    - The only practical way to move a package between x6ibpitl locations is
      to have a vehicle (hctyyy8k) pick it up (load) and later drop it (unload).
    - Vehicle movement across x6ibpitl locations uses `arsslvyt` and follows the
      adjacency graph given by static facts `(miud1izi <from> <to>)`. We treat
      those as directed edges (the static input commonly contains both
      directions when two-way movement is possible).
    - We ignore the kyvrixb5-level constraints (the `do9fmoyo` / `hab0vli4`
      graph) when estimating feasibility of loads/unloads: we count load and
      unload as single actions even though the domain requires certain
      `do9fmoyo`/`hab0vli4` relationships to be present for the concrete
      operators. This makes the heuristic inadmissible but fast and effective
      in guiding greedy best-first search.
    - If no vehicle location is known in the current state for a needed
      vehicle, or if two locations are disconnected in the x6ibpitl graph,
      we use a large finite penalty so the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals).
    - Extract the static x6ibpitl adjacency graph from facts of the form
      `(miud1izi <from> <to>)` into a dictionary for fast shortest-path queries.
    - Keep the set of x6ibpitl location names that appear in the adjacency list.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For every package that appears in the goal (a goal fact is of the form
    `(sjuejzxf <package> <goal_loc>)`) compute an estimated action cost:

    1. If all goals are already satisfied in the current state, return 0.

    2. For each package goal:
       - Determine the package's current representation in the state:
         a) On the ground: there is a fact `(sjuejzxf <pkg> <loc_pkg>)`.
         b) In a vehicle: there is `(un606moq <pkg> <veh>)`.
       - Determine vehicle locations by reading `(sjuejzxf <veh> <loc_veh>)`
         facts for all vehicles that appear in `do9fmoyo` or `sjuejzxf` facts.
         If a necessary vehicle location is missing, treat distances involving
         that vehicle as very large but finite.

    3. Cost model (simple, per-package):
       - If the package is already at its goal location: cost = 0.
       - If the package is in vehicle V (at location loc_V): estimate cost as
         shortest_path(loc_V, goal_loc) + 1 (one unload action).
       - If the package is on the ground at loc_P: choose the vehicle V that
         minimizes:
             shortest_path(loc_V, loc_P)   [drive V to package]
           + 1                             [load]
           + shortest_path(loc_P, goal_loc)[drive while carrying]
           + 1                             [unload]
         So cost = d(loc_V, loc_P) + d(loc_P, goal_loc) + 2 (for load+unload).
       - Sum these estimated costs for all package goals.

    Notes
    -----
    - shortest_path is computed by BFS on the x6ibpitl adjacency graph built
      from static facts `(miud1izi ...)`. If no path exists, BFS returns a
      large finite value (PENALTY) to keep the heuristic finite.
    - The heuristic is intentionally optimistic w.r.t. kyvrixb5-level
      constraints (hab0vli4 / do9fmoyo) but still captures the dominant cost
      contributors (vehicle travel and load/unload actions) and thus helps
      reduce node expansions in practice.
    """

    PENALTY = 100000  # large finite value for unreachable or unknown situations

    def __init__(self, task):
        # Goal facts (frozenset of strings)
        self.goals = task.goals
        # Static facts (frozenset of strings)
        static_facts = task.static

        # Build directed adjacency list for x6ibpitl locations from (miud1izi from to)
        self.loc_adj = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                if a not in self.loc_adj:
                    self.loc_adj[a] = set()
                self.loc_adj[a].add(b)

        # Ensure every known location has an entry (possibly empty neighbors)
        for loc in list(self.locations):
            self.loc_adj.setdefault(loc, set())

    def bfs_distance(self, start, goal):
        """Return shortest path length from start to goal in the x6ibpitl graph.
        Returns PENALTY if start/goal unknown or unreachable.
        """
        if start == goal:
            return 0
        if start not in self.loc_adj or goal not in self.loc_adj:
            return self.PENALTY
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nbr in self.loc_adj.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
        return self.PENALTY

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # If already goal state, heuristic is 0 (check entire goal set)
        if self.goals <= state:
            return 0

        # Parse goal facts to collect packages that must end up somewhere.
        # We only consider goals of form (sjuejzxf <pkg> <goal_loc>)
        package_goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                pkg, goal_loc = parts[1], parts[2]
                package_goal_locations[pkg] = goal_loc

        # Extract current ground locations for objects (from sjuejzxf facts).
        # Map object -> current x6ibpitl location for both packages and vehicles.
        current_sjuejzxf = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                current_sjuejzxf[obj] = loc

        # Extract which packages are inside which vehicle: un606moq <pkg> <veh>
        in_vehicle = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "un606moq" and len(parts) >= 3:
                pkg, veh = parts[1], parts[2]
                in_vehicle[pkg] = veh

        # Collect vehicles known in the current state:
        # Vehicles typically appear as first arg in do9fmoyo facts as well.
        vehicles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "do9fmoyo" and len(parts) >= 3:
                vehicles.add(parts[1])
        # Also include any object that appears as a subject in sjuejzxf and also
        # is present in do9fmoyo in the static/initial description could be a
        # vehicle, but the above is usually sufficient. We'll also scan sjuejzxf
        # subjects and include them heuristically (they might be packages too,
        # but that's harmless for vehicle pool).
        for obj in current_sjuejzxf:
            vehicles.add(obj)

        # For vehicles, their x6ibpitl location is current_sjuejzxf.get(vehicle)
        # (if missing, treat as unknown).

        total_cost = 0

        # For each package that must be at a goal location, estimate cost.
        for pkg, goal_loc in package_goal_locations.items():
            # If already achieved in state, continue
            if f"(sjuejzxf {pkg} {goal_loc})" in state:
                continue

            # Is the package currently in a vehicle?
            if pkg in in_vehicle:
                veh = in_vehicle[pkg]
                veh_loc = current_sjuejzxf.get(veh)
                if veh_loc is None:
                    # Unknown vehicle location: large penalty but finite
                    total_cost += self.PENALTY
                else:
                    dist = self.bfs_distance(veh_loc, goal_loc)
                    # unload cost = 1
                    total_cost += dist + 1
                continue

            # Otherwise the package is on the ground at some x6ibpitl location
            # Try to find its ground location
            pkg_loc = current_sjuejzxf.get(pkg)
            if pkg_loc is None:
                # Unknown package location: large penalty
                total_cost += self.PENALTY
                continue

            # If package is on ground but goal is elsewhere, choose best vehicle
            best_cost = self.PENALTY
            # If there are no vehicles at all, cannot move the package with domain actions.
            if not vehicles:
                total_cost += self.PENALTY
                continue

            for veh in vehicles:
                veh_loc = current_sjuejzxf.get(veh)
                if veh_loc is None:
                    # Unknown vehicle location: treat as very expensive but consider others
                    continue
                to_pkg = self.bfs_distance(veh_loc, pkg_loc)
                pkg_to_goal = self.bfs_distance(pkg_loc, goal_loc)
                # load + unload costs = 2
                cost = to_pkg + pkg_to_goal + 2
                if cost < best_cost:
                    best_cost = cost

            # If best_cost remains PENALTY and we had vehicles but unknown positions,
            # escalate to PENALTY as well.
            total_cost += best_cost

        return int(total_cost)