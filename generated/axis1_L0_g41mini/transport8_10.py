from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards '*'.

    :param fact: fact string like "(at p1 l1)"
    :param args: pattern to match, e.g., ("at", "*", "l1")
    :return: True if matches, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Summary
    -------
    The heuristic estimates the number of actions required to deliver all packages
    to their goal locations. It returns zero if the goal is already reached.

    Assumptions
    -----------
    - Vehicles can carry packages limited by capacity sizes; the capacity hierarchy is given.
    - Packages must be picked up at their current location and dropped at their goal location.
    - Vehicles move only along predefined roads between locations.
    - Multiple vehicles and packages may exist.
    - Picking up and dropping packages consume one action each.
    - Driving between two adjacent locations consumes one action.
    - Capacity sizes restrict how many packages a vehicle can carry simultaneously.
      The heuristic assumes each package requires one size unit.
    - The heuristic is not necessarily admissible but tries to be accurate and efficient.

    Heuristic Initialization
    ------------------------
    During initialization:
    - Extract map of adjacency between locations from static road facts to form a bidirectional graph.
    - Extract and store capacity-predecessor relationships to form a size hierarchy graph.
      This allows comparing capacities or knowing if one capacity is smaller than another.
    - Precompute shortest distances between all pairs of locations using BFS on the road graph to enable quick travel cost estimations.
    - Extract goal locations of all packages from the goal conditions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If the goal is reached (all packages at goal locations), return 0.

    2. For each package:
       - Find current location:
         * If the package is in a vehicle, find the vehicle's location instead.
         * Otherwise, the package location is its own location.

       - Find the goal location.

       - Estimate the minimal number of drive actions needed to get from current location
         (or vehicle location if carried) to the goal location using precomputed shortest paths.

       - Estimate pick-up/drop cost:
         * If the package is carried by a vehicle and at the goal, drop needed (1 action).
         * If package is not carried and not at the goal, pick-up (1 action) + drop at goal (1 action).
         * If package is carried but not at the goal, assume the package needs to be transported and then dropped at goal.

       - Sum these actions for all packages.

    3. Capacity constraints are approximated by counting packages currently in the same vehicle:
       - Vehicles might need multiple trips if their capacity is insufficient.
       - The heuristic penalizes transporting packages one-by-one if necessary, by estimating extra pick/drop costs accordingly.
       - Because arbitrary capacity sizes exist, the heuristic assumes the worst case that all packages must be transported individually if they don't fit simultaneously.

    4. Sum over all packages:
       - The heuristic value is the sum of estimated drive steps + pick-up + drop actions required.

    Efficiency considerations:
    - Precomputing shortest distances avoids costly recalculation.
    - Using BFS on small graphs for shortest paths is efficient.
    - This heuristic uses only sets and dictionaries and avoids complex search during evaluation.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build road adjacency graph (undirected) from static facts "road lX lY"
        self.roads = defaultdict(set)
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.roads[l1].add(l2)
                # Also add reverse edge if not already present
                self.roads[l2].add(l1)

        # Extract capacity predecessor relations and build capacity orders
        # Create a graph of size order: c0 < c1 < c2 < ...
        self.capacity_pred = defaultdict(set)  # c1 -> {c0}, means c0 predecessor of c1
        self.capacity_succ = defaultdict(set)  # c0 -> {c1}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, c0, c1 = get_parts(fact)
                self.capacity_pred[c1].add(c0)
                self.capacity_succ[c0].add(c1)

        # Compute all capacity predecessors for each size to quickly test if size_a <= size_b
        # We'll store a mapping: for each size, all sizes <= it
        self.size_less_equal = {}

        def compute_less_equal(size):
            # Recursive DFS to find all sizes <= current size
            visited = set()

            def dfs(s):
                if s in visited:
                    return
                visited.add(s)
                for pred in self.capacity_pred.get(s, set()):
                    dfs(pred)

            dfs(size)
            return visited

        # Collect all sizes from facts
        all_sizes = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*") or match(fact, "capacity", "*", "*"):
                parts = get_parts(fact)
                all_sizes.update(parts[1:])

        for size in all_sizes:
            self.size_less_equal[size] = compute_less_equal(size)
            self.size_less_equal[size].add(size)  # size <= size

        # Extract goal location for each package from goals (at p l)
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                pkg = parts[1]
                loc = parts[2]
                self.package_goals[pkg] = loc

        # Precompute shortest distances between all locations using BFS
        self.locations = list(self.roads.keys())

        self.distance_cache = {}  # (loc1, loc2) -> distance (int)

        def bfs(start):
            dist = {start: 0}
            queue = deque([start])
            while queue:
                loc = queue.popleft()
                for neigh in self.roads[loc]:
                    if neigh not in dist:
                        dist[neigh] = dist[loc] + 1
                        queue.append(neigh)
            return dist

        # Store distances from each location to all others
        self.shortest_paths = {loc: bfs(loc) for loc in self.locations}

    def __call__(self, node):
        state = node.state

        # Quick goal check: heuristic = 0 if goal reached
        if self.goals <= state:
            return 0

        # Helper to find current vehicle capacities: capacity(v) = size
        vehicle_capacity = {}
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, v, size = get_parts(fact)
                vehicle_capacity[v] = size

        # Helper to get capacity order check: size_a <= size_b?
        def capacity_leq(size_a, size_b):
            # Return True if size_a is <= size_b on the capacity order
            if size_b not in self.size_less_equal:
                return False
            return size_a in self.size_less_equal[size_b]

        # Get current package locations:
        # (at p l) or (in p v)
        package_locations = {}
        # Get vehicle locations: (at v l)
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj = parts[1]
                loc = parts[2]
                if obj.startswith("v"):  # vehicle
                    vehicle_locations[obj] = loc
                else:
                    # can be package or vehicle
                    # But packages carried inside vehicles use (in ...)
                    package_locations[obj] = loc
            elif parts[0] == "in":
                pck = parts[1]
                veh = parts[2]
                package_locations[pck] = veh

        # Count how many packages are currently inside each vehicle
        packages_in_vehicle = defaultdict(set)
        for pck, loc in package_locations.items():
            if loc in vehicle_locations:
                packages_in_vehicle[loc].add(pck)

        # Estimate heuristic costs
        total_cost = 0

        for pkg, goal_loc in self.package_goals.items():
            if pkg not in package_locations:
                # Assume worst case if location unknown: one pick + one drop + some driving
                total_cost += 3
                continue
            curr_loc = package_locations[pkg]

            # Check if package at goal location:
            if curr_loc == goal_loc:
                # Already at goal, no action needed
                continue

            # Determine where the package effectively is located for movement:
            if curr_loc in vehicle_locations:
                # package inside vehicle, so package moves with vehicle
                vehicle = curr_loc
                vehicle_loc = vehicle_locations[vehicle]
                distance = self.shortest_paths.get(vehicle_loc, {}).get(goal_loc, None)
                if distance is None:
                    # No path known, assign high cost
                    distance = 10  # arbitrary penalty
                # Costs:
                # drive distance steps + drop (1 action)
                # Assuming already picked up before
                total_cost += distance + 1
            else:
                # package on the ground
                package_loc = curr_loc
                distance_to_package = 0
                # In transport domain, the vehicle must drive to package location to pick it up.

                # Find the closest vehicle that can carry this package
                # We approximate by taking the minimal distance between package_loc and any vehicle location that can carry it:
                best_vehicle_dist = None
                best_vehicle = None
                package_size_assumed = "c1"  # We do not have explicit package sizes; assume c1

                for v, v_loc in vehicle_locations.items():
                    # Check vehicle can carry the package size:
                    v_cap = vehicle_capacity.get(v, None)
                    if v_cap is None:
                        continue
                    if not capacity_leq(package_size_assumed, v_cap):
                        continue
                    dist_to_package = self.shortest_paths.get(v_loc, {}).get(package_loc, None)
                    if dist_to_package is None:
                        continue
                    if best_vehicle_dist is None or dist_to_package < best_vehicle_dist:
                        best_vehicle_dist = dist_to_package
                        best_vehicle = v

                if best_vehicle_dist is None:
                    # No suitable vehicle reachable; penalize heavily
                    best_vehicle_dist = 10

                # Distance vehicle to package location + pick-up (1) +
                # drive from package location to goal location +
                drive_to_goal = self.shortest_paths.get(package_loc, {}).get(goal_loc, None)
                if drive_to_goal is None:
                    drive_to_goal = 10

                # drop at goal (1)
                total_cost += best_vehicle_dist + 1 + drive_to_goal + 1

        # Add heuristic penalty if some vehicles carry packages exceeding capacity (approximation)
        for v, pkgs in packages_in_vehicle.items():
            v_cap = vehicle_capacity.get(v, None)
            if not v_cap:
                # Unknown capacity, ignore
                continue
            # Assume each package requires capacity unit c1
            # Count how many packages fit in v_cap by going down capacity order
            # We'll count capacity "levels" roughly by number of c1 steps: count minimal capacity levels
            # Here, we just approximate by number of packages exceeding 1 increments
            max_packages = 0
            # Count how many capacity levels above c0 equals to c1, c2, etc.
            # Let's build capacity size to number of c1 units for heuristic:
            cap_levels_ordered = sorted(self.size_less_equal.keys(), key=lambda s: len(self.size_less_equal[s]))
            # We approximate capacity as number of steps above c0
            def capacity_level(size):
                # Assumes chain c0 < c1 < c2 < ...
                # The level corresponds to how many predecessors before c0
                level = 0
                visited = set()
                current = size
                while True:
                    preds = self.capacity_pred.get(current, set())
                    if not preds:
                        break
                    current = next(iter(preds))
                    if current in visited:
                        break
                    visited.add(current)
                    level += 1
                return level

            max_packages = capacity_level(v_cap)

            if len(pkgs) > max_packages:
                # Penalize one extra pick-up and drop per exceeded package
                excess = len(pkgs) - max_packages
                total_cost += excess * 2  # pick + drop

        return total_cost
