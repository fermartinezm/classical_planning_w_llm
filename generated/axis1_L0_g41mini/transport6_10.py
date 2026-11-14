from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by trimming parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches the given pattern with wildcards.

    Parameters:
    - fact: a string representing a PDDL fact e.g. "(at p1 l1)"
    - args: pattern elements to match e.g. "at", "*", "l1"

    Returns:
    - True if fact matches the pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the `transport` domain.

    Summary:
    This heuristic estimates the number of actions required to deliver all packages
    to their goal locations. It considers the cost of driving vehicles along roads,
    picking up packages, and dropping them at goal locations, while respecting vehicle
    capacity constraints in a simplified manner. The heuristic is designed to be 
    efficiently computable and provides a reasonable estimate, guiding a greedy 
    best-first search.

    Assumptions:
    - The cost to move between two adjacent locations (road) is 1 action.
    - Picking up or dropping a package costs 1 action each.
    - Vehicle capacity constraints are considered in a relaxed manner: 
      the heuristic does not fully simulate loading sequences but factors capacity roughly 
      by counting distinct vehicles needed to carry all packages.
    - The domain is connected (every location reachable via roads).
    - Roads are bidirectional (as seen in example instances).
    
    Heuristic Initialization:
    - Extract road connectivity to build a graph over locations.
    - Precompute shortest road distances between all locations using Floyd-Warshall.
    - Extract the capacity size hierarchy for vehicles (capacity-predecessor) to understand capacity levels.
    - Extract initial vehicle sizes and static capacity relationships.
    - Extract goal destinations for all packages.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. Identify current location and loaded status (in which vehicle or at a location) of each package.
    2. For each package not yet at its goal location:
       a) Determine its current location (on ground or inside a vehicle).
       b) Compute the shortest driving distance to its goal location. 
          - If inside a vehicle, get that vehicle's location.
          - If on ground, use package location directly.
       c) Add:
          - 1 action for drop if package is inside a vehicle at its goal,
          - or 
          - 1 action to pick up if package is on ground at its location (assuming vehicle will pick it),
          - plus driving cost from vehicle/package location to goal location.
          - plus 1 action to drop at goal.
          (If package is already at goal on ground, cost is 0.)
    3. Approximate capacity constraints by:
       - Counting sum of package sizes to be transported.
       - Counting sum of vehicle capacities.
       - Estimating minimum number of vehicle loads needed.
    4. Combine the above estimates:
       - Total driving cost to move vehicles to pick up and deliver packages.
       - Number of pick/drop actions proportional to number of packages.
       - A small penalty for packages requiring multiple loads because of capacity limits.
    5. Return the sum as the heuristic value.

    This heuristic is not guaranteed to be admissible (it may overestimate) but aims to 
    provide a strong guidance for the search.

    """

    def __init__(self, task):
        """Extract static information and precompute location distances and capacity relations."""
        self.goals = task.goals  # goal facts
        self.static = task.static  # static facts never changed by actions

        # Extract all locations and build adjacency graph from roads
        self.locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:3]
                self.locations.add(l1)
                self.locations.add(l2)

        self.locations = list(self.locations)
        self.loc_index = {loc: i for i, loc in enumerate(self.locations)}

        n = len(self.locations)
        # Initialize distance matrix with infinity except zero on diagonal
        INF = 10**9
        self.dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            self.dist[i][i] = 0

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:3]
                i1, i2 = self.loc_index[l1], self.loc_index[l2]
                self.dist[i1][i2] = 1  # each road costs 1 action to move
                # Assume roads are bidirectional as in examples:
                self.dist[i2][i1] = 1

        # Floyd-Warshall to compute all-pairs shortest path distances between locations
        for k in range(n):
            dist_k = self.dist[k]
            for i in range(n):
                dist_i = self.dist[i]
                ik = dist_i[k]
                if ik == INF:
                    continue
                for j in range(n):
                    kj = dist_k[j]
                    new_dist = ik + kj
                    if new_dist < dist_i[j]:
                        dist_i[j] = new_dist

        # Extract capacity-predecessor relationships in a dictionary {higher_size: lower_size}
        self.cap_predecessor = {}
        self.size_levels = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                small, large = get_parts(fact)[1:3]
                self.cap_predecessor[large] = small
                self.size_levels.add(small)
                self.size_levels.add(large)

        # We'll generate capacity levels by links up from smallest
        # Build reverse map to get size order (lowest to highest)
        # Detect minimal size (no predecessor pointing to it)
        all_sizes = set(self.cap_predecessor.keys()) | set(self.cap_predecessor.values())
        # Find minimal size: one that does not appear as a large size
        min_sizes = all_sizes - set(self.cap_predecessor.keys())

        # Map sizes to numeric levels, 0 = smallest
        self.size_to_level = {}
        level = 0
        queue = list(min_sizes)
        # Simple iterative assign levels by following capacity-predecessor downwards
        while queue:
            next_queue = []
            for s in queue:
                if s not in self.size_to_level:
                    self.size_to_level[s] = level
                    # Find all sizes that have s as predecessor (i.e., s -> successor)
                    successors = [large for large, small in self.cap_predecessor.items() if small == s]
                    next_queue.extend(successors)
            queue = next_queue
            level += 1

        # Extract vehicles with their current capacities (sizes)
        self.vehicle_sizes = {}  # vehicle -> level of capacity
        for fact in self.static:
            if match(fact, "capacity", "*", "*"):
                v, s = get_parts(fact)[1:3]
                self.vehicle_sizes[v] = self.size_to_level.get(s, -1)

        # Store package and vehicle types to distinguish locatables if needed
        self.vehicles = set(self.vehicle_sizes.keys())

        # Extract goal locations for packages from task.goals
        self.package_goal = {}
        for fact in self.goals:
            if match(fact, "at", "*", "*"):
                package, goal_loc = get_parts(fact)[1:3]
                self.package_goal[package] = goal_loc

    def __call__(self, node):
        """
        Compute heuristic estimate of minimum actions needed to achieve goal from current state.

        Method:
        - For each package not at goal:
            - Determine current position (location or vehicle)
            - Get vehicle location if loaded, or package location if on ground
            - Compute shortest distance to goal location
            - Add 1 pick-up if package on ground (except if at goal)
            - Add 1 drop if package at vehicle (except if vehicle is at goal)
            - Add 1 drop at goal location
        - Approximate capacity constraints by counting total package sizes and vehicles,
          scale the heuristic accordingly to reflect multiple loads needed if capacity is tight.
        """

        state = node.state

        # Helper to extract facts by predicates quickly:
        # Map: object -> current location or vehicle for packages
        package_locations = {}
        # Map: vehicle -> location
        vehicle_locations = {}

        # Map: package -> size level; assume every package has size c0 (smallest) because domain never states package sizes explicitly.
        # We'll treat package size as level 0 (smallest) consistently.
        # Reasoning: no explicit package sizes, we take minimal size = 0 for all.
        PACKAGE_SIZE_LEVEL = 0

        # Find all (at p l) and (in p v) for packages
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.package_goal:
                    package_locations[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif pred == "in":
                pck, veh = parts[1], parts[2]
                if pck in self.package_goal:
                    package_locations[pck] = veh
                elif veh in self.vehicles:
                    # Defensive: in this domain only packages go in vehicles
                    pass

            elif pred == "capacity":
                # Defensive: vehicle location can also be extracted if needed here,
                # but vehicle_locations extracted from 'at' is sufficient.
                pass

        INF = 10**9

        # Compute heuristic sum
        heuristic_cost = 0

        # Count sum of package sizes to be transported
        sum_package_sizes = 0

        # Track number of vehicles and their capacity levels
        vehicle_capacity_levels = list(self.vehicle_sizes.values())
        if not vehicle_capacity_levels:
            # Defensive fallback: if no vehicles known, heuristic 0 if goals met else large cost
            if self.goals <= state:
                return 0
            else:
                return 1000

        total_vehicle_capacity_level_sum = sum(vehicle_capacity_levels)

        for package, goal_loc in self.package_goal.items():
            # Check if goal already achieved for the package
            # Package is at goal on ground if (at package goal_loc) in state
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                # Package at goal on ground; cost 0
                continue

            sum_package_sizes += PACKAGE_SIZE_LEVEL  # all packages are minimal size 0

            current_pos = package_locations.get(package, None)
            if current_pos is None:
                # Package not tracked in state; assume must be at original location, safe estimate large cost
                heuristic_cost += 10
                continue

            # Determine package location type:
            # If current_pos in vehicles, it's in a vehicle; else location
            in_vehicle = current_pos in self.vehicles

            # Determine where the package currently "is" at ground level for distance calculation:
            if in_vehicle:
                # Vehicle location needed for distance
                vehicle_loc = vehicle_locations.get(current_pos, None)
                if vehicle_loc is None:
                    # Vehicle location unknown, penalize
                    dist = 5
                else:
                    dist = self._distance(vehicle_loc, goal_loc)
            else:
                # Package on ground
                dist = self._distance(current_pos, goal_loc)

            # If distance is infinite (no path), assign large cost
            if dist >= INF:
                dist = 10

            # Actions:
            # If in vehicle and vehicle at goal -> 1 drop to put package on ground
            # If in vehicle and vehicle NOT at goal -> heuristic includes driving from vehicle loc to goal
            # and drop at goal (1 action)
            # If on ground and not at goal -> pick-up (1), drive distance, drop (1)

            if in_vehicle:
                # vehicle loc to goal distance included above
                heuristic_cost += dist + 1  # drop at goal

            else:
                # package on ground: pick-up required + drive + drop
                heuristic_cost += 1 + dist + 1

        # Approximate capacity implications:

        # Because all package sizes are 0 (smallest), capacity constraints are not restricting
        # Under more complex instances with multiple sizes, can factor:
        # Estimate number of vehicle loads required:
        # Let max capacity level be max(self.vehicle_sizes.values())
        # Here minimal size 0 means max capacity level vehicle can carry multiple packages,
        # so we apply a heuristic penalty if sum_package_sizes > total_vehicle_capacity_level_sum

        # Penalty factor for capacity overflow:
        if sum_package_sizes > total_vehicle_capacity_level_sum:
            overload = sum_package_sizes - total_vehicle_capacity_level_sum
            heuristic_cost += 5 * overload  # penalty for needing multiple loads

        # If all goals achieved, heuristic 0
        if self.goals <= state:
            return 0

        # Return heuristic cost (rounded or truncated to int)
        return int(heuristic_cost)

    def _distance(self, loc1, loc2):
        """Return shortest distance between two locations, or large value if unreachable."""
        i1 = self.loc_index.get(loc1, None)
        i2 = self.loc_index.get(loc2, None)
        if i1 is None or i2 is None:
            return 10**9
        return self.dist[i1][i2]
