from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain that estimates the number of actions
    required to deliver all packages to their goal locations.

    Summary:
    --------
    The heuristic estimates the minimum number of actions needed to reach a goal state from a given state.
    It takes into account the spatial distribution of packages and vehicles, the road connectivity between locations,
    package loading and unloading actions, and vehicle capacity constraints.

    Assumptions:
    ------------
    - Vehicles can move only along roads between locations (edges in a graph).
    - Packages must be either at a location or inside a vehicle.
    - Vehicles have a capacity ordered by "capacity-predecessor" facts. A vehicle can carry multiple packages
      as long as capacity constraints allow (this is modeled through a size hierarchy; picking up a package
      decreases capacity, dropping increases it).
    - The heuristic is not guaranteed to be admissible but aims to be informative and efficient.
    - Distance between locations is approximated by shortest path lengths on the road graph.
    - Capacity constraints are considered only roughly (using capacity sizes and counting number of packages per vehicle).
    - Ignoring complex interactions between package sizes (assumed uniform for heuristic).

    Heuristic Initialization:
    -------------------------
    - Extracts the road network from static facts and builds an undirected graph of locations.
    - Uses a BFS shortcut to quickly estimate shortest path lengths between any pair of locations.
    - Extracts the capacity-predecessor ordering to reconstruct size hierarchy.
    - Extracts initial capacities of each vehicle (represented as size levels).
    - Extracts the goal destination for each package (from the provided goal facts).
      Packages are identified by their names and types.
    - Prepares data structures to quickly access package status and vehicle capacities during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If the current state is a goal state, return 0.
    2. For each package, determine:
       - Current location: either at a location or inside a vehicle.
       - Goal location: from the goal conditions.
       - Current vehicle (if inside one) and that vehicle's location.
    3. Compute the shortest path distance between locations using the road graph.
    4. For packages not at their goal location, estimate:
       a) Actions needed to move the package from its current position to its goal position.
          This includes:
          - Driving actions to move the vehicle closer to package or towards destination.
          - Pick-up and drop actions needed (usually 2 actions: pick-up and drop).
       b) If the package is inside a vehicle but the vehicle is not at the goal location, also consider vehicle driving cost.
       c) If the package is not yet picked up, consider the vehicle moving to the package location, picking up, moving to the goal location, and dropping.
    5. For each vehicle, approximate capacity used by counting packages currently in it.
       - If capacity is exceeded (heuristic overestimate), add penalty actions for multiple trips.
    6. Accumulate all such estimated costs for all packages.
    7. Return the summed estimate as the heuristic value.

    This heuristic provides a well-informed guidance balancing planning costs between vehicle routing,
    package pick-up/drop, and capacity constraints.
    """

    def __init__(self, task):
        """
        Extract static information:
        - Build a road graph of all locations for shortest path lookup.
        - Extract capacity-predecessor relations to rank capacity sizes.
        - Extract initial vehicle capacities.
        - Map each package to its goal location (from goal facts).
        """
        self.goals = task.goals
        self.static = task.static
        self.vehicles = set()
        self.packages = set()
        self.locations = set()
        self.size_order = []  # From smallest to largest size
        self.capacity_predecessor = {}
        self.vehicle_capacities = {}  # vehicle -> size index
        self.goal_locations = {}  # package -> goal location
        self.road_graph = {}  # location -> set of neighbor locations

        # Identify all relevant objects from facts.
        for fact in task.facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "capacity-predecessor":
                # capacity-predecessor c0 c1 means c0 < c1
                smaller, larger = parts[1], parts[2]
                self.capacity_predecessor[larger] = smaller
            elif pred == "capacity":
                vehicle, size = parts[1], parts[2]
                self.vehicles.add(vehicle)
            elif pred == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.update([l1, l2])

        # Build capacity size order based on capacity-predecessor chain
        # Find smallest size(s) (without predecessor)
        sizes_with_pred = set(self.capacity_predecessor.keys())
        all_sizes = set()
        for _, size in [get_parts(fact)[1:] for fact in self.static if fact.startswith("(capacity ")]:
            all_sizes.add(size)
        all_sizes.update(self.capacity_predecessor.keys())
        all_sizes.update(self.capacity_predecessor.values())

        # Try to construct order by following predecessors repeatedly
        # We reverse it so index=0 is smallest size, index=max is largest capacity
        def build_size_order():
            # find sizes without predecessor (roots)
            roots = all_sizes - sizes_with_pred
            order = []
            visited = set()

            # Follow chain from each root forward by successors (reverse mapping)
            successor_map = {}
            for k, v in self.capacity_predecessor.items():
                successor_map.setdefault(v, []).append(k)

            def dfs(size):
                if size in visited:
                    return
                visited.add(size)
                order.append(size)
                for suc in successor_map.get(size, []):
                    dfs(suc)

            for r in roots:
                dfs(r)
            return order

        self.size_order = build_size_order()
        self.size_to_idx = {size: idx for idx, size in enumerate(self.size_order)}

        # Map vehicle to current capacity index
        for fact in self.static:
            if fact.startswith("(capacity "):
                parts = get_parts(fact)
                vehicle, size = parts[1], parts[2]
                if vehicle in self.vehicles:
                    self.vehicle_capacities[vehicle] = self.size_to_idx[size]

        # Build undirected road graph for shortest path computations
        self.road_graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            if fact.startswith("(road "):
                _, l1, l2 = get_parts(fact)
                self.road_graph[l1].add(l2)
                # The domain seems to allow two-way roads as separate facts,
                # but we add both directions to be robust.
                self.road_graph[l2].add(l1)

        # Extract goal package locations (only "at" goals for packages)
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "at":
                package, loc = parts[1], parts[2]
                self.goal_locations[package] = loc
                self.packages.add(package)

        # Some packages and vehicles may not be in goals, find from facts:
        for fact in task.facts:
            if fact.startswith("(at "):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                # We classify package or vehicle by types known from domain types
                # Domain types known: vehicle and package are locatable
                # We assume naming conventions from instance file:
                if obj.startswith("p"):  # package
                    self.packages.add(obj)
                elif obj.startswith("v"):  # vehicle
                    self.vehicles.add(obj)
                self.locations.add(loc)

        # Precompute shortest paths between all pairs of locations with BFS for efficiency in heuristic
        # Store distances in a dict of dicts: distance[loc1][loc2] = shortest path length
        self.distance = {}
        for loc in self.locations:
            self.distance[loc] = self._bfs_shortest_paths(loc)

    def _bfs_shortest_paths(self, start):
        """Compute shortest path distances from start to all reachable locations using BFS."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def _capacity_index(self, capacity_fact):
        """Translate capacity size to index (smaller index = smaller capacity)."""
        return self.size_to_idx.get(capacity_fact, -1)

    def _increase_capacity(self, capacity_idx):
        """
        Return the next larger capacity index or None if at max capacity.
        This according to capacity-predecessor relation reversed.
        """
        if capacity_idx + 1 < len(self.size_order):
            return capacity_idx + 1
        return None

    def __call__(self, node):
        """
        Compute heuristic estimate of needed actions to reach goal.

        Algorithm:
        - Return 0 if goal reached.
        - For each package:
          * If at goal location: cost=0.
          * Else:
            - Determine current position: at location or inside vehicle.
            - If inside vehicle:
              + Add cost to drive vehicle to package goal location (shortest distance).
              + Add cost for drop action (1).
            - If at location (not in vehicle):
              + Add cost for vehicle to drive to package location.
              + Add cost for pick-up (1).
              + Add cost for vehicle to drive to goal location.
              + Add cost for drop (1).
        - Add cost penalties if vehicle capacity exceeded (assumes packages size=1, capacity indexed by size order).
        - Sum all package costs as heuristic value.

        The heuristic ignores detailed interaction between multiple packages in one vehicle beyond capacity checks,
        which makes it efficient yet informative.
        """

        state = node.state

        # Return 0 if goal reached
        if self.goals <= state:
            return 0

        # Parse current package locations and vehicle locations, capacities, and which packages are in which vehicle
        package_pos = {}  # package -> location or vehicle
        vehicle_pos = {}  # vehicle -> current location
        packages_in_vehicle = {v: [] for v in self.vehicles}
        vehicle_capacity_idx = {}

        # Extract package location info (at or in)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.packages:
                package_pos[parts[1]] = parts[2]
            elif parts[0] == "in" and parts[1] in self.packages:
                package_pos[parts[1]] = parts[2]

        # Extract vehicle locations and their capacities (capacity facts)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in self.vehicles:
                vehicle_pos[parts[1]] = parts[2]
            elif parts[0] == "capacity" and parts[1] in self.vehicles:
                vehicle_capacity_idx[parts[1]] = self.size_to_idx.get(parts[2], -1)

        # Associate packages inside vehicles
        for pkg, location in package_pos.items():
            if location in self.vehicles:
                packages_in_vehicle[location].append(pkg)

        total_cost = 0

        # Helper constants for action costs
        DRIVE_COST = 1   # cost for one drive action between adjacent locations
        PICKUP_COST = 1  # cost for pick-up action
        DROP_COST = 1    # cost for drop action

        # For each package, compute estimated cost to bring it to goal location
        for package in self.packages:
            goal_loc = self.goal_locations.get(package)
            if not goal_loc:
                # No goal specified for this package; assume no cost
                continue

            curr_pos = package_pos.get(package)
            if not curr_pos:
                # Package location unknown means ignore (or max cost?)
                # We assign a heuristic penalty proportional to total locations
                total_cost += 5
                continue

            # Package already at goal location?
            if curr_pos == goal_loc:
                # No cost needed for this package
                continue

            if curr_pos in self.locations:
                # Package is at a location, needs to be picked up and delivered

                # Find all vehicles and compute minimal cost of delivering package with any vehicle

                min_cost_for_package = float('inf')

                for vehicle in self.vehicles:
                    v_loc = vehicle_pos.get(vehicle)
                    if v_loc is None:
                        # Vehicle location unknown: skip it
                        continue

                    # Determine if vehicle has capacity for another package
                    cap_idx = vehicle_capacity_idx.get(vehicle, -1)
                    used_capacity = len(packages_in_vehicle[vehicle])
                    # Estimate minimal capacity index after pick up:
                    # Each pickup reduces capacity size index by 1 predecessor step.
                    # But capacities represent sizes, picking up a package reduces to predecessor.
                    # Simplified: check if capacity can hold one more package assuming unit size packages.
                    # If insufficient capacity, penalize.

                    capacity_available = (cap_idx >= used_capacity)

                    # Driving cost for vehicle to reach package location
                    dist_to_pkg = self.distance.get(v_loc, {}).get(curr_pos)
                    if dist_to_pkg is None:
                        # Not reachable path, skip this vehicle
                        continue

                    # Driving cost from package location to goal location
                    dist_pkg_to_goal = self.distance.get(curr_pos, {}).get(goal_loc)
                    if dist_pkg_to_goal is None:
                        continue

                    # Total driving cost: vehicle moves to package + vehicle moves to goal
                    drive_actions = dist_to_pkg + dist_pkg_to_goal

                    # Pick-up and drop cost for package
                    pickup_drop_actions = PICKUP_COST + DROP_COST

                    # Penalize insufficient capacity by adding extra cost (simulate multiple trips)
                    capacity_penalty = 0
                    if not capacity_available:
                        # We add penalty proportional to how many packages exceed capacity
                        overload = used_capacity + 1 - cap_idx
                        capacity_penalty = overload * (pickup_drop_actions + 2)  # arbitrary penalty (> actions for one extra trip)

                    cost = drive_actions * DRIVE_COST + pickup_drop_actions + capacity_penalty

                    if cost < min_cost_for_package:
                        min_cost_for_package = cost

                if min_cost_for_package == float('inf'):
                    # No vehicle can deliver this package (not reachable), assign a penalty
                    min_cost_for_package = 20
                total_cost += min_cost_for_package

            elif curr_pos in self.vehicles:
                # Package is inside a vehicle; vehicle must drive to goal location and drop package

                v_loc = vehicle_pos.get(curr_pos)
                cap_idx = vehicle_capacity_idx.get(curr_pos, -1)
                if v_loc is None:
                    # Unknown vehicle location, penalize
                    total_cost += 20
                    continue

                dist_vehicle_to_goal = self.distance.get(v_loc, {}).get(goal_loc)
                if dist_vehicle_to_goal is None:
                    # No path, penalize
                    total_cost += 20
                    continue

                # Driving cost for vehicle plus drop action
                drive_actions = dist_vehicle_to_goal
                drop_action = DROP_COST

                total_cost += drive_actions * DRIVE_COST + drop_action

            else:
                # Package position unknown or invalid: penalize
                total_cost += 20

        return total_cost
