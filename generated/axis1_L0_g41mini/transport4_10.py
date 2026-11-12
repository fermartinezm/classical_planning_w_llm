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
    A domain-dependent heuristic for the Transport domain.

    Summary
    -------
    This heuristic estimates the number of actions required to deliver all packages
    to their respective goal locations. The domain is characterized by vehicles 
    with varying capacities moving on a road network to pick up and drop packages.
    The heuristic considers:
    1. The minimal distances in the road network between relevant locations.
    2. The capacity constraints of vehicles limiting how many packages can be carried at once.
    3. The current locations of packages and vehicles.
    4. The goal locations for packages.

    Assumptions
    -----------
    - Vehicles may only drive along roads defined in the static facts.
    - Packages must be picked up and dropped according to vehicle capacities.
    - The heuristic ignores capacity-predecessor complexity beyond the minimal needed capacity to carry packages.
    - Loading and dropping each package costs one action.
    - Driving along one road segment costs one action.
    - The heuristic does not explicitly simulate fine-grained capacity changes during transport but uses minimal capacity needed estimates.
    - The heuristic is not guaranteed admissible but efficient and informative enough to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract the road network as an undirected graph for shortest path computations.
    - Identify all locations, vehicles, and packages.
    - Extract capacities and capacity-predecessor relationships to compute the minimal capacity needed per vehicle.
    - Extract goal locations of all packages from the goal conditions.
    - Precompute shortest paths between all pairs of locations needed for heuristic estimation using BFS.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify each package's current location:
       - If it is "at" some location, note that location.
       - If it is "in" a vehicle, use the vehicle's current location.
    2. For each package:
       - If it is already at its goal location, it requires no actions.
       - Otherwise, estimate the cost to move it from its current location to the goal location.
    3. For vehicles:
       - Account for the capacity to estimate how many trips would be needed.
       - Since capacity is given via a predecessor relation on sizes, interpret the sizes as an ordering; a vehicle with capacity size s2 > s1 can carry more packages.
    4. Estimate cost for a single package delivery as:
       - Actions to load the package (1),
       - Actions to drop the package (1),
       - Driving cost: shortest distance from package current location (or vehicle location if in vehicle) to the package goal location,
         plus optionally the distance vehicle must travel to pick it up if needed.
    5. Since vehicles can carry multiple packages simultaneously limited by capacity, approximate how many trips are needed by summing required load/unload actions and travel segments.
       The heuristic conservatively assumes packages are transported one-by-one but discounts trips by spreading over available vehicles and capacities.
    6. Finally, sum over all packages the costs computed to obtain the heuristic value.

    Note:
    - The heuristic is based on approximations: it does not simulate exact vehicle schedules or capacity changes.
    - It is always zero if the state satisfies the goal.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse road network as adjacency list (undirected)
        self.roads = {}
        self.locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.roads.setdefault(l1, set()).add(l2)
                self.roads.setdefault(l2, set()).add(l1)

        # Extract vehicles and their capacities
        self.vehicle_capacities = {}  # vehicle -> size
        self.size_predecessor = {}  # maps s1 -> s2 if capacity-predecessor (s1 s2)

        # We gather all sizes from capacity-predecessor chains
        self.sizes = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.size_predecessor[s1] = s2
                self.sizes.add(s1)
                self.sizes.add(s2)

        for fact in self.static:
            if match(fact, "capacity", "*", "*"):
                _, v, s = get_parts(fact)
                self.vehicle_capacities[v] = s
                self.sizes.add(s)

        # Compute size order to interpret capacity levels as integers for easy comparison
        # Build a ranking so we can estimate numeric capacity values by counting predecessors
        self.size_rank = {}
        def size_depth(s):
            depth = 0
            curr = s
            while curr in self.size_predecessor:
                curr = self.size_predecessor[curr]
                depth += 1
            return depth
        for s in self.sizes:
            self.size_rank[s] = size_depth(s)

        # Extract all locatables: packages and vehicles
        self.packages = set()
        self.vehicles = set()
        for fact in task.facts:
            # facts from task.facts are strings like 'p1' or 'v1'
            # But types are not given here; we infer from usage in state
            pass
        # Instead, infer packages and vehicles from initial state facts
        # Packages appear in (at ?p ?l) or (in ?p ?v) where ?p is package, ?v vehicle
        self.packages = set()
        self.vehicles = set()
        for fact in task.initial_state | task.goals | self.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at":
                obj = parts[1]
                if obj.startswith("p"):  # crude heuristic: package names start with p
                    self.packages.add(obj)
                if obj.startswith("v"):  # crude heuristic: vehicle names start with v
                    self.vehicles.add(obj)
            elif pred == "in":
                obj1 = parts[1]
                obj2 = parts[2]
                if obj1.startswith("p"):
                    self.packages.add(obj1)
                if obj2.startswith("v"):
                    self.vehicles.add(obj2)

        # Extract goals for each package: package -> goal location
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, p, l = get_parts(goal)
                self.goal_locations[p] = l

        # Precompute all shortest path distances between all pairs of locations for heuristic queries
        # Using BFS on the road graph
        self.shortest_paths = {}
        for loc in self.locations:
            self.shortest_paths[loc] = self._bfs_shortest_paths(loc)

    def _bfs_shortest_paths(self, start):
        """Compute shortest distance from start to all reachable locations as dictionary."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            for neighbor in self.roads.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Estimate the number of actions needed to deliver all packages to their goal locations.

        The heuristic value is 0 if the goal is reached.
        Otherwise, for each undelivered package, estimate:
        - Driving cost to pick it up (if not in vehicle),
        - Driving cost to delivery location (from vehicle or package location),
        - Actions to pick up and drop (1 each),
        Considering vehicle capacities to estimate minimal number of trips.
        """
        state = node.state

        if self.goals <= state:
            return 0

        # Get current location of each vehicle
        vehicle_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc

        # Get locations of packages: either at location or in vehicle, in which case package location = vehicle location
        package_locations = {}
        for p in self.packages:
            package_locations[p] = None  # initialize

        # For packages "at" location
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.packages:
                    package_locations[obj] = loc

        # For packages "in" vehicles, location = vehicle location
        for fact in state:
            if match(fact, "in", "*", "*"):
                p, v = get_parts(fact)[1:]
                if p in self.packages and v in vehicle_locations:
                    package_locations[p] = vehicle_locations[v]

        # Count how many packages each vehicle carries
        vehicle_package_counts = {v: 0 for v in self.vehicles}
        for fact in state:
            if match(fact, "in", "*", "*"):
                p, v = get_parts(fact)[1:]
                if v in vehicle_package_counts:
                    vehicle_package_counts[v] += 1

        # Now compute minimal capacity (as integer rank) each vehicle currently has (capacity from state)
        current_vehicle_capacity = {}
        for v in self.vehicles:
            # Extract capacity fact for vehicle v in this state
            # We search for (capacity v s) in state
            cap_size = None
            for fact in state:
                if match(fact, "capacity", v, "*"):
                    cap_size = get_parts(fact)[2]
                    break
            if cap_size:
                current_vehicle_capacity[v] = self.size_rank.get(cap_size, 0)
            else:
                current_vehicle_capacity[v] = 0  # no capacity known (should not happen)

        # For each package not at goal location, estimate cost to deliver
        total_actions_estimate = 0

        # Helper: driving cost between locations (returns large penalty if no path)
        def driving_cost(loc1, loc2):
            if loc1 == loc2:
                return 0
            dist_map = self.shortest_paths.get(loc1)
            if dist_map is None:
                # loc1 isolated; penalize heavily
                return 1000
            dist = dist_map.get(loc2)
            if dist is None:
                # no road path from loc1 to loc2
                return 1000
            return dist  # driving cost = number of roads traversed

        # Aggregate packages and vehicles for distribution of trips:
        # We approximate that packages are individually handled ignoring complex loading orders,
        # but discount required drive actions by assuming each vehicle can carry up to capacity packages at once.

        # Sum of all required pickups, drops and drive costs for packages individually:
        indiv_package_action_costs = []

        for p in self.packages:
            goal_loc = self.goal_locations.get(p)
            if goal_loc is None:
                # Package not in goal (e.g., no specified goal), skip it
                continue

            current_loc = package_locations.get(p)
            if current_loc is None:
                # Unknown location, penalize heavily
                total_actions_estimate += 1000
                continue

            if current_loc == goal_loc:
                # Already at goal, no cost needed for this package
                continue

            # Cost components:
            # - If package is "in" a vehicle, no pickup needed, else pickup needed.
            # - Drop is always needed.
            # - Driving costs:
            #   * If package is at location, vehicle must go there to pickup,
            #   * Then drive to goal location to drop.

            # Determine if package is in vehicle (by checking facts)
            package_in_vehicle = False
            for fact in state:
                if match(fact, "in", p, "*"):
                    package_in_vehicle = True
                    break

            # Determine vehicle to which package is in (if any)
            vehicle_for_p = None
            for fact in state:
                if match(fact, "in", p, "*"):
                    _, _, vehicle_for_p = get_parts(fact)
                    break

            # Driving cost to get package (if on location)
            # If package is in vehicle, pickup cost zero plus vehicle is already carrying it.
            # So:
            if package_in_vehicle and vehicle_for_p:
                # package is already loaded - no pickup required
                pickup_cost = 0
                # driving cost is from vehicle location to package goal location
                veh_loc = vehicle_locations.get(vehicle_for_p)
                dc = driving_cost(veh_loc, goal_loc)
            else:
                # package at location, pickup needed (1 action)
                pickup_cost = 1

                # We estimate the minimal driving cost for some vehicle to drive to package location to pick it up,
                # and then to deliver package to goal location.

                # Find vehicle closest to package current location:
                min_drive_to_package = min(
                    (driving_cost(vehicle_locations[v], current_loc), v) for v in self.vehicles if v in vehicle_locations
                ) if self.vehicles else (1000, None)

                if min_drive_to_package[1] is None:
                    # No vehicle available (should not happen in solvable states)
                    total_actions_estimate += 1000
                    continue

                drive_to_package = min_drive_to_package[0]

                # Driving cost to goal from package location
                drive_to_goal = driving_cost(current_loc, goal_loc)

                dc = drive_to_package + drive_to_goal

            # Drop cost always 1
            drop_cost = 1

            # Sum action estimate for this package (pickup, drop, drive)
            cost_for_p = pickup_cost + drop_cost + dc
            indiv_package_action_costs.append(cost_for_p)

        if not indiv_package_action_costs:
            # No packages need delivery
            return 0

        # Now approximate reduction of trips by total carrying capacity:
        # Total capacity available: sum capacities of all vehicles minus packages currently carried (since they take capacity slots)
        total_capacity = 0
        capacity_used = 0
        for v in self.vehicles:
            cap = current_vehicle_capacity.get(v, 0)
            total_capacity += cap
            capacity_used += vehicle_package_counts.get(v, 0)
        free_capacity = max(total_capacity - capacity_used, 1)  # at least 1 to avoid division by zero

        # Sum costs of all packages and adjust assuming parallel transport by capacity
        # Divide sum by free_capacity to approximate number of trips
        total_cost = sum(indiv_package_action_costs) / free_capacity

        # Since heuristic value is expected to be integral number of actions, round up
        total_cost = int(total_cost + 0.9999)  # ceil approximation

        # Return finite heuristic cost
        return total_cost
