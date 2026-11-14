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
    This heuristic estimates the number of actions needed to deliver all packages to their goal locations.
    The Transport domain involves picking up packages with vehicles that have limited capacity and driving 
    vehicles along roads between locations.

    Assumptions
    -----------
    - Each vehicle has a capacity hierarchy represented by size constants and capacity-predecessor relations:
      when picking up a package, the vehicle's capacity size decreases to the predecessor size;
      when dropping a package, capacity increases again.
    - Roads connect locations bidirectionally as in the static facts.
    - Vehicles must drive to the package's current location to pick it up, then drive to the package's goal location to drop it.
    - Packages not yet loaded must be moved by pickup, drive, drop sequences.
    - Packages already loaded in vehicles require driving to the goal location and dropping.
    - Driving is the main cost, and the heuristic takes into account the shortest road distances between locations.
    - Capacity constraints restrict how many packages a vehicle can carry simultaneously, but the heuristic ignores complex packing order and load balancing for efficiency.
    - The heuristic does not need to be admissible but aims to give a close estimate of the number of required actions.

    Heuristic Initialization
    ------------------------
    - Extract the goal locations for all packages.
    - Extract all road connections and build a graph for locations to compute shortest path distances efficiently.
    - Map packages and vehicles to their current locations (from initial state and static facts).
    - Parse the capacity-predecessor relations to understand how vehicle capacity changes when picking/dropping packages.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each package:
      1. Find the current location of the package:
         - If it is "in" a vehicle, retrieve the location of that vehicle.
         - If it is "at" a location, use that location directly.
      2. Find the goal location of the package.
      3. Compute the shortest road distance from the vehicle location (where package currently is or in which it resides) 
         to the package's goal location.
      4. Compute approximate actions to:
         - If package is outside a vehicle:
           * Drive from vehicle location to package location (if different).
           * Pick up package (1 action).
           * Drive from package location to goal location.
           * Drop package (1 action).
         - If package is inside a vehicle:
           * Drive from current vehicle location to package goal location.
           * Drop package.
      5. Count the minimal number of drive actions according to the shortest path distances.
      6. Sum these action counts for all packages.
      
    Notes:
    - Picking up or dropping a package costs exactly 1 action each.
    - Each drive step between adjacent locations costs 1 action per road segment.
    - Heuristic is efficient since shortest distances are precomputed once.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goals."""
        self.goals = task.goals
        self.static = task.static

        # Extract road connections and build adjacency graph for locations
        self.locations = set()
        self.road_graph = {}  # location -> set of neighboring locations

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)

        # Precompute all-pairs shortest path distances with BFS for each location
        self.distance_cache = {}
        for loc in self.locations:
            self.distance_cache[loc] = self._bfs_shortest_distances(loc)

        # Extract goal locations for packages
        self.package_goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goal_locations[package] = location

        # Extract capacity predecessor relations (to understand capacity transitions)
        # Maps: capacity_successor -> capacity_predecessor
        self.capacity_pred = {}
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                self.capacity_pred[s2] = s1

    def _bfs_shortest_distances(self, start):
        """
        Compute shortest distances from `start` to all reachable locations using BFS.
        Returns a dict: location -> distance (integer)
        """
        from collections import deque

        distances = {start: 0}
        to_visit = deque([start])
        while to_visit:
            current = to_visit.popleft()
            cur_dist = distances[current]
            for neighbor in self.road_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = cur_dist + 1
                    to_visit.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of actions to reach the goal from the current state.

        Works by summing, for each package, the estimated number of pick-up, drop, and drive actions needed
        from the package's current location (or vehicle location if loaded) to its goal location.
        """
        state = node.state

        # Quick goal check: if goal facts are all satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map from facts to easy lookup
        # Extract locations of packages and vehicles
        # package_location[package] = location OR vehicle in which it is loaded (then resolve vehicle location)
        package_locations = {}
        vehicle_locations = {}

        # Extract which packages are in which vehicle
        package_in_vehicle = {}

        # Extract vehicle capacities - not used directly but could aid future extensions if needed

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "at":
                obj = parts[1]
                loc = parts[2]
                # Vehicles and packages can appear here
                if obj.startswith("v"):  # vehicle
                    vehicle_locations[obj] = loc
                else:  # package
                    package_locations[obj] = loc
            elif pred == "in":
                package = parts[1]
                vehicle = parts[2]
                package_in_vehicle[package] = vehicle

        # For packages that are in vehicles, override their location to vehicle's location.
        for package, vehicle in package_in_vehicle.items():
            # If vehicle location unknown, skip package (should not occur in a consistent state)
            if vehicle in vehicle_locations:
                package_locations[package] = vehicle_locations[vehicle]
            else:
                # Vehicle location unknown, heuristic will be rougher
                package_locations[package] = None

        total_cost = 0

        # For each package, estimate cost to deliver:
        for package, goal_loc in self.package_goal_locations.items():
            current_loc = package_locations.get(package)

            # If package location unknown, we estimate a moderate cost (fail-safe)
            if current_loc is None:
                # Fallback: assume at some non-goal location far away:
                # We add cost = 2 (pick+drop) + a fixed driving penalty (say 5)
                total_cost += 7
                continue

            # If already at goal, zero cost for this package
            if current_loc == goal_loc:
                continue

            # Estimate drive distance (number of drive steps) from current package location to goal location
            # If package is outside a vehicle, we also consider one drive from vehicle to package location (we guess closest vehicle)

            # Find if the package is currently inside a vehicle by comparing with package_in_vehicle
            if package in package_in_vehicle:
                # Package already in vehicle, so:
                # Cost: drive from vehicle location to goal location + drop (1 action)
                vehicle_loc = current_loc
                dist = self.distance_cache.get(vehicle_loc, {}).get(goal_loc, None)

                # If no path found, fallback large cost
                if dist is None:
                    dist = 10  # arbitrary large penalty

                # Actions: dist drives + 1 drop
                cost = dist + 1
                total_cost += cost

            else:
                # Package currently outside any vehicle:
                # We must pick a vehicle to go to package location, pick it, drive to goal location, drop it.

                # Heuristic: pick the closest vehicle to the package location.
                # If no vehicles known, fallback large cost.

                min_vehicle_dist = None
                for vehicle, v_loc in vehicle_locations.items():
                    dist_v_to_p = self.distance_cache.get(v_loc, {}).get(current_loc, None)
                    if dist_v_to_p is not None:
                        if min_vehicle_dist is None or dist_v_to_p < min_vehicle_dist:
                            min_vehicle_dist = dist_v_to_p

                # If no vehicle found reachable, fallback moderate cost
                if min_vehicle_dist is None:
                    min_vehicle_dist = 5

                dist_p_to_goal = self.distance_cache.get(current_loc, {}).get(goal_loc, None)
                if dist_p_to_goal is None:
                    dist_p_to_goal = 5

                # Actions:
                # drive vehicle to package location (min_vehicle_dist)
                # pick-up package (1)
                # drive package location to goal location (dist_p_to_goal)
                # drop package (1)

                cost = min_vehicle_dist + 1 + dist_p_to_goal + 1
                total_cost += cost

        # Return the total estimated cost
        return total_cost
