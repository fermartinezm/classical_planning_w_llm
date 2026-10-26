# Need deque for BFS
from collections import deque
# Need Heuristic base class
from heuristics.heuristic_base import Heuristic
# Need Task class definition (provided in problem description)
from task import Task # Assuming task.py is available in the environment
# import logging # Optional: uncomment for debugging logs

# Helper function to parse fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    return tuple(fact_string[1:-1].split())

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each package that is not yet at its goal location.
    For a package not at its goal, the cost is estimated based on its current
    status (on the ground or in a vehicle) and the shortest path distance
    in the road network to its goal location. This is a non-admissible heuristic
    designed for greedy best-first search.

    Assumptions:
    - Roads are bidirectional (as is typical in transport domains and seen in example instances).
    - Any package can eventually be picked up by some vehicle (vehicle capacity
      is ignored in the cost calculation for simplicity and speed, making it non-admissible).
    - The road network is static and provided in the static facts.
    - Object types (package, vehicle, location) can be inferred from the facts
      they appear in within the initial state, goals, and static facts.
    - The primary goals are of the form (at package location). Other goal types
      are ignored by this heuristic.

    Heuristic Initialization:
    1. Identify object types (packages, vehicles, locations) by parsing initial state, goals, and static facts.
    2. Parse the goal state to map each package that has an (at package location) goal to its target location.
    3. Build the road network graph (adjacency list) from static facts of the form (road l1 l2).
    4. Compute all-pairs shortest paths between all identified locations using BFS on the road network graph. Store distances in a dictionary keyed by (start_loc, end_loc).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `total_heuristic` to 0.
    2. Parse the current state (`node.state`) to extract the current locations of packages on the ground, which packages are inside which vehicles, and the current locations of vehicles. Store these in dictionaries (`current_package_locations`, `package_in_vehicle`, `vehicle_locations`).
    3. Iterate through each package `package_name` and its corresponding goal location `goal_location` stored during initialization (`self.package_goals`).
    4. For the current package:
       - Check if the fact `(at package_name goal_location)` is present in the current state. This requires formatting the fact string correctly, including quotes, e.g., `'(at p1 l2)'`.
       - If the goal fact is in the state, the package is already at its destination. The cost for this package is 0. Continue to the next package.
       - If the goal fact is not in the state, the package needs to be moved. Determine its current status:
         - If `package_name` is found as a key in `current_package_locations` (meaning it's on the ground at `l_current_p`):
           - Look up the shortest distance `dist` from `l_current_p` to `goal_location` using the precomputed distances (`self.location_distances`).
           - If the distance is infinity (locations are disconnected), the state is likely unsolvable or requires traversing disconnected parts (which is impossible). Return `float('inf')` for the total heuristic.
           - The estimated cost for this package is 2 (for the pick-up action and the drop action) plus the driving distance `dist`. Add this cost to `total_heuristic`.
         - If `package_name` is found as a key in `package_in_vehicle` (meaning it's in `vehicle_name`):
           - Find the current location `l_current_v` of `vehicle_name` from `vehicle_locations`. If the vehicle's location is not found (which indicates an inconsistent or invalid state), return `float('inf')`.
           - Look up the shortest distance `dist` from `l_current_v` to `goal_location` using `self.location_distances`.
           - If the distance is infinity, return `float('inf')`.
           - The estimated cost for this package is 1 (for the drop action) plus the driving distance `dist`. Add this cost to `total_heuristic`.
         - If the package is neither on the ground nor in a vehicle (should not happen in valid states), return `float('inf')`.
    5. After iterating through all goal packages, return the accumulated `total_heuristic`. This value will be 0 if and only if all packages listed in self.package_goals
    are currently at their respective goal locations. If the task goal contains other facts, this heuristic will be 0 when the package delivery part is done, but not necessarily the full goal. This is acceptable for
    a non-admissible heuristic focused on the main problem aspect.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self._identify_object_types(task)
        self._parse_goals(task)
        self._build_road_network(task)
        self._compute_all_pairs_shortest_paths()

    def _identify_object_types(self, task):
        """Identifies object types (packages, vehicles, locations, sizes)
           by examining facts in the task description."""
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()

        # Examine initial state, goals, and static facts
        facts_to_examine = set(task.initial_state) | set(task.goals) | set(task.static)

        # First pass: Identify types based on specific predicates
        for fact_string in facts_to_examine:
            fact = parse_fact(fact_string)
            predicate = fact[0]
            if predicate == 'in' and len(fact) == 3:
                # (in ?x - package ?v - vehicle)
                package_name, vehicle_name = fact[1], fact[2]
                self.packages.add(package_name)
                self.vehicles.add(vehicle_name)
            elif predicate == 'road' and len(fact) == 3:
                # (road ?l1 ?l2 - location)
                loc1, loc2 = fact[1], fact[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
            elif predicate == 'capacity' and len(fact) == 3:
                # (capacity ?v - vehicle ?s1 - size)
                vehicle_name, size_name = fact[1], fact[2]
                self.vehicles.add(vehicle_name)
                self.sizes.add(size_name)
            elif predicate == 'capacity-predecessor' and len(fact) == 3:
                # (capacity-predecessor ?s1 ?s2 - size)
                size1, size2 = fact[1], fact[2]
                self.sizes.add(size1)
                self.sizes.add(size2)
            elif predicate == 'at' and len(fact) == 3:
                 # (at ?x - locatable ?v - location)
                 # obj1 is locatable, obj2 is location
                 self.locations.add(fact[2])

        # Second pass: Identify locatables in 'at' facts that weren't identified as packages
        # These must be vehicles
        for fact_string in facts_to_examine:
             fact = parse_fact(fact_string)
             if fact[0] == 'at' and len(fact) == 3:
                 obj1 = fact[1]
                 if obj1 not in self.packages:
                     self.vehicles.add(obj1)


    def _parse_goals(self, task):
        """Parses the goal state to map packages to their target locations."""
        self.package_goals = {}
        for goal_fact_string in task.goals:
            goal_fact = parse_fact(goal_fact_string)
            # We only care about (at package location) goals
            # Ensure the object is actually identified as a package and location
            if goal_fact[0] == 'at' and len(goal_fact) == 3:
                 package_name, goal_location = goal_fact[1], goal_fact[2]
                 if package_name in self.packages and goal_location in self.locations:
                    self.package_goals[package_name] = goal_location
            # Ignore other potential goal types or goals for non-packages

    def _build_road_network(self, task):
        """Builds the road network graph from static facts."""
        # Initialize graph with all identified locations
        self.road_graph = {loc: [] for loc in self.locations}
        for static_fact_string in task.static:
            static_fact = parse_fact(static_fact_string)
            if static_fact[0] == 'road' and len(static_fact) == 3:
                loc1, loc2 = static_fact[1], static_fact[2]
                # Ensure locations are in our identified set before adding roads
                if loc1 in self.locations and loc2 in self.locations:
                    # Assuming roads are bidirectional
                    self.road_graph[loc1].append(loc2)
                    self.road_graph[loc2].append(loc1)
                # else: road fact involves objects not identified as locations, ignore or warn

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations."""
        self.location_distances = {}
        for start_loc in self.locations:
            distances_from_start = self._bfs(start_loc)
            for end_loc, dist in distances_from_start.items():
                self.location_distances[(start_loc, end_loc)] = dist

    def _bfs(self, start_node):
        """Performs BFS starting from start_node to find distances."""
        # Initialize distances for all known locations
        distances = {loc: float('inf') for loc in self.locations}

        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node is in graph and has neighbors
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is a known location (should be if graph built correctly)
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Extract current locations/status from the state
        current_package_locations = {} # {package_name: location_name}
        package_in_vehicle = {}        # {package_name: vehicle_name}
        vehicle_locations = {}         # {vehicle_name: location_name}

        for fact_string in state:
            fact = parse_fact(fact_string)
            predicate = fact[0]
            if predicate == 'at' and len(fact) == 3:
                obj1, obj2 = fact[1], fact[2]
                if obj1 in self.packages and obj2 in self.locations:
                    current_package_locations[obj1] = obj2
                elif obj1 in self.vehicles and obj2 in self.locations:
                    vehicle_locations[obj1] = obj2
                # else: fact involves objects not identified as package/vehicle/location - ignore
            elif predicate == 'in' and len(fact) == 3:
                package_name, vehicle_name = fact[1], fact[2]
                if package_name in self.packages and vehicle_name in self.vehicles:
                     package_in_vehicle[package_name] = vehicle_name
                # else: fact involves objects not identified as package/vehicle - ignore

        total_heuristic = 0

        # Calculate cost for each package with a defined goal location
        for package_name, goal_location in self.package_goals.items():
            # Check if package is already at goal
            # Need to format the fact string exactly as it appears in the state frozenset
            goal_fact_string_in_state = f"'(at {package_name} {goal_location})'"
            if goal_fact_string_in_state in state:
                 continue # Package is at goal, cost is 0 for this package

            # Package is not at goal, determine its current status
            cost_p = float('inf') # Initialize cost for this package

            if package_name in current_package_locations:
                # Package is on the ground
                l_current_p = current_package_locations[package_name]
                # l_current_p is guaranteed to be in self.locations if extracted from state

                dist = self.location_distances.get((l_current_p, goal_location), float('inf'))
                if dist == float('inf'):
                    # Cannot reach goal location from current location
                    return float('inf')
                # Cost: pick-up (1) + drive (dist) + drop (1)
                cost_p = 2 + dist

            elif package_name in package_in_vehicle:
                # Package is in a vehicle
                vehicle_name = package_in_vehicle[package_name]
                l_current_v = vehicle_locations.get(vehicle_name)

                if l_current_v is None:
                    # Vehicle location not found in state - indicates an issue or unreachable state
                    # logging.error(f"Vehicle {vehicle_name} carrying {package_name} has no location in state.")
                    return float('inf')

                # l_current_v is guaranteed to be in self.locations if extracted from state

                dist = self.location_distances.get((l_current_v, goal_location), float('inf'))
                if dist == float('inf'):
                    # Cannot reach goal location from vehicle's current location
                    return float('inf')
                # Cost: drive (dist) + drop (1)
                cost_p = 1 + dist
            else:
                 # Package is not at goal, not on ground, and not in vehicle.
                 # This implies an invalid state or the package doesn't exist as expected.
                 # For robustness, return infinity.
                 # logging.error(f"Package {package_name} not at goal, not on ground, and not in vehicle.")
                 return float('inf')

            # If cost_p is still inf, it means a required distance was inf
            if cost_p == float('inf'):
                 return float('inf')

            total_heuristic += cost_p

        # The heuristic is 0 if and only if all packages listed in self.package_goals
        # are currently at their respective goal locations. If the task goal
        # contains other facts, this heuristic will be 0 when the package delivery
        # part is done, but not necessarily the full goal. This is acceptable for
        # a non-admissible heuristic focused on the main problem aspect.

        return total_heuristic
