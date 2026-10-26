from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extracts predicate and arguments from a fact string."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum actions required
    for each package to reach its goal location, ignoring vehicle capacity and
    availability constraints. For a package not at its goal location, the cost
    is estimated as:
    - If the package is at a location: 1 (pick-up) + shortest_path_distance (drives) + 1 (drop).
    - If the package is in a vehicle: 1 (drop) + shortest_path_distance (drives from vehicle's location).
    The shortest path distances between locations are precomputed using BFS on the road network.

    Assumptions:
    - The goal only consists of (at ?p ?l) predicates for packages.
    - The road network is undirected (if (road l1 l2) exists, (road l2 l1) also exists or is implied and handled by adding both directions).
    - All locations and roads mentioned in the initial state and goals are part of a single connected component, or packages only need to move within their connected component. Unreachable goal locations are assigned a very high cost.
    - Vehicle capacity and availability are not bottlenecks (this is a relaxation).
    - The state representation is valid: all goal packages and vehicles are present and located (either at a location or in a vehicle for packages, at a location for vehicles). Vehicles are identified as objects with 'capacity' or objects 'at' a location that are not goal packages.

    Heuristic Initialization:
    - Parses the goal facts to store the target location for each package in `self.goal_locations`.
    - Collects all packages that are part of the goal into `self.goal_packages`.
    - Collects all locations mentioned in `road` facts.
    - Collects all vehicles mentioned in initial state `at` or `capacity` facts into `self.vehicles`.
    - Builds the road network graph (adjacency list) from `road` facts. Assumes bidirectional roads.
    - Computes all-pairs shortest paths between locations found in road facts using BFS starting from each such location. Stores distances in `self.distances`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize `total_cost` to 0.
    2. Create dictionaries `current_package_locations` and `current_vehicle_locations` to store the current whereabouts of relevant objects.
    3. Iterate through each fact in the current state:
       - If the fact is `(at ?obj ?l)`:
         - If `?obj` is a package we care about (i.e., in `self.goal_packages`), record its location: `current_package_locations[?obj] = ?l`.
         - If `?obj` is a vehicle we know about (i.e., in `self.vehicles`), record its location: `current_vehicle_locations[?obj] = ?l`.
       - If the fact is `(in ?p ?v)`:
         - If `?p` is a package we care about, record that it is in vehicle `?v`: `current_package_locations[?p] = ?v`.
    4. Define a large value `UNREACHABLE_DISTANCE` to represent infinite cost for unreachable locations.
    5. Iterate through each package and its goal location in `self.goal_locations.items()`:
       - Get the package `package` and its `goal_location`.
       - Retrieve the `current_loc_or_veh` of the `package` from `current_package_locations`.
       - If `current_loc_or_veh` is the same as `goal_location`, the package is already delivered; continue to the next package.
       - If `current_loc_or_veh` is a vehicle name (check if it's in `self.vehicles`):
         - This means the package is currently inside this `vehicle`.
         - Get the `vehicle_current_location` from `current_vehicle_locations`. (Assumes vehicle is always located).
         - If vehicle location is not found (invalid state), assign high cost and skip this package.
         - The cost for this package is 1 (for the `drop` action) plus the shortest path distance from the `vehicle_current_location` to the `goal_location`. Look up the distance in `self.distances`, using `UNREACHABLE_DISTANCE` if the path is not found. Add this cost to `total_cost`.
       - If `current_loc_or_veh` is a location name:
         - This means the package is currently `at` this `package_current_location`.
         - The cost for this package is 1 (for the `pick-up` action) plus the shortest path distance from the `package_current_location` to the `goal_location` (for `drive` actions) plus 1 (for the `drop` action). Look up the distance in `self.distances`, using `UNREACHABLE_DISTANCE` if the path is not found. Add this cost to `total_cost`.
    6. Return the final `total_cost`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Parse goal facts to get package goal locations and goal packages
        self.goal_locations = {}
        self.goal_packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.goal_packages.add(package)

        # 2. Collect all locations and vehicles from initial state and static facts
        all_locations = set()
        self.vehicles = set()

        # Locations from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)

        # Vehicles from initial state and static facts (capacity)
        # Objects 'at' a location that are not goal packages are assumed vehicles
        # Objects with 'capacity' are vehicles
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # If obj is not a package we need to deliver, assume it's a vehicle
                # This relies on the domain structure where only packages and vehicles are locatable.
                if obj not in self.goal_packages:
                    self.vehicles.add(obj)
            elif parts[0] == 'capacity' and len(parts) == 3:
                 vehicle, size = parts[1], parts[2]
                 self.vehicles.add(vehicle) # Objects with capacity are definitely vehicles

        # Ensure goal locations are included in the set of all locations considered for distances
        all_locations.update(self.goal_locations.values())

        # 3. Build road network graph
        self.road_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Run BFS from all locations that are endpoints of any road
        all_road_locations = set(self.road_graph.keys()) | set(loc for neighbors in self.road_graph.values() for loc in neighbors)

        for start_node in all_road_locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Get neighbors, handling locations not present as keys (no outgoing roads)
                neighbors = self.road_graph.get(current_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state

        # 2. Extract current locations of packages and vehicles
        current_package_locations = {} # Maps package -> location or vehicle
        current_vehicle_locations = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, location = parts[1], parts[2]
                if obj in self.goal_packages: # It's a package we care about
                    current_package_locations[obj] = location
                elif obj in self.vehicles: # It's a vehicle we know about
                     current_vehicle_locations[obj] = location
            elif parts[0] == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_packages: # It's a package we care about
                    current_package_locations[package] = vehicle

        total_cost = 0
        # Use a large number for unreachable locations. This makes states leading
        # to unreachable goals have a very high heuristic value.
        UNREACHABLE_DISTANCE = 1000000

        # 5. Iterate through each package and its goal location
        for package, goal_location in self.goal_locations.items():
            current_loc_or_veh = current_package_locations.get(package)

            # Assumption: State is valid and all goal packages are present and located.
            # If current_loc_or_veh is None, it indicates an issue with the state or parsing.
            # We proceed assuming it's found.

            # If package is already at goal
            if current_loc_or_veh == goal_location:
                continue # Cost is 0 for this package

            # If package is in a vehicle
            if current_loc_or_veh in self.vehicles: # Check if it's a vehicle name we know about
                vehicle = current_loc_or_veh
                # Get the vehicle's current location.
                # Assumption: Vehicle is always 'at' a location if it exists.
                vehicle_current_location = current_vehicle_locations.get(vehicle)

                # If vehicle location is not found (invalid state), assign high cost
                if vehicle_current_location is None:
                     total_cost += UNREACHABLE_DISTANCE
                     continue # Cannot estimate further for this package

                # Cost: 1 (drop) + distance from vehicle's location to package's goal location
                # Get distance, defaulting to UNREACHABLE_DISTANCE if path doesn't exist
                dist = self.distances.get((vehicle_current_location, goal_location), UNREACHABLE_DISTANCE)
                total_cost += 1 + dist

            # If package is at a location
            else: # current_loc_or_veh must be a location name
                package_current_location = current_loc_or_veh

                # Cost: 1 (pick-up) + distance from package's location to package's goal location + 1 (drop)
                # Get distance, defaulting to UNREACHABLE_DISTANCE if path doesn't exist
                dist = self.distances.get((package_current_location, goal_location), UNREACHABLE_DISTANCE)
                total_cost += 2 + dist

        return total_cost
