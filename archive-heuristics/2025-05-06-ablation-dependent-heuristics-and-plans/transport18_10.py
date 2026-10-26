import collections
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(start_location, road_network, all_locations):
    """
    Performs BFS from start_location on the road_network graph.
    all_locations is a list of all possible locations in the problem.
    """
    distances = {loc: float('inf') for loc in all_locations}
    if start_location in distances: # Ensure start_location is one of the known locations
        distances[start_location] = 0
        queue = collections.deque([start_location])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc has neighbors defined in the network
            if current_loc in road_network:
                for neighbor in road_network[current_loc]:
                    # Ensure neighbor is also a known location
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
    # If start_location was not in all_locations, distances remains all inf, which is correct.
    return distances

def compute_all_pairs_shortest_paths(road_network, all_locations):
    """
    Computes shortest path distances between all pairs of locations.
    road_network is the adjacency list.
    all_locations is a list of all possible locations in the problem.
    """
    all_distances = {}
    for start_loc in all_locations:
        all_distances[start_loc] = bfs(start_loc, road_network, all_locations)
    return all_distances

# Define the Heuristic class
# Note: The base class 'Heuristic' is assumed to be provided elsewhere,
# likely imported from 'heuristics.heuristic_base'.
# A dummy definition is included here for standalone testing if needed,
# but should be removed when integrated into the planner framework.
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the costs for all
    packages independently. It considers pick-up, drop, and drive actions.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles are always available when needed to pick up or drop off a package
      at the package's current location or the goal location, provided a road exists.
    - Vehicle capacity constraints are ignored.
    - Multiple packages can be transported simultaneously by a single vehicle,
      but the heuristic calculates costs per package independently and sums them.
      This might overestimate the cost but is a common simplification for
      non-admissible heuristics.
    - The cost of driving between two locations is the shortest path distance
      in the road network graph.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds the road network graph from static `(road l1 l2)` facts.
    - Collects all relevant locations (from initial state, goals, and roads)
      and computes all-pairs shortest paths between them using BFS.
    - Identifies all packages and vehicles present in the initial state or goals
      (based on naming convention 'p*' and 'v*').

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it at a location on the ground,
       or is it inside a vehicle?
    2. If the package is at a location `l` (and `l` is not the goal `l_goal`):
       - It needs to be picked up (1 action).
       - It needs to be transported from `l` to `l_goal`. The minimum number of
         drive actions required for the vehicle carrying it is the shortest path
         distance `dist(l, l_goal)`.
       - It needs to be dropped at `l_goal` (1 action).
       - The estimated cost for this package is `1 + dist(l, l_goal) + 1`.
    3. If the package is inside a vehicle `v`:
       - Find the current location `l` of vehicle `v`.
       - If `l` is not the goal `l_goal`:
         - It needs to be transported from `l` to `l_goal` by vehicle `v`. The
           minimum number of drive actions is `dist(l, l_goal)`.
         - It needs to be dropped at `l_goal` (1 action).
         - The estimated cost for this package is `dist(l, l_goal) + 1`.
       - If `l` is the goal `l_goal`:
         - It only needs to be dropped at `l_goal` (1 action).
         - The estimated cost for this package is `1`.
    4. The total heuristic value is the sum of the estimated costs for all
       packages that are not yet at their goal locations.
    5. If any package's goal location is unreachable from its current location
       (or its vehicle's current location) via the road network, the heuristic
       value is infinity.
    6. If all packages are at their goal locations, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network for shortest path calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        all_locations = set()
        all_packages = set()
        all_vehicles = set()

        # Collect info from goals
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
                all_locations.add(location)
                all_packages.add(package) # Assume objects in goal 'at' are packages

        # Collect info from static facts (roads)
        self.road_network = collections.defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_network[loc1].append(loc2)
                self.road_network[loc2].append(loc1) # Roads are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Collect info from initial state facts (locations of objects)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, location = parts[1], parts[2]
                 all_locations.add(location)
                 # Infer type based on naming convention for now
                 if obj.startswith('p'): all_packages.add(obj)
                 elif obj.startswith('v'): all_vehicles.add(obj)
             elif parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 all_packages.add(package)
                 all_vehicles.add(vehicle) # Vehicle carrying package must exist

        # Ensure all locations mentioned in the problem are considered for distances
        all_locations_list = list(all_locations)

        # Compute all-pairs shortest paths.
        self.distances = compute_all_pairs_shortest_paths(self.road_network, all_locations_list)

        # Store all packages and vehicles found (useful for __call__)
        self.all_packages = list(all_packages)
        self.all_vehicles = list(all_vehicles)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Track where packages and vehicles are currently located.
        package_locations = {} # Maps package -> location_name or vehicle_name
        vehicle_locations = {} # Maps vehicle -> location_name

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, location = parts[1], parts[2]
                if obj in self.all_packages:
                     package_locations[obj] = location
                elif obj in self.all_vehicles:
                     vehicle_locations[obj] = location

            elif parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 if package in self.all_packages:
                     package_locations[package] = vehicle # Package is inside a vehicle


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at its goal. Calculate cost.
            current_loc_or_vehicle = package_locations.get(package)

            if current_loc_or_vehicle is None:
                 # Package exists (in goal_locations) but is not found in state facts. Invalid.
                 return float('inf')

            cost_for_package = 0

            # Determine if the package is on the ground or in a vehicle
            if current_loc_or_vehicle in self.all_vehicles:
                 # Case 2: Package is inside a vehicle
                 vehicle_name = current_loc_or_vehicle

                 # Find the location of the vehicle
                 vehicle_current_location = vehicle_locations.get(vehicle_name)

                 if vehicle_current_location is None:
                      # Vehicle carrying package is nowhere. Invalid state.
                      return float('inf')

                 # Need distance from vehicle_current_location to goal_location
                 dist = self.distances.get(vehicle_current_location, {}).get(goal_location, float('inf'))

                 if dist == float('inf'):
                     return float('inf') # Unreachable goal

                 # Cost: drive (if needed) + drop
                 # If vehicle is already at goal, drive cost is 0
                 cost_for_package = (dist if vehicle_current_location != goal_location else 0) + 1 # drive + drop


            else:
                 # Case 1: Package is on the ground at current_location
                 # current_loc_or_vehicle must be a location string
                 current_location = current_loc_or_vehicle

                 # Need distance from current_location to goal_location
                 dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))

                 if dist == float('inf'):
                     return float('inf') # Unreachable goal

                 # Cost: pick-up + drive + drop
                 cost_for_package = 1 + dist + 1 # pick + drive + drop

            total_cost += cost_for_package

        return total_cost
