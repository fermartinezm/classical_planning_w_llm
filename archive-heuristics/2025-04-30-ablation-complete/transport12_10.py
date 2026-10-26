import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location,
    summing the costs for each package independently. It ignores vehicle capacity
    and availability constraints, and assumes shortest paths are always traversable
    by an available vehicle.

    # Assumptions
    - The road network is bidirectional.
    - Any vehicle can transport any package (capacity constraints are ignored).
    - Vehicles are always available where needed to pick up or drop packages,
      and their movement cost is simply the shortest path distance.
    - Packages can be moved independently of each other for cost estimation.
    - States are assumed to be valid: packages are either at a location or
      inside a vehicle, and vehicles are always at a location.
    - All locations relevant to the problem (appearing in initial state, goals, or static facts)
      are considered, and shortest paths are computed based on the road network
      defined in static facts.

    # Heuristic Initialization
    - Parses the goal facts to create a mapping from each package to its goal location.
    - Parses static facts to build the road network graph (adjacency list).
    - Collects all unique locations mentioned in the initial state, goals, and static facts.
    - Computes all-pairs shortest paths between these relevant locations using BFS
      on the road network graph.
    - Identifies packages (from goal facts) and vehicles (from initial state 'at'
      facts for non-packages, or 'capacity' facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Create temporary mappings for the current state:
       - `current_package_locations`: maps packages on the ground to their location.
       - `package_in_vehicle`: maps packages inside vehicles to the vehicle.
       - `current_vehicle_locations`: maps vehicles to their location.
       These mappings are populated by iterating through the facts in the current state.
    3. Iterate through each package and its goal location stored during initialization.
    4. For the current package `p` and its goal location `goal_loc_p`:
       - Check if `p` is already at `goal_loc_p` in the current state (i.e., `p` is in `current_package_locations` and its location matches `goal_loc_p`). If true, add 0 cost for this package and proceed to the next package.
       - If `p` is not at its goal:
         - Determine the current status of `p`: Is it on the ground at `current_loc_p` (found in `current_package_locations`) or inside a vehicle `v` (found in `package_in_vehicle`)?
         - If `p` is on the ground at `current_loc_p`:
           - The estimated cost for this package is 1 (for pick-up) + 1 (for drop) + the shortest path distance from `current_loc_p` to `goal_loc_p`.
           - Look up the shortest path distance in the precomputed table. If no path exists (distance is infinity), the state is likely unsolvable, return infinity.
           - Add this cost to the total heuristic cost.
         - If `p` is inside vehicle `v`:
           - Find the current location `current_loc_v` of vehicle `v` (from `current_vehicle_locations`). If the vehicle's location is not found, the state is invalid, return infinity.
           - If `current_loc_v` is the same as `goal_loc_p`:
             - The estimated cost for this package is 1 (for drop).
             - Add this cost to the total heuristic cost.
           - If `current_loc_v` is different from `goal_loc_p`:
             - The estimated cost for this package is 1 (for drop) + the shortest path distance from `current_loc_v` to `goal_loc_p`.
             - Look up the shortest path distance. If no path exists, return infinity.
             - Add this cost to the total heuristic cost.
         - If the package's status is not found (neither at a location nor in a vehicle), the state is invalid, return infinity.
    5. After iterating through all packages, return the accumulated total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing shortest paths."""
        self.goals = task.goals
        self.static = task.static
        initial_state = task.initial_state # Need initial state to identify vehicles

        # Extract goal locations for packages
        self.goal_locations = {}
        # Identify packages from goal facts
        self.packages = set()
        for goal in self.goals:
            # Goal is expected to be (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
                self.packages.add(package)

        # Identify vehicles from initial state 'at' facts (objects not in packages)
        # or from 'capacity' facts. This is a heuristic way to identify vehicles.
        self.vehicles = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 if obj not in self.packages:
                     self.vehicles.add(obj)
             elif parts[0] == "capacity" and len(parts) == 3:
                  obj = parts[1]
                  self.vehicles.add(obj) # Objects with capacity are vehicles

        # Collect all unique locations mentioned in the problem
        all_locations_in_problem = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                all_locations_in_problem.add(parts[1])
                all_locations_in_problem.add(parts[2])

        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 all_locations_in_problem.add(parts[2])

        for fact in self.goals:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 all_locations_in_problem.add(parts[2])


        # Build road network graph using only locations found in road facts
        self.road_graph = collections.defaultdict(set)
        locations_in_road_facts = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Roads are bidirectional
                locations_in_road_facts.add(loc1)
                locations_in_road_facts.add(loc2)


        # Compute shortest paths using BFS from all locations found anywhere in the problem
        self.shortest_paths = {}
        for start_loc in all_locations_in_problem:
            # Run BFS only if the start location is part of the road network graph
            if start_loc in locations_in_road_facts:
                 self._bfs(start_loc, locations_in_road_facts)
            # If start_loc is not in road_graph, it's an isolated location.
            # BFS from it within the road network graph won't find anything,
            # and shortest_paths will not contain entries starting with start_loc.
            # This is handled correctly by the .get() with default float('inf') later.


    def _bfs(self, start_node, locations_in_graph):
        """Compute shortest paths from start_node to all other locations reachable within the graph."""
        # Initialize distances for all locations that are part of the graph
        distances = {loc: float('inf') for loc in locations_in_graph}
        if start_node in distances: # Ensure start_node is in the graph locations
            distances[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                current_loc = queue.popleft()

                # Only explore neighbors if current_loc is in the graph
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        # Ensure neighbor is also considered a valid location in the graph
                        if neighbor in distances and distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)

            # Store computed distances for reachable locations
            for end_loc, dist in distances.items():
                 if dist != float('inf'):
                    self.shortest_paths[(start_node, end_loc)] = dist
        # If start_node is not in locations_in_graph, no paths starting from it are stored.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatables (packages, vehicles) to their current location or container
        current_package_locations = {} # package -> location (if at a location)
        package_in_vehicle = {} # package -> vehicle (if in a vehicle)
        current_vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    current_package_locations[obj] = loc
                elif obj in self.vehicles:
                    current_vehicle_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Ensure they are known types (based on init/goal/static parsing)
                if package in self.packages and vehicle in self.vehicles:
                     package_in_vehicle[package] = vehicle
                # Ignore 'in' facts involving unknown objects for robustness,
                # although in valid states this shouldn't happen.

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if package in current_package_locations and current_package_locations[package] == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal. Calculate cost for this package.
            package_cost = 0

            if package in current_package_locations: # Package is on the ground
                current_loc = current_package_locations[package]
                # Needs pick-up (1) + drive + drop (1)
                package_cost += 1 # pick-up action
                package_cost += 1 # drop action
                # Add drive cost: shortest path from current location to goal location
                path_cost = self.shortest_paths.get((current_loc, goal_location), float('inf'))
                if path_cost == float('inf'):
                     # If the goal location is unreachable from the package's current location
                     # via the road network, the state is likely unsolvable.
                     return float('inf')
                package_cost += path_cost # drive actions

            elif package in package_in_vehicle: # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find vehicle's location
                if vehicle not in current_vehicle_locations:
                     # A package is in a vehicle, but the vehicle's location is unknown.
                     # This indicates an invalid state representation.
                     return float('inf')

                current_loc_v = current_vehicle_locations[vehicle]

                if current_loc_v == goal_location:
                    # Needs drop (1)
                    package_cost += 1 # drop action
                else:
                    # Needs drive + drop (1)
                    package_cost += 1 # drop action
                    # Add drive cost: shortest path from vehicle's location to goal location
                    path_cost = self.shortest_paths.get((current_loc_v, goal_location), float('inf'))
                    if path_cost == float('inf'):
                        # If the goal location is unreachable from the vehicle's current location,
                        # the state is likely unsolvable.
                        return float('inf')
                    package_cost += path_cost # drive actions

            else:
                 # Package is not at a location and not in a vehicle? Invalid state.
                 return float('inf')

            total_cost += package_cost

        # If the loop finishes, all packages with goals were either already at their goal
        # or their cost was added. If all were at their goal, total_cost remains 0.
        return total_cost
