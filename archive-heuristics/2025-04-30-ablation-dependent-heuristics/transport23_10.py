# Required imports
import collections
import math

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    parts = fact_str[1:-1].split()
    if not parts: # Handle empty fact string '()'
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to build road graph
def build_road_graph(static_facts):
    """Builds adjacency list for road network from static facts."""
    graph = collections.defaultdict(list)
    locations = set()
    for fact_str in static_facts:
        predicate, args = parse_fact(fact_str)
        if predicate == 'road':
            if len(args) == 2:
                l1, l2 = args
                graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)
    # Ensure all locations mentioned in roads are keys in graph, even if they have no outgoing roads
    for loc in locations:
        if loc not in graph:
            graph[loc] = []
    return graph, list(locations)

# Helper function for BFS shortest paths
def bfs_shortest_paths(graph, start_node):
    """Computes shortest paths from start_node to all other nodes using BFS."""
    distances = {node: math.inf for node in graph}
    if start_node in graph: # Ensure start_node is in the graph
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is still valid (should be if from queue)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, locations):
    """Computes shortest paths between all pairs of locations."""
    all_paths = {}
    for start_node in locations:
        all_paths[start_node] = bfs_shortest_paths(graph, start_node)
    return all_paths


class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    estimated minimum action costs for each package that is not yet at its
    goal location. For a package needing transport, the estimated cost includes:
    1. The minimum drive cost for a vehicle to reach the package's current
       location (if it's on the ground).
    2. The cost of the 'pick-up' action (if on the ground).
    3. The minimum drive cost to move the package from its current effective
       location (where it is, or where its vehicle is) to its goal location.
    4. The cost of the 'drop' action (as it must end up on the ground at the goal).
    Shortest path distances on the road network are used to estimate driving costs.
    This heuristic is non-admissible as it sums costs independently for each
    package, potentially double-counting shared vehicle movements or vehicle
    movements needed for multiple packages. It ignores vehicle capacity constraints.

    Assumptions:
    - The input state is valid and consistent with the domain definition.
    - Every package mentioned in the goal has a single goal location ('at' fact).
    - The road network is static and correctly represented by 'road' facts.
    - Vehicle locations are always specified in the state if a package is 'in' that vehicle.
    - Objects with 'capacity' facts in the initial state or static facts are vehicles.
    - Locations mentioned in 'at' facts for packages/vehicles or in goal 'at' facts
      are expected to be part of the road network if movement is required.

    Heuristic Initialization:
    The constructor precomputes necessary static information from the task:
    1. Identifies vehicles by looking for objects involved in 'capacity' facts
       in the initial state and static facts.
    2. Parses 'road' facts from static facts to build the road network graph
       (adjacency list). It also collects all unique locations mentioned in roads.
    3. Computes all-pairs shortest paths (APSP) on the road network using BFS
       starting from each location identified in step 2. This provides the
       minimum number of 'drive' actions between any two connected locations.
       Unreachable locations have infinite distance.
    4. Parses goal facts to create a mapping from package names to their goal locations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to quickly access package locations ('at' facts),
       packages inside vehicles ('in' facts), and vehicle locations ('at' facts).
       Store these in dictionaries for efficient lookup.
    2. Initialize the total heuristic value `h` to 0.
    3. Initialize an `unreachable` flag to `False`.
    4. Iterate through each package that has a specified goal location
       (packages identified during initialization from goal facts).
    5. For the current package `p`, get its goal location `L_goal_p` from the
       precomputed map.
    6. Check if `p` is already at `L_goal_p` in the current state. If yes, the
       cost for this package is 0, continue to the next package.
    7. If `p` is not at its goal:
       a. Initialize the estimated cost for package `p` (`package_cost`) to 0.
       b. Determine the package's current status and location:
          - If `p` is `(at p L_p)` in the state (i.e., `p` is a key in `package_location`):
            - Its current location is `L_p = package_location[p]`.

            # Check if current_loc and goal_loc are known locations in the road network
            if current_loc not in self.locations or goal_loc not in self.locations:
                unreachable = True
                break # Package/Goal is at an isolated location

            # Cost: pickup + drive from current_loc to goal_loc + drop
            # Need to get a vehicle to current_loc first
            min_dist_vehicle_to_pickup = math.inf
            
            vehicle_found_in_state = False
            for v, v_loc in vehicle_location.items():
                vehicle_found_in_state = True
                # Check if vehicle location is part of the road network
                if v_loc in self.locations:
                    # Check if current_loc is reachable from v_loc
                    if current_loc in self.shortest_paths.get(v_loc, {}): # Use .get for safety
                         min_dist_vehicle_to_pickup = min(min_dist_vehicle_to_pickup, self.shortest_paths[v_loc][current_loc])

            if not vehicle_found_in_state or min_dist_vehicle_to_pickup == math.inf:
                # No vehicles in state, or no vehicle can reach the package location
                unreachable = True
                break # Cannot get a vehicle to the package

            package_cost += min_dist_vehicle_to_pickup # Drive vehicle to package
            package_cost += 1 # Pickup action

            # Now the package is conceptually in a vehicle at current_loc, needs to go to goal_loc
            # Check if goal_loc is reachable from current_loc
            if goal_loc in self.shortest_paths.get(current_loc, {}): # Use .get for safety
                package_cost += self.shortest_paths[current_loc][goal_loc] # Drive package to goal
            else:
                unreachable = True
                break # Goal location unreachable from package location

            package_cost += 1 # Drop action

          - If `p` is `(in p v)` in the state (i.e., `p` is a key in `package_in_vehicle`):
            vehicle = package_in_vehicle[p]
            current_loc = vehicle_location.get(vehicle) # Use .get for safety, though should exist

            if current_loc is None: # Vehicle location not found
                unreachable = True
                break # Package is in a vehicle whose location is unknown

            # Check if current_loc and goal_loc are known locations in the road network
            if current_loc not in self.locations or goal_loc not in self.locations:
                unreachable = True
                break # Vehicle/Goal is at an isolated location

            # Cost: drive from current_loc to goal_loc + drop
            # Check if goal_loc is reachable from current_loc
            if goal_loc in self.shortest_paths.get(current_loc, {}): # Use .get for safety
                package_cost += self.shortest_paths[current_loc][goal_loc] # Drive package to goal
            else:
                unreachable = True
                break # Goal location unreachable from package location
            package_cost += 1 # Drop action

          - Else (package is neither at a location nor in a vehicle):
            # This indicates an inconsistent state. Should not happen in valid states.
            unreachable = True
            break # Invalid state or unreachable

       c. If not unreachable, add `package_cost` to the total heuristic `h`.
    9. After iterating through all packages, if the `unreachable` flag is `True`, return `math.inf`. Otherwise, return the total heuristic value `h`.
    """
    def __init__(self, task):
        # Identify vehicles (assuming objects with capacity are vehicles)
        self.vehicles = set()
        # Check initial state for capacity facts
        for fact_str in task.initial_state:
             predicate, args = parse_fact(fact_str)
             if predicate == 'capacity':
                 if args: self.vehicles.add(args[0])
        # Check static facts for capacity facts (less common, but possible)
        for fact_str in task.static:
             predicate, args = parse_fact(fact_str)
             if predicate == 'capacity':
                 if args: self.vehicles.add(args[0])

        # Precompute road graph and shortest paths
        self.road_graph, self.locations = build_road_graph(task.static)
        self.shortest_paths = compute_all_pairs_shortest_paths(self.road_graph, self.locations)

        # Precompute package goal locations
        self.package_goal_location = {}
        for goal_fact_str in task.goals:
            predicate, args = parse_fact(goal_fact_str)
            if predicate == 'at':
                if len(args) == 2:
                    package, location = args
                    self.package_goal_location[package] = location

    def __call__(self, state):
        # Efficiently parse state facts
        package_location = {}
        package_in_vehicle = {}
        vehicle_location = {}

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at':
                if len(args) == 2:
                    obj, loc = args
                    if obj in self.package_goal_location: # It's a package we care about
                         package_location[obj] = loc
                    elif obj in self.vehicles: # It's a vehicle we identified
                         vehicle_location[obj] = loc
            elif predicate == 'in':
                if len(args) == 2:
                    p, v = args
                    package_in_vehicle[p] = v
            # Ignore capacity facts in state for heuristic calculation itself

        h = 0
        unreachable = False

        # Iterate through packages that need to reach a goal location
        for package, goal_loc in self.package_goal_location.items():
            # Check if package is already at goal
            if package in package_location and package_location[package] == goal_loc:
                continue # Package is already at its goal

            # Package is not at goal, calculate its contribution to heuristic
            package_cost = 0

            if package in package_location: # Package is on the ground at package_location[package]
                current_loc = package_location[package]

                # Check if current_loc and goal_loc are known locations in the road network
                if current_loc not in self.locations or goal_loc not in self.locations:
                    unreachable = True
                    break # Package/Goal is at an isolated location

                # Cost: pickup + drive from current_loc to goal_loc + drop
                # Need to get a vehicle to current_loc first
                min_dist_vehicle_to_pickup = math.inf
                
                vehicle_found_in_state = False
                for v, v_loc in vehicle_location.items():
                    vehicle_found_in_state = True
                    # Check if vehicle location is part of the road network
                    if v_loc in self.locations:
                        # Check if current_loc is reachable from v_loc
                        if current_loc in self.shortest_paths.get(v_loc, {}): # Use .get for safety
                             min_dist_vehicle_to_pickup = min(min_dist_vehicle_to_pickup, self.shortest_paths[v_loc][current_loc])

                if not vehicle_found_in_state or min_dist_vehicle_to_pickup == math.inf:
                    # No vehicles in state, or no vehicle can reach the package location
                    unreachable = True
                    break # Cannot get a vehicle to the package

                package_cost += min_dist_vehicle_to_pickup # Drive vehicle to package
                package_cost += 1 # Pickup action

                # Now the package is conceptually in a vehicle at current_loc, needs to go to goal_loc
                # Check if goal_loc is reachable from current_loc
                if goal_loc in self.shortest_paths.get(current_loc, {}): # Use .get for safety
                    package_cost += self.shortest_paths[current_loc][goal_loc] # Drive package to goal
                else:
                    unreachable = True
                    break # Goal location unreachable from package location

                package_cost += 1 # Drop action

            elif package in package_in_vehicle: # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                current_loc = vehicle_location.get(vehicle) # Use .get for safety, though should exist

                if current_loc is None: # Vehicle location not found
                    unreachable = True
                    break # Package is in a vehicle whose location is unknown

                # Check if current_loc and goal_loc are known locations in the road network
                if current_loc not in self.locations or goal_loc not in self.locations:
                    unreachable = True
                    break # Vehicle/Goal is at an isolated location

                # Cost: drive from current_loc to goal_loc + drop
                # Check if goal_loc is reachable from current_loc
                if goal_loc in self.shortest_paths.get(current_loc, {}): # Use .get for safety
                    package_cost += self.shortest_paths[current_loc][goal_loc] # Drive package to goal
                else:
                    unreachable = True
                    break # Goal location unreachable from package location
                package_cost += 1 # Drop action

            else:
                # Package is not at a location and not in a vehicle? Should not happen in valid states.
                unreachable = True
                break # Invalid state or unreachable

            # If package_cost became inf during calculation, unreachable would be True
            if unreachable:
                 break

            h += package_cost

        if unreachable:
            return math.inf
        else:
            return h
