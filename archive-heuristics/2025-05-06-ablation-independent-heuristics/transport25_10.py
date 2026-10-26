# Import necessary modules
from heuristics.heuristic_base import Heuristic
from task import Task # Used for type hinting and accessing task details in __init__
import collections # Used for BFS queue
import math # Used for infinity

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into its predicate and arguments.
    e.g., '(at p1 l8)' -> ('at', ['p1', 'l8'])
    """
    # Remove leading/trailing brackets and split by space
    # Assumes standard PDDL fact format with space separation
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function for Breadth-First Search (BFS) to find shortest paths
def bfs(graph, start_node):
    """
    Performs BFS starting from start_node to find shortest distances to all
    reachable nodes in the graph.

    Args:
        graph: An adjacency list representation of the graph
               (dict: node -> set of neighbors).
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from start_node.
        Nodes not reachable will have distance math.inf.
    """
    distances = {node: math.inf for node in graph}

    # If the start node is not in the graph (e.g., an isolated location not
    # connected by roads), it cannot reach any other node via roads.
    if start_node not in graph:
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Process neighbors if the current node is still valid in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal state by summing the estimated costs
    for each package that is not yet at its goal location. The cost for a
    package depends on its current status (at a location or in a vehicle)
    and the shortest path distances on the road network. It relaxes capacity
    constraints and assumes vehicle availability.

    Assumptions:
    - Objects starting with 'p' are packages, 'v' are vehicles, 'l' are locations.
      This is inferred from common PDDL naming conventions and the provided examples.
    - Vehicle capacity constraints are ignored. Any vehicle can pick up any package.
    - The road network defined by (road l1 l2) facts is static and represents
      undirected edges (if l1-l2 road exists, l2-l1 road also exists, as seen
      in examples).
    - The state representation is a frozenset of strings like '(predicate arg1 ...)'.
    - The heuristic value is 0 only for true goal states. For non-goal states
      where the calculation results in 0, it returns 1.

    Heuristic Initialization:
    1. Parses the goal facts to identify the target location for each package
       that needs to be at a specific location.
    2. Builds an undirected graph representation of the road network from the
       static (road l1 l2) facts. It also collects all unique locations mentioned
       in goals and road facts.
    3. Computes all-pairs shortest paths on the road network graph using BFS
       for all collected locations and stores these distances in a dictionary
       mapping (start_loc, end_loc) tuples to distance.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Extract the current locations of all packages and vehicles from the
       current state facts. Store package status (at location or in vehicle)
       and vehicle locations in dictionaries.
    3. Iterate through each package that has a goal location defined in the task.
    4. For a package `p` with goal location `L_goal`:
       - Check the current status of package `p` in the state (`package_status`).
       - If `p` is currently `(at p L_current)`:
         - If `L_current` is the same as `L_goal`, the package is already at its goal; its contribution is 0.
         - If `L_current` is different from `L_goal`:
           - Estimate the cost to get a vehicle to `L_current`, pick up `p`, drive to `L_goal`, and drop `p`.
           - Find the minimum shortest distance from *any* vehicle's current location (`L_v_current`) to `L_current`. If no vehicles exist or are reachable, this part is infinite.
           - Find the shortest distance from `L_current` to `L_goal`. If unreachable, this part is infinite.
           - The estimated cost for this package is (min vehicle travel to pickup) + 1 (pickup action) + (package travel to goal) + 1 (drop action).
           - Add this cost to `h`. If any component distance was infinite, `h` becomes infinite.
       - If `p` is currently `(in p v)`:
         - Find the current location `L_v_current` of vehicle `v`. If the vehicle's location is unknown, this state is problematic, and `h` becomes infinite.
         - Find the shortest distance from `L_v_current` to `L_goal`. If unreachable, this part is infinite.
         - The estimated cost for this package is (vehicle travel to goal) + 1 (drop action).
         - Add this cost to `h`. If the distance was infinite, `h` becomes infinite.
       - If the package's status is not found or is inconsistent (e.g., not 'at' or 'in'), `h` becomes infinite.
    5. After summing costs for all goal packages:
       - If `h` is `math.inf`, return `math.inf`.
       - If `h` is 0, check if the current state is the actual goal state using `self.task.goal_reached(node.state)`. Return 0 if it is, and 1 otherwise.
       - If `h` is greater than 0, return `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goal_package_locations = {}
        self.road_graph = {}
        self.locations = set()
        self.distances = {}

        # 1. Parse goal facts to get package target locations and collect locations
        for goal_fact in task.goals:
            predicate, args = parse_fact(goal_fact)
            # Assuming 'at' predicate for goal locations and 'p' prefix for packages
            if predicate == 'at' and len(args) == 2 and args[0].startswith('p'):
                package_name = args[0]
                location_name = args[1]
                self.goal_package_locations[package_name] = location_name
                self.locations.add(location_name) # Collect goal locations

        # 2. Build road network graph and collect all locations from road facts
        for static_fact in task.static:
            predicate, args = parse_fact(static_fact)
            # Assuming 'road' predicate and 'l' prefix for locations
            if predicate == 'road' and len(args) == 2 and args[0].startswith('l') and args[1].startswith('l'):
                loc1, loc2 = args
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Ensure all locations collected (from goals and roads) are keys in the graph structure
        # This is important for the BFS to correctly handle isolated locations
        # Use list() to iterate over a copy as we might add keys if a location was only in goals
        for loc in list(self.locations):
             self.road_graph.setdefault(loc, set())


        # 3. Compute all-pairs shortest paths using BFS
        # We compute distances between all pairs of locations we've identified
        for start_loc in list(self.locations): # Iterate over a copy
            loc_distances = bfs(self.road_graph, start_loc)
            for end_loc in self.locations:
                 # Store distance, using infinity if end_loc was not reached by BFS
                 self.distances[(start_loc, end_loc)] = loc_distances.get(end_loc, math.inf)


    def __call__(self, node):
        state_facts = node.state
        h = 0
        package_status = {} # { package_name: ('at', location) or ('in', vehicle_name) }
        vehicle_locations = {} # { vehicle_name: location }

        # Extract package and vehicle locations/status from current state
        for fact_string in state_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'at' and len(args) == 2:
                obj_name, loc_name = args
                # Assuming 'p' prefix for packages and 'l' for locations
                if obj_name.startswith('p') and loc_name.startswith('l'):
                    package_status[obj_name] = ('at', loc_name)
                # Assuming 'v' prefix for vehicles and 'l' for locations
                elif obj_name.startswith('v') and loc_name.startswith('l'):
                    vehicle_locations[obj_name] = loc_name
            # Assuming 'in' predicate, 'p' for package, 'v' for vehicle
            elif predicate == 'in' and len(args) == 2 and args[0].startswith('p') and args[1].startswith('v'):
                 pkg_name, veh_name = args
                 package_status[pkg_name] = ('in', veh_name)

        # Calculate heuristic based on goal packages
        for package_name, goal_location in self.goal_package_locations.items():
            current_status = package_status.get(package_name)

            # If package is not in the state facts at all, or its status is unexpected
            # This indicates an inconsistent or unreachable state
            if current_status is None:
                 return math.inf

            status_type, current_loc_or_veh = current_status

            if status_type == 'at':
                current_location = current_loc_or_veh
                # If package is already at its goal location, cost is 0 for this package
                if current_location == goal_location:
                    continue

                # Package is at a location, needs pickup and transport
                min_vehicle_dist_to_pickup = math.inf

                # Find the closest vehicle to the package's current location
                # Only calculate if there are vehicles and the package's location is known
                if vehicle_locations and current_location in self.locations:
                    for veh_name, veh_location in vehicle_locations.items():
                        # Ensure vehicle location is also a known location
                        if veh_location in self.locations:
                            # Look up the precomputed distance
                            dist = self.distances.get((veh_location, current_location), math.inf)
                            min_vehicle_dist_to_pickup = min(min_vehicle_dist_to_pickup, dist)
                        # If veh_location is not in self.locations, its distance to anything is inf

                # If no vehicles, or package location/vehicle location is isolated/unknown
                if min_vehicle_dist_to_pickup == math.inf:
                     # Cannot get a vehicle to the package
                     return math.inf

                # Distance from package location to goal location
                # Ensure current_location and goal_location are known locations
                if current_location in self.locations and goal_location in self.locations:
                    dist_pickup_to_goal = self.distances.get((current_location, goal_location), math.inf)
                else:
                    # One or both locations are isolated/unknown
                    dist_pickup_to_goal = math.inf

                if dist_pickup_to_goal == math.inf:
                     # Cannot reach goal location from package location
                     return math.inf

                # Estimated cost for this package:
                # vehicle travel to pickup + pickup action + package travel to goal + drop action
                cost_p = min_vehicle_dist_to_pickup + 1 + dist_pickup_to_goal + 1
                h += cost_p

            elif status_type == 'in':
                vehicle_name = current_loc_or_veh
                # Package is in a vehicle, needs transport to goal and drop
                if vehicle_name in vehicle_locations:
                    vehicle_location = vehicle_locations[vehicle_name]
                    # Distance from vehicle's current location to goal location
                    # Ensure vehicle_location and goal_location are known locations
                    if vehicle_location in self.locations and goal_location in self.locations:
                        dist_vehicle_to_goal = self.distances.get((vehicle_location, goal_location), math.inf)
                    else:
                        # One or both locations are isolated/unknown
                        dist_vehicle_to_goal = math.inf

                    if dist_vehicle_to_goal == math.inf:
                         # Cannot reach goal location from vehicle location
                         return math.inf

                    # Estimated cost for this package:
                    # vehicle travel to goal + drop action
                    cost_p = dist_vehicle_to_goal + 1
                    h += cost_p
                else:
                    # Package is in a vehicle, but vehicle location is unknown - invalid state
                    return math.inf
            # Else: status_type is something unexpected, this package is problematic
            # The initial check `if current_status is None` handles the case where
            # the package isn't found at all. If it's found but status_type isn't
            # 'at' or 'in', it's also an invalid state.
            else:
                 return math.inf


        # Final check: if h is 0, ensure it's a true goal state
        if h == 0:
            # Check if ALL goal facts are satisfied in the state
            return 0 if self.task.goal_reached(state_facts) else 1
        elif h == math.inf:
             return math.inf
        else:
            return h
