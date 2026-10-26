import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function for Breadth-First Search to find shortest paths
def bfs_distances(start_node, graph):
    """
    Performs BFS from start_node on the graph to find distances to all reachable nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    while queue:
        current_node = queue.popleft()
        distance = distances[current_node]
        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the costs for each package independently.
    It ignores vehicle capacity constraints and assumes any vehicle can be used
    for any package.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds the road network graph from static facts.
    - Precomputes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status:
       - Is it on the ground at a location L_curr?
       - Is it inside a vehicle V, which is currently at location L_v?
    2. Determine the package's goal location L_goal.
    3. Calculate the estimated cost for this package:
       - If the package is on the ground at L_curr:
         - If L_curr is the goal location L_goal, the cost is 0 for this package.
         - If L_curr is not the goal location:
           - Find the minimum driving distance from *any* vehicle's current location
             to L_curr (cost: min_dist_v_to_curr).
           - Add the cost of picking up the package (+1).
           - Add the driving distance from L_curr to L_goal (cost: dist_curr_to_goal).
           - Add the cost of dropping the package (+1).
           - Total cost for this package: min_dist_v_to_curr + 1 + dist_curr_to_goal + 1.
       - If the package is inside a vehicle V, which is at location L_v:
         - If L_v is the goal location L_goal:
           - Add the cost of dropping the package (+1).
           - Total cost for this package: 1.
         - If L_v is not the goal location:
           - Add the driving distance from L_v to L_goal (cost: dist_v_to_goal).
           - Add the cost of dropping the package (+1).
           - Total cost for this package: dist_v_to_goal + 1.
    4. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the road network."""
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            parts = get_parts(goal)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # 2. Extract all locations and build the road graph
        locations = set()
        road_graph = {} # Adjacency list {loc1: [loc2, loc3], ...}

        # Infer locations from road facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                road_graph.setdefault(l1, []).append(l2)
                # Note: Assuming roads are explicitly defined in both directions if bidirectional.
                # The example instance confirms this pattern.

        # 3. Precompute all-pairs shortest paths
        self.distances = {} # {(loc1, loc2): distance}
        for start_loc in locations:
            distances_from_start = bfs_distances(start_loc, road_graph)
            for end_loc, dist in distances_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Parse current state to find locations of packages and vehicles
        package_locations = {} # {package: location} if on ground
        package_in_vehicle = {} # {package: vehicle} if in vehicle
        vehicle_locations = {} # {vehicle: location}

        # We need to know all possible packages and vehicles to parse the state correctly.
        # Assume all objects in goal_locations are packages.
        # Assume any object that is 'at' a location but is not a package is a vehicle.
        # Assume any object a package is 'in' is a vehicle.
        all_packages = set(self.goal_locations.keys())
        all_vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                if obj in all_packages:
                     package_locations[obj] = loc
                else: # Assume it's a vehicle
                     vehicle_locations[obj] = loc
                     all_vehicles.add(obj)
            elif predicate == 'in':
                 p, v = parts[1], parts[2]
                 package_in_vehicle[p] = v
                 all_packages.add(p) # Add package if not in goals (unlikely in valid problems)
                 all_vehicles.add(v) # Add vehicle if not seen via 'at'

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Find package's current status
            current_status = None # 'ground', 'in_vehicle'
            current_location = None # location if on ground or vehicle location if in vehicle
            current_vehicle = None # vehicle if in vehicle

            if package in package_locations:
                current_status = 'ground'
                current_location = package_locations[package]
            elif package in package_in_vehicle:
                current_status = 'in_vehicle'
                current_vehicle = package_in_vehicle[package]
                # Need vehicle's location
                if current_vehicle in vehicle_locations:
                    current_location = vehicle_locations[current_vehicle]
                else:
                    # Vehicle location unknown - problem state is malformed?
                    # Or vehicle is carrying package but not 'at' anywhere? Unlikely in this domain.
                    # Treat as unreachable for safety.
                    return float('inf')
            else:
                 # Package is not 'at' a location and not 'in' a vehicle. Malformed state?
                 # Or maybe package is not in initial state but appears later? No, STRIPS is static objects.
                 # Assume unreachable.
                 return float('inf')

            # Now apply the cost logic based on current status and goal
            if current_status == 'ground':
                if current_location == goal_location:
                    # Package is on the ground at the goal. Cost is 0 for this package.
                    pass
                else:
                    # Package is on the ground, not at the goal.
                    # Cost: find closest vehicle + pickup + drive + drop
                    min_dist_v_to_curr = float('inf')
                    # Consider all vehicles to find the closest one to the package
                    for vehicle, v_loc in vehicle_locations.items():
                        if (v_loc, current_location) in self.distances:
                             min_dist_v_to_curr = min(min_dist_v_to_curr, self.distances[(v_loc, current_location)])

                    if min_dist_v_to_curr == float('inf'):
                         # No vehicle can reach the package's current location
                         return float('inf')

                    dist_curr_to_goal = self.distances.get((current_location, goal_location), float('inf'))
                    if dist_curr_to_goal == float('inf'):
                         # Goal location unreachable from package location
                         return float('inf')

                    # Cost = drive vehicle to package + pickup + drive package to goal + drop
                    total_cost += min_dist_v_to_curr + 1 + dist_curr_to_goal + 1

            elif current_status == 'in_vehicle':
                # Package is in a vehicle. current_location is the vehicle's location.
                if current_location == goal_location:
                    # Vehicle is at the goal location. Need to drop.
                    total_cost += 1
                else:
                    # Vehicle is not at the goal location. Need to drive vehicle to goal + drop.
                    dist_v_to_goal = self.distances.get((current_location, goal_location), float('inf'))
                    if dist_v_to_goal == float('inf'):
                         # Goal location unreachable from vehicle location
                         return float('inf')
                    total_cost += dist_v_to_goal + 1 # Drive + Drop

        return total_cost

