import collections

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def bfs(graph, start):
    """Perform BFS to find shortest distances from a start node in a graph."""
    distances = {start: 0}
    queue = collections.deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        distance = distances[current]
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic: # Inherit from Heuristic
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions (pick-up, drive, drop)
    to move each package from its current location to its goal location,
    summing the minimum costs for each package independently. It uses shortest
    path distances on the road network for the drive cost.

    # Assumptions:
    - The cost of each action (pick-up, drop, drive) is 1.
    - Vehicle capacity is ignored.
    - Vehicle availability is ignored (assumes a vehicle is available when needed).
    - Roads are bidirectional.
    - All locations involved in 'road' facts form the road network.
    - Packages and vehicles are located at points within or reachable from this network.
    - Objects with an 'at' fact that are not packages with goal locations are vehicles.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph representing the road network from static facts.
    - Precomputes shortest path distances between all pairs of locations present in the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every vehicle by looking for `(at v l)` facts. Assume any object with an 'at' fact that is not a package with a goal is a vehicle.
    2. For each package `p` with a goal location `goal_l`:
       - Find the package's current status by looking for `(at p l)` or `(in p v)` facts in the state.
       - Determine the package's effective current location (`current_l`). If `p` is on the ground at `l`, `current_l = l`. If `p` is in vehicle `v` which is at `l_v` (found in step 1), `current_l = l_v`.
       - If the package's status is not found or its vehicle's location is unknown/isolated, the goal is unreachable, return infinity.
       - If `current_l` is the same as `goal_l`:
         - If `p` is on the ground, cost is 0.
         - If `p` is in a vehicle, it needs to be dropped (1 action). Cost is 1.
       - If `current_l` is different from `goal_l`:
         - If `p` is on the ground at `current_l`, it needs pick-up (1 action), transport from `current_l` to `goal_l` (cost is `dist(current_l, goal_l)`), and drop (1 action). Total cost: `1 + dist(current_l, goal_l) + 1`.
         - If `p` is in a vehicle at `current_l`, it needs transport from `current_l` to `goal_l` (cost is `dist(current_l, goal_l)`) and drop (1 action). Total cost: `dist(current_l, goal_l) + 1`.
       - If `dist(current_l, goal_l)` is infinite, the goal is unreachable from the current location, return infinity for the heuristic.
    3. The total heuristic value for the state is the sum of the costs calculated for each package that has a goal location.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        # Assuming task.goals is an iterable of fact strings like state/static
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Goal facts are typically (at package location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road graph and precompute distances.
        self.road_graph = collections.defaultdict(list)
        locations_in_roads = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming roads are bidirectional
                locations_in_roads.add(l1)
                locations_in_roads.add(l2)

        self.distances = {}
        # Run BFS from every location that is part of the road network
        for loc in locations_in_roads:
             self.distances[loc] = bfs(self.road_graph, loc)


    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if loc1 was a starting point for BFS and loc2 was reached from loc1
        # loc1 must be a key in self.distances (i.e., part of the road network)
        if loc1 in self.distances and loc2 in self.distances[loc1]:
            return self.distances[loc1][loc2]
        else:
            # loc1 might not be in the graph, or loc2 is unreachable from loc1.
            # Return infinity.
            return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Track current locations of vehicles.
        vehicle_locations = {}
        # Track current status of packages (location if at, vehicle if in)
        package_status = {}

        # Populate locations and status from the state.
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # If the object is a package we care about (has a goal)
                 if obj in self.goal_locations:
                     package_status[obj] = loc # Package is on the ground
                 # Otherwise, assume it's a vehicle if it has an 'at' fact
                 # This is a heuristic assumption based on domain structure.
                 else:
                     vehicle_locations[obj] = loc # Assume it's a vehicle

            elif predicate == 'in' and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 # If the package is one we care about (has a goal)
                 if pkg in self.goal_locations:
                     package_status[pkg] = veh # Package is in a vehicle


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach a goal.
        for package, goal_location in self.goal_locations.items():
            # If package status is not found, it's an invalid state or parsing issue.
            # Return infinity.
            if package not in package_status:
                 return float('inf')

            current_status = package_status[package]

            # Determine effective current location and calculate cost
            # Case 1: Package is on the ground (current_status is a location name)
            # Check if the status value is a location key we have distances for.
            # This implies it's a location in the road network.
            if current_status in self.distances:
                current_location = current_status
                if current_location != goal_location:
                    # Needs pick-up, drive, drop
                    distance = self.get_distance(current_location, goal_location)
                    if distance == float('inf'):
                         return float('inf')
                    total_cost += 1 + distance + 1 # pick + drive + drop
                # Else: current_location == goal_location, cost is 0

            # Case 2: Package is in a vehicle (current_status is a vehicle name)
            # Check if the status value is a vehicle name whose location we know.
            elif current_status in vehicle_locations:
                vehicle_name = current_status
                veh_location = vehicle_locations[vehicle_name]

                # Check if vehicle location is a location we know about in the graph
                # If the vehicle is at a location not in the road network, distance is inf unless it's the goal.
                if veh_location not in self.distances and veh_location != goal_location:
                     return float('inf')

                if veh_location != goal_location:
                    # Needs drive, drop
                    distance = self.get_distance(veh_location, goal_location)
                    if distance == float('inf'):
                         return float('inf')
                    total_cost += distance + 1 # drive + drop
                else: # veh_location == goal_location
                    # Needs drop
                    total_cost += 1 # drop

            else:
                 # The package's status value is neither a known location in the graph
                 # nor a known vehicle with a location. This indicates an unexpected
                 # state fact or parsing issue. Treat as unreachable goal.
                 return float('inf')

        return total_cost
