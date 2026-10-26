from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest distances
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from a start_node to all other nodes
    in a graph using BFS.

    Args:
        graph: Adjacency list representation (dict: node -> list of neighbors).
        start_node: The starting node for BFS.

    Returns:
        A dictionary mapping node -> distance from start_node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost for a package includes pick-up (if on the ground), driving, and drop-off.
    - Vehicle capacity and availability are not explicitly modeled; it's assumed
      a suitable vehicle is available when needed.
    - The cost of driving between locations is the shortest path distance in the
      road network.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Identifies all vehicles from the initial state's capacity facts.
    - Builds a graph representation of the road network from static facts.
    - Collects all relevant locations (from roads, goals, initial vehicle/package positions).
    - Precomputes all-pairs shortest path distances between all collected locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle. Also, note which packages are inside which vehicles.
    2. For each package that has a goal location:
       a. Check if the package is already at its goal location. If yes, the cost for this package is 0.
       b. If the package is on the ground at a location different from its goal:
          - Estimate the cost as 1 (for pick-up) + the shortest distance from the package's current location to its goal location (for driving) + 1 (for drop-off).
       c. If the package is inside a vehicle:
          - Find the current location of the vehicle.
          - Estimate the cost as the shortest distance from the vehicle's current location to the package's goal location (for driving) + 1 (for drop-off).
    3. The total heuristic value for the state is the sum of the estimated costs for all goal packages.
    4. If any required distance calculation results in infinity (locations are disconnected), the heuristic returns infinity, indicating an likely unsolvable state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, identifying
        vehicles, building the road network graph, and computing distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static
        # Initial state facts
        initial_state = task.initial_state # Access initial state here

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is always (at package location)
                package, location = args[0], args[1]
                self.goal_locations[package] = location

        # Identify all vehicles from initial state capacity facts
        self.vehicles = set()
        for fact in initial_state:
             if match(fact, "capacity", "*", "*"):
                 self.vehicles.add(get_parts(fact)[1])
        # Also check static facts for capacity (though usually in init)
        for fact in static_facts:
             if match(fact, "capacity", "*", "*"):
                 self.vehicles.add(get_parts(fact)[1])


        # Build the road network graph and collect all locations.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional based on example instances
                self.road_graph.setdefault(loc2, []).append(loc1)

        # Add locations from goals and initial state (for packages and vehicles)
        locations.update(self.goal_locations.values())
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 locations.add(loc)
             # 'in' facts don't directly give a location, the location is the vehicle's location


        # Ensure all collected locations are in the graph keys
        for loc in locations:
             self.road_graph.setdefault(loc, [])


        # Compute all-pairs shortest paths
        all_locations = list(self.road_graph.keys())
        self.distances = {}
        for start_loc in all_locations:
            distances_from_start = bfs_shortest_paths(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

        # Define a large value for unreachable locations
        self.unreachable_penalty = float('inf')


    def get_distance(self, loc1, loc2):
        """Looks up the precomputed shortest distance between two locations."""
        # Ensure both locations are known in our distance map.
        # If not, they are likely unreachable from each other or not part of the road network.
        # Check if loc1 and loc2 were part of the locations set used to build distances
        if loc1 not in self.road_graph or loc2 not in self.road_graph:
             return self.unreachable_penalty

        return self.distances.get((loc1, loc2), self.unreachable_penalty)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        package_status = {} # package_name -> (location_or_vehicle_name, 'ground' or 'in')
        vehicle_locations = {} # vehicle_name -> location_name

        # Populate current locations and containment
        # Process 'in' facts first to correctly identify packages inside vehicles
        packages_in_vehicles = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 if package in self.goal_locations: # Only track goal packages
                     package_status[package] = (vehicle, 'in')
                     packages_in_vehicles.add(package)

        # Process 'at' facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations and obj not in packages_in_vehicles:
                    # It's a goal package on the ground
                    package_status[obj] = (loc, 'ground')
                elif obj in self.vehicles: # Use the pre-identified set of vehicles
                    # It's a vehicle
                    vehicle_locations[obj] = loc

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # If a goal package is not found in the current state facts ('at' or 'in'),
            # it implies an invalid state or the package doesn't exist.
            # For a heuristic, we might assume it's unreachable or already at goal if not found.
            # Given valid states, it should always be present.
            if package not in package_status:
                 # This case indicates an issue with state representation or problem definition
                 # For robustness, treat as unreachable
                 return self.unreachable_penalty

            current_status, status_type = package_status[package]

            # Check if package is already at goal (only applies if on the ground)
            if status_type == 'ground' and current_status == goal_location:
                continue # Package is already at its goal location

            cost_for_package = 0

            if status_type == 'ground':
                # Package is on the ground at current_status, needs to go to goal_location
                # Cost: pick-up + drive + drop
                distance = self.get_distance(current_status, goal_location)
                if distance == self.unreachable_penalty:
                    return self.unreachable_penalty # Goal is unreachable for this package

                # Cost is 1 (pick) + distance (drive) + 1 (drop)
                cost_for_package = 1 + distance + 1

            elif status_type == 'in':
                # Package is in a vehicle (current_status is vehicle_name)
                vehicle_name = current_status
                if vehicle_name not in vehicle_locations:
                     # Vehicle location unknown? Should not happen in valid states.
                     # Treat as unreachable
                     return self.unreachable_penalty

                vehicle_loc = vehicle_locations[vehicle_name]

                # Cost: drive + drop
                distance = self.get_distance(vehicle_loc, goal_location)
                if distance == self.unreachable_penalty:
                    return self.unreachable_penalty # Goal is unreachable for this package

                # Cost is distance (drive) + 1 (drop)
                cost_for_package = distance + 1

            total_cost += cost_for_package

        return total_cost
