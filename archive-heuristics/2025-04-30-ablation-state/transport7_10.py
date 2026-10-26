# from heuristics.heuristic_base import Heuristic # This line would be present in the actual environment

# Helper function
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS implementation
def bfs(start_node, graph):
    """
    Performs BFS from a start node on a graph.
    Returns a dictionary of distances from the start node to all reachable nodes.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        if current_node in graph: # Handle nodes with no outgoing edges
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, ignoring vehicle capacity constraints
    and vehicle availability/location relative to packages. It sums the
    estimated costs for each package that is not yet at its goal location.

    # Assumptions
    - The goal is always to have packages at specific locations (not inside vehicles).
    - Any vehicle can carry any package (capacity is ignored).
    - A suitable vehicle is always available at the package's current location
      when needed for pick-up or transport.
    - The cost of each action (drive, pick-up, drop) is 1.
    - Roads are bidirectional (implied by example instances).

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph of locations based on the static 'road' facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package.
    2. For each package that has a specific goal location:
       a. If the package is already at its goal location (on the ground), the cost for this package is 0.
       b. If the package is on the ground at a location different from its goal:
          - It needs to be picked up (1 action).
          - It needs to be transported by a vehicle from its current location to the goal location. The minimum number of drive actions is the shortest path distance between the locations.
          - It needs to be dropped at the goal location (1 action).
          - Total estimated cost for this package: `distance(current_location, goal_location) + 1 (pick-up) + 1 (drop)`.
       c. If the package is inside a vehicle:
          - Find the current location of the vehicle.
          - If the vehicle is at the package's goal location:
             - The package needs to be dropped (1 action).
             - Total estimated cost for this package: `1 (drop)`.
          - If the vehicle is at a location different from the package's goal:
             - The package needs to be transported by the vehicle from the vehicle's current location to the package's goal location. The minimum number of drive actions is the shortest path distance between the locations.
             - The package needs to be dropped at the goal location (1 action).
             - Total estimated cost for this package: `distance(vehicle_location, goal_location) + 1 (drop)`.
    3. The total heuristic value for the state is the sum of the estimated costs for all packages not yet at their goal location.
    4. A large penalty (1000) is added if the goal location is unreachable from the package's or vehicle's current location in the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        # The Heuristic base class is assumed to be initialized correctly
        # super().__init__(task) # If the base class needs task

        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other goal types if any, as per problem examples

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional based on example instances
                self.road_graph.setdefault(l2, []).append(l1)

        self.locations = list(locations) # Store list of locations

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_l in self.locations:
            self.distances[start_l] = bfs(start_l, self.road_graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        # packages_in_vehicles: package -> vehicle
        # object_locations: object (package or vehicle) -> location
        packages_in_vehicles = {}
        object_locations = {}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "in":
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles[package] = vehicle
            elif predicate == "at":
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc

        total_cost = 0  # Initialize action cost counter.
        UNREACHABLE_PENALTY = 1000 # Penalty for unreachable locations

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # A package is at its goal location on the ground if (at package goal_location) is true
            # AND it is NOT inside any vehicle.
            is_at_goal_on_ground = (package in object_locations and
                                    object_locations[package] == goal_location and
                                    package not in packages_in_vehicles)

            if is_at_goal_on_ground:
                 # Package is at goal location on the ground. Cost is 0 for this package.
                 continue

            # Find the package's current location or container
            if package in packages_in_vehicles:
                # Package is inside a vehicle
                vehicle = packages_in_vehicles[package]
                # Find the vehicle's location
                if vehicle in object_locations:
                    vehicle_l = object_locations[vehicle]
                    if vehicle_l != goal_location:
                        # Package is in vehicle, vehicle is not at goal location. Needs drive + drop.
                        # Get distance from vehicle's current location to package's goal location
                        if vehicle_l in self.distances and goal_location in self.distances[vehicle_l]:
                             dist = self.distances[vehicle_l][goal_location]
                             total_cost += dist + 1 # drive actions + drop action
                        else:
                             # Goal location is unreachable from vehicle's current location
                             total_cost += UNREACHABLE_PENALTY
                    else: # vehicle_l == goal_location
                        # Package is in vehicle, vehicle is at goal location. Needs drop.
                        total_cost += 1 # drop action
                else:
                    # Vehicle location unknown? This state is likely invalid or indicates a problem.
                    # Add a large penalty.
                    total_cost += UNREACHABLE_PENALTY
            elif package in object_locations:
                # Package is on the ground, not at goal location (checked by the 'continue' above)
                current_l = object_locations[package]
                # Needs pick-up, drive, drop.
                # Get distance from package's current location to package's goal location
                if current_l in self.distances and goal_location in self.distances[current_l]:
                    dist = self.distances[current_l][goal_location]
                    total_cost += dist + 2 # pick-up + drive actions + drop action
                else:
                    # Goal location is unreachable from package's current location
                    total_cost += UNREACHABLE_PENALTY
            else:
                 # Package location unknown? This state is likely invalid or indicates a problem.
                 # Add a large penalty.
                 total_cost += UNREACHABLE_PENALTY


        return total_cost
