# Import necessary modules
from collections import deque
# Assuming heuristics.heuristic_base exists and defines a Heuristic class
# If not, replace with a dummy class or adjust based on actual framework
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the base class is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found")


# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Helper to parse PDDL fact string like '(predicate arg1 arg2)'."""
    # Remove surrounding parentheses and split by spaces
    # Handle potential empty strings or malformed facts gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Or raise an error, depending on expected input robustness
         return []
    return fact.strip()[1:-1].split()

# Helper functions for graph processing (BFS)
def bfs_shortest_path(graph, start):
    """Computes shortest path distances from a start node in a graph."""
    # Initialize distances for all nodes known in the graph
    distances = {location: float('inf') for location in graph}
    # Add any locations that are destinations but not sources in the graph
    for neighbors in graph.values():
         for neighbor in neighbors:
             if neighbor not in distances:
                 distances[neighbor] = float('inf')

    if start in distances: # Ensure start node is a valid location in the graph
        distances[start] = 0
        queue = deque([start])
        while queue:
            current_loc = queue.popleft()
            # Check if current_loc has outgoing edges defined
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    # Ensure neighbor is a known location in distances
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_paths = {}
    # Collect all unique locations involved in the graph (both sources and destinations)
    all_locations = set(graph.keys())
    for neighbors in graph.values():
        all_locations.update(neighbors)

    for start_node in all_locations:
        paths_from_start = bfs_shortest_path(graph, start_node)
        for end_node, dist in paths_from_start.items():
            if dist != float('inf'):
                all_paths[(start_node, end_node)] = dist
    return all_paths


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum actions
    required for each package to reach its goal location independently.
    It considers the current status of each package (at a location or in a vehicle)
    and its goal location. It uses precomputed shortest path distances
    on the road network for drive actions. Vehicle capacity and conflicts
    are ignored.

    Assumptions:
    - The road network is static and provides connections between locations.
    - Any vehicle can theoretically transport any package (capacity is ignored).
    - Vehicles are available where needed (vehicle location relative to package
      pick-up location is partially ignored, only vehicle location when
      carrying a package is considered for drive cost).
    - The cost of each action (drive, pick-up, drop) is 1.
    - Packages relevant to the goal are those appearing in the goal's (at ?p ?l) facts.
    - Vehicles are objects appearing in static (capacity ?v ?s) facts.
    - State representation is consistent with PDDL predicates: packages are either 'at' a location or 'in' a vehicle; vehicles are 'at' a location and have a 'capacity'.

    Heuristic Initialization:
    - Parses static facts to build the road network graph from (road ?l1 ?l2) facts.
      Ensures all locations mentioned are included as nodes in the graph.
    - Computes all-pairs shortest paths between locations using BFS on the road graph.
      Stores distances in a dictionary mapping (start_loc, end_loc) tuples to distance.
    - Parses goal facts to store the target location for each package
      mentioned in (at ?p ?l) goal facts.
    - Identifies all vehicles from static (capacity ?v ?s) facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. For a given state (represented as a frozenset of fact strings),
       iterate through the facts to identify:
       - The current location for every package that is relevant to the goal
         (i.e., listed in self.goal_locations). A package is either '(at ?p ?l)'
         or '(in ?p ?v)'. Store this status (location or vehicle) for each package.
       - The current location for every vehicle identified during initialization.
         Vehicles are typically '(at ?v ?l)'. Store this location for each vehicle.
       Store these findings in dictionaries: package_status and vehicle_locations.
    2. Initialize the total heuristic value to 0.
    3. Iterate through each package 'p' and its goal location 'l_goal'
       as stored in self.goal_locations.
    4. For package 'p' with goal location 'l_goal':
       a. Retrieve the current status of 'p' from the package_status dictionary.
          If the package is not found in the state facts (which indicates an invalid state),
          the goal is unreachable, and the heuristic should reflect this (e.g., return infinity).
       b. If the current status indicates 'p' is inside a vehicle 'v' (i.e., current_status is a vehicle object string):
          - Retrieve the current location 'l_v' of vehicle 'v' from the vehicle_locations dictionary.
            If the vehicle's location is not found (invalid state), return infinity.
          - If 'l_v' is not 'l_goal':
            - The package needs to be transported from 'l_v' to 'l_goal' and then dropped.
            - The minimum drive actions is the shortest path distance between 'l_v' and 'l_goal'.
            - Look up shortest_path(l_v, l_goal) in the precomputed distances. If the path doesn't exist (distance is infinity), return infinity.
            - Add the drive cost + 1 (for the drop action) to the total heuristic.
          - If 'l_v' is 'l_goal':
            - The package is already at the goal location (inside the vehicle). It only needs to be dropped.
            - Add 1 (for the drop action) to the total heuristic.
       c. If the current status indicates 'p' is at a location 'l_curr' (i.e., current_status is a location object string):
          - If 'l_curr' is not 'l_goal':
            - The package needs to be picked up, transported from 'l_curr' to 'l_goal', and then dropped.
            - The minimum drive actions is the shortest path distance between 'l_curr' and 'l_goal'.
            - Look up shortest_path(l_curr, l_goal) in the precomputed distances. If the path doesn't exist (distance is infinity), return infinity.
            - Add 1 (pick-up) + the drive cost + 1 (drop) to the total heuristic.
          - If 'l_curr' is 'l_goal':
            - The package is already at its goal location.
            - Add 0 to the total heuristic for this package.
    5. Return the total heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Build location graph from road facts
        self.location_graph = {}
        all_locations_in_roads = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road':
                if len(parts) == 3: # Ensure correct number of parts
                    l1, l2 = parts[1], parts[2]
                    all_locations_in_roads.add(l1)
                    all_locations_in_roads.add(l2)
                    if l1 not in self.location_graph:
                        self.location_graph[l1] = []
                    self.location_graph[l1].append(l2)
                # else: ignore malformed road fact

        # Ensure all locations mentioned in roads are keys in the graph dict, even if no outgoing edges
        for loc in all_locations_in_roads:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []

        # Compute all-pairs shortest paths
        self.shortest_paths = compute_all_pairs_shortest_paths(self.location_graph)

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                 if len(parts) == 3: # Ensure correct number of parts
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                 # else: ignore malformed goal fact

        # Identify all vehicles from static capacity facts
        self.all_vehicles = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'capacity':
                  if len(parts) == 3: # Ensure correct number of parts
                     self.all_vehicles.add(parts[1])
                  # else: ignore malformed capacity fact


    def __call__(self, node):
        state = node.state

        # Find current locations/carriers for packages and locations for vehicles
        package_status = {} # package -> location or vehicle
        vehicle_locations = {} # vehicle -> location

        # Populate package_status and vehicle_locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            if parts[0] == 'at':
                if len(parts) == 3:
                    obj, location = parts[1], parts[2]
                    # Check if this object is one of the packages we care about (i.e., in goal_locations)
                    if obj in self.goal_locations:
                         package_status[obj] = location
                    # Check if this object is a vehicle
                    elif obj in self.all_vehicles:
                         vehicle_locations[obj] = location
            elif parts[0] == 'in':
                if len(parts) == 3:
                    p, v = parts[1], parts[2]
                    # Check if this package is one we care about
                    if p in self.goal_locations:
                        package_status[p] = v # Store the vehicle it's in
                # else: ignore malformed 'in' fact
            # else: ignore other fact types

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_status = package_status.get(package)

            # If package is not found in state facts (neither 'at' nor 'in'),
            # it's an invalid state for a package that needs to reach a goal.
            # Goal is unreachable.
            if current_status is None:
                return float('inf')

            if current_status in self.all_vehicles: # Package is in a vehicle
                vehicle = current_status
                vehicle_location = vehicle_locations.get(vehicle)

                # If vehicle location is not found, it's an invalid state.
                if vehicle_location is None:
                     # Vehicle exists but is not 'at' any location? Invalid state.
                     return float('inf')

                if vehicle_location != goal_location:
                    # Needs drive + drop
                    drive_cost = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         return float('inf') # Goal unreachable
                    total_cost += drive_cost + 1 # drive + drop
                else: # Vehicle is at goal_location
                    # Needs drop
                    total_cost += 1

            else: # Package is at a location (current_status is a location string)
                current_location = current_status
                if current_location != goal_location:
                    # Needs pick-up + drive + drop
                    drive_cost = self.shortest_paths.get((current_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         return float('inf') # Goal unreachable
                    total_cost += 1 + drive_cost + 1 # pick-up + drive + drop
                # If current_location == goal_location, cost is 0 for this package, already handled.

        return total_cost
