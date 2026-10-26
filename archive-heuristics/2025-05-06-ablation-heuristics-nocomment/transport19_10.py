from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Helper to parse PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()


def build_location_graph(static_facts):
    """Builds a graph of locations based on road facts."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'road':
            l1, l2 = parts[1], parts[2]
            locations.add(l1)
            locations.add(l2)
            graph.setdefault(l1, []).append(l2)
            # Assuming roads are bidirectional based on example instances
            graph.setdefault(l2, []).append(l1)

    # Ensure all locations mentioned in roads are keys in the graph
    for loc in locations:
        graph.setdefault(loc, [])

    return graph, list(locations) # Return locations list for BFS


def bfs_shortest_paths(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node exists in graph keys (handles isolated locations)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


def compute_all_pairs_shortest_paths(graph, locations):
    """Computes shortest path distances between all pairs of locations."""
    all_paths = {}
    for start_loc in locations:
        paths_from_start = bfs_shortest_paths(graph, start_loc)
        for end_loc, dist in paths_from_start.items():
            all_paths[(start_loc, end_loc)] = dist
    return all_paths


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required actions
    for each package that is not yet at its goal location. The minimum actions
    for a package involve picking it up (if needed), driving it to the goal
    location, and dropping it. The driving cost is estimated by the shortest
    path distance in the road network. The heuristic ignores vehicle capacity
    constraints and the initial location of vehicles relative to packages needing
    pickup. It assumes a suitable vehicle is available for pickup at the package's
    location and uses the vehicle's current location if the package is already
    inside one.

    Assumptions:
    - Roads are bidirectional (based on example instance).
    - Vehicle capacity is ignored.
    - The presence of a suitable vehicle for pickup at the package's location
      is assumed (cost of moving a vehicle to the package is not included).
    - The heuristic returns infinity if any package needs to travel between
      locations that are not connected by a road path.

    Heuristic Initialization:
    - Parses goal facts to identify target locations for each package.
    - Parses static facts to build the road network graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies packages and vehicles based on their appearance in initial,
      goal, and static facts (specifically 'at', 'in', 'capacity').

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic cost to 0.
    2. Initialize a flag `unreachable` to False.
    3. Parse the current state to determine:
       - The current location of each package (either 'at' a location or 'in' a vehicle).
       - The current location of each vehicle.
       - (Vehicle capacity is ignored by this heuristic).
    4. For each package whose goal location is known:
       a. Check if the package is already at its goal location. If yes, cost for this package is 0.
       b. If the package is currently 'at' a location (not the goal):
          - The estimated cost for this package is 1 (pick-up) + shortest_path(current_location, goal_location) + 1 (drop).
          - Look up the shortest path distance using the precomputed table.
          - If the shortest path is infinity, set `unreachable` to True and break the loop.
          - Add the estimated cost (2 + distance) to the total cost.
       c. If the package is currently 'in' a vehicle:
          - Find the current location of that vehicle.
          - If the vehicle is already at the package's goal location:
             - The estimated cost is 1 (drop).
             - Add 1 to the total cost.
          - If the vehicle is not at the package's goal location:
             - The estimated cost is shortest_path(vehicle_location, goal_location) + 1 (drop).
             - Look up the shortest path distance.
             - If the shortest path is infinity, set `unreachable` to True and break the loop.
             - Add the estimated cost (distance + 1) to the total cost.
    5. If the `unreachable` flag is True, return infinity.
    6. Otherwise, return the total computed cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify objects (packages and vehicles)
        self.packages = set()
        self.vehicles = set()

        # Identify packages from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at': # Assuming goals are always (at package location)
                self.packages.add(parts[1])

        # Identify vehicles from static capacity facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'capacity':
                self.vehicles.add(parts[1])

        # Also add vehicles mentioned in initial state 'at' or 'in' facts
        # This covers vehicles that might not have a capacity fact in static (unlikely but safe)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj = parts[1]
                 # If obj is not a package we identified, assume it's a vehicle
                 if obj not in self.packages:
                     self.vehicles.add(obj)
             elif parts[0] == 'in':
                 vehicle = parts[2]
                 self.vehicles.add(vehicle)


        # 2. Store goal locations for packages
        self.goal_locations = {}
        for package in self.packages:
             # Find the goal location for this package
             for goal in self.goals:
                 parts = get_parts(goal)
                 if parts[0] == 'at' and parts[1] == package:
                     self.goal_locations[package] = parts[2]
                     break # Found goal for this package

        # 3. Build road network graph and compute shortest paths
        self.location_graph, self.locations = build_location_graph(static_facts)
        self.shortest_paths = compute_all_pairs_shortest_paths(self.location_graph, self.locations)

    def __call__(self, node):
        state = node.state

        # 3. Parse current state for package and vehicle locations/status
        package_status = {} # {package: ('at', loc) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: loc}
        # current_vehicle_capacities = {} # Not used by this heuristic

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj in self.packages:
                    package_status[obj] = ('at', loc)
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == 'in':
                package = parts[1]
                vehicle = parts[2]
                package_status[package] = ('in', vehicle)
            # Ignore 'capacity' facts for this heuristic

        # 4. Compute total heuristic cost
        total_cost = 0
        unreachable = False

        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state facts (e.g., problem parsing issue), skip or error
            # This check is mostly for safety; valid states should contain info about all relevant packages.
            if package not in package_status:
                 # This might happen if a package object exists but is not mentioned in any 'at' or 'in' fact in the current state.
                 # This shouldn't occur in standard PDDL state representations derived from initial state and actions.
                 # For this heuristic, we assume all packages in goal_locations are tracked.
                 # If a package from goal_locations is missing from state facts, it's an unexpected state.
                 # We could treat this as unreachable or add a large penalty. Let's assume valid states for now.
                 continue

            status, loc_or_vehicle = package_status[package]

            if status == 'at':
                current_location = loc_or_vehicle
                if current_location != goal_location:
                    # Need to pick up, drive, drop
                    drive_cost = self.shortest_paths.get((current_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                        unreachable = True
                        break # Goal is unreachable for this package
                    total_cost += 1 # pick-up action
                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop action
                # Else: current_location == goal_location, cost for this package is 0

            elif status == 'in':
                vehicle = loc_or_vehicle
                # Need vehicle location
                vehicle_location = vehicle_locations.get(vehicle)
                if vehicle_location is None:
                     # Package is in a vehicle, but vehicle location is unknown?
                     # This indicates an invalid state representation or parsing issue.
                     # Treat as unreachable.
                     unreachable = True
                     break

                if vehicle_location != goal_location:
                    # Need to drive, drop
                    drive_cost = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                        unreachable = True
                        break # Goal is unreachable for this package
                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop action
                else: # vehicle_location == goal_location
                    # Need to drop
                    total_cost += 1 # drop action

        # 5. Return result
        if unreachable:
            return float('inf')

        return total_cost
