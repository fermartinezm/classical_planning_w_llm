from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes leading/trailing parentheses and splits by space."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        # print(f"Warning: Malformed fact string: {fact}")
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum actions
        required for each package to reach its goal location, ignoring
        vehicle capacity and availability constraints. The cost for a package
        depends on whether it is currently at its goal location, at a different
        location, or inside a vehicle. It includes costs for pick-up, drop,
        and driving (based on shortest path distance).

    Assumptions:
        - The road network is static and provided by (road ?l1 ?l2) facts.
        - All locations relevant to goal packages (appearing in initial state,
          static road facts, or goal facts) are considered nodes in the graph.
        - Vehicle capacity is ignored.
        - Vehicle availability is ignored (assumes a vehicle is available when
          needed for pick-up or driving).
        - The state representation is consistent, listing the location of
          goal packages via 'at' or 'in' facts, and vehicle locations via 'at' facts.

    Heuristic Initialization:
        1. Parses the goal state to identify the target location for each package
           that needs to be at a specific location.
        2. Parses static facts and initial state facts to identify all relevant
           locations and build the road network graph. Assumes roads are bidirectional.
        3. Computes all-pairs shortest paths between locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
        1. For a given state, identify the current location of every locatable
           object ('at' facts) and which packages are inside which vehicles ('in' facts).
        2. Based on 'at' and 'in' facts, determine the immediate container/location
           for each goal package (either a location string or a vehicle string).
        3. Identify all vehicles (objects with 'capacity' facts) and find their
           current locations from the 'at' facts.
        4. Initialize the total heuristic value to 0.
        5. Iterate through each package that has a goal location specified in the task.
        6. For the current package:
            a. Check if the package is already at its goal location according to the
               goal condition (i.e., if the fact '(at package goal_location)' is
               present in the current state). If yes, add 0 to the total heuristic
               and proceed to the next package.
            b. If the package is not at its goal location, find its current status
               (at a location or in a vehicle) using the information gathered in steps 1-3.
            c. If the package is at a location (not in a vehicle):
               - Calculate the shortest distance from its current location to its goal location.
               - If the goal is unreachable, return infinity for the total heuristic.
               - The estimated cost for this package is 1 (pick-up) + distance + 1 (drop).
            d. If the package is inside a vehicle:
               - Find the current location of the vehicle.
               - If the vehicle's location is the package's goal location:
                 - The estimated cost for this package is 1 (drop).
               - If the vehicle's location is different from the package's goal location:
                 - Calculate the shortest distance from the vehicle's location to the package's goal location.
                 - If the goal is unreachable, return infinity for the total heuristic.
                 - The estimated cost for this package is distance + 1 (drop).
            e. Add the estimated cost for the current package to the total heuristic.
        8. Return the total heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Parse goals to get package goal locations
        self.package_goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                if len(parts) >= 3:
                    package, location = parts[1], parts[2]
                    self.package_goal_locations[package] = location

        # 2. Build road network graph and collect all locations
        self.road_graph = defaultdict(list)
        all_locations = set()

        # Collect locations from static road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                 if len(parts) >= 3:
                    l1, l2 = parts[1], parts[2]
                    self.road_graph[l1].append(l2)
                    # Assuming roads are bidirectional based on examples
                    self.road_graph[l2].append(l1)
                    all_locations.add(l1)
                    all_locations.add(l2)

        # Collect locations from initial state 'at' facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 if len(parts) >= 3:
                    location = parts[2]
                    all_locations.add(location)

        # Collect locations from goal 'at' facts (already done in step 1, but add to all_locations set)
        for loc in self.package_goal_locations.values():
             all_locations.add(loc)

        # Ensure all collected locations are keys in the graph, even if isolated
        for loc in all_locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = [] # Add isolated locations

        # 3. Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in all_locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc, self.road_graph)

    def _bfs(self, start_node, graph):
        """Helper to compute shortest paths from a start node."""
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()
            # Ensure current_node is a valid key in the graph
            if current_node not in graph:
                 continue

            current_dist = distances[current_node]

            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Looks up precomputed shortest path distance."""
        # Return infinity if loc1 or loc2 are not in our precomputed paths
        # or if loc2 is unreachable from loc1.
        if loc1 not in self.shortest_paths or loc2 not in self.shortest_paths.get(loc1, {}):
             # This might happen if a location appears in a state but wasn't in init/static/goals
             # or if the graph is disconnected.
             # print(f"Warning: Distance requested for unknown or unreachable locations: {loc1} to {loc2}")
             return float('inf')
        return self.shortest_paths[loc1][loc2]

    def __call__(self, node):
        state = node.state

        # Map locatable object (package or vehicle) to its current location/container
        # Location can be a location string or a vehicle string (if inside)
        current_locatable_info = {} # obj -> location_string or vehicle_string

        # Map vehicle to its current location string
        vehicle_locations = {} # vehicle_string -> location_string

        # Identify vehicles (objects with capacity)
        vehicles_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "capacity"}

        # First pass: Find where everything is listed as 'at'
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                if len(parts) >= 3:
                    obj, location = parts[1], parts[2]
                    current_locatable_info[obj] = location
                    # If it's a vehicle, store its location separately
                    if obj in vehicles_in_state:
                         vehicle_locations[obj] = location

        # Second pass: Find which packages are 'in' vehicles
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "in":
                if len(parts) >= 3:
                    package, vehicle = parts[1], parts[2]
                    # If a package is 'in' a vehicle, its immediate container is the vehicle
                    current_locatable_info[package] = vehicle


        # Calculate total heuristic
        total_heuristic = 0

        # Iterate through packages with goals
        for package, goal_location in self.package_goal_locations.items():
            # Check if the package is already at its goal location (and not inside a vehicle)
            # This is the goal condition for the package.
            if f'(at {package} {goal_location})' in state:
                 # Package is at goal location, cost is 0 for this package
                 continue # Move to the next package

            # If we are here, the package is NOT at its goal location.
            # Find its current status (at location or in vehicle)
            current_info = current_locatable_info.get(package)

            if current_info is None:
                 # Goal package not found in state facts ('at' or 'in').
                 # This indicates an invalid state or an unreachable problem.
                 # Return infinity to prune this path.
                 # print(f"Warning: Goal package {package} not found in state facts.")
                 return float('inf')

            cost_for_package = 0

            if current_info in vehicles_in_state: # current_info is a vehicle name
                vehicle_name = current_info
                current_location = vehicle_locations.get(vehicle_name)

                if current_location is None:
                     # Vehicle carrying package not found at any location. Invalid state.
                     # print(f"Warning: Vehicle {vehicle_name} carrying {package} not found at any location.")
                     return float('inf')

                # Package is in a vehicle at current_location
                if current_location == goal_location:
                    # Vehicle is at goal location, just need to drop
                    cost_for_package = 1 # Drop action
                else:
                    # Vehicle needs to drive, then drop
                    dist = self.get_distance(current_location, goal_location)
                    if dist == float('inf'):
                        # Goal location unreachable from vehicle's current location
                        # print(f"Warning: Goal location {goal_location} unreachable from vehicle location {current_location} for package {package}.")
                        return float('inf')
                    cost_for_package = dist + 1 # Drive actions + Drop action

            else: # current_info is a location name
                current_location = current_info

                # Package is at current_location (which is not the goal location, checked above)
                # Need to pick up, drive, and drop
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'):
                    # Goal location unreachable from package's current location
                    # print(f"Warning: Goal location {goal_location} unreachable from {current_location} for package {package}.")
                    return float('inf')
                cost_for_package = 1 + dist + 1 # Pick-up action + Drive actions + Drop action

            total_heuristic += cost_for_package

        return total_heuristic
