# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each package not at its goal location,
    the shortest path distance from its current location (or its vehicle's location) to its goal location,
    plus the costs for necessary load/unload actions. It assumes each package move is independent.

    # Assumptions
    - Roads are bidirectional (handled by adding edges in both directions).
    - Shortest path distance between locations is a reasonable estimate for vehicle movement cost (1 action per road segment).
    - Vehicle capacity and availability are not strictly modeled for individual package costs,
      leading to a non-admissible heuristic.
    - Each package requires 1 load action (if on ground) and 1 unload action (if in vehicle or needs unloading at goal).
    - All packages that need to reach a goal location are listed in the task's goals.
    - Object types (package, vehicle) are declared using `(object - type)` facts in the initial state.
    - All relevant locations are mentioned in 'road' facts or 'at' facts in the initial state or goals.
    - The road network is connected for all relevant locations in solvable problems.

    # Heuristic Initialization
    - Parses object types (packages, vehicles) from `(object - type)` facts in the initial state.
    - Collects all relevant locations from 'road' facts in static information and 'at' facts in the initial state and goals.
    - Parses goal locations for each package from the task's goals.
    - Builds a graph of locations based on 'road' facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state by comparing the state facts with the task's goal facts. If all goal facts are present in the state, return 0.
    2. Initialize total heuristic cost to 0.
    3. Identify the current location of every package and vehicle by iterating through the state facts. Keep track of which package is in which vehicle.
    4. For each package that is listed in the task's goals (meaning it needs to reach a specific location):
       a. Get the package's goal location from the precomputed goal information.
       b. Check if the goal fact `(at package goal_location)` is already present in the current state. If it is, this package has reached its goal *on the ground*, so continue to the next package.
       c. Determine the package's effective current location.
          - If the package is currently recorded as being `(at package current_loc)` on the ground, its effective location is `current_loc`.
          - If the package is currently recorded as being `(in package vehicle)`, its effective location is the current location of that `vehicle` (found in step 3).
          - If the package's location/status is not found in the state (should not happen in valid states), return infinity as the state is likely invalid or leads to an unreachable goal.
       d. Calculate the shortest path distance `d` from the effective current location to the goal location using the precomputed distance table. If no path exists between these locations in the road network, return infinity as the goal is unreachable for this package.
       e. Estimate the minimum number of actions required for this package, ignoring vehicle capacity and coordination with other packages:
          - If the package was on the ground at its effective current location: It needs a vehicle to load it (1 action), the vehicle needs to move (`d` actions), and it needs to be unloaded at the goal (1 action). Total = `d + 2`.
          - If the package was inside a vehicle at its effective current location: The vehicle needs to move (`d` actions), and the package needs to be unloaded at the goal (1 action). Total = `d + 1`.
       f. Add this estimated cost for the package to the total heuristic cost.
    5. Return the calculated total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        super().__init__(task)

        # Parse object types (assuming facts like (object - type) exist in initial_state)
        self.packages = set()
        self.vehicles = set()
        all_locations = set() # Will populate from roads/at facts

        # Parse object types from initial state facts like (object - type)
        for fact_string in task.initial_state:
             parts = get_parts(fact_string)
             # Look for facts like (object - type)
             if len(parts) == 3 and parts[1] == '-':
                 obj_name = parts[0]
                 obj_type = parts[2]
                 if obj_type == 'package':
                     self.packages.add(obj_name)
                 elif obj_type == 'vehicle':
                     self.vehicles.add(obj_name)
                 # Locations are typically not listed this way, but let's add if they are.
                 elif obj_type == 'location':
                     all_locations.add(obj_name)


        # Collect locations from road facts
        road_graph = {} # Adjacency list: location -> set of connected locations
        for fact_string in task.static:
            parts = get_parts(fact_string)
            if parts and parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)
                road_graph.setdefault(loc1, set()).add(loc2)
                road_graph.setdefault(loc2, set()).add(loc1) # Assuming bidirectional roads

        # Collect locations from initial 'at' facts (vehicles and packages)
        for fact_string in task.initial_state:
            parts = get_parts(fact_string)
            if parts and parts[0] == 'at' and len(parts) == 3:
                loc = parts[2]
                all_locations.add(loc)

        # Collect locations from goal 'at' facts
        for goal_fact_string in task.goals:
             parts = get_parts(goal_fact_string)
             if parts and parts[0] == 'at' and len(parts) == 3:
                  loc = parts[2]
                  all_locations.add(loc)


        self.locations = all_locations # Final set of all relevant locations
        self.road_graph = road_graph # Store graph

        # Compute all-pairs shortest paths using BFS
        self.distances = {} # Store distances (loc1, loc2) -> dist

        for start_node in self.locations:
            q = deque([(start_node, 0)]) # (node, distance)
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Get neighbors from the graph, handle locations not in road_graph (isolated locations)
                # An isolated location is a valid node in the graph, but has no edges.
                neighbors = self.road_graph.get(current_loc, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Parse goal locations for packages
        self.goal_locations = {} # package -> goal_location
        for goal_fact_string in task.goals:
            parts = get_parts(goal_fact_string)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Only add goals for objects identified as packages
                if package in self.packages:
                    self.goal_locations[package] = location
                # else: print(f"Warning: Goal for non-package object {package}")


        # Store goals as a frozenset for quick goal state check
        self._goal_facts = task.goals


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # If the current state is the goal state, the heuristic is 0.
        if self._goal_facts <= state:
             return 0

        # Track current locations of packages and vehicles
        package_current_status = {} # package -> {'type': 'at', 'loc': loc} or {'type': 'in', 'vehicle': v}
        vehicle_locations = {}      # vehicle -> loc

        for fact_string in state:
            parts = get_parts(fact_string)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj_name = parts[1]
                location = parts[2]
                if obj_name in self.packages:
                    package_current_status[obj_name] = {'type': 'at', 'loc': location}
                elif obj_name in self.vehicles:
                    vehicle_locations[obj_name] = location
            elif predicate == 'in' and len(parts) == 3:
                package_name = parts[1]
                vehicle_name = parts[2]
                # Ensure both are known types before adding
                if package_name in self.packages and vehicle_name in self.vehicles:
                     package_current_status[package_name] = {'type': 'in', 'vehicle': vehicle_name}

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Find the package's current status
            current_status = package_current_status.get(package)

            # If package status is unknown, it's an unexpected state.
            # For a heuristic, we can assume it's not at the goal and needs transport.
            # A robust state representation should always have 'at' or 'in' for packages.
            if current_status is None:
                 # This case indicates the package is not 'at' any location and not 'in' any vehicle.
                 # This shouldn't happen in a valid state for this domain.
                 # Treat as unreachable goal for this package, return infinity.
                 # print(f"Error: Package {package} status unknown in state.")
                 return float('inf')


            # Check if the goal fact `(at package goal_location)` is already in the state.
            # This is the most reliable way to check if the package goal is met *on the ground*.
            if f'(at {package} {goal_location})' in state:
                 continue # This package is already at its goal location on the ground.

            # Determine effective current location and base cost
            effective_current_location = None
            base_actions = 0 # Actions other than movement (load/unload)

            if current_status['type'] == 'at':
                # Package is on the ground at current_status['loc']
                effective_current_location = current_status['loc']
                base_actions = 2 # Needs load and unload
            elif current_status['type'] == 'in':
                # Package is in a vehicle
                vehicle = current_status['vehicle']
                # Get vehicle's location
                vehicle_loc = vehicle_locations.get(vehicle)
                if vehicle_loc is None:
                    # Vehicle location unknown - shouldn't happen in valid states
                    # print(f"Error: Vehicle {vehicle} location unknown for package {package} in vehicle.")
                    return float('inf') # Cannot estimate cost if vehicle location is unknown

                effective_current_location = vehicle_loc
                base_actions = 1 # Needs unload

            # Calculate movement cost (shortest path distance)
            # Ensure effective_current_location and goal_location are valid locations in our graph
            if effective_current_location in self.locations and goal_location in self.locations:
                 # Get distance from precomputed table
                 # Handle case where start_node == end_node (distance is 0)
                 if effective_current_location == goal_location:
                     movement_cost = 0
                 else:
                     movement_cost = self.distances.get((effective_current_location, goal_location))

                 if movement_cost is not None:
                     # Total cost for this package = movement cost + base actions
                     total_cost += movement_cost + base_actions
                 else:
                     # No path found between effective_current_location and goal_location
                     # This implies the goal is unreachable from this state.
                     # Return infinity.
                     # print(f"Error: No path found from {effective_current_location} to {goal_location} for package {package}.")
                     return float('inf')

            else:
                 # effective_current_location or goal_location is not a known location node
                 # This indicates a parsing issue or unexpected state structure.
                 # Treat as impossible or high cost.
                 # print(f"Error: Unknown location {effective_current_location} or {goal_location} for package {package}.")
                 return float('inf')


        return total_cost
