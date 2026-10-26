from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the estimated costs for each package that is not yet at its goal location. The cost for a package is estimated as the sum of:
    1. A pick-up action (if the package is on the ground).
    2. The shortest path distance (number of drive actions) required to move the package (or the vehicle carrying it) from its current location to its goal location.
    3. A drop action (if the package is not already on the ground at the goal).
    Vehicle capacity and availability are ignored in this estimation.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Roads are bidirectional (derived from example instance structure).
    - Vehicle capacity and availability do not block package movement (this is a relaxation).
    - Shortest path distances on the road network represent the minimum number of drive actions needed for transport.
    - Objects starting with 'v' are vehicles, and objects starting with 'p' are packages (based on examples).

    # Heuristic Initialization
    - The road network graph is constructed from the static `(road ?l1 ?l2)` facts.
    - All-pairs shortest path distances are precomputed on the road network using Breadth-First Search (BFS).
    - The goal location for each package is extracted from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package and the location for every vehicle by parsing the `(at ...)` and `(in ...)` facts in the state.
    2. Initialize the total heuristic cost to 0.
    3. For each package specified in the goal:
       a. Determine the package's goal location from the precomputed goal information.
       b. Find the package's current status (location or vehicle) from the parsed state information.
       c. If the package is `(at package current_location)`:
          i. If `current_location` is the goal location, the cost for this package is 0. Continue to the next package.
          ii. If `current_location` is not the goal location, the package needs to be picked up (1 action), transported, and dropped (1 action). The transport cost is estimated as the shortest path distance from `current_location` to the goal location using the precomputed distances. Add `1 (pick-up) + shortest_path_distance(current_location, goal_location) + 1 (drop)` to the total cost. If the goal is unreachable from the current location, return infinity.
       d. If the package is `(in package vehicle)`:
          i. Find the current location of the `vehicle` from the parsed state information.
          ii. The package needs to be transported while in the vehicle and then dropped (1 action). The transport cost is estimated as the shortest path distance from the vehicle's current location to the package's goal location. Add `shortest_path_distance(vehicle_current_location, goal_location) + 1 (drop)` to the total cost. If the goal is unreachable from the vehicle's current location, return infinity.
          iii. Note: If the vehicle is already at the goal location, the shortest path distance is 0, and only the drop action cost (1) is added.
    4. The total heuristic value is the sum of the estimated costs for all goal packages. If any required shortest path is infinite (goal unreachable), the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing shortest paths and storing goal locations."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all locations and road connections
        locations = set()
        road_graph = {} # Adjacency list: location -> set of connected locations

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on example instances
                road_graph.setdefault(l2, set()).add(l1)

        self.locations = list(locations) # Store locations
        self.road_graph = road_graph # Store graph

        # Precompute all-pairs shortest paths using BFS
        self.shortest_path_distances = {} # Map (start_loc, end_loc) -> distance

        for start_loc in self.locations:
            # Run BFS from start_loc
            queue = collections.deque([(start_loc, 0)]) # Use deque for efficiency
            visited = {start_loc}
            self.shortest_path_distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            while queue:
                current_loc, current_dist = queue.popleft()

                # Explore neighbors
                for neighbor in self.road_graph.get(current_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.shortest_path_distances[(start_loc, neighbor)] = current_dist + 1
                        queue.append((neighbor, current_dist + 1))

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Track where packages and vehicles are currently located/contained.
        current_status = {} # Maps object name -> location string or vehicle name
        vehicle_locations = {} # Maps vehicle name -> location string

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                # Assuming vehicles start with 'v' based on examples
                if obj.startswith("v"):
                     vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside this vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Get current status (location string or vehicle name)
            current_status_pkg = current_status.get(package)

            # If a goal package is not found in the state, it's an issue.
            # Assuming all goal packages are present in the initial state and subsequent states.
            # If not found, it cannot be at the goal. Treat as if it needs transport.
            # A robust heuristic might return infinity here, but let's assume valid states.
            # If current_status_pkg is None, it implies the package object isn't in any 'at' or 'in' fact.
            # This shouldn't happen in a valid state representation for a goal object.
            # Let's add a check just in case, although it might indicate a problem outside the heuristic.
            if current_status_pkg is None:
                 # Cannot determine package location, assume unreachable or error state
                 return float('inf') # Indicate very high cost

            # Check if package is at goal location on the ground
            if current_status_pkg == goal_location:
                # Package is already at its goal location on the ground
                continue # Cost for this package is 0

            # Package is not at goal location on the ground.
            # It is either at a different location on the ground, or inside a vehicle.

            if current_status_pkg in vehicle_locations: # Check if the status is a known vehicle name
                vehicle_name = current_status_pkg
                # Find the location of the vehicle
                vehicle_location = vehicle_locations.get(vehicle_name)
                # vehicle_location should exist if vehicle_name is in vehicle_locations

                # Cost: drive vehicle from its current location to package's goal location + drop
                # If vehicle is already at goal location, drive cost is 0.
                drive_cost = self.shortest_path_distances.get((vehicle_location, goal_location))

                if drive_cost is None:
                     # Goal is unreachable from vehicle's current location
                     return float('inf') # Indicate very high cost

                total_cost += drive_cost + 1 # drive + drop

            else: # Package is at a location on the ground (but not the goal location)
                package_location = current_status_pkg # This should be a location name

                # Cost: pick-up + drive from package location to goal location + drop
                # Find shortest path from package_location to goal_location
                drive_cost = self.shortest_path_distances.get((package_location, goal_location))

                if drive_cost is None:
                     # Goal is unreachable from package's current location
                     return float('inf') # Indicate very high cost

                total_cost += 1 + drive_cost + 1 # pick-up + drive + drop

        return total_cost
