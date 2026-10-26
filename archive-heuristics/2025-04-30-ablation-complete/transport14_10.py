# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class transportHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity and shared vehicle usage. The cost
    includes shortest path vehicle travel and necessary pick-up/drop actions.

    # Assumptions
    - Each package can be moved independently.
    - Vehicles are always available when needed at the package's current location
      or the vehicle's current location.
    - Vehicle capacity constraints are ignored.
    - The cost of moving a vehicle between two locations is the shortest path
      distance in the road network (each drive action costs 1).
    - The cost of pick-up and drop actions is 1 each.
    - Unreachable goals for a package incur a large penalty.

    # Heuristic Initialization
    - Extracts all location, vehicle, and package objects from the task definition's object list.
    - Builds a graph representing the road network from static facts.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Extracts the goal location for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every locatable object (packages and vehicles) by scanning 'at' and 'in' facts in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Get the package's current position (either a location name if 'at', or a vehicle name if 'in').
       b. If the package's current position is its goal location: The package is at its goal on the ground. Cost for this package is 0.
       c. If the package is currently on the ground at a location different from its goal:
          - Estimate the cost as the shortest path distance from its current location to its goal location (for vehicle travel) plus 2 actions (1 for pick-up, 1 for drop). If the goal is unreachable, add a large penalty.
       d. If the package is currently inside a vehicle:
          - Find the current location of the vehicle using the 'at' facts for vehicles in the current state.
          - Estimate the cost as the shortest path distance from the vehicle's location to the package's goal location (for vehicle travel) plus 1 action (1 for drop). If the goal is unreachable from the vehicle's location, add a large penalty.
    4. Sum the estimated costs for all packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        all_objects = task.objects # Assuming task.objects is a list of strings like "obj - type"

        # 1. Extract all location, vehicle, and package objects based on type
        self.locations = set()
        self.vehicles = set()
        self.packages = set() # Store packages to distinguish from vehicles if needed
        for obj_str in all_objects:
            parts = obj_str.split()
            if len(parts) == 3 and parts[1] == '-':
                name, obj_type = parts[0], parts[2]
                if obj_type == 'location':
                    self.locations.add(name)
                elif obj_type == 'vehicle':
                    self.vehicles.add(name)
                elif obj_type == 'package':
                    self.packages.add(name)
                # Assuming locatable objects are either vehicles or packages

        # 2. Build the road network graph
        self.graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                if l1 in self.graph and l2 in self.graph: # Ensure locations are valid
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Roads are bidirectional

        # 3. Compute all-pairs shortest path distances
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                package, location = parts[1], parts[2]
                # Only consider goals for packages that are actually packages and target valid locations
                if package in self.packages and location in self.locations:
                    self.goal_locations[package] = location
                # else: ignore goals for non-packages or invalid locations

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             return distances # Start node is not a valid location

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is a valid location and has neighbors in the graph
            if current_node not in self.graph:
                 continue # Should not happen if start_node was valid

            for neighbor in self.graph.get(current_node, set()): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current location/container for all locatable objects (packages and vehicles)
        current_positions = {} # Map object -> location or vehicle
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'at':
                    obj, loc = parts[1], parts[2]
                    current_positions[obj] = loc
                elif parts[0] == 'in':
                    pkg, veh = parts[1], parts[2]
                    current_positions[pkg] = veh # Package is inside a vehicle

        total_cost = 0
        UNREACHABLE_PENALTY = 1000 # Define a large penalty

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            current_pos_pkg = current_positions.get(package)

            # If package is not in the state (shouldn't happen in valid problems)
            if current_pos_pkg is None:
                 total_cost += UNREACHABLE_PENALTY
                 continue

            # Check if the package is already at its goal location on the ground
            # current_pos_pkg will be a location name if the package is 'at' that location
            if current_pos_pkg == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # If package is on the ground at a different location
            if current_pos_pkg in self.locations: # Check if the value is a location name
                current_l = current_pos_pkg
                # Cost: drive distance + pick-up + drop
                # Need to handle unreachable locations - BFS returns inf.
                drive_cost = self.distances[current_l].get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    total_cost += UNREACHABLE_PENALTY
                else:
                    total_cost += drive_cost + 2 # 1 for pick-up, 1 for drop

            # If package is inside a vehicle
            elif current_pos_pkg in self.vehicles: # Check if the value is a vehicle name
                 vehicle = current_pos_pkg
                 vehicle_l = current_positions.get(vehicle) # Get vehicle's location

                 # Check if vehicle location is known and is a valid location
                 if vehicle_l is None or vehicle_l not in self.locations:
                     # Vehicle location unknown or not a valid location. Add penalty.
                     total_cost += UNREACHABLE_PENALTY
                 else:
                     # Cost: drive distance + drop
                     drive_cost = self.distances[vehicle_l].get(goal_location, float('inf'))
                     if drive_cost == float('inf'):
                         total_cost += UNREACHABLE_PENALTY
                     else:
                         total_cost += drive_cost + 1 # 1 for drop

            # If current_pos_pkg is neither a location nor a known vehicle (should not happen in valid states)
            else:
                 # Unexpected state representation for the package's position
                 total_cost += UNREACHABLE_PENALTY


        return total_cost
