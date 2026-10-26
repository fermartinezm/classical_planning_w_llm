# Assuming Heuristic base class is available elsewhere,
# but not strictly required for this standalone code.
# from heuristics.heuristic_base import Heuristic

from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format, maybe log a warning or raise error
        # For robustness in a heuristic, returning an empty list is safer
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for safety

    return fact[1:-1].split()

class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions (pick-up, drive, drop) required
    to move each package to its goal location independently.

    Heuristic value for a package:
    - If at goal: 0
    - If on ground at location L (L != Goal): 1 (pick-up) + dist(L, Goal) (drive) + 1 (drop)
    - If in vehicle V, and V is at location L_v: dist(L_v, Goal) (drive) + 1 (drop)

    Total heuristic is the sum of costs for all packages not at their goal.
    Uses precomputed shortest path distances on the road network.
    Returns float('inf') if any goal package is unreachable or state is malformed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Identifies packages and vehicles. Precomputes shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.packages = set()
        self.vehicles = set()
        all_locations = set()
        self.graph = {} # Road network graph: location -> list of connected locations

        # 1. Identify packages, vehicles, and collect all locations from initial state and goals
        # A package is any object that appears in a goal (at pkg loc) or in an (in pkg veh) fact.
        # A vehicle is any object that appears in a (capacity veh size) or (in pkg veh) fact.
        # Any other locatable object appearing in an (at obj loc) fact is assumed to be a vehicle.

        # Collect packages and vehicles from initial state and goals
        for fact in initial_state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                if loc: # Ensure loc is not empty from get_parts error
                    all_locations.add(loc)
            elif predicate == "in":
                package, vehicle = args
                if package and vehicle: # Ensure parts are not empty
                    self.packages.add(package)
                    self.vehicles.add(vehicle)
            elif predicate == "capacity":
                 vehicle, size = args
                 if vehicle: # Ensure vehicle is not empty
                    self.vehicles.add(vehicle)

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                if package and location: # Ensure parts are not empty
                    self.packages.add(package)
                    all_locations.add(location)
            # Assuming goals are only (at package location)

        # Add any objects from initial state 'at' facts that weren't identified as packages/vehicles yet
        # Assume they are vehicles if they are locatable but not packages
        for fact in initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "at":
                 obj, loc = args
                 if obj and loc: # Ensure parts are not empty
                     if obj not in self.packages and obj not in self.vehicles:
                         # This object is 'at' a location but wasn't seen in 'in' or 'capacity'.
                         # Based on domain types, it must be a vehicle.
                         self.vehicles.add(obj)


        # 2. Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                if package and location: # Ensure parts are not empty
                    self.package_goals[package] = location
            # Assuming goals are only (at package location)


        # 3. Build the road network graph from static facts
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                if l1 and l2: # Ensure parts are not empty
                    self.graph.setdefault(l1, []).append(l2)
                    all_locations.add(l1)
                    all_locations.add(l2)

        # Ensure all identified locations are keys in the graph, even if they have no outgoing roads
        for loc in all_locations:
             self.graph.setdefault(loc, [])


        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        locations_list = list(all_locations) # Use a list for consistent ordering if needed

        for start_node in locations_list:
            q = deque([(start_node, 0)])
            visited = {start_node: 0}
            self.distances[(start_node, start_node)] = 0 # Distance to self is 0

            while q:
                current_loc, dist = q.popleft()

                # Explore neighbors
                if current_loc in self.graph:
                    for neighbor in self.graph[current_loc]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            self.distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

            # Store infinity for unreachable locations from start_node
            for end_node in locations_list:
                 if (start_node, end_node) not in self.distances:
                      self.distances[(start_node, end_node)] = float('inf')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Sum of (pick-up + drive + drop) costs for each package not at goal.
        Returns float('inf') if any goal package is unreachable or state is malformed.
        """
        state = node.state

        # Track current locations of packages and vehicles
        package_current_locations = {} # Maps package -> location or vehicle
        vehicle_current_locations = {} # Maps vehicle -> location

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                if obj and loc: # Ensure parts are not empty
                    if obj in self.packages:
                         package_current_locations[obj] = loc
                    elif obj in self.vehicles:
                         vehicle_current_locations[obj] = loc
                # Ignore other 'at' facts if any
            elif predicate == "in":
                package, vehicle = args
                if package and vehicle: # Ensure parts are not empty
                    if package in self.packages and vehicle in self.vehicles:
                        package_current_locations[package] = vehicle # Package is inside vehicle
                # Ignore 'in' facts for non-packages/non-vehicles if any

        total_cost = 0

        # Check if all goal packages are present in the state facts
        # This is a basic check for state validity relative to goals
        if not all(pkg in package_current_locations for pkg in self.package_goals):
             # Some goal package is not found in the state facts provided.
             # This might indicate an invalid state representation or a package was lost.
             # Return infinity as it's likely unsolvable or malformed.
             return float('inf')


        for package, goal_location in self.package_goals.items():
            current_location = package_current_locations[package] # We checked existence above

            # If package is already at the goal
            if current_location == goal_location:
                continue

            # Package is not at the goal. Calculate cost for this package.
            cost_for_package = 0

            if current_location in self.graph: # current_location is a location string (package is on the ground)
                # Needs pick-up, drive, drop
                dist = self.distances.get((current_location, goal_location), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable from here
                cost_for_package = 1 + dist + 1 # pick-up + drive + drop
            elif current_location in self.vehicles: # current_location is a vehicle string (package is inside a vehicle)
                vehicle = current_location
                vehicle_location = vehicle_current_locations.get(vehicle)

                if vehicle_location is None:
                    # Vehicle carrying the package is not 'at' any location in the state
                    # This indicates an issue with state representation
                    return float('inf')

                # Needs drive (by vehicle) + drop
                dist = self.distances.get((vehicle_location, goal_location), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal unreachable from vehicle's location
                cost_for_package = dist + 1 # drive + drop
            else:
                 # current_location is neither a known location nor a known vehicle
                 # This indicates an issue with state representation
                 return float('inf')


            total_cost += cost_for_package

        # Heuristic is 0 only if total_cost is 0, which happens when all packages
        # are already at their goal locations.
        return total_cost
