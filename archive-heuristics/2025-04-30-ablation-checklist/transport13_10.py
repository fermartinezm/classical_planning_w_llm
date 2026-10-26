import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    cost for each package that is not yet at its goal location. The cost for
    a package is estimated based on its current state (on the ground or in a vehicle)
    and the shortest path distance from its current location (or its vehicle's
    location) to its goal location. Capacity constraints are ignored.

    # Assumptions
    - Packages can be on the ground or in a vehicle.
    - Vehicles can move between connected locations.
    - The cost of pick-up and drop actions is 1.
    - The cost of a drive action is 1 per road segment traversed (shortest path).
    - Capacity constraints are ignored for heuristic calculation.
    - Vehicle availability is not explicitly modeled; any package on the ground
      is assumed to eventually find a vehicle, and a package in a vehicle is
      assumed to be transported by that vehicle.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task.
    - Build a graph of locations based on `road` facts from static information.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify packages and vehicles from initial state and goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of every package (either on the ground or inside a vehicle).
    2. Identify the current location of every vehicle.
    3. For each package that is not yet at its goal location:
       a. If the package is on the ground at location `l_current`:
          The estimated cost for this package is 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) + 1 (drop).
       b. If the package is inside a vehicle `v`:
          Find the current location `l_v` of vehicle `v`.
          The estimated cost for this package is shortest_distance(`l_v`, `l_goal`) + 1 (drop).
       c. If the goal location is unreachable from the package's current location/vehicle's location, the cost is infinity.
    4. The total heuristic value is the sum of the estimated costs for all packages not at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each package.
        self.goal_locations = {}  # {package_name: goal_location_name}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Collect all locations and build the road graph.
        self.locations = set()
        self.graph = collections.defaultdict(list) # {location: [neighbor1, neighbor2, ...]}

        # Collect objects to distinguish packages from vehicles later
        self.packages_in_goals = set(self.goal_locations.keys())
        self.vehicles_in_init_or_goals = set() # Objects appearing as vehicles

        # Parse static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph[l1].append(l2)
                # Assuming roads are bidirectional unless specified otherwise
                # The example shows bidirectional roads, so let's assume that.
                self.graph[l2].append(l1)
            elif parts and parts[0] == "capacity-predecessor":
                 # Sizes are not needed for this heuristic
                 pass
            # Add any locations mentioned in static facts even if not in roads
            for part in parts[1:]:
                 if part.startswith('l'): # Simple heuristic for location names
                     self.locations.add(part)


        # Parse initial state facts to find more locations and identify vehicles
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 if obj not in self.packages_in_goals:
                     # If it's not a package we care about for the goal, assume it's a vehicle or other object
                     # We primarily care about vehicles' locations
                     self.vehicles_in_init_or_goals.add(obj)
             elif parts and parts[0] == "in":
                 pkg, veh = parts[1], parts[2]
                 self.packages_in_goals.add(pkg) # Ensure packages mentioned in 'in' are tracked
                 self.vehicles_in_init_or_goals.add(veh)
             elif parts and parts[0] == "capacity":
                 veh, size = parts[1], parts[2]
                 self.vehicles_in_init_or_goals.add(veh)
             # Add any locations mentioned in initial state facts
             for part in parts[1:]:
                 if part.startswith('l'): # Simple heuristic for location names
                     self.locations.add(part)


        # Ensure all locations from goals are included
        for loc in self.goal_locations.values():
             self.locations.add(loc)


        # Compute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        Returns a dictionary distances[start_loc][end_loc] = distance.
        Unreachable locations have distance float('inf').
        """
        distances = {loc: {other_loc: float('inf') for other_loc in self.locations}
                     for loc in self.locations}

        for start_loc in self.locations:
            distances[start_loc][start_loc] = 0
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state.

        # Track current state of packages relevant to goals and vehicle locations
        package_locations = {}      # {package_name: location} if on ground
        package_in_vehicles = {}    # {package_name: vehicle_name} if in vehicle
        vehicle_locations = {}      # {vehicle_name: location}

        # Parse the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we need to deliver
                    package_locations[obj] = loc
                elif obj in self.vehicles_in_init_or_goals: # It's a vehicle
                    vehicle_locations[obj] = loc
                # Ignore other 'at' facts (e.g., at-robby in gripper example, though not in transport)
            elif predicate == "in":
                pkg, veh = parts[1], parts[2]
                if pkg in self.goal_locations: # It's a package we need to deliver
                    package_in_vehicles[pkg] = veh
                # Ignore 'in' facts for objects not in goal_locations if any
            # Ignore capacity facts for heuristic calculation

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location in the current state
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal. Estimate cost based on its current state.
            cost_for_package = float('inf') # Initialize cost for this package

            if package in package_locations:
                # Package is on the ground at package_locations[package]
                current_loc = package_locations[package]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                drive_cost = self.distances.get(current_loc, {}).get(goal_location, float('inf'))
                if drive_cost != float('inf'):
                     cost_for_package = 1 + drive_cost + 1

            elif package in package_in_vehicles:
                # Package is in vehicle package_in_vehicles[package]
                vehicle = package_in_vehicles[package]
                # Find vehicle's current location
                vehicle_loc = vehicle_locations.get(vehicle)

                if vehicle_loc is not None:
                    # Cost: drive (distance) + drop (1)
                    drive_cost = self.distances.get(vehicle_loc, {}).get(goal_location, float('inf'))
                    if drive_cost != float('inf'):
                        cost_for_package = drive_cost + 1
                # If vehicle_loc is None, something is wrong with state parsing or state itself,
                # cost_for_package remains inf.

            # Add the estimated cost for this package to the total
            total_cost += cost_for_package

        # If total_cost is infinity, it means at least one package goal is unreachable
        # from its current state based on shortest paths.
        return total_cost if total_cost != float('inf') else -1 # Return -1 for unsolvable/unreachable

