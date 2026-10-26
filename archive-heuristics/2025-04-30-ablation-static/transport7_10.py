# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each package not at its goal location,
    the minimum number of actions (pick-up, drop) and the shortest road distance the package (or the vehicle carrying it)
    needs to travel. It relaxes vehicle capacity constraints and assumes vehicles are available when needed.

    # Assumptions:
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored (assumes a vehicle is always available for a package).
    - Vehicle movement cost for a package is the shortest path distance in the road network.
    - The heuristic sums costs for each package independently, potentially overcounting shared vehicle travel.
    - The road network is static and defined by (road l1 l2) facts.
    - All locations relevant to the problem (initial package/vehicle locations, goal locations, locations in road facts) are considered.
    - If a state is solvable, all necessary locations are connected by roads. Unreachable locations are assigned infinite distance.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goal conditions.
    - Identifies packages and vehicles based on predicates in initial state and goals.
    - Builds the road network graph from static (road l1 l2) facts.
    - Computes all-pairs shortest path distances between all relevant locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package (either on the ground or inside a vehicle) and every vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `goal_l` defined in the task:
       a. Check if package `p` is already at its goal location `goal_l` in the current state. If yes, the cost for this package is 0; continue to the next package.
       b. If package `p` is not at its goal:
          i. Determine if `p` is currently on the ground at some location `current_l`, or inside a vehicle `v` which is at location `current_l_v`.
          ii. If `p` is on the ground at `current_l`:
              - Add 1 to the cost for the required pick-up action.
              - Add 1 to the cost for the required drop action at the goal.
              - Add the shortest road distance from `current_l` to `goal_l` to the cost (representing vehicle travel).
          iii. If `p` is inside vehicle `v` which is at `current_l_v`:
              - Add 1 to the cost for the required drop action at the goal.
              - Add the shortest road distance from `current_l_v` to `goal_l` to the cost (representing vehicle travel).
    4. The total heuristic value is the sum of costs calculated for all packages not yet at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        identifying objects, building the road graph, and computing distances.
        """
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages (objects that appear in 'at' goals)
        self.packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        # Identify vehicles (objects that have 'capacity' in initial state)
        self.vehicles = set()
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if parts[0] == "capacity":
                 vehicle = parts[1]
                 self.vehicles.add(vehicle)

        # Build the road graph and compute all-pairs shortest paths.
        all_locations = set()
        road_graph = {}

        # Collect locations and build graph from road facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)
                if l1 not in road_graph:
                    road_graph[l1] = []
                road_graph[l1].append(l2)

        # Collect locations from initial state and goals
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                all_locations.add(parts[2])
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == 'at':
                all_locations.add(parts[2])

        # Ensure all locations are keys in the graph, even if they have no outgoing roads
        for loc in all_locations:
            if loc not in road_graph:
                road_graph[loc] = []

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in all_locations:
            q = [(start_loc, 0)]
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0
            head = 0
            while head < len(q):
                curr_loc, dist = q[head]
                head += 1 # Dequeue

                if curr_loc in road_graph: # Handle locations with no outgoing roads
                    for neighbor in road_graph[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Assign infinite distance to unreachable pairs
        for l1 in all_locations:
            for l2 in all_locations:
                if (l1, l2) not in self.distances:
                     self.distances[(l1, l2)] = float('inf') # Use infinity for unreachable

    def get_distance(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance between two locations.
        Returns float('inf') if there is no path.
        """
        # Distance from a location to itself is 0, handled by BFS initialization
        return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_locations = {} # {package: location} if on ground
        vehicle_locations = {} # {vehicle: location}
        package_in_vehicle = {} # {package: vehicle} if inside vehicle

        # Populate current locations from the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_locations[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal
            if (f"(at {package} {goal_location})") in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at its goal, calculate cost
            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Ensure the vehicle's location is known (should be if state is valid)
                if vehicle not in vehicle_locations:
                     # This indicates an invalid state or missing fact, return infinity
                     return float('inf')

                current_location_vehicle = vehicle_locations[vehicle]

                # Cost includes drop action + vehicle movement
                total_cost += 1 # Drop action
                total_cost += self.get_distance(current_location_vehicle, goal_location) # Vehicle drive

            elif package in package_locations:
                # Package is on the ground
                current_location_package = package_locations[package]

                # Cost includes pick-up action + drop action + vehicle movement
                total_cost += 1 # Pick up action
                total_cost += 1 # Drop action
                total_cost += self.get_distance(current_location_package, goal_location) # Vehicle drive
            else:
                 # Package is not at goal, not on ground, not in vehicle? Invalid state.
                 return float('inf')


        return total_cost
