import math
from collections import deque
from fnmatch import fnmatch

# Try to import the base class Heuristic. If it fails, define a dummy class.
# This makes the code runnable even if the environment setup is slightly different.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class if the real one is not available."""
        def __init__(self, task):
            self.task = task # Store task for potential use by subclasses
        def __call__(self, node):
            raise NotImplementedError("Heuristic calculation not implemented.")

def get_parts(fact):
    """
    Helper function to parse PDDL fact strings.
    Removes surrounding parentheses and splits the string by spaces.
    Returns a list of strings (predicate name and arguments).
    Returns an empty list if the fact string is malformed.

    Example: "(at package1 locationA)" -> ["at", "package1", "locationA"]
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the remaining number of actions required to reach the
    goal state. It calculates the cost for each package individually based on its
    current state (at a location or inside a vehicle) and its goal location. The
    total heuristic value is the sum of these individual costs. It assumes unit
    costs for actions (drive=1, pick-up=1, drop=1). This heuristic is designed
    for use with Greedy Best-First Search and is not required to be admissible.

    # Assumptions
    - Action costs are uniform: drive=1, pick-up=1, drop=1.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored.
      It assumes that a suitable vehicle will be available and have capacity when needed.
    - The cost of moving a vehicle to a package's location *before* picking it up
      is not included. The heuristic only counts actions starting from the pickup
      (if the package is on the ground) or from the vehicle's current location
      (if the package is already inside).
    - The road network is defined by static `(road l1 l2)` predicates. These define
      a directed graph, and shortest paths are calculated based on this structure.
    - All packages that need to reach a specific destination are specified in the
      goal conditions using `(at package location)` predicates. Packages not
      mentioned in the goals do not contribute to the heuristic value.

    # Heuristic Initialization
    - Extracts the set of packages and their target locations directly from the
      task's goal predicates of the form `(at package location)`.
    - Parses the static `(road l1 l2)` facts from the task's static information
      to build a directed graph representing the locations and connections.
    - Computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS). Stores these distances in a dictionary, using
      `float('inf')` to represent unreachable location pairs.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state node (`node`):
    1. Retrieve the set of facts representing the current state (`state = node.state`).
    2. Check if the current state satisfies all goal conditions (`self.goals <= state`).
       If true, the goal is reached; return a heuristic value of 0.
    3. Initialize the total heuristic estimate `h = 0.0` (using float for intermediate math).
    4. Identify all vehicles currently present in the state. This is done by finding
       objects that appear as the first argument in `(capacity vehicle size)` facts.
    5. Parse the `state` facts to determine the current status of packages and vehicles:
       - `current_package_locations`: Maps package -> location for packages `(at p l)`.
       - `package_in_vehicle`: Maps package -> vehicle for packages `(in p v)`.
       - `vehicle_locations`: Maps vehicle -> location for vehicles `(at v l)`. Only stores
         locations for objects confirmed to be vehicles (from step 4).
    6. Iterate through each package `p` that was identified during initialization (i.e., those with a specified goal location):
       a. Get the goal location `loc_goal` for package `p`.
       b. Check if the specific goal for `p`, `(at p loc_goal)`, is already true in `state`. If yes, the cost contribution for this package is 0.
       c. If `p` is currently on the ground at `loc_p` (found in `current_package_locations`):
          i. Retrieve the precomputed shortest path distance `d = distance(loc_p, loc_goal)`.
          ii. If `loc_p` or `loc_goal` are not valid locations in the precomputed distance map, or if `d` is infinity (meaning `loc_goal` is unreachable from `loc_p`), then the state is considered unsolvable from this point; return `float('inf')`.
          iii. Add the estimated cost `1 (pickup) + d (drive) + 1 (drop)` to the total heuristic value `h`.
       d. If `p` is currently inside vehicle `v` (found in `package_in_vehicle`):
          i. Find the current location `loc_v` of vehicle `v` using `vehicle_locations`.
          ii. If `loc_v` cannot be found (e.g., the vehicle has no `(at v l)` fact), the state is inconsistent; return `float('inf')`.
          iii. Retrieve the precomputed shortest path distance `d = distance(loc_v, loc_goal)`.
          iv. If `loc_v` or `loc_goal` are invalid locations or `d` is infinity, return `float('inf')`.
          v. Add the estimated cost `d (drive) + 1 (drop)` to `h`.
       e. If the status of package `p` cannot be determined (it's neither `at` a location nor `in` a vehicle), this indicates a potential state inconsistency; return `float('inf')`.
    7. After iterating through all relevant packages:
       - If `h` accumulated to `float('inf')`, return `float('inf')`.
       - Convert the calculated `h` to an integer `final_h` (rounding may handle minor float issues).
       - If `final_h` is 0 (and the state is not a goal state, as checked in step 2), return 1. This ensures the heuristic value is strictly positive for any non-goal state.
       - Otherwise, return the calculated integer value `final_h`.
    """

    def __init__(self, task):
        super().__init__(task) # Call base class constructor
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify packages and their goal locations from goal facts
        self.packages = set()
        self.goal_locations = {} # package -> goal_location
        for goal in self.goals:
            parts = get_parts(goal)
            # Assume goals relevant to packages are strictly (at package location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                 package, location = parts[1], parts[2]
                 # Assume the first argument of 'at' in a goal is a package
                 self.packages.add(package)
                 self.goal_locations[package] = location

        # 2. Build road graph and compute all-pairs shortest paths
        self.locations = set()
        adj = {} # Adjacency list: location -> [list of reachable locations in 1 step]

        for fact in static_facts:
            parts = get_parts(fact)
            # Process (road l1 l2) facts
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Initialize adjacency list for l1 if not present
                if l1 not in adj: adj[l1] = []
                # Ensure l2 is also prepared as a potential key, even if it has no outgoing roads
                if l2 not in adj: adj[l2] = []
                # Add directed edge l1 -> l2, avoid duplicates if listed multiple times
                if l2 not in adj[l1]:
                    adj[l1].append(l2)

        # Ensure all identified locations are keys in the adjacency list, even isolated ones
        for loc in self.locations:
            if loc not in adj:
                adj[loc] = []

        # Compute and store shortest path distances
        self.distances = self._compute_all_pairs_shortest_paths(self.locations, adj)


    def _compute_all_pairs_shortest_paths(self, locations, adj):
        """
        Computes shortest path distances using BFS starting from each location.
        Returns a dictionary of dictionaries: distances[start_loc][end_loc].
        Uses float('inf') for unreachable pairs.
        """
        # Initialize distance matrix with infinity
        distances = {loc: {other_loc: float('inf') for other_loc in locations} for loc in locations}

        for start_node in locations:
            # Check if start_node is a valid location key (it should be)
            if start_node not in locations: continue

            distances[start_node][start_node] = 0 # Distance to self is 0
            queue = deque([(start_node, 0)]) # Queue stores (node, distance_from_start)
            # visited_dist keeps track of shortest distance found *during this BFS run*
            visited_dist = {start_node: 0}

            while queue:
                current_node, current_dist = queue.popleft()

                # Explore neighbors if the current node has outgoing roads
                if current_node in adj:
                    for neighbor in adj[current_node]:
                        # Ensure the neighbor is a known location
                        if neighbor in locations:
                            # If neighbor hasn't been visited or a shorter path is found
                            # (the second condition is mainly for non-unit costs, but safe)
                            if neighbor not in visited_dist or visited_dist[neighbor] > current_dist + 1:
                                 new_dist = current_dist + 1
                                 visited_dist[neighbor] = new_dist
                                 # Update the main distance matrix for the start_node
                                 distances[start_node][neighbor] = new_dist
                                 queue.append((neighbor, new_dist))
                        # else: Optional: Log warning about neighbor not in known locations set

        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # --- Goal Check ---
        # If the current state satisfies all goal conditions, heuristic is 0.
        if self.goals <= state:
            return 0

        h_value = 0.0 # Use float for calculations involving infinity

        # --- State Parsing ---
        # Identify vehicles based on 'capacity' facts in the current state
        vehicles = set()
        # Store current locations of packages and vehicles, and package containment
        current_package_locations = {} # package -> location
        package_in_vehicle = {}      # package -> vehicle
        vehicle_locations_all = {}   # Stores 'at' for all objects initially

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'capacity' and len(args) == 2:
                vehicles.add(args[0]) # This object is a vehicle
            elif predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                vehicle_locations_all[obj] = loc # Store location for all objects 'at' something
                if obj in self.packages: # Check if it's a package we track
                    current_package_locations[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                if package in self.packages: # Check if it's a package we track
                    package_in_vehicle[package] = vehicle

        # Filter vehicle locations to only include actual vehicles identified
        vehicle_locations = {v: loc for v, loc in vehicle_locations_all.items() if v in vehicles}

        # --- Heuristic Calculation per Package ---
        for package in self.packages:
            # Packages considered are only those listed in goal_locations (from init)
            if package not in self.goal_locations: continue
            goal_loc = self.goal_locations[package]

            package_cost = 0.0

            # Check if this package's specific goal is already met
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                package_cost = 0.0 # Already at destination
            elif package in current_package_locations:
                # Case 1: Package is on the ground
                current_loc = current_package_locations[package]
                # Check if locations are valid keys in the distance map
                if current_loc not in self.distances or goal_loc not in self.distances:
                     # This indicates an issue, e.g., location not in static 'road' facts
                     return float('inf')
                dist = self.distances[current_loc].get(goal_loc, float('inf'))

                if dist == float('inf'): return float('inf') # Goal is unreachable
                # Cost: pickup(1) + drive(dist) + drop(1)
                package_cost = 1.0 + dist + 1.0
            elif package in package_in_vehicle:
                # Case 2: Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_locations:
                    vehicle_loc = vehicle_locations[vehicle]
                    # Check if locations are valid keys
                    if vehicle_loc not in self.distances or goal_loc not in self.distances:
                         return float('inf') # Invalid location
                    dist = self.distances[vehicle_loc].get(goal_loc, float('inf'))

                    if dist == float('inf'): return float('inf') # Goal is unreachable
                    # Cost: drive(dist) + drop(1)
                    package_cost = dist + 1.0
                else:
                    # Inconsistency: Package is in a vehicle, but vehicle has no location
                    return float('inf')
            else:
                 # Case 3: Package is not 'at' and not 'in'. State inconsistency.
                 return float('inf')

            h_value += package_cost

        # --- Final Value Calculation ---
        # If any calculation resulted in infinity, propagate it
        if h_value == float('inf'):
            return float('inf')

        # Convert the final heuristic value to an integer
        # Rounding handles potential minor floating point inaccuracies if any occurred
        final_h = int(round(h_value))

        # Ensure the heuristic is 0 *only* for goal states.
        # If h=0 but state is not goal, return 1.
        if final_h == 0:
            # We already established state is not goal (at the beginning)
            return 1
        else:
            # Return the calculated positive integer heuristic value
            return final_h

