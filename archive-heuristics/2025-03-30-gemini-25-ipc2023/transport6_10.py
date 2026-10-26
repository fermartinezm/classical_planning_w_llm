import collections
from fnmatch import fnmatch
import heapq # Not strictly needed for BFS distance, but good practice for graph algorithms

# Try to import the base class; provide a dummy if it fails.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found.")

# Helper function to parse PDDL facts safely
def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Returns a list of strings, or an empty list if the fact is malformed.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    This heuristic estimates the minimum number of actions required to move each
    package specified in the goal to its target location. It calculates the cost
    for each package independently based on its current state (on the ground or
    in a vehicle) and sums these costs. The cost includes estimated driving
    actions (based on precomputed shortest paths in the road network) and the
    necessary pick-up and drop actions (1 action each).

    # Assumptions
    - The heuristic estimates the cost by summing the estimated costs for each
      package individually. It ignores interactions like vehicle capacity limits
      or the potential synergy of multiple packages sharing a single vehicle trip.
    - The driving cost between two locations is the shortest path length (number
      of 'drive' actions) in the static 'road' network.
    - When a package needs to be picked up from the ground, the heuristic assumes
      a vehicle is already present at the package's location for the pick-up.
      This simplifies the calculation but means the heuristic might underestimate
      the true cost (by omitting the cost to drive the vehicle to the package first).
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored
      during the heuristic calculation.
    - The goals of the task are solely defined by `(at package location)` predicates.
      Other types of goals are not considered by this heuristic.
    - Objects are identified as packages based on the naming convention that they
      start with 'p' (e.g., 'p1', 'packageA'). This is derived from the examples.
      A more robust implementation might use PDDL type information if available
      from the task representation.

    # Heuristic Initialization
    - Parses static facts (`task.static`) to build an adjacency list representation
      of the 'road' network. Locations are identified from these facts.
    - Computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS). Distances are stored in `self.distances`.
    - Identifies all objects that are considered packages based on the 'p*' naming
      convention and stores them in `self.packages`.
    - Parses the goal conditions (`task.goals`) to determine the target location
      for each package required by the goal. Stores this mapping in `self.goal_locations`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Initialization (`__init__`)**:
        *   Extract all unique locations from `(road l1 l2)` facts in `task.static`.
        *   Build `self.adj`, an adjacency list where `self.adj[l1]` contains all locations `l2` reachable directly from `l1` via a road.
        *   Run BFS starting from each location `l` to compute the shortest path distance (number of `drive` actions) to all other reachable locations. Store these in `self.distances[l][l_other]`. Initialize unreachable pairs with infinity.
        *   Identify all object names from the task's facts (initial state, goals, static facts). Filter these to find package names (assuming they start with 'p' and are not locations). Store these in `self.packages`.
        *   Iterate through `task.goals`. For each goal fact `(at p goal_loc)` where `p` is in `self.packages` and `goal_loc` is a known location, store the mapping `p -> goal_loc` in `self.goal_locations`.
    2.  **Heuristic Calculation (`__call__`)**:
        *   Initialize the total heuristic estimate `h` to 0.
        *   Parse the current `state` (a set of fact strings) to determine the current status of relevant objects:
            *   `package_location`: Map `p -> l` for packages `p` currently `(at p l)`.
            *   `package_in_vehicle`: Map `p -> v` for packages `p` currently `(in p v)`.
            *   `vehicle_location`: Map `v -> l` for vehicles `v` currently `(at v l)`.
            *   Keep track of all packages (`current_packages`) present in the state.
        *   If `self.goal_locations` is empty (no package goals), return 0.
        *   Iterate through each package `p` and its target location `goal_loc` defined in `self.goal_locations`.
        *   **Check if goal for `p` is met:** If the fact `(at p goal_loc)` exists in the current `state`, this package contributes 0 to `h`. Continue to the next package.
        *   **Check if package `p` exists in state:** If `p` is required for the goal but is not found in `current_packages` (i.e., not `at` anywhere and not `in` anything), the state is considered invalid or the goal unreachable from this state. Return `float('inf')`.
        *   **Determine cost based on `p`'s state:**
            *   **If `p` is in a vehicle `v` (`p` in `package_in_vehicle`):**
                *   Find the vehicle's location `current_vehicle_loc` from `vehicle_location`. If `v` has no location in the state, return `float('inf')` (invalid state).
                *   Look up the shortest distance `drive_dist = self.distances[current_vehicle_loc][goal_loc]`. Use `.get()` for safe access, defaulting to `float('inf')`.
                *   If `drive_dist` is `inf`, the goal is unreachable. Return `float('inf')`.
                *   The estimated cost for `p` is `cost_p = drive_dist + 1` (1 action for `drop`).
            *   **If `p` is on the ground (`p` in `package_location`):**
                *   Find the package's location `current_package_loc`.
                *   Look up the shortest distance `drive_dist = self.distances[current_package_loc][goal_loc]`. Use `.get()` for safe access, defaulting to `float('inf')`.
                *   If `drive_dist` is `inf`, the goal is unreachable. Return `float('inf')`.
                *   The estimated cost for `p` is `cost_p = 1 + drive_dist + 1` (1 for `pick-up`, `drive_dist` for driving, 1 for `drop`).
            *   **If `p` is neither `at` nor `in` (but was found in `current_packages`):** This case should not be logically possible if parsing is correct. Return `float('inf')` as a safeguard.
        *   Add `cost_p` to the total heuristic value `h`.
    3.  After checking all goal packages, return the total calculated `h`. If `h` is 0, it implies all package goals are satisfied in the current state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-computing road network distances,
        identifying packages, and parsing goal locations.
        """
        # If Heuristic base class needs initialization:
        # super().__init__(task)
        self.task = task # Store task for potential future use
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute Road Network and Distances ---
        locations = set()
        self.adj = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Basic validation: assume locations don't start with 'p' or 'v'
                if not loc1.startswith('p') and not loc1.startswith('v'):
                    locations.add(loc1)
                if not loc2.startswith('p') and not loc2.startswith('v'):
                    locations.add(loc2)
                # Add edge if both seem like valid locations
                if loc1 in locations and loc2 in locations:
                     self.adj[loc1].append(loc2)
            # Ignore capacity-predecessor for distance calculation

        self.locations = frozenset(locations)
        self.distances = self._compute_all_pairs_shortest_paths()

        # --- Identify Packages ---
        # Assumes packages start with 'p'. Robust implementation would use PDDL types.
        all_objects = set()
        # Gather objects from initial state, goals, and static facts
        for fact_set in [task.initial_state, task.goals, task.static]:
             for fact in fact_set:
                 parts = get_parts(fact)
                 # Add all arguments (potential objects) except the predicate name
                 if len(parts) > 1:
                     all_objects.update(parts[1:])

        # Filter objects: must start with 'p' and not be a known location
        self.packages = {obj for obj in all_objects if obj.startswith('p') and obj not in self.locations}

        # --- Precompute Goal Locations ---
        self.goal_locations = {} # package -> location
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            # Expect goals like (at package location)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Check if the object is a known package and the location is valid
                if package in self.packages and location in self.locations:
                    self.goal_locations[package] = location
                # Silently ignore goals not matching the expected format or objects


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of known locations
        using Breadth-First Search (BFS).
        Returns a dict of dicts: distances[start][end] = distance.
        """
        distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        for start_node in self.locations:
            # Initialize distance to self as 0
            distances[start_node][start_node] = 0
            # Queue for BFS: stores (node, distance_from_start)
            queue = collections.deque([(start_node, 0)])
            # Keep track of visited nodes within this BFS run to avoid cycles
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors using the precomputed adjacency list
                # Use .get() for safe access in case a location has no outgoing roads
                for neighbor in self.adj.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Record the shortest distance found so far
                        distances[start_node][neighbor] = dist + 1
                        # Add neighbor to the queue for further exploration
                        queue.append((neighbor, dist + 1))
        return distances


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        Estimates the number of actions required to reach the goal state.
        Returns float('inf') if the goal seems unreachable from the state.
        """
        state = node.state
        h_value = 0

        # --- Parse current state ---
        package_location = {} # Map: package -> location (if on ground)
        package_in_vehicle = {} # Map: package -> vehicle (if inside vehicle)
        vehicle_location = {} # Map: vehicle -> location
        current_packages = set() # Set of packages present in the state

        # Identify vehicles - assuming they start with 'v' based on examples
        vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages: # Check if it's a known package
                    package_location[obj] = loc
                    current_packages.add(obj)
                # Assume objects starting with 'v' are vehicles
                elif obj.startswith('v') and obj not in self.locations:
                    vehicle_location[obj] = loc
                    vehicles.add(obj)

            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                if package in self.packages: # Check if it's a known package
                    package_in_vehicle[package] = vehicle
                    current_packages.add(package)
                    # Track the vehicle involved
                    if vehicle.startswith('v'): # Assume 'v' convention
                        vehicles.add(vehicle)

            elif predicate == 'capacity' and len(args) == 2:
                 vehicle = args[0]
                 # Track vehicles mentioned in capacity facts too
                 if vehicle.startswith('v'): # Assume 'v' convention
                     vehicles.add(vehicle)

        # --- Calculate cost for each goal package ---
        if not self.goal_locations: # No package goals defined
             # Heuristic should be 0 if the goal state is already achieved.
             # Check if the goal set is empty or satisfied by the state.
             # Assuming self.task.goal_reached(state) exists and is accurate.
             # If not, we assume 0 if no package goals exist.
             return 0

        goal_achieved_count = 0
        for package, goal_loc in self.goal_locations.items():

            # Construct the goal fact string to check against the state
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                goal_achieved_count += 1
                continue # This package's goal is met, cost is 0.

            # If package is required for goal but not found anywhere in state
            if package not in current_packages:
                 # This implies the goal is unreachable as the package is missing.
                 # print(f"Debug: Goal package {package} not found in state.")
                 return float('inf')

            cost_p = 0 # Cost estimate for this package
            if package in package_in_vehicle:
                # Case 1: Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle not in vehicle_location:
                     # Vehicle exists (carries package) but has no location. Invalid state.
                     # print(f"Debug: Vehicle {vehicle} carrying {package} has no location.")
                     return float('inf')

                current_vehicle_loc = vehicle_location[vehicle]
                # Safely get distance from vehicle's location to goal location
                # Uses precomputed shortest paths (number of drive actions)
                drive_dist = self.distances.get(current_vehicle_loc, {}).get(goal_loc, float('inf'))

                if drive_dist == float('inf'):
                    # Goal location is unreachable from where the vehicle currently is.
                    # print(f"Debug: Goal {goal_loc} unreachable from vehicle {vehicle} at {current_vehicle_loc}.")
                    return float('inf')

                # Cost = drive actions + 1 drop action
                cost_p = drive_dist + 1

            elif package in package_location:
                # Case 2: Package is on the ground at some location
                current_package_loc = package_location[package]
                # Safely get distance from package's location to goal location
                drive_dist = self.distances.get(current_package_loc, {}).get(goal_loc, float('inf'))

                if drive_dist == float('inf'):
                     # Goal location is unreachable from where the package currently is.
                     # print(f"Debug: Goal {goal_loc} unreachable from package location {current_package_loc}.")
                     return float('inf')

                # Cost = 1 pick-up action + drive actions + 1 drop action
                # (Assumes vehicle is magically available at current_package_loc)
                cost_p = 1 + drive_dist + 1
            else:
                # Package is required for goal, exists in state, but neither 'at' nor 'in'.
                # This indicates an inconsistency or error in state representation.
                # print(f"Debug: Package {package} exists but has no location or container.")
                return float('inf')

            # Add the estimated cost for this package to the total heuristic value
            h_value += cost_p

        # Final heuristic value is the sum of costs for all unmet package goals.
        # If h_value is 0, it means all goal packages were already in place.
        return h_value

