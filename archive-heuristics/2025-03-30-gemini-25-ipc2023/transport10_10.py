import collections
import itertools
from fnmatch import fnmatch
# Assuming the Heuristic base class is available at this path
# You might need to adjust the import path based on your project structure
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    # Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    return fact[1:-1].split()

# Optional: A match function similar to the examples (though direct parsing might be sufficient)
def match(fact, *args):
    """Checks if a PDDL fact string matches a given pattern using fnmatch for wildcards."""
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their specified goal locations. It calculates the estimated cost for each
    package independently based on its current state (location or inside a vehicle)
    and sums these costs. The cost estimation includes the minimum number of
    pick-up, drive, and drop actions required for that package, assuming an
    abstract vehicle is always available where needed for transport. It relies on
    pre-calculated shortest path distances between all locations based on the
    static 'road' predicates.

    # Assumptions:
    - Roads defined by `(road l1 l2)` predicates allow travel in both directions
      (l1 to l2 and l2 to l1). This is based on the structure of example instances.
      If roads were unidirectional, the graph construction in `__init__` would need
      to be adjusted accordingly.
    - The cost of each action (drive, pick-up, drop) is uniformly 1.
    - Vehicle capacity constraints (related to `capacity` and `capacity-predecessor`
      predicates) are ignored for simplicity. This means the heuristic does not model
      limitations on how many packages a vehicle can carry simultaneously.
    - Vehicle availability and positioning are ignored. The heuristic assumes a
      vehicle can magically appear at a package's location for pick-up without
      incurring extra driving cost for the vehicle to get there initially.
    - The heuristic primarily focuses on achieving the `(at package location)` goals.
      Other potential goal types (if any were added to the domain) are not
      considered in this calculation.
    - If a package's goal location is determined to be unreachable from its current
      location (due to disconnected parts of the road network), the heuristic
      returns infinity, signaling that the goal is impossible to reach from the
      current state under these assumptions.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's static information (facts
      that never change, like roads) and goal conditions.
    - It extracts all unique location names mentioned in the `road` predicates.
    - It builds an adjacency list representation of the road network graph based
      on the `road` facts.
    - It computes all-pairs shortest path distances between reachable locations using
      Breadth-First Search (BFS) starting from each location. These distances
      (number of `drive` actions) are stored in `self.distances`. Unreachable pairs
      retain a distance of infinity.
    - It parses the task's goal conditions to identify the target location for each
      package specified in an `(at package goal_location)` fact. These package-goal
      pairs are stored in the `self.package_goals` dictionary. It also identifies
      the set of all packages relevant to the goals.

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state node:
    1. Goal Check: First, it checks if the current state `node.state` already satisfies
       all the goal conditions specified in `self.goals`. If yes, the estimated cost
       to reach the goal is 0, and the method returns 0.
    2. State Parsing: If the goal is not met, it iterates through the facts in the
       current state (`node.state`, a frozenset of strings) to determine the current
       status of relevant objects:
       - The location of each vehicle is recorded using `(at vehicle loc)` facts.
       - The status of each package relevant to the goals is recorded:
         - If `(at package loc)` is present, the package is on the ground at `loc`.
         - If `(in package vehicle)` is present, the package is inside `vehicle`.
       These are stored in `vehicle_location` and `package_location` dictionaries.
    3. Cost Aggregation: It initializes a total estimated heuristic cost `h = 0`.
    4. Per-Package Cost Calculation: It iterates through each package `p` that has a
       defined goal location in `self.package_goals`.
       a. Check if package `p` is already at its `goal_loc` on the ground. This is
          checked by seeing if the fact `(at p goal_loc)` exists in the current state.
          If yes, this specific package goal is satisfied, and it contributes 0 to `h`.
       b. If the package goal is not yet satisfied:
          i. Determine if `p` is currently on the ground (`at loc_p`) or in a vehicle (`in v`).
             This information comes from the `package_location` dictionary populated in step 2.
          ii. **Case 1: `p` is on the ground at `loc_p`:**
              - The estimated minimum sequence of actions is: `pick-up` at `loc_p`, `drive`
                from `loc_p` to `goal_loc`, `drop` at `goal_loc`.
              - It retrieves the shortest path distance `d = self.get_dist(loc_p, goal_loc)`.
              - If `d` is infinity, the goal is unreachable for this package; the method
                immediately returns `float('inf')`.
              - The estimated cost contribution for this package is `1 (pickup) + d (drive) + 1 (drop) = d + 2`.
              - This cost (`d + 2`) is added to the total heuristic value `h`.
          iii. **Case 2: `p` is in vehicle `v`:**
              - First, find the current location of vehicle `v` from `vehicle_location`. Let this be `loc_v`.
              - **Subcase 2a: `loc_v == goal_loc`:** The package is in a vehicle that is already
                at the target location. Only a `drop` action is needed.
                  - Cost contribution is `1 (drop)`.
                  - Add 1 to `h`.
              - **Subcase 2b: `loc_v != goal_loc`:** The vehicle needs to drive from `loc_v` to
                `goal_loc`, and then the package needs to be dropped.
                  - Retrieve the shortest path distance `d = self.get_dist(loc_v, goal_loc)`.
                  - If `d` is infinity, return `float('inf')`.
                  - Cost contribution is `d (drive) + 1 (drop) = d + 1`.
                  - Add `d + 1` to `h`.
    5. Final Value: After iterating through all packages with goals, the method returns the
       total accumulated cost `h`. If at any point a required path was found to be impossible
       (distance is infinity), `float('inf')` would have been returned earlier.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts (roads) and goals.
        Pre-computes all-pairs shortest path distances between locations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # --- Graph Construction and Shortest Path Calculation ---
        self.locations = set()
        self.adj = collections.defaultdict(list)
        # Parse static facts to find locations and build road graph
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Check if the fact represents a road connection
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add edge l1 -> l2
                self.adj[l1].append(l2)
                # Assumption: Roads are bidirectional based on examples. Add reverse edge.
                # If domain specified unidirectional roads, this line should be removed or conditional.
                self.adj[l2].append(l1)

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = collections.defaultdict(lambda: float('inf'))
        for loc in self.locations:
            # Distance from a location to itself is 0
            self.distances[loc, loc] = 0
            # Initialize queue for BFS starting from loc
            queue = collections.deque([(loc, 0)])
            # Keep track of visited nodes and their distances in this specific BFS run
            visited_dist_from_loc = {loc: 0}

            while queue:
                curr_loc, dist = queue.popleft()

                # Explore neighbors
                for neighbor in self.adj[curr_loc]:
                    # If neighbor hasn't been reached yet in this BFS
                    if neighbor not in visited_dist_from_loc:
                        new_dist = dist + 1
                        visited_dist_from_loc[neighbor] = new_dist
                        # Store the shortest distance found so far
                        self.distances[loc, neighbor] = new_dist
                        # Add neighbor to the queue for further exploration
                        queue.append((neighbor, new_dist))

        # --- Goal Processing ---
        self.package_goals = {} # Dictionary mapping package name -> goal location name
        self.packages = set()   # Set of all package names mentioned in goals
        # Parse goal facts to find target locations for packages
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal is of the form (at package location)
            if parts[0] == 'at' and len(parts) == 3:
                 # We need a reliable way to know if parts[1] is a package.
                 # Assuming objects mentioned as the first arg of 'at' in goals are packages.
                 # A robust solution might involve checking object types from the task definition.
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
                self.packages.add(package)
                # Sanity check: Warn if the goal location isn't part of the known road network.
                if location not in self.locations:
                     print(f"Warning: Goal location '{location}' for package '{package}' "
                           f"is not found among locations connected by roads. Goal might be unreachable.")


    def get_dist(self, loc1, loc2):
        """Returns the pre-calculated shortest distance between two locations."""
        # Handle cases where one or both locations might not be in the network
        # (e.g., if a goal location was isolated).
        if loc1 not in self.locations or loc2 not in self.locations:
            return float('inf')
        # Return the precomputed distance (defaults to infinity if unreachable)
        return self.distances[loc1, loc2]

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Estimates the minimum number of actions required to achieve the goals.
        """
        state = node.state

        # Check if the current state already satisfies all goal conditions
        if self.goals <= state:
            # If all goals are met, the cost to reach the goal is 0
            return 0

        # --- State Parsing: Find current locations of packages and vehicles ---
        package_location = {} # Maps package name -> current location name or vehicle name
        vehicle_location = {} # Maps vehicle name -> current location name
        packages_in_vehicle = set() # Stores names of packages currently inside a vehicle

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            # Track 'at' facts for packages and vehicles
            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                # Check if the object is a package relevant to our goals
                if obj in self.package_goals:
                    package_location[obj] = loc
                # Basic check if the object is a vehicle (e.g., starts with 'v').
                # This should be replaced with type checking if type info is available.
                elif obj.startswith('v'):
                    vehicle_location[obj] = loc
            # Track 'in' facts for packages
            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                # Check if the package is relevant to our goals
                if package in self.package_goals:
                    # Store the vehicle name as the package's 'location'
                    package_location[package] = vehicle
                    packages_in_vehicle.add(package)

        # --- Heuristic Calculation: Sum costs for each unmet package goal ---
        heuristic_value = 0
        # Iterate through all packages that have a defined goal location
        for package, goal_loc in self.package_goals.items():

            # Construct the goal fact string for this package
            goal_fact_for_package = f"(at {package} {goal_loc})"
            # Check if this specific goal is already satisfied in the current state
            if goal_fact_for_package in state:
                # If yes, this package contributes 0 to the heuristic cost
                continue

            # If the package goal is not met, calculate the estimated cost to satisfy it.
            # First, ensure we know the current state of the package.
            if package not in package_location:
                # If a package needed for a goal is not found in the state ('at' or 'in'),
                # something is wrong. Return infinity to signal an issue or unreachable goal.
                # print(f"Warning: State of package {package} needed for goal is unknown in state.")
                return float('inf')

            # Get the current position/container of the package
            current_pos = package_location[package]

            # Case 1: Package is currently inside a vehicle
            if package in packages_in_vehicle:
                vehicle = current_pos # current_pos stores the vehicle name
                # We need the location of this vehicle. Check if it's known.
                if vehicle not in vehicle_location:
                    # Inconsistent state: package is in a vehicle whose location is unknown.
                    # print(f"Warning: Location of vehicle {vehicle} carrying {package} is unknown.")
                    return float('inf')

                current_vehicle_loc = vehicle_location[vehicle]

                # Subcase 1a: Vehicle is already at the goal location
                if current_vehicle_loc == goal_loc:
                    # Only needs 1 action: drop the package.
                    heuristic_value += 1
                # Subcase 1b: Vehicle is not at the goal location
                else:
                    # Needs drive actions + 1 drop action.
                    dist = self.get_dist(current_vehicle_loc, goal_loc)
                    # Check if the goal location is reachable from the vehicle's location
                    if dist == float('inf'):
                        # If not reachable, the overall goal is impossible from this state.
                        return float('inf')
                    # Add cost: drive distance + 1 for the drop action
                    heuristic_value += dist + 1

            # Case 2: Package is currently on the ground
            else:
                current_loc = current_pos # current_pos stores the location name
                # Needs 1 pickup action + drive actions + 1 drop action.
                dist = self.get_dist(current_loc, goal_loc)
                # Check if the goal location is reachable from the package's current location
                if dist == float('inf'):
                    # If not reachable, the overall goal is impossible.
                    return float('inf')
                # Add cost: 1 for pickup + drive distance + 1 for drop
                heuristic_value += 1 + dist + 1

        # Return the total estimated cost. It should be non-negative.
        return heuristic_value
