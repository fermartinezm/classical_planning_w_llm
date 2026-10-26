import math
from collections import deque
# Assuming the Heuristic base class is available, e.g., from heuristics.heuristic_base import Heuristic
# If not available, define a placeholder base class:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Example: "(at package1 locationA)" -> ["at", "package1", "locationA"]
    """
    return fact[1:-1].split()

# BFS for computing shortest paths in the road network
def compute_shortest_paths(locations, roads):
    """
    Computes shortest path distances between all pairs of locations using BFS.

    Args:
        locations (set): A set of all location names.
        roads (dict): An adjacency list representation of the road network,
                      where roads[loc1] = {loc2, loc3} means there is a road
                      from loc1 to loc2 and loc1 to loc3. Assumes directed edges
                      as defined in the PDDL 'road' predicates.

    Returns:
        dict: A nested dictionary where distances[loc1][loc2] is the shortest
              distance (minimum number of 'drive' actions) from loc1 to loc2.
              Returns math.inf if loc2 is unreachable from loc1.
    """
    # Initialize distances: infinite distance between all pairs initially
    distances = {loc: {other_loc: math.inf for other_loc in locations} for loc in locations}

    if not locations:
        return {} # Return empty dict if there are no locations

    # Run BFS starting from each location to find shortest paths to all others
    for start_node in locations:
        # Check if the start node is valid (present in the distance matrix keys)
        if start_node not in distances: continue

        # Distance from start node to itself is 0
        distances[start_node][start_node] = 0
        # Queue for BFS: stores tuples of (location, current_distance_from_start)
        queue = deque([(start_node, 0)])
        # Dictionary to keep track of the shortest distance found so far *during this specific BFS run*
        visited_dist_from_start = {start_node: 0}

        while queue:
            current_loc, current_dist = queue.popleft()

            # Explore neighbors if roads exist from the current location
            if current_loc in roads:
                for neighbor in roads[current_loc]:
                    # Process neighbor only if it's a known location
                    if neighbor in distances:
                        new_dist = current_dist + 1
                        # Update distance if neighbor hasn't been reached yet OR
                        # if we found a shorter path to it through current_loc
                        if neighbor not in visited_dist_from_start or new_dist < visited_dist_from_start[neighbor]:
                             visited_dist_from_start[neighbor] = new_dist
                             # Update the main distance matrix with the shortest path found
                             distances[start_node][neighbor] = new_dist
                             # Add neighbor to the queue for further exploration
                             queue.append((neighbor, new_dist))

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach the goal state.
    It calculates the cost by summing the estimated minimum actions required for each package
    that is not yet at its final destination, based on its current state (at a location or
    inside a vehicle). The heuristic uses precomputed shortest path distances for 'drive' actions.
    It is designed to be informative for guiding a greedy best-first search and is not
    required to be admissible.

    # Assumptions
    - Vehicle Capacity: The heuristic ignores vehicle capacity constraints and the effects
      of pick-up/drop actions on the '(capacity ?v ?s)' predicates. It assumes a vehicle
      can always pick up a package if needed.
    - Vehicle Availability: It assumes an appropriate vehicle is instantly available at the
      package's location for pick-up, ignoring the cost for the vehicle to travel there first.
    - No Conflicts/Synergies: It calculates costs per package independently, ignoring potential
      interactions like multiple packages traveling in the same vehicle, vehicle contention,
      or optimal routing choices involving multiple packages.
    - Uniform Action Cost: Assumes all actions defined in the domain (drive, pick-up, drop)
      have a cost of 1.
    - Shortest Paths: Uses the minimum number of 'drive' actions between locations based on
      the road network graph.

    # Heuristic Initialization
    - Identifies all unique locations, packages mentioned in goals, and vehicles from the
      static facts and initial state predicates (using 'road', 'at', 'in', 'capacity').
    - Parses static 'road' facts to build a directed graph representing the road network.
    - Precomputes all-pairs shortest path distances between all known locations using BFS
      and stores them. `math.inf` is used for unreachable pairs.
    - Parses the goal conditions (assuming they are of the form '(at <package> <location>)')
      to store the target location for each goal package.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic estimate `h` to 0.
    2. Check if the current state (`node.state`) satisfies all goal conditions using
       `self.task.goal_reached(state)`. If true, return 0 immediately.
    3. Parse the current state (a set of fact strings) to efficiently determine the status
       of relevant objects (goal packages and all vehicles):
       - `package_location`: Maps package `p` to location `l` if `(at p l)` is true.
       - `package_in_vehicle`: Maps package `p` to vehicle `v` if `(in p v)` is true.
       - `vehicle_location`: Maps vehicle `v` to location `l` if `(at v l)` is true.
    4. Iterate through each package `p` and its corresponding goal location `g_loc`
       (stored in `self.goal_locations`).
    5. For each package `p`:
       a. Check if the specific goal fact `(at p g_loc)` is present in the current state.
          If yes, this goal is met; add 0 to `h` for this package and continue.
       b. **If package `p` is at location `p_loc`** (found in `package_location`):
          i. Retrieve the precomputed shortest distance `d` from `p_loc` to `g_loc`
             using `self.distances[p_loc][g_loc]`.
          ii. If `d` is `math.inf`, the goal is unreachable from this state; return `math.inf`.
          iii. Estimate cost = `1 (pick-up) + d (drive actions) + 1 (drop)`.
          iv. Add this cost to `h`.
       c. **If package `p` is inside vehicle `v`** (found in `package_in_vehicle`):
          i. Find the vehicle's current location `v_loc` from `vehicle_location`. If the
             vehicle's location is missing (state inconsistency), return `math.inf`.
          ii. Retrieve the shortest distance `d` from `v_loc` to `g_loc` using
             `self.distances[v_loc][g_loc]`.
          iii. If `d` is `math.inf`, the goal is unreachable; return `math.inf`.
          iv. Estimate cost = `d (drive actions) + 1 (drop)`.
          v. Add this cost to `h`.
       d. **If package `p` (required for a goal) is neither `at` nor `in`**: This indicates
          an unexpected state; return `math.inf`.
    6. After processing all goal packages, return the total accumulated heuristic value `h`.
       This sum represents the estimated actions needed under the heuristic's relaxations.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static
        init_facts = task.initial_state

        # --- Initialization Step 1: Identify Objects and Locations ---
        self.locations = set()
        self.packages = set()
        self.vehicles = set()

        # Scan all static and initial facts to identify objects and locations
        facts_to_scan = static_facts.union(init_facts)
        for fact in facts_to_scan:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'road' and len(parts) == 3:
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])
             elif pred == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 # Type (package/vehicle) will be refined later
             elif pred == 'capacity' and len(parts) == 3:
                 # Objects with capacity are vehicles
                 self.vehicles.add(parts[1])
             elif pred == 'in' and len(parts) == 3:
                 # Object 'in' a vehicle implies package and vehicle
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])

        # Identify packages and locations specifically mentioned in goals
        self.goal_locations = {} # Map: package -> goal_location
        for goal in self.goals:
            parts = get_parts(goal)
            # Assume goals are strictly '(at package location)'
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package) # Ensure goal packages are known
                self.locations.add(location) # Ensure goal locations are known

        # Refine vehicle identification: assume anything initially 'at' a location
        # that isn't identified as a package must be a vehicle.
        for fact in init_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj = parts[1]
                if obj not in self.packages:
                    self.vehicles.add(obj)

        # --- Initialization Step 2: Build Road Graph ---
        # Create an adjacency list representation of the road network
        self.roads = {loc: set() for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations exist in the graph structure (even if isolated)
                if loc1 not in self.roads: self.roads[loc1] = set()
                if loc2 not in self.roads: self.roads[loc2] = set()
                # Add directed edge based on PDDL definition
                self.roads[loc1].add(loc2)

        # --- Initialization Step 3: Precompute Shortest Paths ---
        self.distances = compute_shortest_paths(self.locations, self.roads)


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state (node).
        """
        state = node.state
        h_cost = 0

        # --- Heuristic Calculation Step 1: Check Goal Reached ---
        if self.task.goal_reached(state):
            return 0

        # --- Heuristic Calculation Step 2: Parse Current State ---
        # Find current locations of relevant packages and vehicles
        package_location = {} # Map: package -> location (if 'at')
        package_in_vehicle = {} # Map: package -> vehicle (if 'in')
        vehicle_location = {} # Map: vehicle -> location (if 'at')

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            # Process 'at' facts for goal packages and vehicles
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # Check if it's a package relevant to a goal
                    package_location[obj] = loc
                elif obj in self.vehicles: # Check if it's a known vehicle
                     vehicle_location[obj] = loc
            # Process 'in' facts for goal packages
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # Check if it's a package relevant to a goal
                    package_in_vehicle[package] = vehicle
                    # Ensure consistency: if a package is 'in' a vehicle, it cannot be 'at' a location
                    if package in package_location:
                        del package_location[package]

        # --- Heuristic Calculation Step 3 & 4: Estimate Cost per Unsatisfied Goal ---
        for package, goal_loc in self.goal_locations.items():
            # Check if this specific goal is already satisfied in the current state
            current_state_goal_fact = f"(at {package} {goal_loc})"
            if current_state_goal_fact in state:
                continue # Goal satisfied, cost contribution is 0

            cost_for_package = 0
            # Case A: Package is at a location
            if package in package_location:
                p_loc = package_location[package]

                # Safety check for valid locations in distance matrix
                if p_loc not in self.distances or goal_loc not in self.distances.get(p_loc, {}):
                     # This indicates an internal error or invalid state/locations
                     # print(f"Heuristic Error: Distance lookup failed for {p_loc} -> {goal_loc}")
                     return math.inf

                distance = self.distances[p_loc][goal_loc]

                # If goal is unreachable, return infinity
                if distance == math.inf:
                    return math.inf

                # Estimate: 1 action for pick-up, 'distance' actions for drive, 1 action for drop
                cost_for_package = 1 + distance + 1

            # Case B: Package is inside a vehicle
            elif package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Check if the vehicle containing the package has a known location
                if vehicle not in vehicle_location:
                    # This signifies an invalid or inconsistent state
                    # print(f"Heuristic Error: Vehicle {vehicle} carrying {package} has no location.")
                    return math.inf

                v_loc = vehicle_location[vehicle]

                # Safety check for valid locations in distance matrix
                if v_loc not in self.distances or goal_loc not in self.distances.get(v_loc, {}):
                     # print(f"Heuristic Error: Distance lookup failed for {v_loc} -> {goal_loc}")
                     return math.inf

                distance = self.distances[v_loc][goal_loc]

                # If goal is unreachable from vehicle's location, return infinity
                if distance == math.inf:
                    return math.inf

                # Estimate: 'distance' actions for drive, 1 action for drop
                cost_for_package = distance + 1

            # Case C: Package is neither 'at' nor 'in'
            else:
                # This goal package is required but doesn't seem to exist in the state description
                # print(f"Heuristic Error: Goal package {package} not found 'at' or 'in'.")
                return math.inf

            # Add the estimated cost for this package to the total heuristic value
            h_cost += cost_for_package

        # --- Heuristic Calculation Step 5: Return Total Estimated Cost ---
        # The calculated cost should be non-negative or infinity
        return h_cost
