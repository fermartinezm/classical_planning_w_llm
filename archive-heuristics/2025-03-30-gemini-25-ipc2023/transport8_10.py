import itertools
from collections import deque
# Assuming the base class `Heuristic` is available at this path
# from planning_framework.heuristics.heuristic_base import Heuristic
# Using the provided path:
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Parses a PDDL fact string into a list of its components.
    Removes parentheses and splits by space.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    """
    return fact.strip()[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    Estimates the cost to reach the goal state in the Transport domain.
    The heuristic sums the estimated costs for each package that is not
    yet in its target location. The cost for a package includes the actions
    to pick it up (if not already in a vehicle), drive it to the destination,
    and drop it. Driving cost is estimated using precomputed shortest path
    distances on the road network.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - The heuristic ignores vehicle capacity constraints. It assumes any vehicle
      can potentially transport any package. This makes the heuristic potentially
      inadmissible but can be informative for greedy search algorithms.
    - It assumes any package can be transported between any two connected locations
      if a path exists in the road network.
    - It calculates the cost for each package independently and sums them up.
      This might overcount vehicle movements if multiple packages share trips,
      contributing to the non-admissibility.
    - The road network defined by 'road' predicates is static throughout planning.

    # Heuristic Initialization
    - Extracts all known locations, packages, and vehicles from the task definition.
      It attempts to get this information directly from the task object attributes
      (e.g., task.locations, task.packages, task.vehicles). If these are not
      available or empty, it falls back to inferring these object sets by parsing
      the initial state and static facts.
    - Parses goal conditions to identify target locations for packages specified
      using the 'at' predicate (e.g., '(at p1 l2)'). Stores these in `self.goal_locations`.
    - Parses static 'road' facts to build an adjacency list representation
      of the road network graph (`self.adj`).
    - Precomputes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS). Stores these distances in `self.distances`, using
      `float('inf')` for unreachable pairs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost `H = 0`.
    2. Check if the current state `node.state` already satisfies all goal conditions
       defined in `self.goals`. If yes, the goal is reached, return `H = 0`.
    3. Parse the current state `node.state` to determine the current situation:
       - Find the location of each vehicle using `(at ?v ?l)` facts -> `vehicle_at` map.
       - Find the location of each package on the ground using `(at ?p ?l)` facts -> `package_at` map.
       - Find which vehicle each package is inside using `(in ?p ?v)` facts -> `package_in` map.
    4. Iterate through each package `p` that has a target location `loc_goal` defined in `self.goal_locations`:
       a. Check if the goal fact `(at p loc_goal)` is already present in the current state.
          If yes, this package is already settled, contributes 0 to `H`, so continue to the next package.
       b. **If the package `p` is currently at a location `loc_curr`** (i.e., `p` is in `package_at`):
          i. Retrieve the precomputed shortest driving distance `dist` from `loc_curr` to `loc_goal` using `self._get_loc_dist(loc_curr, loc_goal)`.
          ii. If `dist` is `float('inf')`, it means `loc_goal` is unreachable from `loc_curr`. The overall goal cannot be reached from this state. Return `float('inf')`.
          iii. Estimate the cost for this package as: `1` (for pick-up action) + `dist` (for drive actions) + `1` (for drop action).
          iv. Add this estimated cost to the total heuristic value `H`.
       c. **If the package `p` is currently inside a vehicle `veh`** (i.e., `p` is in `package_in`):
          i. Find the current location `loc_vehicle` of the vehicle `veh` from the `vehicle_at` map.
          ii. If the vehicle `veh` is not found in `vehicle_at` (e.g., inconsistent state), the goal is unreachable. Return `float('inf')`.
          iii. Retrieve the precomputed shortest driving distance `dist` from `loc_vehicle` to `loc_goal` using `self._get_loc_dist(loc_vehicle, loc_goal)`.
          iv. If `dist` is `float('inf')`, the goal location is unreachable from the vehicle's current location. Return `float('inf')`.
          v. Estimate the cost for this package as: `dist` (for drive actions) + `1` (for drop action).
          vi. Add this estimated cost to `H`.
       d. **If the package `p` (required for the goal) is not found** either 'at' a location or 'in' a vehicle in the current state, this indicates an unexpected or invalid state. Return `float('inf')`.
    5. After iterating through all goal packages, the final heuristic value is the total sum `H`. It returns `float('inf')` if any package's goal was determined to be unreachable.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information: goals, static facts,
        and object types. Precomputes road network distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Attempt to get typed object sets directly from the task object attributes.
        # Use getattr with default to empty set for safety.
        self.packages = getattr(task, 'packages', set())
        self.vehicles = getattr(task, 'vehicles', set())
        self.locations = getattr(task, 'locations', set())

        # If any object set is empty, fall back to inferring types from facts.
        # This assumes an empty set means the info wasn't provided, not that there are zero objects.
        if not self.locations or not self.packages or not self.vehicles:
            # print("Info: Object sets not fully provided or empty; inferring types...")
            self._infer_objects(task)
        # else:
            # print(f"Info: Using provided object sets. Locations: {len(self.locations)}, "
            #       f"Vehicles: {len(self.vehicles)}, Packages: {len(self.packages)}")

        # Extract goal locations for packages specified via 'at' predicate in goals.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure goal is of the form (at ?p - package ?l - location)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Store goal only if package and location types match known objects
                if package in self.packages and location in self.locations:
                    self.goal_locations[package] = location
                # else:
                    # Optionally log warning about mismatched/unknown objects in goal
                    # print(f"Warning: Goal '{goal}' involves unknown package/location or type mismatch. Ignoring this goal for heuristic.")

        # Build road graph (adjacency list) using known locations and static 'road' facts.
        self.adj = {loc: set() for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edge only if both locations are known members of self.locations
                if l1 in self.locations and l2 in self.locations:
                    self.adj[l1].add(l2)
                # else:
                     # Optionally log warning about road facts with unknown locations
                     # print(f"Warning: Static fact '{fact}' involves unknown location(s). Ignoring.")

        # Precompute all-pairs shortest paths using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()
        # print("Info: Precomputed all-pairs shortest paths.")

    def _infer_objects(self, task):
        """
        Infers locations, packages, and vehicles by parsing initial state and static facts.
        This is a fallback if typed object sets are not directly provided by the task object.
        """
        # Reset inferred sets
        self.packages = set()
        self.vehicles = set()
        self.locations = set()

        # Combine initial state and static facts for a comprehensive scan
        facts_to_scan = task.initial_state.union(task.static)

        # First pass: Identify locations from 'road', vehicles from 'capacity',
        # packages/vehicles from 'in'. These predicates strongly imply types.
        for fact in facts_to_scan:
             parts = get_parts(fact)
             pred = parts[0]
             if len(parts) >= 3: # Consider predicates with at least two arguments
                 if pred == 'road':
                     self.locations.add(parts[1])
                     self.locations.add(parts[2])
                 elif pred == 'in':
                     # (in ?p - package ?v - vehicle)
                     self.packages.add(parts[1])
                     self.vehicles.add(parts[2])
                 elif pred == 'capacity':
                     # (capacity ?v - vehicle ?s - size)
                     self.vehicles.add(parts[1])
             # 'capacity-predecessor' involves sizes, ignore for these object types.
             # 'at' predicate involves locatables and locations - handle in second pass.

        # Second pass: Identify locations from 'at', and try to type the locatable object.
        for fact in facts_to_scan:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc) # Ensure location is registered
                 # If the object 'obj' is 'at' a location and not already known as a vehicle,
                 # tentatively assume it's a package if it's not a location itself.
                 # This relies on 'locatable' primarily being 'vehicle' or 'package' in this domain.
                 if obj not in self.vehicles and obj not in self.locations:
                     self.packages.add(obj)

        # Clean up: Ensure no object is categorized incorrectly (e.g., as both location and package).
        self.packages -= self.locations
        self.vehicles -= self.locations
        self.packages -= self.vehicles # Ensure packages and vehicles are disjoint

        # print(f"Info: Inferred types - Locations: {len(self.locations)}, "
        #       f"Vehicles: {len(self.vehicles)}, Packages: {len(self.packages)}")

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from `start_node` on the road graph
        represented by `self.adj`.
        Returns a dictionary mapping reachable locations to their shortest distance (number of road segments)
        from `start_node`. Unreachable locations will have distance `float('inf')`.
        """
        # Initialize distances to infinity for all known locations
        distances = {loc: float('inf') for loc in self.locations}

        # Check if the start node is a valid location in our graph
        if start_node not in self.locations:
            # This might happen if a state contains an object 'at' an unknown location.
            # print(f"Error: BFS start node '{start_node}' is not a known location.")
            return distances # Return all infinities, indicating unreachability

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Iterate through neighbors using the precomputed adjacency list `self.adj`
            # Use .get() for safety, returning empty set if node has no outgoing roads
            for neighbor in self.adj.get(current_node, set()):
                # If neighbor hasn't been reached yet (distance is inf), update its distance
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
                # Since edge costs are uniform (1), the first time we reach a node is via a shortest path.

        return distances

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes and stores shortest path distances between all pairs of known locations
        by running BFS from each location.
        Returns a dictionary where keys are `(loc1, loc2)` tuples and values are distances.
        """
        all_distances = {}
        if not self.locations:
            # print("Warning: No locations found to compute distances.")
            return {}

        # print(f"Info: Computing shortest paths for {len(self.locations)} locations...")
        for loc1 in self.locations:
            # Run BFS starting from loc1 to find distances to all other locations
            distances_from_loc1 = self._bfs(loc1)
            # Store the computed distances in the main dictionary
            for loc2 in self.locations:
                all_distances[(loc1, loc2)] = distances_from_loc1[loc2] # Will be inf if unreachable

        return all_distances

    def _get_loc_dist(self, loc1, loc2):
         """
         Retrieves the precomputed shortest path distance between two locations `loc1` and `loc2`.
         Returns `float('inf')` if either location is unknown or if they are unreachable.
         """
         # Check if both locations are known members of the location set used for precomputation
         if loc1 not in self.locations or loc2 not in self.locations:
             # print(f"Error: Requesting distance involving unknown location: {loc1} or {loc2}")
             return float('inf') # Indicate error or unreachability

         # Retrieve distance from the precomputed dictionary. Default to infinity if the pair is missing (should not happen).
         return self.distances.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.
        Returns 0 for goal states, a positive integer estimate for non-goal states,
        or float('inf') if the goal seems unreachable from the current state.
        """
        state = node.state

        # Check if the current state satisfies all goal conditions. If so, heuristic value is 0.
        if self.goals <= state:
             return 0

        heuristic_value = 0

        # Parse the current state to find current locations/containment of objects
        package_at = {} # Maps package -> location
        package_in = {} # Maps package -> vehicle
        vehicle_at = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if len(parts) == 3: # Focus on binary predicates: at, in
                obj1, obj2 = parts[1], parts[2]
                if pred == "at":
                    # Check if obj1 is a known package or vehicle
                    if obj1 in self.packages:
                        package_at[obj1] = obj2
                    elif obj1 in self.vehicles:
                        vehicle_at[obj1] = obj2
                elif pred == "in":
                    # Check if obj1 is a package and obj2 is a vehicle
                    if obj1 in self.packages and obj2 in self.vehicles:
                        package_in[obj1] = obj2

        # Calculate heuristic by summing estimated costs for each package not yet at its goal
        for package, goal_loc in self.goal_locations.items():
            # Construct the goal fact string for this package
            goal_fact = f"(at {package} {goal_loc})"

            # If this package's goal is already met in the current state, skip it
            if goal_fact in state:
                continue

            # Initialize cost estimate for this package to infinity (unreachable by default)
            cost_for_package = float('inf')

            if package in package_at:
                # Case 1: Package is currently at a location `loc_curr`
                loc_curr = package_at[package]
                # Get shortest path distance from current location to goal location
                dist = self._get_loc_dist(loc_curr, goal_loc)
                # If reachable (distance is not infinity)
                if dist != float('inf'):
                    # Estimate cost: pickup(1) + drive(dist) + drop(1)
                    cost_for_package = 1 + dist + 1
                # If dist is inf, cost_for_package remains inf

            elif package in package_in:
                # Case 2: Package is currently inside a vehicle `vehicle`
                vehicle = package_in[package]
                # Find the location `loc_vehicle` of this vehicle
                if vehicle in vehicle_at:
                    loc_vehicle = vehicle_at[vehicle]
                    # Get shortest path distance from vehicle's location to goal location
                    dist = self._get_loc_dist(loc_vehicle, goal_loc)
                    # If reachable
                    if dist != float('inf'):
                        # Estimate cost: drive(dist) + drop(1)
                        cost_for_package = dist + 1
                    # If dist is inf, cost_for_package remains inf
                else:
                    # Error condition: Vehicle containing the package has no location defined in the state.
                    # This implies an inconsistent or unexpected state. Treat as unreachable.
                    # print(f"Error: Vehicle '{vehicle}' carrying package '{package}' has no 'at' predicate in state.")
                    cost_for_package = float('inf')

            else:
                # Error condition: A package required for the goal is not found either 'at' a location
                # or 'in' a vehicle in the current state. This shouldn't happen in valid states.
                # print(f"Error: Goal package '{package}' not found 'at'/'in' in current state.")
                cost_for_package = float('inf')


            # If the cost for any package remains infinity, it means the goal is unreachable
            # from this state (at least according to the heuristic's logic).
            # Return infinity immediately.
            if cost_for_package == float('inf'):
                return float('inf')

            # Add the estimated cost for this package to the total heuristic value
            heuristic_value += cost_for_package

        # Return the total estimated cost for all misplaced packages.
        # This value is 0 if the state is a goal state, > 0 otherwise, or infinity if unreachable.
        return heuristic_value

