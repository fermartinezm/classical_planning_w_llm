import math
from collections import deque
# Assuming the heuristic base class is available at this path
# If the environment uses a different path, adjust the import accordingly.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_string):
    """
    Removes the surrounding parentheses and splits the fact string by spaces.
    Example: "(at p1 l1)" becomes ["at", "p1", "l1"]
    Returns an empty list if the string is not a valid fact format (e.g., empty or malformed).
    """
    if isinstance(fact_string, str) and len(fact_string) > 2 and fact_string.startswith("(") and fact_string.endswith(")"):
        return fact_string[1:-1].split()
    return []

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their designated goal locations. It calculates the cost for each package
    individually based on its current state (at a location or inside a vehicle)
    and sums these costs. The cost for a package includes the estimated driving
    distance (shortest path using BFS) and the necessary pick-up (1 action)
    and/or drop actions (1 action). It is designed for Greedy Best-First Search
    and does not need to be admissible.

    # Assumptions
    - The heuristic calculates the cost for each package independently. It does not
      explicitly model vehicle capacity constraints or optimize vehicle routing
      for multiple packages simultaneously. This makes it non-admissible but
      computationally efficient.
    - It assumes that *some* vehicle will perform the transport, estimating the
      cost based on the package's required journey steps (pickup, drive, drop),
      not the vehicle's full trip (e.g., it doesn't add the cost for a vehicle
      to reach the package initially if the package is on the ground).
    - Driving costs are based on the shortest path distance between locations in
      the road network graph, assuming each 'drive' action has a uniform cost of 1.
    - The road network might not be fully connected. If a required path between
      a package's location/vehicle's location and the goal location doesn't exist,
      the distance is considered infinite, resulting in an infinite heuristic value,
      correctly indicating an unreachable goal from the current state.
    - Object types (package, vehicle, location) are inferred based on common PDDL
      conventions (e.g., prefixes 'p', 'v', 'l') if not explicitly available
      through the task representation. This heuristic relies on prefixes found
      in the example instances ('p' for package, 'v' for vehicle).

    # Heuristic Initialization
    - Extracts all package goal locations specified by `(at package location)`
      predicates in the task's goals.
    - Parses static `road` facts to build an undirected graph representation of
      the locations and their connectivity.
    - Computes all-pairs shortest path distances between all connected locations
      using Breadth-First Search (BFS) for unweighted edges.
    - Stores the identified package goals and the precomputed distances in
      internal data structures (`self.package_goals`, `self.distances`) for
      efficient lookup during heuristic evaluation.
    - Attempts to identify package and vehicle objects based on prefixes ('p', 'v')
      found in goal specifications and initial/static facts, storing them in
      `self.packages` and `self.vehicles` sets.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current `state` already satisfies all `task.goals`. If yes, return 0.
    2.  Initialize `total_cost` to 0.
    3.  Parse the current `state` (a set of fact strings) to determine the status of each relevant object:
        - `package_location`: Map of package -> location for packages `at` a location.
        - `package_in_vehicle`: Map of package -> vehicle for packages `in` a vehicle.
        - `vehicle_location`: Map of vehicle -> location for vehicles `at` a location.
    4.  Iterate through each package `p` that has a defined goal location `goal_loc`
        (stored in `self.package_goals` during initialization):
        a. Construct the goal fact string `(at p goal_loc)`.
        b. Check if this goal fact is already true in the current `state`.
           If yes, this package's goal is met; continue to the next package.
        c. **If the package `p` is currently `at` location `p_loc` (found in `package_location`):**
           - Calculate the shortest distance `dist = self.get_dist(p_loc, goal_loc)`.
           - If `dist` is infinite (unreachable), the goal cannot be reached; return `math.inf`.
           - The estimated cost for this package is `1 (pick-up) + dist (drive actions) + 1 (drop)`.
        d. **If the package `p` is currently `in` vehicle `v` (found in `package_in_vehicle`):**
           - Find the current location of vehicle `v`, `v_loc`, from `vehicle_location`.
           - If `v_loc` cannot be determined (vehicle's 'at' predicate is missing - inconsistent state), return `math.inf`.
           - Calculate the shortest distance `dist = self.get_dist(v_loc, goal_loc)`.
           - If `dist` is infinite (unreachable), return `math.inf`.
           - The estimated cost for this package is `dist (drive actions) + 1 (drop)`.
        e. **If the package `p` is neither `at` a location nor `in` a vehicle:**
           - This indicates an unexpected or error state (e.g., package doesn't exist in the current state
             but is required for a goal, or the state is corrupted). Return `math.inf`.
        f. Add the calculated `cost_package` for package `p` to `total_cost`. If the cost was
           infinite at any step, the function would have already returned `math.inf`.
    5.  After checking all packages, if `total_cost` is 0 but the state is not a goal
        state (meaning all package goals considered by the heuristic are met, but other
        goals might not be), return 1. This ensures the heuristic value is strictly positive
        for non-goal states, preventing potential issues in search algorithms like GBFS.
    6.  Return the final `total_cost` (integer or `math.inf`).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task goals and static facts.
        - Extracts package goals.
        - Builds the road network graph.
        - Computes all-pairs shortest paths using BFS.
        - Identifies packages and vehicles based on naming conventions (prefixes).
        """
        if task is None:
             raise ValueError("Task object cannot be None for heuristic initialization.")

        self.goals = frozenset(task.goals) if task.goals else frozenset()
        static_facts = frozenset(task.static) if task.static else frozenset()
        initial_state = frozenset(task.initial_state) if task.initial_state else frozenset()

        # Combine initial state and static facts for broader info gathering
        all_initial_facts = initial_state.union(static_facts)

        # --- Identify objects based on prefixes (fallback mechanism) ---
        # Using sets for efficient membership checking
        self.packages = set()
        self.vehicles = set()
        for fact in all_initial_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]
             # Identify objects based on predicates they appear in and prefixes 'p'/'v'
             if predicate == "at" and len(args) == 2:
                 obj = args[0]
                 if obj.startswith("p"): self.packages.add(obj)
                 elif obj.startswith("v"): self.vehicles.add(obj)
             elif predicate == "in" and len(args) == 2:
                 obj = args[0] # The first argument is the package
                 if obj.startswith("p"): self.packages.add(obj)
             elif predicate == "capacity" and len(args) == 2:
                 obj = args[0] # The first argument is the vehicle
                 if obj.startswith("v"): self.vehicles.add(obj)

        # --- Extract package goal locations ---
        self.package_goals = {} # map: package_name -> goal_location_name
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure it's an 'at' predicate with 3 parts (predicate, object, location)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 # Check if the object is identified as a package (or starts with 'p')
                 if obj in self.packages or obj.startswith("p"):
                     location = parts[2]
                     self.package_goals[obj] = location
                     self.packages.add(obj) # Ensure package is tracked even if only in goal

        # --- Build road network and compute distances ---
        roads = []
        found_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                found_locations.add(l1)
                found_locations.add(l2)
                # Store roads for graph construction
                roads.append((l1, l2))
        self.locations = found_locations
        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths(self.locations, roads)


    def _compute_all_pairs_shortest_paths(self, locations, roads):
        """
        Computes shortest path distances between all pairs of locations using BFS.
        Assumes roads define an undirected graph where each step costs 1.
        Returns a dictionary mapping (from_loc, to_loc) tuples to their shortest
        distance (integer). Unreachable pairs will not have an entry.
        """
        distances = {}
        # Build adjacency list representation of the graph
        adj = {loc: set() for loc in locations}
        for u, v in roads:
            # Ensure locations are valid before adding edges
            if u in locations and v in locations:
                adj[u].add(v)
                adj[v].add(u) # Add edge in both directions for undirected graph

        # Run BFS from each location to find shortest paths to all reachable locations
        for start_node in locations:
            # Distance from start_node to itself is 0
            distances[(start_node, start_node)] = 0
            # Queue for BFS: stores tuples of (node, distance_from_start)
            queue = deque([(start_node, 0)])
            # Set to keep track of visited nodes in the current BFS run from start_node
            visited_bfs = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors of the current node
                for neighbor in adj.get(current_node, set()):
                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        new_dist = dist + 1
                        # Store the shortest distance found
                        distances[(start_node, neighbor)] = new_dist
                        queue.append((neighbor, new_dist))
        return distances

    def get_dist(self, l1, l2):
        """
        Retrieves the precomputed shortest distance between location l1 and l2.
        Returns 0 if l1 == l2.
        Returns math.inf if l2 is unreachable from l1 (no entry in self.distances).
        """
        if l1 == l2:
            return 0
        # Lookup the distance; default to infinity if the key doesn't exist
        return self.distances.get((l1, l2), math.inf)

    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.
        The state is accessed via node.state.
        Returns 0 for goal states, math.inf for potentially unsolvable states,
        and a positive integer estimate otherwise.
        """
        if node is None or node.state is None:
             # Handle invalid input node
             return math.inf # Or raise an error

        state = node.state # State is typically a frozenset of fact strings

        # Check if the goal is already satisfied
        # self.goals is a frozenset of goal facts. Check if it's a subset of the current state.
        if self.goals <= state:
            return 0

        total_cost = 0

        # --- Parse current state for object locations/statuses ---
        package_location = {} # map: package_name -> location_name
        package_in_vehicle = {} # map: package_name -> vehicle_name
        vehicle_location = {} # map: vehicle_name -> location_name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed
            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj, loc = args
                # Use identified sets or prefixes to determine object type
                if obj in self.packages or obj.startswith("p"):
                    package_location[obj] = loc
                elif obj in self.vehicles or obj.startswith("v"):
                    vehicle_location[obj] = loc
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                # Check if it's a known/plausible package and vehicle
                if (package in self.packages or package.startswith("p")) and \
                   (vehicle in self.vehicles or vehicle.startswith("v")):
                    package_in_vehicle[package] = vehicle

        # --- Calculate cost for each unmet package goal ---
        for package, goal_loc in self.package_goals.items():
            # Construct the goal fact string to check against the state
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                continue # This package's goal is already met

            cost_package = 0 # Estimated cost for this specific package
            if package in package_location:
                # Case 1: Package is at a location p_loc
                p_loc = package_location[package]
                dist = self.get_dist(p_loc, goal_loc)
                if dist == math.inf:
                    # Goal location is unreachable from package's current location
                    return math.inf # State is likely unsolvable regarding this goal
                # Estimated cost = pick-up(1) + drive(dist) + drop(1)
                cost_package = 1 + dist + 1
            elif package in package_in_vehicle:
                # Case 2: Package is inside a vehicle v
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_location:
                    # Find the vehicle's current location v_loc
                    v_loc = vehicle_location[vehicle]
                    dist = self.get_dist(v_loc, goal_loc)
                    if dist == math.inf:
                        # Goal location is unreachable from vehicle's current location
                        return math.inf # State is likely unsolvable
                    # Estimated cost = drive(dist) + drop(1)
                    cost_package = dist + 1
                else:
                    # Error condition: Package is in a vehicle, but the vehicle's
                    # location ('at' predicate) is missing from the state.
                    # This indicates an inconsistent or invalid state.
                    # Consider logging this error if a logging mechanism is available.
                    # print(f"Error: Vehicle {vehicle} carrying {package} has no 'at' predicate in state.")
                    return math.inf # Treat as unsolvable / error state
            else:
                # Error condition: A package required for a goal is neither 'at'
                # a location nor 'in' a vehicle in the current state.
                # This implies the state is invalid or the package doesn't exist.
                # Consider logging this error.
                # print(f"Error: Package {package} (goal: {goal_loc}) not found 'at' or 'in' state.")
                return math.inf # Treat as unsolvable / error state

            # Add the cost for this package to the total heuristic value
            total_cost += cost_package

        # Final check: If the total calculated cost is 0, but the state is not
        # actually a goal state (checked at the beginning), return 1.
        # This ensures the heuristic value is strictly positive for non-goal states,
        # preventing infinite loops in search algorithms like GBFS if a non-goal
        # state evaluates to 0. This might happen if goals involve non-package
        # predicates not considered by this heuristic.
        if total_cost == 0 and not (self.goals <= state):
             return 1

        # Return the calculated total estimated cost (can be 0 if goal met, inf if unsolvable)
        return total_cost
