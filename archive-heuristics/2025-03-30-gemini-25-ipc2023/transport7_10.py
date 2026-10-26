import math
from collections import deque
# Assume the Heuristic base class is available from the planner's infrastructure,
# e.g., from heuristics.heuristic_base. If not, define a placeholder:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    class Heuristic:
        def __init__(self, task):
            """Initialize the heuristic with the planning task."""
            pass
        def __call__(self, node):
            """
            Calculate the heuristic value for the state in the given node.
            Must be implemented by subclasses.
            """
            raise NotImplementedError

def get_parts(fact: str):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(at package1 locationA)" -> ["at", "package1", "locationA"]
    """
    # Remove parentheses and split by space
    return fact[1:-1].split()


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    Estimates the cost (number of actions) to reach the goal state in the Transport domain.
    The heuristic calculates the sum of estimated costs for moving each package to its
    target destination, considering shortest path distances on the road network and
    the necessary pickup/drop actions. This heuristic is designed for guiding
    greedy best-first search and is not necessarily admissible (it might overestimate).

    # Assumptions
    - Roads defined by `(road l1 l2)` predicates are bidirectional, as suggested by the examples.
      If roads were unidirectional, the graph construction and BFS would need adjustment.
    - The cost of the `drive` action between adjacent locations is 1.
    - Vehicle capacity constraints (predicates `capacity`, `capacity-predecessor`) are ignored
      to keep the heuristic simple and computationally efficient. This is a major simplification.
    - The heuristic estimates the cost for each package independently and sums these costs.
      It does not model potential synergies (e.g., one vehicle carrying multiple packages
      simultaneously) or complex vehicle routing and assignment problems.
    - The shortest path distance (number of road segments) is used as the estimate for the
      number of `drive` actions required.

    # Heuristic Initialization
    - Extracts all unique location objects. These are identified from `road` predicates
      in the static facts and from the second argument of `at` predicates in the
      initial state and goal conditions.
    - Parses static `road` facts to build an adjacency list representation of the map graph.
    - Computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS) starting from each location. Stores these distances.
      If two locations are disconnected, the distance is treated as infinity.
    - Parses the goal facts `(at package location)` to identify the target location
      for each package required in the goal state. Stores this in `goal_locations`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** If the current state `node.state` already satisfies all goal conditions
        (i.e., `self.goals <= state`), the goal is reached, and the heuristic value is 0.
    2.  **Initialization:** Initialize the total heuristic cost `h = 0.0`.
    3.  **State Parsing:** Parse the current state `node.state` to determine:
        - The current location of all objects (packages and vehicles) using `(at object location)` facts.
        - Which packages are currently inside which vehicles using `(in package vehicle)` facts.
        Store this information, e.g., in dictionaries `current_at_locations` and `package_in_vehicle`.
    4.  **Iterate Through Goal Packages:** For each package `p` that has a goal condition `(at p L_goal)`:
        a.  **Find Package State:** Determine the package's current effective location `L_current`
            and whether it is currently inside a vehicle (`is_in_vehicle`).
            - If `(at p L_p)` exists in the state: `L_current = L_p`, `is_in_vehicle = False`.
            - If `(in p v)` exists: Find the vehicle `v`'s location `L_v` using `(at v L_v)`.
              If `L_v` is found, then `L_current = L_v`, `is_in_vehicle = True`.
            - If the package's location (directly or via its vehicle) cannot be determined from the
              state, this indicates an inconsistent or problematic state. Return `float('inf')`
              to signify a dead end or error.
        b.  **Check if Goal Met for Package:** If the package `p` is already at its goal location
            (`L_current == L_goal`) AND it is not inside a vehicle (`is_in_vehicle == False`),
            then this specific goal condition is met. The cost contribution for this package is 0.
            Continue to the next package.
        c.  **Calculate Cost for Unmet Goal:** If the package's goal is not met:
            i.  **Get Distance:** Retrieve the shortest path distance `dist` between the package's
                current effective location `L_current` and its goal location `L_goal` from the
                precomputed `self.distances` dictionary. If the locations are disconnected,
                `self.distances.get((L_current, L_goal), float('inf'))` will return infinity.
            ii. **Check Reachability:** If `dist` is `float('inf')`, the goal location is unreachable
                for this package from its current state. The overall goal cannot be reached.
                Return `float('inf')`.
            iii. **Estimate Actions:** Calculate the estimated number of actions required for this package:
                 - If `is_in_vehicle` is True: The package needs to be driven to the goal and dropped.
                   Estimated cost = `dist (drive actions) + 1 (drop action)`.
                 - If `is_in_vehicle` is False: The package needs to be picked up, driven to the goal, and dropped.
                   Estimated cost = `1 (pickup action) + dist (drive actions) + 1 (drop action)`.
        d.  **Accumulate Cost:** Add the calculated `cost_for_package` to the total heuristic cost `h`.
    5.  **Final Value:** After iterating through all goal packages:
        - If `h` is `float('inf')` (because any package goal was unreachable), return `float('inf')`.
        - Otherwise, return `max(0, int(round(h)))`. Rounding handles potential floating point inaccuracies,
          `int()` converts to integer, and `max(0, ...)` ensures non-negativity.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating distances and goal information.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract all unique locations from static facts and state/goal facts
        locations = set()
        roads_list = [] # Store (loc1, loc2) tuples for road connections
        for fact in static_facts:
            parts = get_parts(fact)
            # Extract locations from 'road' predicates
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                roads_list.append((loc1, loc2))
            # Add checks here if other static predicates might define locations

        # Ensure locations mentioned in initial state or goals are included
        all_facts_for_locations = task.initial_state.union(task.goals)
        for fact in all_facts_for_locations:
             parts = get_parts(fact)
             # Locations appear as the second argument in 'at' predicates
             if parts[0] == 'at' and len(parts) == 3:
                 locations.add(parts[2])

        self.locations = frozenset(locations)
        # Store unique road pairs, assuming bidirectionality by sorting
        self.roads = frozenset(tuple(sorted(road)) for road in roads_list)

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = self._compute_shortest_paths()

        # 3. Store goal locations for packages specified in the goal
        self.goal_locations = {} # Map: package_name -> goal_location_name
        packages_in_goal = set() # Set of package names mentioned in goals
        for goal in self.goals:
            parts = get_parts(goal)
            # We only care about goals of the form (at package location)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Validate that the goal location is one of the known locations
                if location in self.locations:
                    self.goal_locations[package] = location
                    packages_in_goal.add(package)
                else:
                    # This indicates a potential issue with the PDDL instance
                    print(f"Warning: Goal location '{location}' for package '{package}' "
                          f"is not among the known locations derived from static facts and state. "
                          f"This goal atom will be ignored by the heuristic.")

        self.packages = frozenset(packages_in_goal)


    def _compute_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of known locations
        using Breadth-First Search (BFS).
        Assumes roads are bidirectional and have a uniform cost of 1.
        Returns a dictionary mapping (loc1, loc2) -> distance.
        Missing pairs indicate infinite distance (unreachable).
        """
        distances = {}
        # Build adjacency list representation of the road network
        adj = {loc: [] for loc in self.locations}
        for loc1, loc2 in self.roads:
             # Ensure both locations are valid before adding edges
             if loc1 in self.locations and loc2 in self.locations:
                adj[loc1].append(loc2)
                adj[loc2].append(loc1) # Add edge in both directions

        # Run BFS starting from each location to find distances to all reachable locations
        for start_node in self.locations:
            # Skip locations that have no outgoing roads (isolated)
            if start_node not in adj: continue

            # Distance from start_node to itself is 0
            distances[(start_node, start_node)] = 0
            # Queue for BFS: stores (node, distance_from_start)
            queue = deque([(start_node, 0)])
            # Keep track of visited nodes and their shortest distance found so far from start_node
            visited_dist = {start_node: 0}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors of the current node
                for neighbor in adj.get(current_node, []): # Use .get for safety
                    # Process neighbor only if it's a known location and hasn't been visited yet
                    # in this specific BFS run (starting from start_node)
                    if neighbor in self.locations and neighbor not in visited_dist:
                        # Record distance and add neighbor to the queue
                        visited_dist[neighbor] = dist + 1
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

        # The resulting 'distances' dictionary contains distances for reachable pairs.
        # Unreachable pairs will be missing.
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the state represented by the search node.
        """
        state = node.state

        # Check if the current state satisfies all goal conditions
        if self.goals <= state:
            # Goal state reached, heuristic value is 0
            return 0

        # --- Parse the current state to find locations and containment ---
        current_at_locations = {} # Map: object_name -> location_name
        package_in_vehicle = {}   # Map: package_name -> vehicle_name

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                # Store location for any object (package or vehicle)
                obj, loc = args[0], args[1]
                current_at_locations[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                # Store containment info only for packages relevant to the goal
                package, vehicle = args[0], args[1]
                if package in self.packages:
                    package_in_vehicle[package] = vehicle

        # --- Calculate heuristic: sum of estimated costs for unmet package goals ---
        total_heuristic_value = 0.0

        for package, goal_loc in self.goal_locations.items():
            cost_for_package = 0.0
            current_loc = None
            is_in_vehicle = False

            # Determine the package's current effective location and status
            if package in current_at_locations:
                # Package is directly at a location
                current_loc = current_at_locations[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's current location
                if vehicle in current_at_locations:
                    current_loc = current_at_locations[vehicle]
                    is_in_vehicle = True
                else:
                    # The vehicle containing the package has no known location.
                    # This indicates an invalid or unreachable state.
                    return float('inf')
            else:
                # The package (which is required for the goal) has no location info.
                # This indicates an invalid or unreachable state.
                return float('inf')

            # Check if this specific package goal is already satisfied
            # Goal is satisfied if package is AT the goal_loc and NOT in a vehicle
            if not is_in_vehicle and current_loc == goal_loc:
                continue # This goal is met, add 0 cost for this package.

            # --- Calculate cost for this unmet package goal ---
            # Retrieve the shortest path distance. Use infinity if unreachable.
            dist = self.distances.get((current_loc, goal_loc), float('inf'))

            # If the goal location is unreachable, the overall goal is unreachable
            if dist == float('inf'):
                return float('inf')

            # Estimate actions based on whether the package is currently in a vehicle
            if is_in_vehicle:
                # Needs drive (dist actions) + drop (1 action)
                cost_for_package = dist + 1.0
            else:
                # Needs pickup (1 action) + drive (dist actions) + drop (1 action)
                cost_for_package = 1.0 + dist + 1.0

            # Add the cost for this package to the total heuristic value
            total_heuristic_value += cost_for_package

        # --- Return the final heuristic value ---
        if total_heuristic_value == float('inf'):
            # If any goal was found unreachable, return infinity
            return float('inf')
        else:
            # Return the calculated cost, rounded to the nearest integer, ensuring non-negativity
            return max(0, int(round(total_heuristic_value)))

