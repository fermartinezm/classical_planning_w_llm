import itertools
from collections import deque
# Assuming the heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    Returns an empty list if the fact is malformed or empty.
    """
    try:
        # Remove leading/trailing whitespace and parentheses, then split
        return fact.strip()[1:-1].split()
    except IndexError:
        # Handle cases like "()" or empty strings gracefully
        return []

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for moving each package to its target destination.
    It focuses on the necessary pick-up, drop, and driving actions directly related
    to each package's journey. The heuristic ignores complex interactions like
    vehicle capacity limits or optimal vehicle assignment to maintain computational
    efficiency. It is designed for use with greedy best-first search and is not
    guaranteed to be admissible.

    # Assumptions
    - Roads defined by `(road l1 l2)` predicates are bidirectional. If the domain
      instance uses unidirectional roads, the distance calculation would need modification.
    - The primary cost driver considered is the sequence of actions needed for each
      package (pick-up, drive, drop). The cost of moving a vehicle *to* a package's
      location before pick-up is ignored to simplify the estimation.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored
      during heuristic calculation.
    - The goal consists solely of `(at package location)` predicates. All packages
      and locations mentioned in goals are assumed to be valid objects defined or
      inferrable from the task definition.
    - Locations, packages, and vehicles can be reliably identified by their usage
      patterns in the domain's predicates (e.g., subject of `in` is package,
      object of `in` is vehicle).

    # Heuristic Initialization
    - The constructor (`__init__`) performs one-time setup:
    - It infers the sets of all locations, packages, and vehicles by analyzing
      predicates found in the initial state, static facts, and goal conditions.
    - It parses the static `(road l1 l2)` facts to build an adjacency list
      representation of the location graph.
    - It computes all-pairs shortest path distances between all known locations
      using Breadth-First Search (BFS). These distances are stored for quick lookup.
      A large distance value signifies unreachability.
    - It parses the goal conditions to create a mapping from each package to its
      required goal location `(at package goal_loc)`.

    # Step-By-Step Thinking for Computing Heuristic
    - The `__call__` method computes the heuristic value for a given state:
    1. It first checks if the current state satisfies all goal conditions using
       `self.task.goal_reached(state)`. If true, the goal is reached, and the
       heuristic value is 0.
    2. Initialize `total_cost = 0`.
    3. Parse the current `state` (a set of fact strings) to determine the current
       location or container of each package and the location of each vehicle:
       - Store package locations (can be a location name or a vehicle name if inside).
       - Store vehicle locations.
    4. Iterate through each package `p` that has a defined goal location `goal_loc`
       (from the `self.goal_locations` map created during initialization).
       a. Check if the specific goal fact `(at p goal_loc)` is already present in
          the current `state`. If yes, this package is already settled, so add 0
          to the cost for this package and continue to the next.
       b. If the goal for package `p` is not met, determine its current status:
          Is it `at` a location or `in` a vehicle?
       c. **Case 1: Package `p` is at location `current_loc`.**
          - Since the goal is not met, `current_loc` must be different from `goal_loc`.
          - The estimated minimum actions are: pick-up `p` (cost 1), drive a vehicle
            carrying `p` from `current_loc` to `goal_loc` (cost = shortest distance),
            drop `p` (cost 1).
          - Retrieve the precomputed distance `dist = self.distances[(current_loc, goal_loc)]`.
          - Add `1 + dist + 1` to `total_cost`.
          - If `goal_loc` is unreachable from `current_loc` (indicated by a large `dist`),
            add a large penalty value instead to signify this difficulty.
       d. **Case 2: Package `p` is inside vehicle `v`.**
          - Find the current location of vehicle `v`, let it be `vehicle_loc`.
          - If `vehicle_loc == goal_loc`, the package only needs to be dropped. Add 1
            (for the drop action) to `total_cost`.
          - If `vehicle_loc != goal_loc`, the vehicle needs to drive from `vehicle_loc`
            to `goal_loc` (cost = shortest distance) and then drop the package (cost 1).
          - Retrieve the precomputed distance `dist = self.distances[(vehicle_loc, goal_loc)]`.
          - Add `dist + 1` to `total_cost`.
          - If `goal_loc` is unreachable from `vehicle_loc`, add a large penalty.
       e. Handle potential inconsistencies (e.g., package needed for goal not found
          in state, vehicle containing package has no location) by adding a large
          penalty (`self.max_dist`) to `total_cost`.
    5. Return the final `total_cost`, ensuring it's non-negative.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static
        init_facts = task.initial_state

        # --- Extract Objects (Infer from predicates) ---
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        # Combine all known facts to maximize chance of finding all objects
        all_facts_for_objects = init_facts | static_facts | self.goals

        temp_locatables = set() # Objects that appear in 'at' subject position

        for fact in all_facts_for_objects:
             try: # Add try-except for robustness against malformed facts
                 parts = get_parts(fact)
                 if not parts: continue # Skip empty or malformed facts
                 pred = parts[0]
                 args = parts[1:]

                 if pred == 'at' and len(args) == 2:
                     temp_locatables.add(args[0])
                     self.locations.add(args[1])
                 elif pred == 'in' and len(args) == 2:
                     # 'in' subject is always package, object is always vehicle per domain
                     self.packages.add(args[0])
                     self.vehicles.add(args[1])
                     temp_locatables.add(args[0]) # Packages are locatable
                 elif pred == 'road' and len(args) == 2:
                     self.locations.add(args[0])
                     self.locations.add(args[1])
                 elif pred == 'capacity' and len(args) == 2:
                     # 'capacity' subject is always vehicle per domain
                     self.vehicles.add(args[0])
                 elif pred == 'capacity-predecessor' and len(args) == 2:
                     # These are sizes, not locations/packages/vehicles
                     pass
                 # Check goal facts specifically for packages/locations if needed
                 elif fact in self.goals and pred == 'at' and len(args) == 2:
                     # If it appears as subject in goal 'at' and not yet ID'd as vehicle, assume package
                     if args[0] not in self.vehicles:
                         self.packages.add(args[0])
                     self.locations.add(args[1])

             except IndexError:
                 # Handle potential errors if get_parts fails or parts list is too short
                 # print(f"Warning: Could not parse fact during object extraction: {fact}")
                 pass


        # Refine locatables: those not identified as vehicles are likely packages
        # (unless other locatable types exist, which they don't in this domain)
        for item in temp_locatables:
            if item not in self.vehicles:
                self.packages.add(item)

        # Ensure vehicles found via 'in' or 'capacity' are not misclassified as packages
        self.packages = self.packages - self.vehicles
        # Ensure packages are not misclassified as vehicles
        self.vehicles = self.vehicles - self.packages

        # --- Parse Static Facts (Roads) ---
        self.roads = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for valid road predicate with 2 arguments
            if parts and parts[0] == 'road' and len(parts) == 3:
                 loc1, loc2 = parts[1], parts[2]
                 # Ensure both locations were identified before adding road
                 if loc1 in self.locations and loc2 in self.locations:
                    self.roads.add(tuple(parts[1:]))

        # --- Compute Shortest Path Distances ---
        # Handle case where there are no locations found
        if not self.locations:
             self.distances = {}
             self.max_dist = 1 # Define a minimal max_dist if no locations
        else:
             # Define max_dist as a value larger than any possible shortest path
             self.max_dist = len(self.locations) + 1
             self.distances = self._compute_shortest_paths()


        # --- Parse Goal Locations ---
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Only consider 'at' goals for objects identified as packages
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in self.packages:
                package, location = parts[1], parts[2]
                # Ensure goal location is a known location before storing
                if location in self.locations:
                    self.goal_locations[package] = location
                # else: print(f"Warning: Goal location '{location}' for package '{package}' is unknown.")


    def _compute_shortest_paths(self):
        """
        Computes all-pairs shortest paths using Breadth-First Search (BFS).
        Assumes roads are bidirectional.
        Returns a dictionary mapping (from_loc, to_loc) tuples to distances.
        Unreachable pairs will have a distance of self.max_dist.
        """
        distances = {}
        # Build adjacency list representation of the road network
        adj = {loc: [] for loc in self.locations}
        for r1, r2 in self.roads:
            # Add edges in both directions
            adj.setdefault(r1, []).append(r2)
            adj.setdefault(r2, []).append(r1)

        # Run BFS starting from each location
        for start_node in self.locations:
            # Initialize distances from start_node to max_dist (unreachable)
            for loc in self.locations:
                distances[(start_node, loc)] = self.max_dist
            # Distance from a node to itself is 0
            distances[(start_node, start_node)] = 0

            queue = deque([(start_node, 0)]) # Queue stores (node, distance_from_start)
            # Keep track of visited nodes *within this specific BFS run* to avoid cycles
            visited_this_run = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                # Use adj.get(node, []) to handle locations with no outgoing roads
                for neighbor in adj.get(current_node, []):
                    if neighbor not in visited_this_run:
                        visited_this_run.add(neighbor)
                        new_dist = dist + 1
                        distances[(start_node, neighbor)] = new_dist
                        queue.append((neighbor, new_dist))

        return distances


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # Check if goal is already reached using the task's method
        if self.task.goal_reached(state):
             return 0

        # --- Get Current Locations from State ---
        package_location = {} # Maps package name to its location (loc name or vehicle name)
        vehicle_location = {} # Maps vehicle name to its location name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    # Ensure location is known, otherwise treat package state as uncertain
                    if loc in self.locations:
                        package_location[obj] = loc
                    # else: print(f"Warning: Package {obj} at unknown location {loc} in state.")
                elif obj in self.vehicles:
                     # Ensure location is known
                     if loc in self.locations:
                        vehicle_location[obj] = loc
                     # else: print(f"Warning: Vehicle {obj} at unknown location {loc} in state.")

            elif pred == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                # Check if both package and vehicle are known objects
                if package in self.packages and vehicle in self.vehicles:
                     package_location[package] = vehicle # Store vehicle name as location marker

        total_cost = 0

        # --- Calculate Cost per Package Goal ---
        for package, goal_loc in self.goal_locations.items():
            # Construct the goal fact string to check if it's already satisfied
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                continue # This package's goal is met, cost is 0 for this package.

            # Find current state of the package if goal is not met
            if package not in package_location:
                 # If goal not met, package must be *somewhere*. If not in map, implies error.
                 # print(f"Warning: Package '{package}' needed for goal '{goal_loc}' not found in current state.")
                 total_cost += self.max_dist # Penalize inconsistent/unknown state
                 continue

            current_loc_or_vehicle = package_location[package]

            # Case 1: Package is at a location on the map
            if current_loc_or_vehicle in self.locations:
                current_loc = current_loc_or_vehicle
                # Goal is not met, so current_loc != goal_loc
                # Use .get for safety, defaulting to max_dist if pair not found (shouldn't happen with BFS init)
                dist = self.distances.get((current_loc, goal_loc), self.max_dist)

                if dist >= self.max_dist: # Unreachable
                    total_cost += self.max_dist * 2 # High penalty for unreachability
                else:
                    # Cost: pick-up(1) + drive(dist) + drop(1)
                    total_cost += (1 + dist + 1)

            # Case 2: Package is in a vehicle
            elif current_loc_or_vehicle in self.vehicles:
                vehicle = current_loc_or_vehicle
                # Find the vehicle's current location using .get for safety
                current_vehicle_loc = vehicle_location.get(vehicle)

                if current_vehicle_loc is None:
                    # Invalid state: vehicle containing package has no location predicate.
                    # print(f"Warning: Vehicle '{vehicle}' carrying '{package}' has no 'at' predicate in state.")
                    total_cost += self.max_dist # Penalize inconsistent state
                    continue

                # If vehicle is already at the goal location, just need to drop
                if current_vehicle_loc == goal_loc:
                     total_cost += 1 # Drop cost
                else:
                    # Need to drive then drop
                    dist = self.distances.get((current_vehicle_loc, goal_loc), self.max_dist)

                    if dist >= self.max_dist: # Unreachable
                        total_cost += self.max_dist * 2 # High penalty for unreachability
                    else:
                        # Cost: drive(dist) + drop(1)
                        total_cost += (dist + 1)
            else:
                 # Package location is neither a known location nor a known vehicle. Error state.
                 # print(f"Warning: Unknown location type '{current_loc_or_vehicle}' for package '{package}'.")
                 total_cost += self.max_dist # Penalize unexpected state

        # Ensure heuristic is non-negative (should be guaranteed by logic, but safe)
        return max(0, total_cost)

