import math
from collections import deque
# Assuming Heuristic base class is available from this path.
# If the environment uses a different structure, adjust the import accordingly.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_string):
    """
    Removes parentheses and splits the fact string into predicate and arguments.
    Handles potential errors if the string is not in the expected format "(pred arg1 arg2...)".
    Example: "(at p1 l1)" -> ['at', 'p1', 'l1']
    Returns an empty list if parsing fails (e.g., not starting/ending with parentheses).
    """
    if not isinstance(fact_string, str) or not fact_string or not fact_string.startswith("(") or not fact_string.endswith(")"):
        return []
    # Remove parentheses and split by space
    return fact_string.strip()[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their specified goal locations. It calculates an estimated cost for each package
    individually based on its current state (either at a location or inside a vehicle)
    and sums these costs. The cost estimation includes pick-up/drop actions and the
    driving distance based on the shortest path in the road network. This heuristic
    is designed for use with greedy best-first search and does not guarantee admissibility.

    # Assumptions
    - Roads defined by the `(road l1 l2)` predicate are bidirectional, as suggested
      by the structure of example instances where `(road l1 l2)` often implies `(road l2 l1)`.
    - The cost of moving a vehicle to a package's current location before a pick-up
      action is ignored. The heuristic focuses primarily on the package's journey
      from its current location (or the location of the vehicle carrying it) to its
      destination.
    - Vehicle capacity constraints (`capacity` predicate and `capacity-predecessor`)
      are ignored for simplicity and computational efficiency.
    - The heuristic assumes any available vehicle could potentially transport a package;
      it doesn't perform specific vehicle assignment or consider complex interactions
      like a vehicle needing to drop one package to pick up another due to capacity.
    - The shortest path distance (minimum number of 'drive' actions) between locations
      is used to estimate travel cost.
    - If a goal location is determined to be unreachable from a package's current
      location (based on the road network), a large finite cost (`UNREACHABLE_COST`)
      is assigned for that package's transport, signaling a potentially difficult or
      unsolvable state from the current node.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task definition (`task`) passed to it.
    - It extracts all unique locations, packages, and vehicles by analyzing the initial
      state, static facts, and goal conditions. Object types are inferred based on
      their usage in relevant predicates (e.g., `in`, `capacity`, `road`, `at` in goals).
    - It builds an adjacency list representation of the road network from `(road l1 l2)`
      static facts.
    - It precomputes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS) and stores these distances in the `self.distances` map.
    - It identifies the specific goal location for each package from the task's goals
      and stores this mapping in `self.package_goals`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize the total heuristic cost `h = 0`.
    2.  Parse the current `state` (a set of fact strings provided in the `node` object)
        to determine the current situation:
        - Find the location of each vehicle using `(at vehicle loc)` facts. Store in `vehicle_locations` dict.
        - Find the location of each package using `(at package loc)` facts. Store in `package_locations` dict.
        - Identify which package is inside which vehicle using `(in package vehicle)` facts. Store in `package_in_vehicle` dict.
    3.  Iterate through each package `p` and its corresponding `goal_loc` stored in `self.package_goals`:
        a. Check if the goal for this package `p` is already satisfied in the current state.
           This is true if `package_locations.get(p) == goal_loc`. If satisfied, this package
           contributes 0 to the heuristic value; skip to the next package.
        b. **If package `p` is currently at a location `current_loc`** (i.e., `p` is found as a key in `package_locations`):
           - Retrieve the precomputed shortest path distance: `dist = self.get_dist(current_loc, goal_loc)`.
           - Add the estimated cost for this package to `h`: `cost = 1 (for pick-up) + dist (for driving) + 1 (for drop)`.
           - Note: `self.get_dist` returns `self.UNREACHABLE_COST` if no path exists, correctly propagating the high cost.
        c. **If package `p` is currently inside a vehicle `v`** (i.e., `p` is found as a key in `package_in_vehicle`):
           - Find the current location of vehicle `v` using the `vehicle_locations` dictionary (`vehicle_loc = vehicle_locations.get(v)`).
           - If the vehicle's location is known:
             - Retrieve the precomputed shortest path distance: `dist = self.get_dist(vehicle_loc, goal_loc)`.
             - Add the estimated cost to `h`: `cost = dist (for driving) + 1 (for drop)`.
           - If the vehicle's location is *not* known (which indicates an inconsistent or unexpected state), add `self.UNREACHABLE_COST` to `h` as a penalty.
        d. **If package `p`'s location cannot be determined** (it's neither found in `package_locations` nor `package_in_vehicle`):
           - This signifies an unexpected state (e.g., the package is mentioned in goals but not present in the current state description). Add `self.UNREACHABLE_COST` to `h` as a penalty.
    4.  Return the total calculated cost `h`. This value serves as the heuristic estimate:
        - It is 0 if and only if all package goals (as defined in `self.package_goals`) are met in the current state.
        - It is positive otherwise, providing a quantitative estimate of the remaining work focused on package delivery actions and associated travel.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task, identifying objects (locations,
        packages, vehicles), precomputing road network distances using BFS, and
        storing package goal locations.
        """
        self.task = task
        self.goals = task.goals # Set of goal facts, e.g., {'(at p1 l2)'}
        static_facts = task.static # Set of static facts, e.g., roads, capacity-preds

        # Define a large finite cost for unreachable locations to avoid math issues with infinity
        self.UNREACHABLE_COST = 1_000_000

        # --- Object Identification ---
        # Initialize sets to store identified objects of each type
        self.locations = set()
        self.packages = set()
        self.vehicles = set()
        all_object_names = set() # Keep track of all mentioned object names for later refinement

        # Combine initial state and static facts for comprehensive object parsing
        facts_to_parse = task.initial_state.union(static_facts)
        for fact in facts_to_parse:
             parts = get_parts(fact)
             if not parts: continue # Skip if fact string is malformed

             predicate = parts[0]
             args = parts[1:]
             all_object_names.update(args) # Add all mentioned object names

             # Infer object types and identify locations based on predicate structure
             if predicate == 'at' and len(args) == 2: # (at ?x - locatable ?l - location)
                 # The second argument is always a location
                 self.locations.add(args[1])
             elif predicate == 'in' and len(args) == 2: # (in ?p - package ?v - vehicle)
                 # First arg is package, second is vehicle
                 self.packages.add(args[0])
                 self.vehicles.add(args[1])
             elif predicate == 'road' and len(args) == 2: # (road ?l1 ?l2 - location)
                 # Both arguments are locations
                 self.locations.add(args[0])
                 self.locations.add(args[1])
             elif predicate == 'capacity' and len(args) == 2: # (capacity ?v - vehicle ?s - size)
                 # First argument is a vehicle
                 self.vehicles.add(args[0])
             # 'capacity-predecessor' involves sizes, not directly used for object typing here

        # Identify packages and their goal locations specifically from the task goals
        self.package_goals = {} # Dictionary mapping package -> goal_location
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue
            # Goal is typically (at package location)
            if parts[0] == 'at' and len(parts) == 3:
                package, goal_loc = parts[1], parts[2]
                self.packages.add(package) # Ensure package is known
                self.package_goals[package] = goal_loc
                all_object_names.add(package)
                all_object_names.add(goal_loc)
                self.locations.add(goal_loc) # Ensure goal locations are in the set of locations

        # Refine object sets:
        # Remove known locations from the set of all object names to get potential locatables
        potential_locatables = all_object_names - self.locations
        # Assume any remaining potential locatable that isn't already identified as a vehicle must be a package.
        # This relies on the domain structure where locatables are either packages or vehicles.
        self.packages.update(potential_locatables - self.vehicles)

        # --- Build Road Graph Adjacency List ---
        # Initialize adjacency list for graph representation
        adj = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edge only if both l1 and l2 are confirmed locations
                if l1 in self.locations and l2 in self.locations:
                    adj[l1].append(l2)
                    # Assume roads are bidirectional based on domain examples
                    adj[l2].append(l1)

        # --- Compute All-Pairs Shortest Paths using BFS ---
        # Stores distances as {(loc1, loc2): distance}
        self.distances = self._compute_all_pairs_shortest_paths(adj)

    def _compute_all_pairs_shortest_paths(self, adj):
        """
        Computes shortest path distances between all pairs of known locations using BFS.
        Args:
            adj: Adjacency list representation of the road network ({location: [neighbors]}).
        Returns:
            A dictionary mapping (start_loc, end_loc) tuples to their shortest distance (int),
            or self.UNREACHABLE_COST if no path exists.
        """
        distances = {}
        for start_node in self.locations:
            # Initialize distance from start node to itself as 0
            distances[(start_node, start_node)] = 0
            # Queue for BFS: stores tuples of (node, current_distance)
            queue = deque([(start_node, 0)])
            # Set to keep track of visited nodes in the current BFS run from start_node
            visited_in_run = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                for neighbor in adj.get(current_node, []):
                    if neighbor not in visited_in_run:
                        visited_in_run.add(neighbor)
                        # Record distance and add neighbor to queue
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Ensure all pairs have an entry in the distances map.
        # If a pair (l1, l2) was not reached, assign UNREACHABLE_COST.
        for l1 in self.locations:
            for l2 in self.locations:
                if (l1, l2) not in distances:
                    distances[(l1, l2)] = self.UNREACHABLE_COST
        return distances

    def get_dist(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance between two locations.
        Args:
            loc1: The starting location name (string).
            loc2: The ending location name (string).
        Returns:
            The shortest distance (integer) if a path exists, otherwise self.UNREACHABLE_COST.
            Also returns UNREACHABLE_COST if either location is not recognized.
        """
        # Basic validation: Check if locations are known to the heuristic
        if loc1 not in self.locations or loc2 not in self.locations:
             # This might occur if the state contains unexpected/unknown location names.
             # print(f"Warning: Invalid location(s) requested for distance: {loc1}, {loc2}")
             return self.UNREACHABLE_COST
        # Return the precomputed distance, defaulting to UNREACHABLE_COST if the pair is somehow missing
        return self.distances.get((loc1, loc2), self.UNREACHABLE_COST)

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        The state is accessed via node.state, which is expected to be a frozenset of fact strings.
        """
        state = node.state # The current state as a set/frozenset of facts
        heuristic_value = 0

        # --- Parse current state efficiently to find object locations ---
        package_locations = {} # Maps package -> current location string
        vehicle_locations = {} # Maps vehicle -> current location string
        package_in_vehicle = {} # Maps package -> vehicle it's inside string

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            try:
                # Process 'at' facts for packages and vehicles
                if predicate == 'at' and len(args) == 2:
                    obj, loc = args[0], args[1]
                    # Check object type before storing location
                    if obj in self.packages:
                        package_locations[obj] = loc
                    elif obj in self.vehicles:
                        vehicle_locations[obj] = loc
                # Process 'in' facts for packages inside vehicles
                elif predicate == 'in' and len(args) == 2:
                    package, vehicle = args[0], args[1]
                    # Check types before storing relationship
                    if package in self.packages and vehicle in self.vehicles:
                        package_in_vehicle[package] = vehicle
            except IndexError:
                # Catch errors if a fact has fewer arguments than expected
                # print(f"Warning: Fact '{fact}' has unexpected structure.")
                continue

        # --- Calculate cost contribution for each package goal ---
        for package, goal_loc in self.package_goals.items():
            # 1. Check if this package's goal is already met
            current_pkg_loc = package_locations.get(package)
            if current_pkg_loc == goal_loc:
                continue # This package is done, contributes 0 cost.

            # 2. Determine the state of the package and calculate cost
            if package in package_locations:
                # Case A: Package is at a location (on the ground)
                current_loc = package_locations[package]
                dist = self.get_dist(current_loc, goal_loc)
                # Cost = pick-up action + drive actions + drop action
                heuristic_value += 1 + dist + 1

            elif package in package_in_vehicle:
                # Case B: Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's current location
                vehicle_loc = vehicle_locations.get(vehicle)
                if vehicle_loc is not None:
                    # Vehicle location is known
                    dist = self.get_dist(vehicle_loc, goal_loc)
                    # Cost = drive actions + drop action
                    heuristic_value += dist + 1
                else:
                    # Vehicle's location is unknown - this indicates an inconsistent state. Penalize.
                    # print(f"Warning: Location of vehicle {vehicle} carrying package {package} not found in state {state}.")
                    heuristic_value += self.UNREACHABLE_COST
            else:
                 # Case C: Package is missing (neither at a location nor in a vehicle)
                 # This implies an issue with the state or problem definition. Penalize.
                 # print(f"Warning: Package {package} needed for goal is not found at a location or in a vehicle in state {state}.")
                 heuristic_value += self.UNREACHABLE_COST

        # The final heuristic value is the sum of estimated costs for all unmet package goals.
        # It correctly returns 0 if and only if all package goals are satisfied.
        return heuristic_value

