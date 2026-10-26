import collections
import sys
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper function to parse PDDL facts like '(predicate obj1 obj2)'
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

# Helper function to match parsed fact parts against a pattern
def match(fact_parts, *pattern):
    """Check if the parts of a fact match a given pattern, allowing wildcards '*'."""
    # Example: match(['at', 'p1', 'l1'], "at", "*", "l1") -> True
    return len(fact_parts) == len(pattern) and all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by summing the estimated costs for moving each package to its target location.
    It considers the shortest path drive actions based on the 'road' network,
    plus the necessary pickup (1 action) and drop (1 action) actions for each package.
    It is designed for guiding Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The heuristic calculates the cost for each package goal `(at package location)` independently and sums them up.
    - It does not account for potential synergies (e.g., one vehicle carrying multiple packages simultaneously)
      or conflicts (e.g., vehicle capacity limits are ignored).
    - It assumes a suitable vehicle is implicitly available whenever a package needs to be picked up at its current location.
    - The cost estimate for moving a package currently on the ground (`at package loc`) includes one pickup action,
      the shortest driving distance from its current location to the goal location, and one drop action. It ignores the cost
      for a vehicle to initially travel *to* the package's location for pickup.
    - The cost estimate for moving a package already inside a vehicle (`in package vehicle`) includes the shortest
      driving distance from the vehicle's current location to the package's goal location, and one drop action.
    - The road network defined by `(road l1 l2)` facts is static and treated as potentially directed (an edge exists only if the fact exists).
      Shortest path distances (number of `drive` actions) between all connected locations are precomputed using BFS.

    # Heuristic Initialization
    - Identifies all unique location objects mentioned in static `road` facts, initial state `at` facts, and goal `at` facts.
    - Parses the goal conditions to identify the target location for each package specified in an `(at package location)` goal. Stores these target locations and the set of packages involved in goals.
    - Parses static `road` facts to build a directed graph representing the road network connectivity.
    - Precomputes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS) on the graph.
      Stores these distances (number of `drive` actions) for efficient lookup during heuristic evaluation. Unreachable locations have infinite distance.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal State:** If the current state satisfies all goal conditions (`task.goal_reached(state)` is true), the heuristic value is 0.
    2.  **Parse State:** Determine the current status of relevant entities:
        a.  Find the location of each vehicle using `(at vehicle loc)` facts.
        b.  Find the status of each package mentioned in the goals (`packages_in_goal`): is it `at` a location or `in` a vehicle?
    3.  **Iterate Package Goals:** For each package `p` that has a target `goal_loc` defined in `self.goal_locations`:
        a.  **Check if Goal Met:** If the fact `(at p goal_loc)` is already present in the current state, this specific goal is satisfied, and the cost contribution for this package is 0. Continue to the next package.
        b.  **Determine Current State of Package:**
            i.  Find if `p` is currently `at` some `current_loc` or `in` some `vehicle`.
            ii. If `p` is `in` a `vehicle`, find the vehicle's current location (`vehicle_loc`). The effective starting location for `p`'s journey is `vehicle_loc`. Mark `p` as being `in_vehicle`.
            iii.If `p` is `at` `current_loc`, the effective starting location is `current_loc`. Mark `p` as not `in_vehicle`.
            iv. If `p`'s status cannot be determined (neither `at` nor `in`), or if its carrying vehicle's location is unknown, the state is considered invalid or inconsistent. Return `sys.maxsize`.
        c.  **Calculate Path Cost:** Find the shortest path distance `dist` from the package's effective starting location to its `goal_loc` using the precomputed `self.distances`.
            i.  If `dist` is infinite (`float('inf')`), the goal location is unreachable for this package from its current state. The overall goal is impossible to reach. Return `sys.maxsize`.
        d.  **Estimate Action Cost for Package:**
            i.  If `p` was `in_vehicle`: Estimated cost = `dist` (drive actions) + 1 (drop action).
            ii. If `p` was not `in_vehicle`: Estimated cost = 1 (pickup action) + `dist` (drive actions) + 1 (drop action).
        e.  **Add Cost:** Add the estimated action cost for package `p` to the total heuristic value `h`.
    4.  **Return Total:** Return the total calculated sum `h` as an integer. If any step indicated an invalid or unreachable state, `sys.maxsize` would have been returned earlier.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing the task's static facts,
        initial state, and goals to precompute necessary information like
        shortest path distances between locations.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Gather all unique location objects from various sources
        all_locations = set()
        # Locations from static 'road' facts
        for fact in static_facts:
            parts = get_parts(fact)
            if match(parts, "road", "?l1", "?l2"):
                all_locations.add(parts[1])
                all_locations.add(parts[2])
        # Locations from initial state 'at' facts (second argument)
        for fact in task.initial_state:
            parts = get_parts(fact)
            if match(parts, "at", "?x", "?l"):
                 all_locations.add(parts[2])
        # Locations from goal 'at' facts (second argument)
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if match(parts, "at", "?p", "?l"):
                 all_locations.add(parts[2])
        self.locations = frozenset(all_locations) # Use frozenset for immutability

        # 2. Extract package goal locations and identify packages mentioned in goals
        self.goal_locations = {} # Stores mapping: package -> goal_location
        self.packages_in_goal = set() # Stores set of packages appearing in goals
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # We only consider 'at' goals for packages for this heuristic
            if match(parts, "at", "?p", "?l"):
                package, location = parts[1], parts[2]
                # Ensure the goal location is one of the identified valid locations
                if location in self.locations:
                    self.goal_locations[package] = location
                    self.packages_in_goal.add(package)
                else:
                    # This might indicate an issue in the PDDL problem definition
                    print(f"Warning: Goal location '{location}' for package '{package}' "
                          f"is not among the known locations derived from the problem. "
                          f"This goal might be unsatisfiable or heuristic may be inaccurate.")

        # 3. Build road network graph (adjacency list) from static 'road' facts
        self.adj = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if match(parts, "road", "?l1", "?l2"):
                l1, l2 = parts[1], parts[2]
                # Add directed edge only if both locations are known/valid
                if l1 in self.locations and l2 in self.locations:
                    self.adj[l1].add(l2)

        # 4. Precompute all-pairs shortest paths using BFS
        self.distances = {} # Stores shortest path distances: (loc1, loc2) -> distance
        for start_node in self.locations:
            # Initialize distances from start_node to all locations as infinity
            dist = {loc: float('inf') for loc in self.locations}

            # Check if start_node is valid before starting BFS
            if start_node in dist:
                 dist[start_node] = 0 # Distance to self is 0
                 queue = collections.deque([start_node]) # Queue for BFS

                 while queue:
                     current_loc = queue.popleft()
                     current_dist = dist[current_loc]

                     # Explore neighbors based on the road network graph
                     for neighbor in self.adj.get(current_loc, set()):
                         # Check if neighbor is a valid location and if it hasn't been reached yet
                         if neighbor in dist and dist[neighbor] == float('inf'):
                             dist[neighbor] = current_dist + 1 # Update distance
                             queue.append(neighbor) # Add neighbor to queue

            # Store computed distances originating from start_node
            for end_node, d in dist.items():
                self.distances[(start_node, end_node)] = d

    def get_shortest_dist(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance (number of drive actions)
        between two locations. Returns float('inf') if unreachable.
        """
        if loc1 == loc2:
            return 0
        # Lookup the precomputed distance; default to infinity if the pair is not found
        # (which implies unreachability or invalid locations)
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates the total number of actions (drive, pickup, drop) required
        to move all packages specified in the goals to their target locations.
        """
        state = node.state

        # If the state already satisfies all goal conditions, heuristic value is 0.
        if self.task.goal_reached(state):
             return 0

        h_value = 0.0 # Initialize heuristic value (use float for infinity checks)

        # Parse the current state to find vehicle locations and package statuses
        vehicle_location = {} # Maps vehicle name -> current location
        package_status = {} # Maps package_in_goal -> ('at', location) or ('in', vehicle)

        for fact in state:
            parts = get_parts(fact)
            # Check for 'at' facts: (at ?x ?l)
            if match(parts, "at", "?x", "?l"):
                obj, loc = parts[1], parts[2]
                # If the object is a package relevant to the goal, store its status
                if obj in self.packages_in_goal:
                    package_status[obj] = ('at', loc)
                else:
                    # Otherwise, assume it's a vehicle and store its location
                    # Note: This assumes objects are either goal packages or vehicles.
                    # A more robust approach might use type information if available.
                    vehicle_location[obj] = loc
            # Check for 'in' facts: (in ?p ?v)
            elif match(parts, "in", "?p", "?v"):
                package, vehicle = parts[1], parts[2]
                # If the package is relevant to the goal, store its status
                if package in self.packages_in_goal:
                    package_status[package] = ('in', vehicle)

        # Calculate cost contribution for each unmet package goal
        for package, goal_loc in self.goal_locations.items():
            # Construct the goal fact string to check if it's already met
            goal_fact_str = f"(at {package} {goal_loc})"
            if goal_fact_str in state:
                continue # This package goal is already satisfied.

            # Check if the package status was found in the current state
            if package not in package_status:
                # If a package mentioned in the goal is not found in the state ('at' or 'in'),
                # it implies an inconsistent or invalid state.
                # print(f"Error: Goal package '{package}' not found in state facts.")
                return sys.maxsize # Indicate unsolvable or invalid state

            status, loc_or_veh = package_status[package]

            current_physical_loc = None # The actual location relevant for distance calculation
            is_in_vehicle = False     # Flag indicating if the package is currently in a vehicle

            if status == 'at':
                current_physical_loc = loc_or_veh
                is_in_vehicle = False
                # Sanity check: Ensure the location where the package is 'at' is valid
                if current_physical_loc not in self.locations:
                    # print(f"Error: Package '{package}' is 'at' an invalid location '{current_physical_loc}'.")
                    return sys.maxsize
            elif status == 'in':
                vehicle = loc_or_veh
                # Check if the location of the vehicle carrying the package is known
                if vehicle not in vehicle_location:
                    # If the vehicle's location is unknown, we cannot calculate the path. Invalid state.
                    # print(f"Error: Vehicle '{vehicle}' carrying package '{package}' has no known location.")
                    return sys.maxsize
                current_physical_loc = vehicle_location[vehicle]
                is_in_vehicle = True
                # Sanity check: Ensure the vehicle's location is valid
                if current_physical_loc not in self.locations:
                     # print(f"Error: Vehicle '{vehicle}' is 'at' an invalid location '{current_physical_loc}'.")
                     return sys.maxsize

            # Calculate the shortest path distance from the package's effective location to its goal
            distance = self.get_shortest_dist(current_physical_loc, goal_loc)

            # If the distance is infinite, the goal is unreachable for this package
            if distance == float('inf'):
                # print(f"Error: Goal location '{goal_loc}' is unreachable for package '{package}' "
                #       f"from its current effective location '{current_physical_loc}'.")
                return sys.maxsize # Indicate that the overall goal is unsolvable

            # Estimate the number of actions required for this package based on its status
            if is_in_vehicle:
                # Needs driving (distance actions) + dropping (1 action)
                cost = distance + 1.0
                h_value += cost
            else: # Package is 'at' a location on the ground
                # Needs picking up (1 action) + driving (distance actions) + dropping (1 action)
                cost = 1.0 + distance + 1.0
                h_value += cost

        # Return the final heuristic value as an integer.
        # Rounding might be needed if intermediate float calculations introduce minor errors,
        # but costs here are integers (1 for pickup/drop, integer distance).
        # Casting to int should be sufficient.
        return int(h_value)

