import collections
import itertools
from fnmatch import fnmatch
# Import the base class - adjust the path if necessary based on project structure
# Assuming the base class `Heuristic` is available in `heuristics.heuristic_base`
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts "(predicate obj1 obj2 ...)" -> ["predicate", "obj1", "obj2", ...]
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    # Handles facts like "(at p1 l1)" -> ["at", "p1", "l1"]
    return fact[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the cost for each package individually
    based on its current state (at a location or in a vehicle) and sums these costs.
    The cost for a package includes the necessary pick-up, drive, and drop actions,
    using pre-calculated shortest path distances for driving. It serves as an estimate
    for Greedy Best-First Search and does not need to be admissible.

    # Assumptions
    - The heuristic ignores vehicle capacities (`capacity` predicate and `size` types).
      It assumes a vehicle is always available and capable of picking up a package
      if they are at the same location, regardless of current load or capacity limits.
    - The heuristic calculates the cost for each package independently. It does not
      account for potential synergies (one vehicle carrying multiple packages to
      the same or nearby locations) or resource conflicts (multiple packages
      needing the same vehicle simultaneously).
    - The cost estimation for a package currently on the ground (`at`) only considers
      the pickup action (cost 1), the drive from its current location to the goal
      (cost = shortest path distance), and the drop action (cost 1). It does not
      explicitly model the cost of moving a vehicle *to* the package's location first.
    - Roads (`road l1 l2`) define directed connections between locations. The cost
      of a `drive` action between directly connected locations is 1. Shortest path
      distances based on the `road` predicates are used to estimate the total driving cost.
      (Note: PDDL instances often define bidirectional roads explicitly by including
      both `(road l1 l2)` and `(road l2 l1)` facts).

    # Heuristic Initialization
    - The constructor (`__init__`) parses static facts (`road`) to build a graph
      representation of the road network (locations and directed connections).
    - It computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS) starting from each location. Distances are stored.
      Unreachable locations have infinite distance.
    - It extracts the goal location for each package from the task's goal specification
      (looking for `(at package location)` goals).
    - It identifies all package, vehicle, and location objects by inspecting static facts,
      initial state facts, goals, and potentially operator definitions to get a
      complete set of relevant objects.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal:** If the current state `node.state` satisfies all goal conditions
        (`self.task.goal_reached(node.state)`), the heuristic value is 0.
    2.  **Initialize Cost:** Set the total estimated cost `h = 0`.
    3.  **State Parsing:** Iterate through the facts in the current `node.state`:
        - Identify the location of each vehicle using `(at vehicle location)` facts.
        - Identify the location of each package using `(at package location)` facts.
        - Identify which packages are inside vehicles using `(in package vehicle)` facts.
        Store this information efficiently (e.g., in dictionaries).
    4.  **Iterate Packages:** For each package `p` that has a defined goal location in `self.goal_locations`:
        a.  Retrieve its goal location `loc_goal`.
        b.  Formulate the goal fact string, e.g., `"(at p loc_goal)"`.
        c.  Check if this goal fact is already present in the current `node.state`. If yes, this package is done; continue to the next package.
        d.  **Determine Package State:** Find out if the package `p` is currently `at` a location or `in` a vehicle based on the parsed state information.
        e.  **Calculate Package Cost:**
            - **If `p` is at `loc_curr`:**
                - Calculate the shortest path distance `dist = self.distances[loc_curr].get(goal_loc, float('inf'))`. Using `.get` provides robustness if `goal_loc` isn't reachable or wasn't found during BFS (though it should be if it's a valid location).
                - If `dist` is infinity, the goal is unreachable for this package from its current location; return `float('inf')` for the entire state heuristic, indicating a dead end or unsolvable state from here.
                - The estimated cost for this package is `1 (pick-up) + dist (drive actions) + 1 (drop)`. Add this cost to `h`.
            - **If `p` is in `vehicle`:**
                - Find the current location of that `vehicle`, let's say `loc_vehicle`. Check if the vehicle's location is known.
                - Calculate the shortest path distance `dist = self.distances[loc_vehicle].get(goal_loc, float('inf'))`.
                - If `dist` is infinity, the goal is unreachable for this package via this vehicle's current path; return `float('inf')`.
                - The estimated cost for this package is `dist (drive actions) + 1 (drop)`. Add this cost to `h`.
            - **Error Handling:** If the package's state cannot be determined (e.g., it's not `at` or `in` but its goal isn't met), this indicates an inconsistent state or problem; return `float('inf')`. Similarly, if the location of the vehicle carrying the package is unknown.
    5.  **Return Total Cost:** Return the accumulated value `h`. This sum represents the estimated total actions needed to satisfy all package goals individually, ignoring complex interactions.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information: building the
        road network, computing shortest paths, and identifying goals/objects.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.locations = set()
        self.packages = set()
        self.vehicles = set()
        # Adjacency list for the road graph: location -> [neighbor_location, ...]
        self.adj = collections.defaultdict(list)
        # Stores shortest path distances: distances[loc1][loc2] = distance
        # Default distance is infinity.
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        # Stores goal location for each package: goal_locations[package] = location
        self.goal_locations = {}

        # --- Identify Objects and Build Road Graph ---
        # Use a comprehensive set of facts from static, init, and goals
        # to ensure all relevant objects (locations, packages, vehicles) are found.
        all_known_facts = static_facts.union(task.initial_state).union(task.goals)

        for fact in all_known_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            predicate = parts[0]

            # Identify objects based on predicate structure
            if predicate == 'at' and len(parts) == 3: # (at ?x ?l)
                obj, loc = parts[1], parts[2]
                self.locations.add(loc)
                # We'll refine whether obj is package/vehicle later
            elif predicate == 'in' and len(parts) == 3: # (in ?p ?v)
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])
            elif predicate == 'road' and len(parts) == 3: # (road ?l1 ?l2)
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.adj[l1].append(l2) # Add directed edge l1 -> l2
            elif predicate == 'capacity' and len(parts) == 3: # (capacity ?v ?s)
                 self.vehicles.add(parts[1])
            # Other predicates like capacity-predecessor, type predicates are ignored by this heuristic

        # Extract package goals specifically and ensure objects are registered
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                 package, loc = parts[1], parts[2]
                 # Assume the first argument of 'at' in a goal is a package
                 self.packages.add(package)
                 self.locations.add(loc)
                 self.goal_locations[package] = loc

        # Refine object sets based on initial state 'at' facts,
        # assuming anything 'at' a location that isn't a package must be a vehicle.
        for fact in task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # If obj is at a location and not already identified as a package,
                 # classify it as a vehicle. This relies on the domain structure.
                 if obj not in self.locations and obj not in self.packages:
                     self.vehicles.add(obj)

        # --- Compute All-Pairs Shortest Paths using BFS ---
        # Convert location set to list to ensure stable iteration order
        all_locations_list = sorted(list(self.locations))
        for start_node in all_locations_list:
            # Skip if start_node is somehow invalid (e.g., empty string)
            if not start_node: continue

            # Initialize distance from start_node to itself as 0
            self.distances[start_node][start_node] = 0
            # Queue for BFS: stores (location, current_distance)
            queue = collections.deque([(start_node, 0)])
            # visited_bfs stores distances found so far *in this specific BFS run*
            visited_bfs = {start_node: 0}

            while queue:
                current_loc, dist = queue.popleft()

                # Explore neighbors based on the road graph adjacency list
                for neighbor in self.adj.get(current_loc, []):
                    # If neighbor hasn't been reached yet in this BFS run
                    if neighbor not in visited_bfs:
                        new_dist = dist + 1
                        visited_bfs[neighbor] = new_dist
                        # Update the main distance matrix
                        self.distances[start_node][neighbor] = new_dist
                        queue.append((neighbor, new_dist))
                    # No need to check for shorter paths in unweighted BFS


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the minimum number of actions (drive, pick-up, drop)
        needed to reach the goal state from the current state.
        """
        state = node.state

        # Check if the goal is already reached using the task's method.
        # This is the only condition under which the heuristic should be 0.
        if self.task.goal_reached(state):
             return 0

        # --- Parse current state to find locations of packages and vehicles ---
        current_package_locations = {} # Maps package -> location string (if 'at')
        package_in_vehicle = {}        # Maps package -> vehicle string (if 'in')
        current_vehicle_locations = {} # Maps vehicle -> location string

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential empty parts
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    # Record package location, ensuring it's not also marked 'in'
                    if obj not in package_in_vehicle:
                         current_package_locations[obj] = loc
                elif obj in self.vehicles:
                    current_vehicle_locations[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
                # If package was previously thought to be 'at', remove that entry
                if package in current_package_locations:
                    del current_package_locations[package]

        # --- Calculate heuristic value by summing costs for each unmet package goal ---
        total_heuristic_value = 0
        for package in self.packages:
            # Consider only packages that have a specific goal location defined
            if package not in self.goal_locations:
                 continue

            goal_loc = self.goal_locations[package]
            # Construct the goal fact string to check if it's already met
            goal_fact = f"(at {package} {goal_loc})"

            # If this specific package's goal is already met, skip cost calculation for it
            if goal_fact in state:
                continue

            package_cost = 0

            # Determine current state of the package and calculate its estimated cost to goal
            if package in current_package_locations:
                # Case 1: Package is at location `loc_curr`
                loc_curr = current_package_locations[package]
                # We know loc_curr != goal_loc because goal_fact is not in state.
                # Get shortest path distance, default to infinity if not found.
                dist = self.distances[loc_curr].get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # If distance is infinite, the goal is unreachable for this package
                    return float('inf') # Indicate unsolvable state / dead end
                # Estimated cost = pickup(1) + drive(dist) + drop(1)
                package_cost = 1 + dist + 1

            elif package in package_in_vehicle:
                # Case 2: Package is in vehicle `vehicle`
                vehicle = package_in_vehicle[package]
                if vehicle not in current_vehicle_locations:
                    # Error condition: Vehicle carrying the package has no known location.
                    # This implies an invalid or inconsistent state.
                    print(f"Error: Location of vehicle {vehicle} carrying package {package} not found in state {state}")
                    return float('inf') # Cannot estimate cost reliably

                loc_vehicle = current_vehicle_locations[vehicle]
                # Get shortest path distance from vehicle's location to goal location.
                dist = self.distances[loc_vehicle].get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # Goal is unreachable via this vehicle's current path
                    return float('inf')
                # Estimated cost = drive(dist) + drop(1)
                package_cost = dist + 1
            else:
                 # Error condition: Package has an unmet goal but is neither 'at' a location nor 'in' a vehicle.
                 # This suggests an inconsistency in the state representation or domain logic.
                 print(f"Error: Package {package} has unmet goal {goal_fact} but is not found 'at' or 'in' state {state}")
                 return float('inf') # Cannot estimate cost reliably

            total_heuristic_value += package_cost

        # The final heuristic value is the sum of estimated costs for all packages
        # that are not yet at their goal location.
        # If h=0 is returned here, it implies all package goals were met. Since the
        # initial check `if self.task.goal_reached(state): return 0` handles the
        # overall goal state, this case should only occur if all package goals are met
        # but potentially other non-package goals (if any existed) are not.
        # However, for this domain, goals typically only involve packages.
        return total_heuristic_value
