import collections
from fnmatch import fnmatch
# Import the base class required by the planning system environment
from heuristics.heuristic_base import Heuristic
# No other external libraries like heapq or itertools are needed

# Helper function to parse PDDL facts like "(predicate obj1 obj2)" into a list ["predicate", "obj1", "obj2"]
def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()

# Helper function to match facts against a pattern with potential wildcards (*) using fnmatch
def match(fact, *args):
    """Check if a PDDL fact matches a given pattern using fnmatch for wildcards."""
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain, designed for Greedy Best-First Search.

    # Summary
    This heuristic estimates the total number of actions required to move all packages
    to their specified goal locations. It calculates an estimated cost for each package
    individually based on its current state (either at a location or inside a vehicle)
    and sums these costs. The cost for a package includes the necessary pick-up action
    (if not already in a vehicle), the minimum number of drive actions required to
    transport it to the goal location (based on precomputed shortest paths), and the
    final drop action.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is assumed to be 1.
    - The heuristic focuses on the essential actions (pickup, drive, drop) directly
      related to moving each package towards its goal.
    - It simplifies the problem significantly by ignoring vehicle capacity constraints.
      It implicitly assumes some vehicle will eventually have the required capacity
      when a pick-up or drop action is needed.
    - It ignores the cost of driving a vehicle *to* a package's location before picking it up.
      This makes the heuristic non-admissible (it might underestimate the true cost)
      but keeps it computationally cheap and potentially provides strong guidance for
      a greedy search algorithm by focusing on package progress.
    - It assumes that some vehicle will always be available to perform the required actions.
    - It assumes the road network defined by `(road l1 l2)` facts is static and
      correctly represents connectivity.

    # Heuristic Initialization
    - Parses the task's goal conditions (`task.goals`) to identify the target location
      for each package. Stores this mapping in `self.goal_locations`.
    - Identifies all unique package objects that have a goal condition associated with them
      and stores them in `self.packages`.
    - Parses the static facts (`task.static`) containing `(road ?l1 ?l2)` predicates
      to build a directed graph representation of the locations and road network (`self.road_graph`).
    - Identifies all unique locations involved in roads, the initial state (`task.initial_state`),
      or goals (`self.locations`).
    - Computes the shortest path distances (minimum number of drive actions) between all
      pairs of locations using Breadth-First Search (BFS) on the `road_graph`. Stores
      these distances in `self.shortest_paths`. Unreachable locations have a distance of infinity.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** For the given state (`node.state`), efficiently extract the
        current situation using dictionaries:
        - `package_locations`: Maps package `p` to location `l` if `(at p l)` is true.
        - `package_vehicles`: Maps package `p` to vehicle `v` if `(in p v)` is true.
        - `vehicle_locations`: Maps vehicle `v` (or any non-package locatable object)
          to location `l` if `(at v l)` is true.
    2.  **Check State Consistency:** Verify that every vehicle `v` found in `package_vehicles`
        (i.e., carrying a package relevant to the goal) also has a known location in
        `vehicle_locations`. If not, the state is considered inconsistent or invalid,
        and the heuristic returns infinity.
    3.  **Iterate Through Packages:** For every package `p` in `self.packages` (those with goals):
        a.  Retrieve the package's goal location `l_goal` from `self.goal_locations`.
        b.  **Determine Package Status & Cost:**
            i.  **If `p` is at `l_current` (in `package_locations`):**
                - If `l_current == l_goal`, the goal for `p` is met. Cost contribution = 0.
                - If `l_current != l_goal`, calculate the shortest path distance `d` from
                  `l_current` to `l_goal` using `self.shortest_paths`. If `d` is infinity
                  (unreachable), return `float('inf')`. Otherwise, the cost contribution
                  is `1 (pick-up) + d (drive) + 1 (drop)`.
            ii. **If `p` is in vehicle `v` (in `package_vehicles`):**
                - Get vehicle's location `l_vehicle` from `vehicle_locations`.
                - Calculate the shortest path distance `d` from `l_vehicle` to `l_goal`.
                  If `d` is infinity, return `float('inf')`. Otherwise, the cost
                  contribution is `d (drive) + 1 (drop)`.
            iii.**If `p` is neither `at` nor `in`:** This indicates an invalid state.
                  Return `float('inf')`.
        c.  **Accumulate Cost:** Add the calculated cost contribution for package `p` to
            the `total_heuristic_value`.
    4.  **Return Value:** The final `total_heuristic_value` is returned. It estimates the
        minimum number of pick-up, drive (while carrying), and drop actions needed.
        It is 0 if and only if all packages are at their respective goal locations.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task goals and static information.
        Builds the road graph and computes all-pairs shortest paths between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations and identify packages with goals
        self.goal_locations = {} # Stores {package_name: goal_location_name}
        self.packages = set()    # Stores names of packages mentioned in goals
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Assume goals are primarily (at package location) predicates
            if len(parts) == 3 and parts[0] == 'at':
                 # Assumption: First argument of 'at' in a goal is a package.
                 # This might need adjustment if the domain uses different goal types
                 # or if type information is available and more reliable.
                 package = parts[1]
                 location = parts[2]
                 self.goal_locations[package] = location
                 self.packages.add(package)

        # 2. Identify all locations and build the directed road graph
        self.locations = set()
        self.road_graph = collections.defaultdict(set) # Stores {location: {neighbor1, neighbor2,...}}
        for fact in static_facts:
            parts = get_parts(fact)
            # Process (road l1 l2) facts
            if len(parts) == 3 and parts[0] == 'road':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph[loc1].add(loc2) # Add directed edge l1 -> l2

        # Ensure locations mentioned in initial state or goals (but potentially not in roads) are included
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                 # The object at a location could be a package or vehicle.
                 # The location itself is the third part.
                 loc = parts[2]
                 self.locations.add(loc)
        for loc in self.goal_locations.values():
             self.locations.add(loc)

        # 3. Compute all-pairs shortest paths using BFS (suitable for unweighted graphs)
        self.shortest_paths = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        # Stores {start_loc: {end_loc: distance}}

        for start_node in self.locations:
            self.shortest_paths[start_node][start_node] = 0 # Distance to self is 0
            queue = collections.deque([(start_node, 0)]) # Queue stores (node, distance_from_start)
            # visited_dist stores the shortest distance found so far to prevent cycles and redundant work
            visited_dist = {start_node: 0}

            while queue:
                current_node, current_dist = queue.popleft()

                # Explore neighbors based on the directed road graph
                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        # If this neighbor hasn't been reached yet in this BFS run
                        if neighbor not in visited_dist:
                            new_dist = current_dist + 1
                            visited_dist[neighbor] = new_dist
                            self.shortest_paths[start_node][neighbor] = new_dist
                            queue.append((neighbor, new_dist))
                        # For unweighted graphs, the first time BFS reaches a node is via a shortest path,
                        # so no need to check for shorter paths to already visited nodes.


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates the remaining actions to move packages to their goals.
        """
        state = node.state
        total_heuristic_value = 0

        # Parse current state efficiently to find locations of packages and vehicles
        package_locations = {} # package -> location
        package_vehicles = {}  # package -> vehicle
        vehicle_locations = {} # vehicle (or other locatable) -> location

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if len(parts) == 3:
                if predicate == 'at':
                    obj, loc = parts[1], parts[2]
                    if obj in self.packages: # Check if the object is a package we care about
                        package_locations[obj] = loc
                    else:
                        # Assume other objects found 'at' a location might be vehicles.
                        # This is a simplification; ideally, use type info if available.
                        vehicle_locations[obj] = loc
                elif predicate == 'in':
                    package, vehicle = parts[1], parts[2]
                    if package in self.packages: # Check if it's a package we care about
                        package_vehicles[package] = vehicle

        # --- State Consistency Check ---
        # Ensure every vehicle carrying a package we track has a known location.
        for pkg, veh in package_vehicles.items():
            if veh not in vehicle_locations:
                 # If a vehicle is holding a package but its location isn't specified in the state,
                 # the state is inconsistent or something is wrong. Return infinity.
                 # print(f"Heuristic Error: Vehicle {veh} carrying package {pkg} has no 'at' fact in state {state}.")
                 return float('inf')

        # --- Calculate Heuristic Cost ---
        # Iterate through all packages that have a goal defined
        for package in self.packages:
            goal_loc = self.goal_locations.get(package)
            # This check should ideally not fail if self.packages is built correctly from goals
            if goal_loc is None:
                continue # Skip if no goal location is known for this package

            cost_for_package = 0
            # Case 1: Package is currently at a location
            if package in package_locations:
                current_loc = package_locations[package]
                if current_loc == goal_loc:
                    cost_for_package = 0 # Package is already at its goal location
                else:
                    # Package needs: pickup(1) + drive(d) + drop(1)
                    drive_dist = self.shortest_paths[current_loc].get(goal_loc, float('inf'))
                    if drive_dist == float('inf'):
                        # Goal location is unreachable from the package's current location
                        # print(f"Heuristic Info: Goal {goal_loc} unreachable from {current_loc} for {package}.")
                        return float('inf')
                    cost_for_package = 1 + drive_dist + 1

            # Case 2: Package is currently inside a vehicle
            elif package in package_vehicles:
                vehicle = package_vehicles[package]
                # Vehicle location is guaranteed by the consistency check above
                vehicle_loc = vehicle_locations[vehicle]

                # Package needs: drive(d) + drop(1)
                drive_dist = self.shortest_paths[vehicle_loc].get(goal_loc, float('inf'))
                if drive_dist == float('inf'):
                     # Goal location is unreachable from the vehicle's current location
                     # print(f"Heuristic Info: Goal {goal_loc} unreachable from vehicle {vehicle} at {vehicle_loc} for {package}.")
                     return float('inf')
                cost_for_package = drive_dist + 1

            # Case 3: Package state is unknown (neither 'at' nor 'in')
            else:
                 # This indicates an invalid state according to the domain logic
                 # (a package must be either at a location or in a vehicle).
                 # print(f"Heuristic Error: Package {package} has no location ('at' or 'in') in state {state}.")
                 return float('inf')

            # Add the cost for this package to the total heuristic value
            total_heuristic_value += cost_for_package

        # Return the total estimated cost for all packages
        # This value is 0 if and only if all packages are at their goal locations.
        return total_heuristic_value
