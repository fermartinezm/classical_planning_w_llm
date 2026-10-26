import sys
from collections import deque

# Assuming the Heuristic base class is available in the path.
# If the planner environment provides a different way to load base classes, adjust accordingly.
# Example: Adjust the path based on where heuristic_base.py is located relative to this file.
# try:
#     from heuristics.heuristic_base import Heuristic
# except ImportError:
#     # Define a dummy base class if the import fails (e.g., for standalone testing)
#     print("Warning: Heuristic base class not found. Using dummy base class.", file=sys.stderr)
#     class Heuristic:
#         def __init__(self, task): pass
#         def __call__(self, node): raise NotImplementedError

# If the planner framework already provides the Heuristic base class,
# this import might not be needed, or might be handled differently.
# For this example, we assume it needs to be imported or defined.
# Let's assume it's available via:
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(predicate arg1 arg2)" -> ["predicate", "arg1", "arg2"]
    """
    return fact[1:-1].split()


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL 'transport' domain.

    # Summary
    Estimates the cost to reach the goal state in the Transport domain for use
    with greedy best-first search. The heuristic sums the estimated costs for
    moving each package to its target location, ignoring vehicle capacities and
    potential conflicts/synergies between packages. The cost is measured in
    the number of actions (drive, pick-up, drop).

    # Assumptions
    - Ignores vehicle capacity constraints (`capacity`, `capacity-predecessor`).
      The heuristic does not check if a vehicle has enough capacity to pick up a package.
    - Ignores conflicts where multiple packages might need the same vehicle simultaneously.
      The cost for each package is calculated independently.
    - Assumes a vehicle is readily available for pickup actions. It estimates the cost
      to move the package from its current location (or the location of the vehicle carrying it)
      to the goal location, plus the pickup/drop actions, but *not* the cost of bringing a
      vehicle to the package initially. This makes the heuristic non-admissible but simpler.
    - Assumes `road` predicates define a bidirectional graph based on typical PDDL usage
      and the provided examples. If the domain uses one-way roads, the graph building
      logic in `__init__` would need adjustment.
    - The cost for a package is estimated as the shortest path driving distance
      (number of `drive` actions) plus the necessary `pick-up` (if applicable) and `drop` actions (always 1 if not at goal).

    # Heuristic Initialization
    - Parses `task.goals` to identify the target location for each package specified
      in an `(at <package> <location>)` goal predicate. Stores these in `self.goal_loc`.
    - Identifies all unique packages, vehicles, and locations relevant to the problem
      by scanning the initial state, static facts, and goals. It uses predicates like
      `at`, `in`, `road`, and `capacity` to infer object types (vehicle, package, location).
      Relies partly on object naming conventions (e.g., 'p' prefix for packages, 'v' for vehicles)
      as a fallback if type information is not explicitly derivable from predicates.
    - Parses static `road` facts to build an adjacency list representation (`self.adj`)
      of the location graph, assuming roads are bidirectional.
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS) starting
      from each location. Stores the minimum number of `drive` actions between any two
      locations in a nested dictionary `self.dist[loc1][loc2]`. Unreachable locations
      are assigned an infinite distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost H = 0.
    2. Check if the current state (`node.state`) already satisfies all goal conditions
       using `self.task.goal_reached(state)`. If yes, the goal is reached, return 0.
    3. Parse the current `state` to determine the current location of each package
       (either directly via `(at p l)` or indirectly via `(in p v)` and `(at v l)`)
       and each vehicle (`(at v l)`). Store these findings in `package_location`
       (mapping package to its location name or the vehicle name it's in) and
       `vehicle_location` (mapping vehicle to its location name) dictionaries.
    4. Iterate through each package `p` and its target location `target_loc` stored in `self.goal_loc`.
       a. Check if the specific goal fact `(at p target_loc)` is present in the
          current `state`. If yes, this particular goal is satisfied, so add 0 cost
          for this package and continue to the next package.
       b. If the goal `(at p target_loc)` is not satisfied, find the package's current
          situation using the `package_location` map:
          i.  **Package `p` is on the ground at location `current_loc`:**
              - Look up the shortest path distance `d = self.get_location_dist(current_loc, target_loc)`.
              - If `d` is infinity (meaning `target_loc` is unreachable from `current_loc`),
                the goal for `p` is unreachable from this state. Return `float('inf')`.
              - The estimated cost for `p` is `d` (drive actions) + `1` (pick-up action) + `1` (drop action).
              - Add `d + 2` to the total heuristic cost H.
          ii. **Package `p` is inside vehicle `v`:** (`current_state_p` will be the vehicle name `v`)
              - Find the location `v_loc` of vehicle `v` using the `vehicle_location` map.
              - If `v_loc` is not found (e.g., vehicle `v` has no `(at v ...)` fact in the state),
                this indicates an invalid or unexpected state. Return `float('inf')`.
              - Look up the shortest path distance `d = self.get_location_dist(v_loc, target_loc)`.
              - If `d` is infinity, the goal is unreachable. Return `float('inf')`.
              - The estimated cost for `p` is `d` (drive actions) + `1` (drop action).
              - Add `d + 1` to H.
          iii. **Package `p` is not found in the state:** If `p` is required for the goal
               but doesn't have an `(at ...)` or `(in ...)` fact associated with it in the
               current state, the state is problematic or the package doesn't exist.
               Return `float('inf')`.
    5. After iterating through all packages in `self.goal_loc`, return the total calculated
       heuristic cost H. If any package calculation resulted in infinity, H will be infinity,
       signaling potential unsolvability or an invalid state. If H is 0, it implies all
       package goals considered by the heuristic are met.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task, building the road graph,
        and pre-calculating all-pairs shortest paths (APSP).
        """
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # --- Data Structures ---
        self.locations = set()
        self.packages = set()
        self.vehicles = set()
        self.goal_loc = {} # package -> goal_location
        self.adj = {} # location -> set(neighbor_locations)
        self.dist = {} # loc1 -> {loc2 -> shortest_path_distance}

        # --- Parse Goals to find package goals ---
        for goal in self.goals:
            parts = get_parts(goal)
            # Focus on 'at' goals for packages
            if parts[0] == "at":
                # Assume the first argument of 'at' in goal is the package
                package, loc = parts[1], parts[2]
                self.goal_loc[package] = loc
                self.packages.add(package) # Tentatively mark as package
                self.locations.add(loc) # Add goal location

        # --- Identify All Objects and Locations from Static and Init ---
        facts_for_obj_scan = self.static_facts.union(task.initial_state)
        known_vehicles = set()
        initial_packages = set() # Packages explicitly mentioned in init state

        for fact in facts_for_obj_scan:
            parts = get_parts(fact)
            predicate = parts[0]
            if not parts: continue # Skip empty facts if any

            if predicate == "road":
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    self.adj.setdefault(l1, set()).add(l2)
                    self.adj.setdefault(l2, set()).add(l1) # Assume bidirectional
            elif predicate == "at":
                 if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    self.locations.add(loc)
                    # Note object, type will be refined later
            elif predicate == "capacity":
                if len(parts) >= 2: # Should be (capacity vehicle size)
                    vehicle = parts[1]
                    known_vehicles.add(vehicle)
            elif predicate == "in": # Found in initial state
                 if len(parts) == 3:
                     package = parts[1]
                     initial_packages.add(package) # Mark as package based on 'in' predicate

        # Refine vehicles list (vehicles are those with capacity)
        self.vehicles = known_vehicles
        # Refine packages list: includes those in goals + those initially 'in' something
        # Remove any known vehicles that might have been added tentatively
        self.packages.update(initial_packages)
        self.packages = self.packages - self.vehicles

        # Ensure all locations found are keys in the adjacency list
        for loc in self.locations:
            self.adj.setdefault(loc, set())

        # Final check: ensure all goal locations are known locations
        # This handles cases where a goal location might not appear elsewhere
        for loc in self.goal_loc.values():
            if loc not in self.locations:
                self.locations.add(loc)
                self.adj.setdefault(loc, set()) # Add isolated goal location to graph

        # --- Compute All-Pairs Shortest Paths (APSP) using BFS ---
        self.dist = self._compute_apsp()

    def _compute_apsp(self):
        """
        Computes all-pairs shortest paths using BFS from each location.
        Returns a nested dictionary: distances[start_loc][end_loc] = distance.
        """
        distances = {}
        max_dist = float('inf') # Use infinity for unreachable locations

        for start_node in self.locations:
            distances[start_node] = {} # Initialize distances from start_node
            for loc in self.locations:
                distances[start_node][loc] = max_dist
            distances[start_node][start_node] = 0

            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Explore neighbors using the adjacency list built from 'road' facts
                for neighbor in self.adj.get(current_node, set()):
                    # If distance is still infinity, this neighbor hasn't been reached yet
                    if distances[start_node][neighbor] == max_dist:
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
                    # In standard BFS on unweighted graphs, the first time a node is reached,
                    # it's via a shortest path, so no need to check for shorter paths.

        return distances

    def get_location_dist(self, loc1, loc2):
        """
        Returns the pre-computed shortest distance between two locations.
        Returns float('inf') if locations are invalid or unreachable.
        """
        # Check if loc1 is a valid starting point in our precomputed distances
        if loc1 not in self.dist:
            # print(f"Warning: Start location '{loc1}' not found in distance map.", file=sys.stderr)
            return float('inf')
        # Check if loc2 is a valid destination from loc1
        if loc2 not in self.dist[loc1]:
            # print(f"Warning: End location '{loc2}' not found in distance map from '{loc1}'.", file=sys.stderr)
            return float('inf')

        return self.dist[loc1][loc2]

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Estimates the minimum number of actions required to move packages
        to their goal locations.
        """
        state = node.state
        h_cost = 0

        # Check if goal is already reached using the task's method
        # This is the most reliable way to check for goal state.
        if self.task.goal_reached(state):
            return 0

        # --- Parse current state to find package and vehicle locations ---
        package_location = {} # package -> location name (if at) or vehicle name (if in)
        vehicle_location = {} # vehicle -> location name

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if not parts: continue

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if obj is a package or vehicle we identified during init
                if obj in self.packages:
                    package_location[obj] = loc
                elif obj in self.vehicles:
                    vehicle_location[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Store that the package is 'in' the vehicle
                if package in self.packages:
                    package_location[package] = vehicle # Use vehicle name as indicator

        # --- Calculate cost for each package goal defined in self.goal_loc ---
        for package, target_loc in self.goal_loc.items():
            # Construct the goal fact string to check against the current state
            goal_fact = f"(at {package} {target_loc})"

            # If this specific package goal is already satisfied, skip it
            if goal_fact in state:
                continue

            # Find where the package is currently located based on parsed state
            current_state_p = package_location.get(package)

            if current_state_p is None:
                # If a package required for the goal is not found in the state
                # (neither 'at' nor 'in'), something is wrong. The goal is likely unreachable.
                # print(f"Warning: Goal package '{package}' not found in state.", file=sys.stderr)
                return float('inf')

            # Check if the package's current state indicates it's inside a vehicle
            is_in_vehicle = current_state_p in self.vehicles

            if is_in_vehicle:
                # Package is in vehicle 'current_state_p'
                vehicle = current_state_p
                current_vehicle_loc = vehicle_location.get(vehicle)

                if current_vehicle_loc is None:
                    # The vehicle containing the package doesn't have a location? Invalid state.
                    # print(f"Warning: Vehicle '{vehicle}' containing '{package}' has no 'at' fact.", file=sys.stderr)
                    return float('inf')

                # Cost = drive distance from vehicle's location + 1 drop action
                dist = self.get_location_dist(current_vehicle_loc, target_loc)
                if dist == float('inf'):
                    # Target location is unreachable from the vehicle's current location.
                    return float('inf')
                h_cost += dist + 1

            else:
                # Package is at location `current_loc` (current_state_p is the location name)
                current_loc = current_state_p
                # Ensure current_loc is a known location for distance calculation
                if current_loc not in self.locations:
                     # print(f"Warning: Package '{package}' at unknown location '{current_loc}'.", file=sys.stderr)
                     return float('inf') # Cannot calculate distance from unknown location

                # Cost = drive distance from package's loc + 1 pickup action + 1 drop action
                dist = self.get_location_dist(current_loc, target_loc)
                if dist == float('inf'):
                    # Target location is unreachable from the package's current location.
                    return float('inf')
                h_cost += dist + 2

        # If h_cost is 0 here, it means all package 'at' goals were satisfied.
        # Since we checked task.goal_reached() at the beginning, this case implies
        # either the goal involves other conditions, or the heuristic calculation
        # somehow missed an unsatisfied package goal (which shouldn't happen with this logic).
        # Returning 0 is correct based on the heuristic's calculation.
        return h_cost
