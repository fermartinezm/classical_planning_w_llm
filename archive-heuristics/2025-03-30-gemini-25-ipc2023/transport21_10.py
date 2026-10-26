import collections
from math import inf
import sys
import os

# This code assumes that it can import Heuristic from heuristics.heuristic_base
# If the environment setup is different, this import might need adjustment.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder if the import fails (e.g., for running standalone)
    print("Warning: heuristics.heuristic_base not found. Using placeholder Heuristic class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

def get_parts(fact_string):
    """
    Parses a PDDL fact string "(predicate arg1 arg2 ...)" into a list
    of strings ["predicate", "arg1", "arg2", ...].
    Returns an empty list if the fact string is malformed.

    Args:
        fact_string: The PDDL fact as a string.

    Returns:
        A list containing the predicate and arguments, or an empty list
        if parsing fails.
    """
    fact_string = fact_string.strip()
    if not fact_string.startswith("(") or not fact_string.endswith(")"):
        # Avoid crashing on potentially malformed input from state representation
        # print(f"Warning: Malformed fact string encountered: {fact_string}", file=sys.stderr)
        return []
    # Split the content within parentheses
    return fact_string[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all packages
    to their specified goal locations. It calculates an estimated cost for each package
    independently based on its current state (at a location or inside a vehicle) and
    sums these costs. The cost calculation includes actions for pick-up (if needed),
    driving (using precomputed shortest path distances), and drop. It is designed
    for use with greedy best-first search and does not need to be admissible.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - The heuristic assumes an "ideal" scenario where a vehicle is readily available
      for transport tasks when a package needs to be picked up. It does not model
      specific vehicle starting locations for initiating pickup, nor does it consider
      vehicle capacities or potential conflicts between tasks for different packages.
      This simplification helps keep the heuristic efficiently computable.
    - It assumes the road network defined by `(road l1 l2)` predicates might not be
      fully connected. Goals requiring travel between disconnected locations will result
      in an infinite heuristic value, signaling unreachability.
    - Package goals are defined solely by `(at package location)` predicates in the
      goal specification. Other types of goals (e.g., vehicle location goals) are
      ignored by this heuristic.

    # Heuristic Initialization
    - Parses static `road` facts from `task.static` to build an adjacency list
      representing the road network graph.
    - Identifies all unique location names mentioned in the road facts.
    - Computes all-pairs shortest path distances between all known locations using
      Breadth-First Search (BFS) starting from each location. Distances are stored
      in `self.distances[start_location][end_location]`. Unreachable pairs have a
      distance of infinity (`math.inf`).
    - Parses `task.goals` to identify package goals of the form `(at package goal_location)`
      and stores them in `self.goal_locations` dictionary mapping package names to
      their target locations. It assumes objects appearing as the first argument in
      such goals are packages that need to be transported.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Initialization (`__init__`)**:
        a. Build the road graph (adjacency list) from `(road l1 l2)` static facts.
        b. Collect all unique location names involved in roads.
        c. Run BFS starting from each location `l` to compute shortest path distances `dist(l, l')`
           to all other reachable locations `l'`. Store these distances in `self.distances`.
           Initialize distances to `inf`, set `dist(l, l) = 0`.
        d. Extract `(at package location)` goals from `task.goals` into `self.goal_locations`.
           Also, store the set of goal facts `self.package_goals` for quick checking.
    2.  **Goal Check (`__call__`)**:
        a. Given the current state `node.state`, first check if all goal facts in
           `self.package_goals` are present in the state.
        b. If yes, the state is a goal state, return heuristic value 0.
    3.  **State Parsing (`__call__`)**: If not a goal state:
        a. Create dictionaries: `package_locations` (mapping package to its current location if `at`),
           `package_vehicles` (mapping package to the vehicle it's `in`), and `vehicle_locations`
           (mapping vehicle to its current location if `at`).
        b. Iterate through the facts in `node.state` to populate these dictionaries.
           Identify vehicles based on their appearance in `in` predicates or `at` predicates
           where the object is not a known package or location.
    4.  **Calculate Cost per Package (`__call__`)**: Initialize `total_heuristic_value = 0`.
        For each `package`, `goal_loc` pair in `self.goal_locations`:
        a. Define the target fact string `package_goal_fact = f"(at {package} {goal_loc})"`.
        b. **If `package_goal_fact` is in state**: The cost for this package goal is 0.
        c. **If package is in a vehicle**: Check `package_vehicles`. Let package `p` be in vehicle `v`.
           - Find `vehicle_loc` from `vehicle_locations`. If `v` has no location defined (indicates an error or inconsistent state), return `inf`.
           - If `vehicle_loc == goal_loc`: The cost is 1 (representing the required `drop` action).
           - If `vehicle_loc != goal_loc`: Calculate the shortest driving distance `d = self.distances[vehicle_loc][goal_loc]`. If `d` is `inf` (unreachable), return `inf`. The estimated cost is `d (drive) + 1 (drop)`.
        d. **If package is at a location**: Check `package_locations`. Let package `p` be at `current_loc`.
           - Since the goal fact check (4b) failed, `current_loc != goal_loc`.
           - Calculate the shortest driving distance `d = self.distances[current_loc][goal_loc]`. If `d` is `inf`, return `inf`.
           - The estimated cost is `1 (pick-up) + d (drive) + 1 (drop)`.
        e. **If package is missing**: If the package (from a goal) is not found `at` a location or `in` a vehicle in the current state, this indicates an anomaly. Return `inf`.
        f. Add the calculated `cost_for_package` to `total_heuristic_value`.
    5.  **Return Value (`__call__`)**:
        a. After iterating through all packages, return the computed `total_heuristic_value`.
        b. A final safety check: if `total_heuristic_value` happens to be 0 but the state
           was determined not to be a goal state in step 2, return 1. This ensures that
           only true goal states have a heuristic value of 0. (This situation should not
           occur with the current logic if goals are only `at` predicates).
    """

    def __init__(self, task):
        self.task = task
        # Store goal facts efficiently for checking goal state
        self.package_goals = {
            goal for goal in task.goals if get_parts(goal) and get_parts(goal)[0] == "at"
        }
        # Store mapping from package name to its goal location
        self.goal_locations = {}
        packages_in_goals = set() # Keep track of objects that are packages based on goals

        for goal in self.package_goals:
             parts = get_parts(goal)
             # Assuming goals are strictly (at package location)
             if len(parts) == 3:
                 package, location = parts[1], parts[2]
                 self.goal_locations[package] = location
                 packages_in_goals.add(package)

        static_facts = task.static

        # --- Build Road Graph & Compute Distances ---
        self.locations = set()
        adj = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip potential malformed static facts
            # Check for road predicate
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                adj[l1].append(l2)
                self.locations.add(l1)
                self.locations.add(l2)

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: inf))

        for start_node in self.locations:
            # Skip if start_node is somehow invalid (e.g., empty string)
            if not start_node: continue
            # Distance from node to itself is 0
            self.distances[start_node][start_node] = 0
            # Initialize BFS queue and visited dictionary (stores distances)
            queue = collections.deque([start_node])
            visited_dist = {start_node: 0}

            while queue:
                current_loc = queue.popleft()
                current_dist = visited_dist[current_loc]

                # Explore neighbors
                for neighbor in adj[current_loc]:
                    if neighbor not in visited_dist:
                        # Record distance and add to queue
                        visited_dist[neighbor] = current_dist + 1
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

    def __call__(self, node):
        state = node.state

        # --- Check if goal is already reached ---
        # A state is a goal state if all package goal facts are present
        is_goal_state = all(goal in state for goal in self.package_goals)
        if is_goal_state:
            # If it's a goal state, heuristic value must be 0
            return 0

        heuristic_value = 0

        # --- Parse Current State ---
        package_locations = {} # Maps package -> location for (at p l) facts
        package_vehicles = {}  # Maps package -> vehicle for (in p v) facts
        vehicle_locations = {} # Maps vehicle -> location for (at v l) facts

        # Identify vehicles and package locations from the state facts
        potential_vehicles = set() # Keep track of objects that might be vehicles

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts in state

            predicate = parts[0]
            obj1 = parts[1] if len(parts) > 1 else None
            obj2 = parts[2] if len(parts) > 2 else None

            if predicate == "at":
                # Check if obj1 is a package we care about (i.e., has a goal location)
                if obj1 in self.goal_locations:
                    package_locations[obj1] = obj2
                # If obj2 is a known location and obj1 is not a package with a goal,
                # assume obj1 might be a vehicle.
                elif obj2 in self.locations and obj1 not in self.goal_locations:
                    vehicle_locations[obj1] = obj2
                    potential_vehicles.add(obj1)
            elif predicate == "in":
                # obj1 is the package, obj2 is the vehicle
                if obj1 in self.goal_locations:
                    package_vehicles[obj1] = obj2
                    # The container obj2 must be a vehicle
                    potential_vehicles.add(obj2)

        # Ensure all vehicles identified (especially those from 'in' facts) have a known location
        # This loop might be redundant if the state is guaranteed to be consistent,
        # but serves as a check.
        for v in potential_vehicles:
            if v not in vehicle_locations:
                 # Try to find the location of vehicle 'v' from an 'at' fact
                 v_loc_found = False
                 for fact in state:
                     parts = get_parts(fact)
                     if parts and parts[0] == "at" and parts[1] == v:
                         vehicle_locations[v] = parts[2]
                         v_loc_found = True
                         break
                 # If a vehicle is mentioned (e.g., carrying a package) but has no 'at' fact,
                 # the state might be inconsistent or intermediate. For heuristic calculation,
                 # we need its location. If missing, we cannot calculate distances accurately.
                 if not v_loc_found:
                     # print(f"Warning: Vehicle {v} mentioned in state but has no location defined.", file=sys.stderr)
                     # Returning infinity signals a problem or uncomputable state for this heuristic.
                     return inf


        # --- Calculate Heuristic Value ---
        for package, goal_loc in self.goal_locations.items():
            cost_for_package = 0
            package_goal_fact = f"(at {package} {goal_loc})"

            # Check the state of the package
            if package_goal_fact in state:
                # Package is already at its goal location and dropped.
                cost_for_package = 0
            elif package in package_vehicles:
                # Package is inside a vehicle.
                vehicle = package_vehicles[package]
                # We need the vehicle's location. Check if it was found.
                if vehicle not in vehicle_locations:
                    # This case should ideally be caught by the check above.
                    # If vehicle location is unknown, cannot compute heuristic.
                    # print(f"Error: Location unknown for vehicle {vehicle} containing {package}.", file=sys.stderr)
                    return inf
                vehicle_loc = vehicle_locations[vehicle]

                if vehicle_loc == goal_loc:
                    # Package is in a vehicle at the goal location. Needs 1 drop action.
                    cost_for_package = 1
                else:
                    # Package is in a vehicle, needs driving then dropping.
                    dist = self.distances[vehicle_loc][goal_loc]
                    if dist == inf:
                        # Goal location is unreachable from the vehicle's current location.
                        # print(f"Goal for {package} unreachable from vehicle {vehicle} at {vehicle_loc}", file=sys.stderr)
                        return inf
                    # Cost is drive distance + 1 drop action.
                    cost_for_package = dist + 1
            elif package in package_locations:
                # Package is at some location on the ground.
                current_loc = package_locations[package]
                # Since it's not at the goal (checked earlier), current_loc != goal_loc.
                # Needs pick-up, driving, and dropping.
                dist = self.distances[current_loc][goal_loc]
                if dist == inf:
                    # Goal location is unreachable from the package's current location.
                    # print(f"Goal for {package} unreachable from location {current_loc}", file=sys.stderr)
                    return inf
                # Cost is 1 pick-up + drive distance + 1 drop action.
                cost_for_package = 1 + dist + 1
            else:
                # The package (which has a goal) is not 'at' any location and not 'in' any vehicle.
                # This indicates an inconsistent state or an error.
                # print(f"Error: Package {package} (required for goal) not found in state.", file=sys.stderr)
                return inf # Cannot compute heuristic if package state is unknown.

            # Add the cost for this package to the total heuristic value.
            heuristic_value += cost_for_package

        # Final safety check: If the heuristic calculation resulted in 0,
        # but we already determined it's not a goal state, return 1.
        # This ensures h=0 if and only if the state is a goal state.
        if heuristic_value == 0 and not is_goal_state:
             # This path should theoretically not be reached if the logic above is correct
             # for 'at' goals (as cost=1 is added for packages in vehicles at goal).
             # If it ever happens, returning 1 prevents h=0 for non-goals.
             # print("Warning: Heuristic calculated 0 for a non-goal state. Returning 1.", file=sys.stderr)
             return 1

        # Return the total estimated cost.
        return heuristic_value

