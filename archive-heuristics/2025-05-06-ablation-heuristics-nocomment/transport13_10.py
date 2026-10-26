from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Assumes fact is like '(predicate arg1 arg2)'
    # Handle potential empty facts or malformed strings gracefully
    fact = fact.strip()
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Estimates the cost to reach the goal by summing up the estimated costs
    for each package that is not yet at its goal location. The estimated cost
    for a single package is the sum of actions needed: pick-up (if on ground),
    driving (shortest path distance), and drop (if in vehicle). This heuristic
    is non-admissible as it ignores resource constraints (vehicle capacity,
    vehicle availability) and assumes independent package movements.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Heuristic Initialization:
        - Stores the entire task object for access to goals and initial state.
        - Stores the goal locations for each package specified in the goal state.
        - Builds the road network graph from static 'road' facts.
        - Collects all relevant locations mentioned in road facts, initial state
          'at' facts, and goal 'at' facts.
        - Computes shortest path distances between all pairs of these relevant
          locations using BFS, storing them in a dictionary. Unreachable locations
          have a distance of float('inf').
        """
        self.task = task # Store the task object
        self.goals = task.goals # Keep direct access to goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are only (at ?p ?l) for packages
            if parts and parts[0] == "at" and len(parts) == 3:
                 # Assume the first argument of goal 'at' is always a package
                 package_name = parts[1]
                 location_name = parts[2]
                 self.goal_locations[package_name] = location_name

        # 2. Build road network graph and collect all locations
        road_graph = {}
        locations = set()

        # Add locations from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                if l1 not in road_graph:
                    road_graph[l1] = []
                road_graph[l1].append(l2)

        # Add locations from initial state facts (at ?x ?l)
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                locations.add(parts[2])

        # Add locations from goal facts (at ?p ?l)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 locations.add(parts[2])

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations_list = list(locations) # Use a list for consistent ordering if needed, set is fine too
        for start_loc in all_locations_list:
            dist_from_start = self._bfs(start_loc, road_graph, locations)
            for target_loc, dist in dist_from_start.items():
                self.distances[(start_loc, target_loc)] = dist

    def _bfs(self, start_location, road_graph, all_locations):
        """Helper function to perform BFS from a start location."""
        distances = {loc: float('inf') for loc in all_locations}
        if start_location in distances: # Ensure start_location is one of the known locations
            distances[start_location] = 0
            queue = deque([start_location])

            while queue:
                current_location = queue.popleft()
                current_dist = distances[current_location]

                # Check if current_location has outgoing roads
                if current_location in road_graph:
                    for neighbor in road_graph[current_location]:
                        # Ensure neighbor is a known location and hasn't been visited yet
                        if neighbor in distances and distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        # If start_location was not in all_locations, distances remains all inf, which is correct.
        return distances


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using the task's goal_reached method.
           If it is, return 0.
        2. Initialize the total heuristic cost to 0.
        3. Parse the current state to find the location of each package
           (either on the ground or inside a vehicle) and the location
           of each vehicle. Classify objects into packages and vehicles
           based on the predicates they appear with ('at', 'in'). Store
           these locations/containment relationships in temporary dictionaries.
        4. Iterate through each package that has a specified goal location
           in the problem's goal state (stored in self.goal_locations).
        5. For the current package, check if the goal fact `(at package goal_location)`
           is already present in the current state. If it is, this package
           contributes 0 to the heuristic, and we move to the next package.
        6. If the goal fact is not present, the package needs to be moved.
           Determine the package's current status:
           - If the package is on the ground at `current_location` (found in pkg_loc):
             - It needs a pick-up action (cost +1).
             - It needs to be transported from `current_location` to `goal_location`.
               The minimum number of drive actions is the shortest path distance
               `dist(current_location, goal_location)` (cost +dist). Look up this
               distance in the precomputed self.distances.
             - It needs a drop action at `goal_location` (cost +1).
             - Total cost for this package: `1 + dist(current_location, goal_location) + 1`.
           - If the package is inside a vehicle `vehicle` (found in pkg_in_veh)
             which is at `current_location` (found in veh_loc):
             - It needs to be transported from `current_location` (the vehicle's location)
               to `goal_location`. The minimum number of drive actions is the shortest
               path distance `dist(current_location, goal_location)` (cost +dist).
               Look up this distance.
             - It needs a drop action at `goal_location` (cost +1).
             - Total cost for this package: `dist(current_location, goal_location) + 1`.
           - If the package is not found in pkg_loc or pkg_in_veh, or if a vehicle
             carrying a package is not found in veh_loc, this indicates an issue
             (e.g., package/vehicle not in a known location, or package from goal
             is missing from the state). In such cases, the goal is likely unreachable
             or the state is malformed; return float('inf').
        7. If the distance lookup for any required drive segment returns float('inf')
           (meaning the target location is unreachable from the current location),
           return float('inf') for the total heuristic value, indicating an unreachable goal.
        8. Sum up the costs calculated for all packages that are not yet at their
           goal locations. This sum is the final heuristic value.

        Assumptions:
        - The goal state primarily contains facts of the form `(at ?p ?l)` where `?p` is a package.
        - Objects (packages, vehicles, locations) mentioned in the initial state
          and goal state are consistent with the domain definition.
        - The state representation uses strings for facts, e.g., `'(at p1 l1)'`.
        - The `Heuristic` base class provides the expected `__init__` and `__call__` structure,
          and the `Task` object provides `goals`, `static`, and `initial_state` attributes,
          as well as a `goal_reached(state)` method.
        - Object types (package vs vehicle) can be inferred from the predicates they appear with
          ('at' and 'in').
        """
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # 2 & 3. Parse current state and classify objects
        pkg_loc = {} # package -> location (if on ground)
        pkg_in_veh = {} # package -> vehicle (if in vehicle)
        veh_loc = {} # vehicle -> location

        potential_vehicles = set() # Objects appearing as the second arg of 'in'
        potential_packages = set() # Objects appearing as the first arg of 'at' or 'in'
        
        # First pass: Identify potential types based on predicate structure
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj_name = parts[1]
                # loc_name = parts[2] # Not needed in this pass
                potential_packages.add(obj_name) # Could be package or vehicle
            elif predicate == 'in' and len(parts) == 3:
                pkg_name = parts[1]
                veh_name = parts[2]
                potential_packages.add(pkg_name)
                potential_vehicles.add(veh_name)

        # Refine classification: An object is a vehicle if it's a potential_vehicle.
        # An object is a package if it's a potential_package but not a vehicle.
        vehicles = potential_vehicles
        packages = potential_packages - vehicles # Objects that are potentially packages but not vehicles

        # Second pass: Populate location/in dictionaries based on refined types
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate == 'at' and len(parts) == 3:
                 obj_name = parts[1]
                 loc_name = parts[2]
                 if obj_name in vehicles:
                     veh_loc[obj_name] = loc_name
                 elif obj_name in packages:
                     pkg_loc[obj_name] = loc_name
                 # else: object type not classified (e.g., location, size) - ignore
             elif predicate == 'in' and len(parts) == 3:
                 pkg_name = parts[1]
                 veh_name = parts[2]
                 if pkg_name in packages and veh_name in vehicles:
                     pkg_in_veh[pkg_name] = veh_name
                 # else: malformed state or unexpected object types - ignore

        total_cost = 0

        # 4. Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # 5. Check if the goal (at package goal_location) is satisfied
            goal_fact_str = f"'(at {package} {goal_location})'"
            if goal_fact_str in state:
                continue # This package is already at its goal location on the ground

            # 6. Goal is not satisfied. Calculate cost for this package.
            current_location = None
            current_vehicle = None

            if package in pkg_loc:
                # Package is on the ground
                current_location = pkg_loc[package]
                # Needs pickup, drive, drop
                cost_for_package = 1 # pick-up
                if current_location != goal_location:
                     # Add drive cost if not already at the goal location
                     # Handle potential unreachable locations
                     drive_cost = self.distances.get((current_location, goal_location), float('inf'))
                     if drive_cost == float('inf'):
                         # If goal location is unreachable from current location, return infinity
                         return float('inf')
                     cost_for_package += drive_cost
                cost_for_package += 1 # drop
                total_cost += cost_for_package

            elif package in pkg_in_veh:
                # Package is in a vehicle
                current_vehicle = pkg_in_veh[package]
                if current_vehicle in veh_loc:
                    current_location = veh_loc[current_vehicle]
                    # Needs drive (if not already at goal location), drop
                    if current_location != goal_location:
                        # Add drive cost if vehicle is not already at the goal location
                        # Handle potential unreachable locations
                        drive_cost = self.distances.get((current_location, goal_location), float('inf'))
                        if drive_cost == float('inf'):
                            return float('inf')
                        total_cost += drive_cost
                    total_cost += 1 # drop
                else:
                    # Vehicle location unknown - should not happen in valid state
                    # If vehicle exists but its location is not in state, goal is unreachable
                    return float('inf')
            else:
                # Package from goal not found in pkg_loc or pkg_in_veh.
                # This means the package object exists (it's in goal_locations)
                # but is not currently located anywhere in the state.
                # This indicates an invalid state or an unreachable goal.
                 return float('inf')

        # 8. Return the total summed cost
        return total_cost
