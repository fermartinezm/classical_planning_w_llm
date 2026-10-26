from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts: return False # Handle malformed facts

    # Check if the number of parts matches the number of arguments in the pattern
    # A pattern like ("at", "*", "l1") expects 3 parts.
    if len(parts) != len(args):
        return False

    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package from its current location to its goal location. It sums
    the estimated costs for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location defined in the task goals.
    - Vehicles are assumed to be available and have sufficient capacity when needed.
    - The cost of moving a package includes getting a vehicle to the package (if on ground),
      picking it up, driving the vehicle to the goal location, and dropping the package.
    - The cost of driving between two locations is the shortest path distance
      in the road network, where each road segment costs 1 drive action.
    - The heuristic ignores vehicle capacity constraints and the specific vehicle used.
    - The heuristic assumes all locations involved in package transport are connected
      via the road network, or that unreachable goals imply a very high cost.

    # Heuristic Initialization
    - Parses static facts to build the road network graph (locations and connections).
    - Identifies all relevant locations from static facts, initial state, and goals.
    - Computes all-pairs shortest path distances between these relevant locations
      using BFS on the road graph. Isolated locations have infinite distance to others.
    - Extracts the goal location for each package from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Create dictionaries to track the current location of each locatable object
       (`locatable_locations`: maps object -> location_name) and which packages
       are inside which vehicles (`package_in_vehicle`: maps package -> vehicle_name)
       by iterating through the facts in the current state.
       - `(at ?x ?l)`: locatable `?x` is on the ground at location `?l`.
       - `(in ?p ?v)`: package `?p` is inside vehicle `?v`.
       - `(at ?v ?l)`: vehicle `?v` is at location `?l`.
       The package's physical location is `l` if on the ground, or `l_v` if inside vehicle `v` which is at `l_v`.
    3. For each package `p` that has a goal location `l_goal` defined in the task goals:
       a. Check if the goal condition `(at p l_goal)` is already satisfied in the current state. If yes, the cost for this package is 0, proceed to the next package.
       b. If the goal is not satisfied, determine the package's current physical location (`current_loc`).
          - If package `p` is in the `package_in_vehicle` map (meaning `(in p v)` is in the state): It is inside vehicle `v`. Find vehicle `v`'s location `l_v` from the `locatable_locations` map (`(at v l_v)`). `current_loc = l_v`. The package is in a vehicle.
          - If package `p` is not in the `package_in_vehicle` map: It must be on the ground. Find its location `l` from the `locatable_locations` map (`(at p l)`). `current_loc = l`. The package is on the ground.
          - If the package's location cannot be determined from the state (should not happen in valid states), add a large penalty and skip.
       c. Calculate the estimated cost to move the package from `current_loc` to `l_goal`.
          - Find the shortest distance `dist` between `current_loc` and `l_goal` using the precomputed distances. If `current_loc` or `l_goal` are not in the road network or are unreachable, `dist` is infinity.
          - If `dist` is infinity, add a large penalty.
          - If `dist` is finite:
             - If the package is on the ground at `current_loc` (i.e., it was not found in the `package_in_vehicle` map): Cost = `pick-up` (1) + `drive` (`dist`) + `drop` (1) = `2 + dist`.
             - If the package is inside a vehicle at `current_loc` (i.e., it was found in the `package_in_vehicle` map): Cost = `drive` (`dist`) + `drop` (1) = `dist + 1`.
       d. Add the calculated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Road network graph and location distances.
        - Goal locations for each package.
        - All relevant locations from the problem.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Build the road network graph and collect all locations
        self.road_graph = {}
        all_relevant_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Roads are bidirectional
                all_relevant_locations.add(l1)
                all_relevant_locations.add(l2)

        # Add locations from initial state and goals
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  all_relevant_locations.add(loc)

        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at" and len(parts) == 3:
                  package, loc = parts[1], parts[2]
                  all_relevant_locations.add(loc)

        self.locations = list(all_relevant_locations) # Use a list for consistent ordering if needed, set is fine too

        # Compute all-pairs shortest path distances using BFS
        self.distances = {}
        LARGE_DIST = 1000000 # Use a large number for unreachable locations

        # Initialize distances for all pairs to LARGE_DIST
        for l1 in self.locations:
            self.distances[l1] = {}
            for l2 in self.locations:
                 self.distances[l1][l2] = LARGE_DIST
            self.distances[l1][l1] = 0 # Distance to self is 0

        # Run BFS from each location that is part of the road network
        locations_in_graph = set(self.road_graph.keys())

        for start_loc in self.locations:
            # Only run BFS if the start location is part of the road network graph
            if start_loc not in locations_in_graph:
                 continue # This location is isolated from the network

            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            # Distance to self is already 0

            while queue:
                current_loc, dist = queue.popleft()

                # Check if current_loc is in the graph before accessing neighbors
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where locatable objects are currently located on the ground
        # locatable_locations: maps object -> location_name (for objects with 'at' fact)
        locatable_locations = {}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  locatable_locations[obj] = loc

        # Track which packages are inside which vehicles
        # package_in_vehicle: maps package -> vehicle_name (for packages with 'in' fact)
        package_in_vehicle = {}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "in" and len(parts) == 3:
                  package, vehicle = parts[1], parts[2]
                  if package in self.goal_locations: # Only track packages we need to move
                       package_in_vehicle[package] = vehicle


        total_cost = 0
        LARGE_PENALTY = 1000 # Penalty for unreachable goals or unknown locations

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the goal condition (at p l_goal) is already satisfied
            if f"(at {package} {goal_location})" in state:
                continue # Goal achieved for this package, cost is 0

            # Determine the package's current physical location (`current_loc`)
            current_location = None
            is_in_vehicle = False

            if package in package_in_vehicle:
                 is_in_vehicle = True
                 vehicle_name = package_in_vehicle[package]
                 current_location = locatable_locations.get(vehicle_name) # Get vehicle's location
                 if current_location is None:
                     # Vehicle location unknown? Add a large penalty.
                     # print(f"Warning: Location of vehicle {vehicle_name} (carrying {package}) unknown in state.")
                     total_cost += LARGE_PENALTY
                     continue
            else:
                 # Package is not in a vehicle, must be on the ground
                 current_location = locatable_locations.get(package)
                 if current_location is None:
                     # Package location unknown? Add a large penalty.
                     # print(f"Warning: Location of package {package} unknown in state.")
                     total_cost += LARGE_PENALTY
                     continue

            # Now current_location holds the physical location of the package

            # Get the distance between current location and goal location
            # Use .get() with default LARGE_PENALTY for unreachable
            dist = self.distances.get(current_location, {}).get(goal_location, LARGE_PENALTY) # Use penalty if start/goal not in distance map

            if dist >= LARGE_PENALTY: # Check against penalty value
                # Goal location is unreachable from current location via roads or locations are unknown.
                total_cost += LARGE_PENALTY
            else:
                # Calculate cost based on whether the package is on the ground or in a vehicle
                if is_in_vehicle:
                    # Package is in a vehicle at current_location, needs drive + drop
                    total_cost += dist + 1
                else:
                    # Package is on the ground at current_location, needs pick-up + drive + drop
                    total_cost += 1 + dist + 1 # pick-up + drive + drop

        return total_cost
