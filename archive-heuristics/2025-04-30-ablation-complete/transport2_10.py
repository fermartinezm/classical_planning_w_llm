from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    to its goal location. It sums the estimated costs for each package
    independently. The cost for a package is estimated as:
    1 (pick-up) + shortest_path_distance (drives) + 1 (drop),
    unless the package is already in a vehicle (skip pick-up) or already
    at its goal location (cost 0 for this package). It ignores vehicle capacity
    and assumes any vehicle can transport any package along the shortest path.

    # Assumptions
    - Vehicle capacity is ignored. Any vehicle can carry any package.
    - The shortest path distance between locations represents the minimum
      number of drive actions required between those specific locations.
    - The cost of pick-up, drive (per step), and drop actions is 1.
    - The goal is always for packages to be at specific locations on the ground,
      represented by `(at ?p ?l)` goals.
    - Object types (package, vehicle) are inferred from initial state and static facts.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task goals.
    - Identifies packages and vehicles by scanning initial state and static facts
      for predicates like `capacity` and `in`.
    - Builds a graph of locations based on `road` facts found in static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state (i.e., all `(at ?p ?l)` goal facts are true). If yes, return 0.
    2. Initialize total heuristic cost `h = 0`.
    3. Determine the current location of every package and vehicle by scanning the state facts.
       Use the pre-identified sets of packages and vehicles to classify objects.
       Note whether a package is on the ground (`(at ?p ?l)`) or inside a vehicle (`(in ?p ?v)`).
       Store vehicle locations (`(at ?v ?l)`).
    4. For each package `p` that has a goal location `l_goal` (extracted during initialization):
        a. If the package is currently on the ground at `l_goal` (i.e., `(at p l_goal)` is in the state), the cost for this package is 0. Continue to the next package.
        b. If the package is not at `l_goal`:
            i. Determine the package's effective current location. If it's on
               the ground at `l_current` (`(at p l_current)`), the effective location is `l_current`.
               If it's inside a vehicle `v` (`(in ?p ?v)`), find the vehicle's
               location `l_v` (`(at ?v ?l_v)`). The effective location is `l_v`.
               If the package or its vehicle is not found in the state facts, assign a large penalty and continue.
            ii. Calculate the shortest path distance `d` from the effective
                current location to `l_goal` using the precomputed distances.
                If no path exists, the state is likely unsolvable; add a large penalty to `h` and continue.
            iii. Estimate the actions needed for this package:
                 - If the package is on the ground: Add 1 (pick-up) + `d` (drives) + 1 (drop).
                 - If the package is in a vehicle: Add `d` (drives) + 1 (drop).
            iv. Add this estimated cost to the total heuristic `h`.
    5. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, road network,
        and identifying object types (packages and vehicles).
        """
        super().__init__(task)
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Identify vehicles and packages based on predicates in initial state and static facts
        self.vehicles = set()
        self.packages = set()
        locations = set() # Collect all locations mentioned

        # Scan initial state and static facts to identify types and locations
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if len(parts) > 1:
                 predicate = parts[0]
                 if predicate == 'capacity' and len(parts) == 3:
                     self.vehicles.add(parts[1])
                 elif predicate == 'in' and len(parts) == 3:
                     self.packages.add(parts[1])
                     self.vehicles.add(parts[2]) # The second arg of 'in' is a vehicle
                 elif predicate == 'at' and len(parts) == 3:
                     # Add location
                     locations.add(parts[2])
                 elif predicate == 'road' and len(parts) == 3:
                     locations.add(parts[1])
                     locations.add(parts[2])

        # Refine packages set: Any object appearing in 'at' or 'in' in initial/static/goals
        # that is not identified as a vehicle is considered a package.
        locatable_objects_in_problem = set()
        for fact in initial_state | static_facts | self.goals:
             parts = get_parts(fact)
             if len(parts) > 1:
                 predicate = parts[0]
                 if predicate in ['at', 'in']:
                     locatable_objects_in_problem.add(parts[1])
                     if predicate == 'in':
                         locatable_objects_in_problem.add(parts[2]) # The vehicle is also locatable

        # Packages are locatable objects found in the problem definition that are not vehicles
        self.packages = {obj for obj in locatable_objects_in_problem if obj not in self.vehicles}


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Only add goals for objects identified as packages
                if package in self.packages:
                    self.goal_locations[package] = location
                # else: ignore goals for non-packages if any exist (unlikely in this domain)


        # Build the road graph and compute distances.
        self.road_graph = {}
        # Locations set is already populated from initial/static facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)
                # locations set already populated

        self.distances = {}
        for start_node in locations:
            self.distances[start_node] = self._bfs(start_node) # BFS only needs start_node and graph

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find shortest distances to all reachable locations.
        Uses the pre-built self.road_graph.
        Returns a dictionary {location: distance}. Unreachable locations are not included.
        """
        distances_from_start = {}
        queue = deque([(start_node, 0)])
        visited = {start_node}
        distances_from_start[start_node] = 0

        while queue:
            current_node, dist = queue.popleft()

            # Get neighbors from the road graph, handle nodes with no outgoing roads
            neighbors = self.road_graph.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances_from_start[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return distances_from_start


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is a goal state.
        if self.goals.issubset(state):
             return 0

        # Track where packages and vehicles are currently located.
        package_status = {} # {package_name: {'location': loc, 'in_vehicle': vehicle_name or None}}
        vehicle_locations = {} # {vehicle_name: location}

        # Scan state facts to populate status dictionaries
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                predicate, obj1, obj2 = parts

                if predicate == "at":
                    if obj1 in self.packages:
                         package_status[obj1] = {'location': obj2, 'in_vehicle': None}
                    elif obj1 in self.vehicles:
                         vehicle_locations[obj1] = obj2
                    # else: ignore 'at' facts for objects not identified as packages or vehicles

                elif predicate == "in":
                     if obj1 in self.packages and obj2 in self.vehicles:
                         package_status[obj1] = {'location': None, 'in_vehicle': obj2}
                     # else: ignore 'in' facts for non-package/non-vehicle combinations

        total_cost = 0  # Initialize action cost counter.
        LARGE_PENALTY = 1000000 # Penalty for unreachable goals

        # Iterate through packages that have goals
        for package, goal_location in self.goal_locations.items():

            # If package is not found in state facts (very unlikely in valid states),
            # it means its status (at/in) is unknown. Assign penalty.
            if package not in package_status:
                 # print(f"Warning: Package {package} from goal not found in state facts.") # Debugging
                 total_cost += LARGE_PENALTY
                 continue

            current_status = package_status[package]
            package_cost = 0
            effective_current_location = None

            if current_status['in_vehicle'] is None:
                # Package is on the ground at current_location.
                effective_current_location = current_status['location']
                package_cost += 1 # Cost for pick-up

            else:
                # Package is inside a vehicle.
                vehicle_name = current_status['in_vehicle']
                # Need to find the vehicle's location.
                if vehicle_name not in vehicle_locations:
                     # Vehicle carrying the package is not found at any location.
                     # This indicates a malformed state. Assign a large cost.
                     # print(f"Warning: Vehicle {vehicle_name} carrying {package} not found at any location.") # Debugging
                     total_cost += LARGE_PENALTY
                     continue # Skip this package

                effective_current_location = vehicle_locations[vehicle_name]
                # No pick-up cost needed, already in vehicle.

            # Now calculate drive and drop costs from effective_current_location to goal_location
            if effective_current_location is not None:
                 if effective_current_location in self.distances and goal_location in self.distances[effective_current_location]:
                      package_cost += self.distances[effective_current_location][goal_location] # Cost for drives
                 else:
                      # Goal location is unreachable from effective current location.
                      # This state is likely unsolvable. Assign a large cost.
                      # print(f"Warning: Goal location {goal_location} unreachable from {effective_current_location} for package {package}.") # Debugging
                      total_cost += LARGE_PENALTY
                      continue # Skip this package

                 package_cost += 1 # Cost for drop

            total_cost += package_cost

        return total_cost
