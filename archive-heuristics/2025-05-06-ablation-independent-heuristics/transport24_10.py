from heuristics.heuristic_base import Heuristic
from task import Operator, Task
from collections import deque
# import logging # Removed logging calls in __call__

# Helper function to parse a fact string like '(predicate arg1 arg2)'
def parse_fact_string(fact_string):
    """Parses a PDDL fact string into a list of strings [predicate, arg1, arg2, ...]."""
    # Remove leading '(' and trailing ')' and split by space
    parts = fact_string.strip().strip('()').split()
    return parts

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required actions
    for each package that is not yet at its goal location. It calculates the
    shortest path distance on the road network for transportation and adds
    fixed costs for pick-up and drop-off actions. Capacity constraints and
    vehicle availability are ignored for simplicity and efficiency.

    Assumptions:
    - The state representation is valid, meaning every package is either
      '(at package location)' or '(in package vehicle)', and every vehicle
      is '(at vehicle location)'.
    - Roads are bidirectional (inferred from example PDDL).
    - The road network is static.
    - Object types (package, vehicle, location, size) can be inferred from
      the predicates they appear with in the initial state, goals, and static facts.

    Heuristic Initialization:
    1. Parses static facts, initial state, and goals to identify objects
       of type vehicle, package, location, and size based on the predicates
       they appear with ('capacity', 'in', 'road', 'at', 'capacity-predecessor').
    2. Builds a graph representation of the road network from '(road l1 l2)' facts.
    3. Computes all-pairs shortest paths between all identified locations
       using Breadth-First Search (BFS) starting from each location. Stores
       these distances. Unreachable locations have infinite distance.
    4. Stores the goal location for each package from the goal facts '(at p l)'.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Parse the current state to determine the current location of each
       package and vehicle. Store this information in dictionaries.
    3. For each package that has a goal location:
        a. Check if the package is currently '(at package current_location)'
           or '(in package vehicle)'.
        b. If the package is '(at package current_location)':
            i. If `current_location` is the goal location, the package goal
               is satisfied for this package; add 0 to `h`.
            ii. If `current_location` is not the goal location, the package
                needs to be picked up, transported, and dropped. Estimate the
                cost as `shortest_path_distance(current_location, goal_location) + 2`
                (1 for pick-up, 1 for drop). Add this cost to `h`. If the goal
                location is unreachable, the state is likely unsolvable, return infinity.
        c. If the package is '(in package vehicle)':
            i. Find the current location of the vehicle, `current_vehicle_location`.
            ii. If `current_vehicle_location` is the goal location, the package
                is at the destination location (inside the vehicle) and only
                needs to be dropped. Estimate the cost as 1 (for drop). Add
                this cost to `h`.
            iii. If `current_vehicle_location` is not the goal location, the
                 package needs to be transported further and dropped. Estimate
                 the cost as `shortest_path_distance(current_vehicle_location, goal_location) + 1`
                 (1 for drop). Add this cost to `h`. If the goal location is
                 unreachable, return infinity.
        d. If the package is not found in the state (neither 'at' nor 'in'),
           it indicates an invalid state or an unsolvable problem; return infinity.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for potential future use

        # --- Type Inference ---
        # Identify object types based on predicates they appear with
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()
        self.packages = set()

        all_facts = set(task.initial_state) | set(task.static) | set(task.goals)
        for fact_string in all_facts:
            parsed = parse_fact_string(fact_string)
            pred = parsed[0]
            args = parsed[1:]
            if pred == 'capacity':
                if args: self.vehicles.add(args[0])
                if len(args) > 1: self.sizes.add(args[1])
            elif pred == 'road':
                if args: self.locations.add(args[0])
                if len(args) > 1: self.locations.add(args[1])
            elif pred == 'capacity-predecessor':
                if args: self.sizes.add(args[0])
                if len(args) > 1: self.sizes.add(args[1])
            elif pred == 'in':
                if args: self.packages.add(args[0])
                if len(args) > 1: self.vehicles.add(args[1])
            elif pred == 'at':
                 if len(args) > 1: self.locations.add(args[1]) # Add location from 'at' fact

        # --- Road Network Graph and Shortest Paths ---
        self.location_graph = {}
        for fact_string in task.static:
            parsed = parse_fact_string(fact_string)
            if parsed[0] == 'road' and len(parsed) == 3:
                l1, l2 = parsed[1], parsed[2]
                self.location_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional
                self.location_graph.setdefault(l2, []).append(l1)

        self.shortest_paths = {}
        all_locations_list = list(self.locations) # Use a list for consistent iteration order

        for start_loc in all_locations_list:
            # BFS to find shortest paths from start_loc
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            distances = {start_loc: 0}

            while queue:
                current_loc, dist = queue.popleft()
                self.shortest_paths[(start_loc, current_loc)] = dist

                for neighbor in self.location_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

            # Mark unreachable locations with infinity
            for end_loc in all_locations_list:
                 if (start_loc, end_loc) not in self.shortest_paths:
                      self.shortest_paths[(start_loc, end_loc)] = float('inf')


        # --- Package Goals ---
        self.package_goals = {}
        for goal_fact_string in task.goals:
            parsed = parse_fact_string(goal_fact_string)
            if parsed[0] == 'at' and len(parsed) == 3:
                package, location = parsed[1], parsed[2]
                self.package_goals[package] = location
            # Ignore other potential goal types if any (domain only shows 'at')


    def get_shortest_path(self, loc1, loc2):
        """Looks up the pre-computed shortest path distance."""
        # Handle cases where locations might not be in the graph (e.g., isolated)
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        return self.shortest_paths.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        h = 0

        # Determine current status of packages and vehicles
        package_current_location = {} # package -> location (if at)
        package_current_vehicle = {}  # package -> vehicle (if in)
        vehicle_current_location = {} # vehicle -> location (if at)

        for fact_string in state:
            parsed = parse_fact_string(fact_string)
            pred = parsed[0]
            args = parsed[1:]
            if pred == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    package_current_location[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_location[obj] = loc
            elif pred == 'in' and len(args) == 2:
                p, v = args[0], args[1]
                package_current_vehicle[p] = v

        # Calculate heuristic contribution for each package with a goal
        for package, goal_location in self.package_goals.items():
            package_found_in_state = False

            if package in package_current_location:
                package_found_in_state = True
                current_location = package_current_location[package]
                if current_location != goal_location:
                    # Package is on the ground, not at goal
                    dist = self.get_shortest_path(current_location, goal_location)
                    if dist == float('inf'):
                        return float('inf') # Unsolvable
                    h += dist + 2 # drive + pick-up + drop
                # Else: package is at goal location on the ground, contribution is 0

            elif package in package_current_vehicle:
                package_found_in_state = True
                vehicle = package_current_vehicle[package]
                if vehicle in vehicle_current_location:
                    current_vehicle_location = vehicle_current_location[vehicle]
                    # Package is in a vehicle
                    if current_vehicle_location != goal_location:
                         # Vehicle is not at goal location
                         dist = self.get_shortest_path(current_vehicle_location, goal_location)
                         if dist == float('inf'):
                             return float('inf') # Unsolvable
                         h += dist + 1 # drive + drop
                    else:
                         # Vehicle is at goal location, just need to drop
                         h += 1 # drop
                else:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # This implies an invalid state or unsolvable problem.
                    return float('inf')

            # If a package has a goal but is not found in the state facts ('at' or 'in'),
            # it indicates an invalid state or an unsolvable problem.
            if not package_found_in_state:
                 return float('inf')

        return h
