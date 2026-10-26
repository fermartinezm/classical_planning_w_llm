# Assuming heuristics.heuristic_base provides the base class
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

# Helper function to parse facts
def get_parts(fact):
    """Removes surrounding brackets and splits by space."""
    # Assuming valid PDDL fact strings like '(predicate arg1 arg2)'
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts
        return []
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern
# Using direct comparison instead of fnmatch for simplicity and speed
def match(fact, *args):
    """Checks if the fact parts match the given arguments."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(parts[i] == args[i] for i in range(len(args)))

# BFS function
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([(start_node, 0)]) # Use deque for efficient pop(0)
    visited = {start_node}

    while queue:
        current_node, dist = queue.popleft() # Use popleft() for deque

        # Check if current_node exists in graph keys before iterating
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
    return distances

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost for each package not at its goal location
        as the sum of:
        1. Cost to pick up the package (1 action, if not already in a vehicle).
        2. Cost to drive the package's carrier vehicle from its current location
           to the package's goal location (shortest path distance).
        3. Cost to drop off the package (1 action).
        The total heuristic is the sum of these costs for all misplaced packages.
        This heuristic ignores vehicle capacity and availability constraints.

    Assumptions:
        - The state and goal facts are represented as strings like '(predicate arg1 ...)'.
        - Road network is static and defined by '(road l1 l2)' facts in static_facts.
        - Roads are bidirectional (if road l1 l2 exists, road l2 l1 also exists).
        - Packages need to be picked up, driven, and dropped off.
        - All packages are distinct and have a single goal location specified in the goals.
        - Objects appearing as the first argument of an 'at' or 'in' predicate that also appear as the first argument of an 'at' predicate in the goals are considered packages.
        - Objects appearing as the first argument of an 'at' predicate that start with 'v' are considered vehicles (based on example naming convention).
        - The task object provides 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts).
        - The node object provides 'state' (frozenset of current facts).

    Heuristic Initialization:
        - Parses goal facts to map each package to its goal location (`self.goal_locations`).
        - Builds a graph of the road network from static facts (`self.road_graph`). Collects all unique locations mentioned in road facts.
        - Computes all-pairs shortest paths on the road network using BFS and stores them (`self.shortest_paths`). Distances to unreachable locations are stored as `float('inf')`.
        - Defines a large cost (`self.unreachable_cost`) used when a package's goal is unreachable from its current location or carrier's location.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the input node.
        2. Check if the current state is a goal state by verifying if all goal facts are present in the state. If yes, return 0.
        3. Initialize the total heuristic cost to 0.
        4. Create dictionaries to track the current status of each package (`package_status`: maps package name to ('at', location) or ('in', vehicle)) and the current location of each vehicle (`vehicle_locations`: maps vehicle name to location).
        5. Iterate through the facts in the current state:
            - Parse the fact string into parts. Skip malformed facts.
            - If the predicate is 'at' and there are three parts:
                - Let `obj` be the second part and `loc` be the third part.
                - If `obj` is a key in `self.goal_locations` (indicating it's a package with a goal), record its status as `('at', loc)` in `package_status`.
                - If `obj` starts with 'v' (assuming it's a vehicle based on naming convention), record its location in `vehicle_locations`.
            - If the predicate is 'in' and there are three parts:
                - Let `package` be the second part and `vehicle` be the third part.
                - If `package` is a key in `self.goal_locations` (indicating it's a package with a goal), record its status as `('in', vehicle)` in `package_status`.
        6. Iterate through each package and its goal location stored in `self.goal_locations`:
            - Construct the string representation of the goal fact for this package.
            - If this goal fact string is present in the current state, this package is already at its goal; continue to the next package.
            - If the package is not at its goal:
                - Retrieve the package's current status from `package_status`. If the package is not found in `package_status` (meaning it's not 'at' any location and not 'in' any vehicle, indicating an unexpected or invalid state), return `self.unreachable_cost`.
                - If the package status is `('at', current_location)`:
                    - Calculate the estimated drive cost as the shortest path distance from `current_location` to `goal_location` using `self.shortest_paths`. If no path exists (distance is `float('inf')`), the drive cost is `float('inf')`.
                    - The cost for this package is 1 (pick-up) + drive_cost + 1 (drop-off).
                - If the package status is `('in', vehicle)`:
                    - Retrieve the vehicle's current location from `vehicle_locations`. If the vehicle is not found (indicating an unexpected state where a package is in a vehicle whose location is unknown), return `self.unreachable_cost`.
                    - Let the vehicle's location be `current_location`.
                    - Calculate the estimated drive cost as the shortest path distance from `current_location` to `goal_location`. If no path exists (distance is `float('inf')`), the drive cost is `float('inf')`.
                    - The cost for this package is drive_cost + 1 (drop-off).
                - Add the calculated cost for this package to the `total_heuristic`. If any package cost is `float('inf')`, the total will become `float('inf')`.
        7. Return the `total_heuristic`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        # Assuming goal facts are always (at package location)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        self.road_graph = defaultdict(list)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming bidirectional roads
                locations.add(l1)
                locations.add(l2)

        self.shortest_paths = {}
        # Compute all-pairs shortest paths using BFS
        all_locations = list(locations)
        for start_loc in all_locations:
            distances = bfs(self.road_graph, start_loc)
            for end_loc in all_locations:
                 # Store distance, use infinity if unreachable
                 self.shortest_paths[(start_loc, end_loc)] = distances.get(end_loc, float('inf'))

        # Define a large cost for unreachable goals in the heuristic sum
        # Using float('inf') is appropriate here.
        self.unreachable_cost = float('inf')


    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
            return 0

        package_status = {} # Maps package -> ('at', loc) or ('in', vehicle)
        vehicle_locations = {} # Maps vehicle -> loc

        # Populate current locations/status from state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Identify packages based on goal_locations keys
                if obj in self.goal_locations:
                     package_status[obj] = ('at', loc)
                # Identify vehicles based on naming convention (less robust)
                # A more robust way would be to get vehicle objects from task definition
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 # Identify packages based on goal_locations keys
                 if package in self.goal_locations:
                     package_status[package] = ('in', vehicle)
                 # Vehicle location is tracked by 'at' predicate

        total_heuristic = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal using the precomputed goal fact string
            goal_fact_str = '(at {} {})'.format(package, goal_location)
            if goal_fact_str in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal, calculate its cost
            if package not in package_status:
                 # Package is not 'at' any location and not 'in' any vehicle.
                 # This indicates an unexpected state or an unreachable goal.
                 # Return a very high cost (infinity).
                 return self.unreachable_cost

            status, current_loc_or_vehicle = package_status[package]

            if status == 'at':
                # Package is at a location, needs pick-up, drive, drop
                current_location = current_loc_or_vehicle
                # Get drive cost (shortest path distance)
                drive_cost = self.shortest_paths.get((current_location, goal_location), self.unreachable_cost)

                # If drive is impossible, the package goal is unreachable from here.
                if drive_cost == float('inf'):
                    return self.unreachable_cost

                # Cost = pick (1) + drive (distance) + drop (1)
                package_cost = 1 + drive_cost + 1
            elif status == 'in':
                # Package is in a vehicle, needs drive, drop
                vehicle = current_loc_or_vehicle
                if vehicle not in vehicle_locations:
                     # Vehicle carrying package has no location? Invalid state.
                     return self.unreachable_cost
                current_location = vehicle_locations[vehicle]

                # Get drive cost (shortest path distance)
                drive_cost = self.shortest_paths.get((current_location, goal_location), self.unreachable_cost)

                # If drive is impossible, the package goal is unreachable from here.
                if drive_cost == float('inf'):
                    return self.unreachable_cost

                 # Cost = drive (distance) + drop (1)
                package_cost = drive_cost + 1
            else:
                 # Unknown status? Should not happen with current logic, but defensive.
                 return self.unreachable_cost

            # Add cost for this package. If any package cost is inf, total becomes inf.
            total_heuristic += package_cost

        # Return the total heuristic. If any package goal was unreachable, this will be inf.
        return total_heuristic
