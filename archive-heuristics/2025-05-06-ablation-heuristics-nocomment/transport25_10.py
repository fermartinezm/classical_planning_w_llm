# from fnmatch import fnmatch # Not used
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """
    Parses a PDDL fact string into a list of its components.
    E.g., '(at p1 l1)' becomes ['at', 'p1', 'l1'].
    """
    # Remove parentheses and split by space
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing up the estimated costs
        for each package that is not yet at its goal location. The estimated
        cost for a package depends on whether it is currently at a location
        or inside a vehicle, and involves the shortest path distance on the
        road network plus costs for pick-up and drop actions. This heuristic
        is non-admissible and designed for greedy best-first search.

    Assumptions:
        - The road network defined by 'road' predicates is static and bidirectional.
        - Vehicles can carry at most one package at a time (inferred from capacity predicate structure).
        - The shortest path distance between locations is a reasonable estimate for drive costs.
        - The heuristic does not consider vehicle availability conflicts or optimal vehicle assignment beyond using the vehicle's current location if the package is already inside it.
        - Vehicles are identified based on their appearance as the first argument in 'capacity' facts or the second argument in 'in' facts in the initial state.
        - Packages relevant to the heuristic are those listed in the goal state.

    Heuristic Initialization:
        - Parses the goal facts to store the target location for each package in `self.package_goals`.
        - Collects all unique locations mentioned in goals, initial state 'at' facts, and static 'road' facts into `self.locations`.
        - Identifies vehicles by collecting objects appearing as the first argument in 'capacity' facts or the second argument in 'in' facts from the initial state into `self.vehicles`.
        - Builds a graph representation of the road network from static 'road' predicates into `self.road_graph`. Ensures all collected locations are nodes in the graph.
        - Computes all-pairs shortest paths between locations using BFS on `self.road_graph`. Stores distances in `self.distances`.
        - Identifies the minimum capacity size 'c0' from 'capacity-predecessor' facts. This is noted but not directly used in the current heuristic cost calculation logic, based on the single-package-per-vehicle assumption simplifying capacity needs.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Initialize total heuristic cost to 0.
        3. Iterate through the current state facts to determine the current status ('at' or 'in') and location for each package listed in `self.package_goals`, and the current location for each vehicle listed in `self.vehicles`. Store this information in `package_current_status` and `vehicle_locations`.
        4. For each package and its goal location stored in `self.package_goals`:
            a. Retrieve the package's current status and location/vehicle from `package_current_status`. If the package is not found in the state (which indicates an invalid state), return `float('inf')`.
            b. If the package is currently `(at L_current)`:
                i. If `L_current` is the same as the goal location `L_goal`, the package is already delivered; add 0 to the total cost for this package.
                ii. If `L_current` is different from `L_goal`, the package needs to be picked up, transported, and dropped. Estimate the cost as the shortest distance from `L_current` to `L_goal` (for the drive action) plus 1 for the pick-up action and 1 for the drop action. Add `dist(L_current, L_goal) + 2` to the total cost. If `L_current` and `L_goal` are disconnected, `dist` will be `float('inf')`, causing the total to become `float('inf')`.
            c. If the package is currently `(in V)`:
                i. Find the current location of vehicle `V`, `L_v_current`, from `vehicle_locations`. If the vehicle is not found (which indicates an invalid state), return `float('inf')`.
                ii. If `L_v_current` is the same as the goal location `L_goal`, the package needs only to be dropped. Add 1 to the total cost.
                iii. If `L_v_current` is different from `L_goal`, the vehicle needs to drive to `L_goal` and drop the package. Estimate the cost as the shortest distance from `L_v_current` to `L_goal` (for the drive action) plus 1 for the drop action. Add `dist(L_v_current, L_goal) + 1` to the total cost. If `L_v_current` and `L_goal` are disconnected, `dist` will be `float('inf')`, causing the total to become `float('inf')`.
        5. Return the final `total_heuristic_cost`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.package_goals = {}
        self.vehicles = set()
        self.locations = set()
        self.road_graph = {}
        self.min_capacity_size = None
        capacity_sizes = set()
        predecessors = {}
        successors = {}

        # Process goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
                self.locations.add(location)

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "road":
                _, l1, l2 = parts
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = set()
                if l2 not in self.road_graph:
                    self.road_graph[l2] = set()
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assuming bidirectional
            elif predicate == "capacity-predecessor":
                _, s1, s2 = parts
                predecessors[s2] = s1
                successors[s1] = s2
                capacity_sizes.add(s1)
                capacity_sizes.add(s2)

        # Process initial state facts to find vehicles and more locations/sizes
        for fact in initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == "at":
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 # Objects in 'at' facts could be packages or vehicles.
                 # We identify vehicles more reliably from 'capacity' or 'in' facts.
             elif predicate == "capacity":
                 vehicle, size = parts[1], parts[2]
                 self.vehicles.add(vehicle) # Found a vehicle
                 capacity_sizes.add(size)
             elif predicate == "in":
                 package, vehicle = parts[1], parts[2]
                 # Assume second arg of 'in' is a vehicle
                 self.vehicles.add(vehicle) # Found a vehicle from 'in' fact
                 # Assume first arg of 'in' is a package. We only care about packages in goals.

        # Find the minimum capacity size 'c0'
        # This is the size that is not a successor of any other size.
        # If capacity_sizes is empty or the graph is circular, min_capacity_size remains None.
        is_successor = set(predecessors.keys()) # These are sizes that are successors
        for size in capacity_sizes:
            if size not in is_successor:
                 self.min_capacity_size = size
                 break # Found the smallest size

        # Ensure all collected locations are nodes in the graph
        for loc in self.locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = set()

        # Compute all-pairs shortest paths using BFS
        all_locations = list(self.road_graph.keys())
        self.distances = {}
        for start_node in all_locations:
            self.distances[start_node] = self._bfs(start_node, all_locations)

    def _bfs(self, start_node, all_nodes):
        """Helper to compute shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys before accessing neighbors
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves precomputed distance, handles disconnected nodes."""
        # Check if loc1 was a start node for BFS and if loc2 was reached from loc1
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This indicates a location exists in state/goals/initial state
             # but was not connected to anything via 'road' facts, or loc2
             # is unreachable from loc1 in the graph.
             return float('inf')

        return self.distances[loc1][loc2]


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        package_current_status = {} # {package: ('at', loc) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: loc}

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.package_goals: # It's a package we care about
                     package_current_status[obj] = ('at', loc)
                elif obj in self.vehicles: # It's a vehicle we identified
                     vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                if package in self.package_goals: # It's a package we care about
                     package_current_status[package] = ('in', vehicle)
            # Ignore capacity facts in __call__ as they are not used in this heuristic's cost calculation

        total_heuristic_cost = 0

        for package, goal_location in self.package_goals.items():
            # If package is not in the current state at all, it's likely an error state or unreachable.
            # Assume valid states where packages are either 'at' or 'in'.
            if package not in package_current_status:
                 # This package is missing from the state. Should not happen in valid states.
                 # Return infinity to prune this branch.
                 return float('inf')

            status, current_loc_or_veh = package_current_status[package]

            if status == 'at':
                current_location = current_loc_or_veh
                if current_location == goal_location:
                    # Package is already at goal
                    continue

                # Package is at a location, needs pickup, drive, drop
                # Estimated cost: drive distance + pickup + drop
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'):
                    return float('inf') # Goal unreachable for this package

                # Cost = drive_cost + pick_cost + drop_cost
                # drive_cost = dist
                # pick_cost = 1
                # drop_cost = 1
                total_heuristic_cost += dist + 2

            elif status == 'in':
                vehicle = current_loc_or_veh
                # Package is in a vehicle, needs drive, drop
                # Estimated cost: drive distance + drop

                if vehicle not in vehicle_locations:
                     # Vehicle carrying package is not at any location? Should not happen.
                     # Return infinity to prune this branch.
                     return float('inf')

                vehicle_current_location = vehicle_locations[vehicle]

                # Check if dropping at current vehicle location satisfies goal
                if vehicle_current_location == goal_location:
                     # Package is in vehicle, vehicle is at goal location.
                     # Needs 1 action: drop.
                     total_heuristic_cost += 1
                     continue # Done with this package

                # Vehicle needs to drive to goal location and drop
                dist = self.get_distance(vehicle_current_location, goal_location)
                if dist == float('inf'):
                    return float('inf') # Goal unreachable for this package

                # Cost = drive_cost + drop_cost
                # drive_cost = dist
                # drop_cost = 1
                total_heuristic_cost += dist + 1

        return total_heuristic_cost
