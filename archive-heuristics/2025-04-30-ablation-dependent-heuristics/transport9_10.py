import collections
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum required
        actions for each package that is not yet at its goal location.
        For a package not at its goal, the estimated cost includes:
        - If the package is at a location: 1 (pick-up) + shortest_distance(current_loc, goal_loc) (drive) + 1 (drop).
        - If the package is inside a vehicle: shortest_distance(vehicle_loc, goal_loc) (drive) + 1 (drop).
        Shortest distances are precomputed using BFS on the road network.

    Assumptions:
        - The road network defined by (road l1 l2) facts is static.
        - Package goal conditions are always of the form (at package location).
        - In any valid state, every package is either (at package package_location) or (in package vehicle).
        - The heuristic ignores vehicle capacity constraints and the possibility of transporting multiple packages together.
        - The road network is assumed to be traversable in both directions if a (road l1 l2) fact exists.
        - Objects appearing in (at obj loc) facts that are not packages with goals are assumed to be vehicles.

    Heuristic Initialization:
        1. Parse static facts to build the road network graph (adjacency list).
        2. Identify all unique locations from road facts and goal facts.
        3. For each location, run BFS to compute shortest distances to all other locations. Store these distances.
        4. Parse goal facts to identify packages and their target locations. Store this mapping.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic value `h = 0`.
        2. Iterate through the current state to identify the current location of each package (either at a location or in a vehicle) and the location of each vehicle.
           - Facts like `(at p l)` tell us a package `p` is at location `l`.
           - Facts like `(in p v)` tell us a package `p` is in vehicle `v`.
           - Facts like `(at v l)` tell us a vehicle `v` is at location `l`. (Assuming objects at locations that are not goal packages are vehicles).
        3. For each package `p` that has a goal location `l_goal` (identified during initialization):
        4. Check if the fact `(at p l_goal)` is present in the current state.
        5. If `(at p l_goal)` is in the state, the package is already at its goal. Add 0 to `h`. Continue to the next package.
        6. If `(at p l_goal)` is not in the state, find the current status of package `p` using the information gathered in step 2:
            a. If package `p` is currently at location `l_current`:
                - The estimated cost for this package is 2 (pick-up + drop) + the precomputed shortest distance from `l_current` to `l_goal`. Add this cost to `h`. If the distance is infinity, return `math.inf`.
            b. If package `p` is currently inside vehicle `v`:
                - Find the current location `l_v` of vehicle `v` (from information gathered in step 2).
                - The estimated cost for this package is 1 (drop) + the precomputed shortest distance from `l_v` to `l_goal`. Add this cost to `h`. If the distance is infinity, return `math.inf`.
            c. If the package is neither at a location nor in a vehicle (should not happen in valid states), return `math.inf`.
        7. After iterating through all packages with goal conditions, the total value of `h` is the heuristic estimate for the current state.
        8. If `h` is 0, the state is a goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing road network distances
        and identifying package goals.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.package_goals = {}
        self.road_graph = collections.defaultdict(set)
        self.locations = set()
        self.distances = {} # distances[start_loc][end_loc] = distance

        # 1. Parse static facts to build the road network graph
        # 2. Identify all unique locations from road facts
        for fact_str in task.static:
            if fact_str.startswith('(road '):
                parts = fact_str.strip("()").split()
                l1 = parts[1]
                l2 = parts[2]
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assume roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)

        # 4. Parse goal facts to identify packages and their target locations
        # Also identify locations mentioned in goals
        for goal_fact_str in task.goals:
            # Goal facts are typically (at package location)
            if goal_fact_str.startswith('(at '):
                parts = goal_fact_str.strip("()").split()
                package = parts[1]
                goal_loc = parts[2]
                self.package_goals[package] = goal_loc
                self.locations.add(goal_loc) # Add goal location to locations set

        # 3. For each location, run BFS to compute shortest distances
        for start_loc in list(self.locations): # Iterate over a copy as BFS might add locations if not careful
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all other nodes
        in the road graph.

        Args:
            start_node: The starting location.

        Returns:
            A dictionary mapping locations to their shortest distance from start_node.
        """
        distances = {loc: math.inf for loc in self.locations}

        # If the start_node is not in the graph (e.g., an isolated location
        # mentioned only in init/goal but not connected by roads),
        # it can only reach itself.
        if start_node not in self.locations:
             # This case should ideally not happen if locations from goals/init
             # are added to self.locations before BFS, but as a safeguard:
             # If a location is somehow missing, it's unreachable from anywhere else.
             # If it's the start_node itself, its distance to itself is 0.
             if start_node in distances: # Check if it was added from goals/init
                 distances[start_node] = 0
             return distances # All others remain inf

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is one of the locations we are tracking distances for
                    if neighbor in distances and distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal.
        """
        h = 0
        current_package_locations = {} # package -> location
        current_package_in_vehicle = {} # package -> vehicle
        current_vehicle_locations = {} # vehicle -> location

        # Collect locations and containment from the state
        for fact_str in state:
            parts = fact_str.strip("()").split()
            predicate = parts[0]
            if predicate == 'at':
                obj = parts[1]
                loc = parts[2]
                # We need to know object types. Without explicit type info,
                # we assume objects with goals are packages, others at locations are vehicles.
                if obj in self.package_goals:
                    current_package_locations[obj] = loc
                else: # Assume it's a vehicle or other non-goal-package locatable
                     current_vehicle_locations[obj] = loc
            elif predicate == 'in':
                package = parts[1]
                vehicle = parts[2]
                # Only track packages that have goals
                if package in self.package_goals:
                    current_package_in_vehicle[package] = vehicle

        # Calculate heuristic contribution for each package with a goal
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at goal
            if package in current_package_locations and current_package_locations[package] == goal_loc:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal, find its current status
            if package in current_package_locations:
                # Package is at a location
                current_loc = current_package_locations[package]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                # Check if locations are in our precomputed distances table
                if current_loc not in self.distances or goal_loc not in self.distances[current_loc]:
                     # This implies an issue with the graph building or an unreachable location
                     return math.inf # Cannot reach goal

                distance = self.distances[current_loc][goal_loc]
                if distance == math.inf:
                    return math.inf # Goal is unreachable from this location
                h += 2 + distance
            elif package in current_package_in_vehicle:
                # Package is inside a vehicle
                vehicle_carrying_package = current_package_in_vehicle[package]
                # Find vehicle's location
                if vehicle_carrying_package in current_vehicle_locations:
                    vehicle_loc = current_vehicle_locations[vehicle_carrying_package]
                    # Cost: drive (distance) + drop (1)
                    # Check if locations are in our precomputed distances table
                    if vehicle_loc not in self.distances or goal_loc not in self.distances[vehicle_loc]:
                         return math.inf # Cannot reach goal

                    distance = self.distances[vehicle_loc][goal_loc]
                    if distance == math.inf:
                        return math.inf # Goal is unreachable via this vehicle's location
                    h += 1 + distance
                else:
                    # Vehicle carrying package is not at any location in the state.
                    # This indicates an inconsistent state or a vehicle type not handled.
                    # Assume unreachable.
                    return math.inf
            else:
                 # Package is neither at a location nor in a vehicle. Should not happen in valid states.
                 # Indicates an inconsistent state representation.
                 # Return infinity as goal is likely unreachable from such a state.
                 return math.inf

        return h
