from collections import deque

# Define a large number for unreachable locations
INF = float('inf')

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum costs
        to deliver each package that is not yet at its goal location.
        The cost for a package is estimated based on its current status
        (at a location or in a vehicle) and the shortest path distance
        to its goal location. Capacity constraints and vehicle availability
        are ignored.

    Assumptions:
        - The state representation is a frozenset of strings like '(predicate arg1 arg2)'.
        - The static information is a frozenset of strings containing 'road' and 'capacity-predecessor' facts.
        - The goal is a frozenset of strings, primarily '(at package location)' facts.
        - Vehicles are always located at some location in valid states.
        - Package goal facts are always of the form '(at package location)'.
        - Objects appearing as the first argument in goal facts like '(at obj loc)' are packages if they are listed in the goal facts, otherwise they are assumed to be vehicles.

    Heuristic Initialization:
        1. Parses static facts to build the road network graph (adjacency list).
        2. Identifies all unique locations from road facts.
        3. Computes all-pairs shortest path distances between locations using BFS.
        4. Parses goal facts to store the target location for each package.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Parse the current state to determine the current location of each package
           (if at a location) and which package is inside which vehicle. Also,
           determine the current location of each vehicle. Distinguish packages
           from vehicles based on whether they appear as the first argument in
           a goal fact of the form `(at obj loc)`.
        3. Iterate through each package that has a defined goal location (identified during initialization).
        4. For the current package `p` and its goal location `l_goal`:
           a. Check if the goal condition `(at p l_goal)` is already satisfied in the state. If yes, this package contributes 0 to the heuristic; move to the next package.
           b. If the goal is not satisfied, determine the package's current status:
              i. If the package `p` is currently at a location `l_p` (i.e., `(at p l_p)` is in the state, and `l_p != l_goal`):
                 - The package needs to be picked up (1 action), transported from `l_p` to `l_goal` (estimated by the shortest path distance `dist(l_p, l_goal)` drive actions), and dropped off (1 action).
                 - Add `2 + dist(l_p, l_goal)` to `h`.
              ii. If the package `p` is currently inside a vehicle `v` (i.e., `(in p v)` is in the state):
                  - Find the current location `l_v` of vehicle `v` (i.e., `(at v l_v)` is in the state).
                  - The package needs to be transported from `l_v` to `l_goal` (estimated by `dist(l_v, l_goal)` drive actions) and dropped off (1 action).
                  - Add `1 + dist(l_v, l_goal)` to `h`.
              iii. If the package's status is neither 'at' a location nor 'in' a vehicle (should not happen in valid states), handle as an error by returning infinity.
        5. Return the total heuristic value `h`.
        6. If any required shortest path distance is infinite (locations are disconnected), the total heuristic will be infinite, indicating an unreachable goal.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing shortest path distances
        and parsing goal locations.

        @param task: The planning task object (instance of Task).
        """
        self.package_goals = {}
        self.road_graph = {}
        self.locations = set()
        self.dist = {}

        # Parse static facts to build the road graph and collect locations
        for fact_string in task.static:
            pred, *args = self._parse_fact(fact_string)
            if pred == 'road':
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)

        # Ensure all locations mentioned in road facts are in the graph keys
        # This handles locations that only have incoming roads or no roads
        for loc in self.locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self.dist[start_loc] = {}
            q = deque([start_loc])
            self.dist[start_loc][start_loc] = 0
            visited = {start_loc}

            while q:
                curr_loc = q.popleft()
                curr_dist = self.dist[start_loc][curr_loc]

                # Get neighbors, handling locations with no outgoing roads
                neighbors = self.road_graph.get(curr_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist[start_loc][neighbor] = curr_dist + 1
                        q.append(neighbor)

            # Fill in unreachable locations with INF
            for loc in self.locations:
                if loc not in self.dist[start_loc]:
                    self.dist[start_loc][loc] = INF


        # Parse goal facts to get package goal locations
        # Assuming goals are primarily (at package location) for packages
        for goal_fact_string in task.goals:
            pred, *args = self._parse_fact(goal_fact_string)
            if pred == 'at' and len(args) == 2:
                 # Assume the first argument of an 'at' goal fact is a package
                 package_name, goal_loc_name = args
                 self.package_goals[package_name] = goal_loc_name
            # Ignore other types of goal facts if any exist


    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding brackets and split by space
        # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
        parts = fact_string.strip("()'").split()
        if not parts:
            return None, [] # Handle empty string case
        return parts[0], parts[1:]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated cost (integer or float('inf')).
        """
        h = 0
        package_current_location = {}
        package_current_vehicle = {}
        vehicle_current_location = {}

        # Parse current state to find locations of packages and vehicles,
        # and which packages are in which vehicles.
        # We distinguish packages from vehicles based on whether they are
        # expected to be at a goal location (i.e., are keys in self.package_goals).
        # This is a heuristic assumption based on typical transport problems.
        known_packages_from_goals = set(self.package_goals.keys())

        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at' and len(args) == 2:
                obj_name, loc_name = args
                if obj_name in known_packages_from_goals:
                     package_current_location[obj_name] = loc_name
                else:
                     # Assume it's a vehicle if not a package listed in goals
                     vehicle_current_location[obj_name] = loc_name

            elif pred == 'in' and len(args) == 2:
                package_name, vehicle_name = args
                package_current_vehicle[package_name] = vehicle_name


        # Calculate heuristic cost for each package that has a goal
        for package, goal_loc in self.package_goals.items():
            # Check if the package is already at its goal location
            goal_fact_string = f"'(at {package} {goal_loc})'"
            if goal_fact_string in state:
                # Package is delivered, contributes 0 to heuristic
                continue

            # Package is not delivered, calculate its contribution
            cost_for_package = 0

            if package in package_current_location:
                # Package is at a location, not the goal
                current_loc = package_current_location[package]
                # Needs pickup (1) + drive (dist) + drop (1)
                # Ensure current_loc and goal_loc are in our distance map
                if current_loc in self.dist and goal_loc in self.dist.get(current_loc, {}):
                    drive_cost = self.dist[current_loc][goal_loc]
                    if drive_cost == INF:
                         # Goal is unreachable from package's current location
                         return INF # Problem likely unsolvable from this state
                    cost_for_package = 2 + drive_cost
                else:
                    # Should not happen if locations are parsed correctly
                    # Indicates an issue with location names or graph
                    return INF # Unknown location or goal location in distance map

            elif package in package_current_vehicle:
                # Package is in a vehicle
                vehicle = package_current_vehicle[package]
                # Find vehicle's location
                if vehicle in vehicle_current_location:
                    current_loc = vehicle_current_location[vehicle]
                    # Needs drive (dist) + drop (1)
                    # Ensure current_loc and goal_loc are in our distance map
                    if current_loc in self.dist and goal_loc in self.dist.get(current_loc, {}):
                        drive_cost = self.dist[current_loc][goal_loc]
                        if drive_cost == INF:
                             # Goal is unreachable from vehicle's current location
                             return INF # Problem likely unsolvable from this state
                        cost_for_package = 1 + drive_cost
                    else:
                         # Should not happen if locations are parsed correctly
                         return INF # Unknown vehicle location or goal location in distance map
                else:
                    # Vehicle location unknown - indicates invalid state representation
                    # Or vehicle is not 'at' any location (should not happen in valid states)
                    return INF # Vehicle location unknown

            else:
                # Package status is unknown (neither at location nor in vehicle)
                # This indicates an invalid state representation
                return INF # Package status unknown

            h += cost_for_package

        return h
