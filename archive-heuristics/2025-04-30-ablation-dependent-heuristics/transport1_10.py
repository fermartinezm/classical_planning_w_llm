import collections

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the minimum estimated costs for each package that is not yet
        at its goal location. For a package not at its goal, the cost includes
        the minimum number of pick-up/drop actions required and the shortest
        driving distance for a vehicle to transport it from its current
        location (or vehicle's location) to its goal location.

    Assumptions:
        - The road network is connected, or at least all locations relevant
          to package movements are within the same connected component.
        - Roads are bidirectional.
        - Vehicle capacity is not explicitly considered in the cost calculation,
          potentially leading to overestimation but simplifying computation.
        - The heuristic assumes a greedy strategy where each package is moved
          independently towards its goal, ignoring potential synergies from
          batching packages in vehicles.
        - The heuristic relies on parsing fact strings in the format '(predicate arg1 arg2 ...)'.
        - The state representation is consistent with the PDDL domain, meaning
          every locatable object (package or vehicle) is either at a location
          or (for packages) inside a vehicle.

    Heuristic Initialization:
        The heuristic constructor precomputes the shortest path distances
        between all pairs of locations based on the 'road' facts in the
        static information. This is done using a Breadth-First Search (BFS)
        algorithm starting from each location. It also pre-parses the goal
        facts to quickly look up the target location for each package.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state by verifying if all
           goal facts are present in the state. If yes, return 0.
        2. Initialize the total heuristic value `h` to 0.
        3. Parse the current state facts to determine:
           - The current location of each package (`package_locations`).
           - Which package is currently inside which vehicle (`package_in_vehicle`).
           - The current location of each vehicle (`vehicle_locations`).
           (Goal locations for packages were pre-parsed during initialization).
           Identify all objects that are packages based on goal facts or
           current 'in' facts. Objects in 'at' facts that are not packages
           are assumed to be vehicles.
        4. Iterate through each package `p` that has a goal location `l_goal`
           defined in `self.package_goals`.
        5. For the current package `p` and its goal `l_goal`:
           - If `p` is currently at a location `l_current`
             (i.e., `(at p l_current)` is in the state) and `l_current` is not `l_goal`:
             This package needs to be picked up and dropped. Minimum 2 actions.
             A vehicle must travel from `l_current` to `l_goal`. The minimum
             driving cost is the shortest path distance `dist(l_current, l_goal)`.
             Add `2 + dist(l_current, l_goal)` to `h`. If the goal is unreachable
             from the current location, return `float('inf')`.
           - If `p` is currently inside a vehicle `v`
             (i.e., `(in p v)` is in the state):
             Find the current location `l_v` of vehicle `v`
             (i.e., `(at v l_v)` is in the state).
             This package needs to be dropped. Minimum 1 action.
             Vehicle `v` must travel from its current location `l_v` to the
             goal location `l_goal`. The minimum driving cost is `dist(l_v, l_goal)`.
             Add `1 + dist(l_v, l_goal)` to `h`. If the goal is unreachable
             from the vehicle's location, return `float('inf')`.
           - If `p` is at `l_current` and `l_current` is `l_goal`:
             The package is already at its goal location and not in a vehicle.
             This state contributes 0 to the heuristic for this package.
           - If the package is a goal package but is neither at a location
             nor in a vehicle in the current state, or if a vehicle's location
             is unknown, this indicates an unexpected state structure or
             unsolvable path. Return `float('inf')`.
        6. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing shortest path distances
        and parsing goal facts.

        Args:
            task: The planning task object.
        """
        self.task = task

        # Pre-parse goal locations for packages
        self.package_goals = {}
        for goal_fact in task.goals:
            parsed_goal = self._parse_fact(goal_fact)
            if parsed_goal[0] == 'at':
                package, location = parsed_goal[1], parsed_goal[2]
                self.package_goals[package] = location

        # Precompute road network graph and shortest path distances
        self.location_graph, self.locations = self._build_road_graph(task.static)
        self.location_distances = self._compute_all_pairs_shortest_paths(
            self.location_graph, self.locations
        )

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal, or float('inf')
            if the goal appears unreachable from this state.
        """
        # Check if the state is the goal state
        if self.task.goals <= state:
             return 0

        h = 0

        # Parse current state information
        package_locations = {}
        package_in_vehicle = {}
        vehicle_locations = {}
        # vehicle_capacities = {} # Not used in this heuristic calculation

        # Collect all objects that are explicitly packages based on goal or 'in' predicate
        all_packages = set(self.package_goals.keys())
        # Add packages currently in vehicles from the current state
        for fact in state:
             parsed_fact = self._parse_fact(fact)
             if parsed_fact[0] == 'in':
                 all_packages.add(parsed_fact[1]) # Add package name

        for fact in state:
            parsed_fact = self._parse_fact(fact)
            predicate = parsed_fact[0]
            if predicate == 'at':
                obj, loc = parsed_fact[1], parsed_fact[2]
                if obj in all_packages:
                     package_locations[obj] = loc
                else: # Assume it's a vehicle if not a known package
                     vehicle_locations[obj] = loc
            elif predicate == 'in':
                package, vehicle = parsed_fact[1], parsed_fact[2]
                package_in_vehicle[package] = vehicle
            # elif predicate == 'capacity':
            #     vehicle, size = parsed_fact[1], parsed_fact[2]
            #     vehicle_capacities[vehicle] = size

        # Calculate heuristic based on packages not at goal
        for package, goal_location in self.package_goals.items():
            # Check if the package is currently at its goal location (and not in a vehicle)
            # A package at its goal location needs to be *at* that location, not *in* a vehicle there.
            if package in package_locations and package_locations[package] == goal_location:
                # Package is at goal location and not in a vehicle. Cost is 0 for this package.
                continue

            # Package is not at its goal location in the desired final state.
            # It's either at a wrong location or inside a vehicle.

            if package in package_locations:
                # Package is at a location, but it's not the goal location.
                current_location = package_locations[package]
                # This package needs to be picked up and dropped. Minimum 2 actions.
                # A vehicle needs to drive from current_location to goal_location.
                drive_cost = self.location_distances.get(current_location, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                     # Goal location is unreachable from the package's current location
                     return float('inf')
                h += 2 + drive_cost

            elif package in package_in_vehicle:
                # Package is inside a vehicle.
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_locations:
                    vehicle_location = vehicle_locations[vehicle]
                    # This package needs to be dropped. Minimum 1 action.
                    # The vehicle needs to drive from its current location to the goal location.
                    drive_cost = self.location_distances.get(vehicle_location, {}).get(goal_location, float('inf'))
                    if drive_cost == float('inf'):
                        # Goal location is unreachable from the vehicle's current location
                        return float('inf')
                    h += 1 + drive_cost
                else:
                    # Package is in a vehicle, but the vehicle's location is unknown.
                    # This indicates an unexpected state structure. Treat as unreachable.
                    return float('inf')
            else:
                 # Package is a goal package but is neither at a location nor in a vehicle.
                 # This indicates an unexpected state structure. Treat as unreachable.
                 return float('inf')


        return h

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove parentheses and split by spaces
        return fact_string[1:-1].split()

    def _build_road_graph(self, static_facts):
        """Builds an adjacency list representation of the road network."""
        graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            parsed = self._parse_fact(fact)
            if parsed[0] == 'road':
                l1, l2 = parsed[1], parsed[2]
                graph[l1].append(l2)
                graph[l2].append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)
        return graph, list(locations)

    def _compute_all_pairs_shortest_paths(self, graph, locations):
        """Computes shortest path distances between all pairs of locations using BFS."""
        dist = {}
        for start_node in locations:
            dist[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, d = queue.popleft()
                dist[start_node][current_node] = d
                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        # Ensure all pairs have a distance (inf if unreachable)
        for l1 in locations:
            if l1 not in dist: # Handle isolated locations not in graph keys initially
                 dist[l1] = {l2: float('inf') for l2 in locations}
            for l2 in locations:
                if l2 not in dist[l1]:
                    dist[l1][l2] = float('inf')

        return dist
