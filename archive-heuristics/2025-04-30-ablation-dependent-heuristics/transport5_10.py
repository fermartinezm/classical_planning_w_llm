import collections

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        by summing up the estimated costs for each package that is not yet at
        its goal location. The cost for a package is estimated based on its
        current location (either at a location or inside a vehicle) and its
        goal location, using precomputed shortest path distances between locations.

    Assumptions:
        - The road network is static and provides paths between any two locations
          that need to be connected for goal achievement.
        - Vehicle capacity is simplified: we don't explicitly model capacity
          constraints or the need to free up capacity. We assume a suitable
          vehicle is eventually available for pickup/dropoff.
        - The heuristic sums costs per package independently, ignoring potential
          synergies (e.g., one vehicle transporting multiple packages) or conflicts.
          This makes it non-admissible but potentially effective for greedy search.
        - All packages relevant to the goal have a defined goal location in the task's goals.
        - States are valid according to the domain rules (e.g., a package is
          either at a location or in a vehicle, not both or neither, if it's a goal package).

    Heuristic Initialization:
        The heuristic is initialized with the planning task. During initialization,
        it performs the following steps:
        1. Parses static facts to build the road network graph.
        2. Parses goal facts to store the target location for each package that
           needs to be at a specific location in the goal state.
        3. Computes all-pairs shortest paths between locations using BFS on the
           road graph. These distances represent the minimum number of 'drive'
           actions required to travel between any two locations.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (a frozenset of facts):
        1. Initialize the total heuristic value `h` to 0.
        2. Create dictionaries to quickly look up the current location of any
           object (`current_locations`) and which package is inside which vehicle
           (`package_in_vehicle`) by iterating through the state facts.
        3. Iterate through each package `p` that has a goal location defined
           in the task's goals (`self.package_goals`).
        4. For the current package `p`, retrieve its goal location `loc_p_goal`
           from the `self.package_goals` dictionary.
        5. Determine the current effective location of package `p`.
           - If `p` is found as a key in `current_locations` (meaning `(at p current_loc)` is in the state), its current location is `current_locations[p]`.
           - If `p` is found as a key in `package_in_vehicle` (meaning `(in p v)` is in the state), its current effective location is the location of vehicle `v`, found by looking up `v` in `current_locations`.
           - If the package's location cannot be determined, it's treated as unreachable (infinity).
        6. If the package's current effective location is the same as its goal location (`current_loc == loc_p_goal`), the package is already at its goal, and contributes 0 to `h`.
        7. If the package `p` is not at its goal location:
           a. Get the shortest path distance `dist` from the package's current effective location to its goal location using the precomputed `self.distances`. If no path exists, the distance is infinity.
           b. If `dist` is infinity, the state is likely unsolvable, return infinity.
           c. If `p` was found to be `(at p current_loc)`:
              - It needs a pick-up, the drive(s), and a drop.
              - Add `dist + 2` to `h`.
           d. If `p` was found to be `(in p v)`:
              - It needs the drive(s) (by the vehicle) and a drop.
              - Add `dist + 1` to `h`.
        8. After iterating through all goal packages, the total value `h` is
           the heuristic estimate for the state. Return `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information
        and shortest paths.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.package_goals = self._parse_package_goals(task.goals)
        self.road_graph = self._build_road_graph(task.static)
        self.distances = self._compute_all_pairs_shortest_paths(self.road_graph)
        # Size hierarchy is not used in this simplified heuristic
        # self.size_hierarchy = self._parse_size_hierarchy(task.static)

    def _parse_fact(self, fact_str):
        """Parses a fact string into a list of strings."""
        # Remove surrounding brackets and split by spaces
        return fact_str[1:-1].split()

    def _parse_package_goals(self, goals):
        """Extracts package goal locations from the goal facts."""
        package_goals = {}
        for goal_fact_str in goals:
            parts = self._parse_fact(goal_fact_str)
            # Goal facts can be complex, but in transport domain examples,
            # they are typically (at package location).
            # We only extract 'at' goals for packages.
            if parts[0] == 'at' and len(parts) == 3:
                # Assuming the first argument after 'at' is the object (package/vehicle)
                # and the second is the location. We only care about objects
                # that are explicitly listed in the goal as needing to be 'at' a location.
                obj, location = parts[1], parts[2]
                package_goals[obj] = location
        return package_goals

    def _build_road_graph(self, static_facts):
        """Builds the road graph from static facts."""
        road_graph = collections.defaultdict(set)
        locations = set()
        for fact_str in static_facts:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                road_graph[loc1].add(loc2)
                road_graph[loc2].add(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all locations mentioned in roads are keys in the graph, even if they have no outgoing roads
        for loc in locations:
             if loc not in road_graph:
                 road_graph[loc] = set()

        return road_graph

    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest paths between all pairs of locations using BFS."""
        distances = {}
        locations = list(graph.keys())

        for start_node in locations:
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            distances[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # If there are locations not connected to anything (not in graph keys),
        # distances involving them will not be in the dict, resulting in get() returning inf.
        # This is correct.

        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            An integer, the estimated number of actions to reach a goal state.
            Returns float('inf') if the state is estimated to be unsolvable
            (e.g., goal location unreachable).
        """
        h = 0

        # Determine current locations of all locatables (packages and vehicles)
        # and which packages are inside which vehicles.
        current_locations = {} # obj -> loc
        package_in_vehicle = {} # package -> vehicle

        for fact_str in state: # Iterate directly on frozenset
            parts = self._parse_fact(fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in' and len(parts) == 3:
                p, v = parts[1], parts[2]
                package_in_vehicle[p] = v

        # Compute heuristic contribution for each package not at its goal
        for package, goal_loc in self.package_goals.items():
            # Find the package's current effective location
            current_loc = None
            is_in_vehicle = False

            if package in current_locations:
                # Package is at a location
                current_loc = current_locations[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                # Package is in a vehicle, its effective location is the vehicle's location
                vehicle = package_in_vehicle[package]
                current_loc = current_locations.get(vehicle) # Get vehicle's location
                is_in_vehicle = True

            # If current_loc is None, the package's location/status is unknown/invalid for a goal package
            if current_loc is None:
                 # This should not happen in valid states for goal packages.
                 # Treat as unreachable.
                 # print(f"Warning: Location of goal package {package} not found in state.")
                 return float('inf')

            # If package is already at its goal location, it contributes 0
            if current_loc == goal_loc:
                 continue

            # Package is not at goal, calculate cost
            dist = self.distances.get((current_loc, goal_loc), float('inf'))

            if dist == float('inf'):
                # Goal is unreachable from the package's current location (or vehicle's location)
                return float('inf') # Unsolvable state branch

            if is_in_vehicle:
                # Package is in a vehicle, needs drive(s) and a drop
                h += dist + 1
            else: # package is at a location
                # Package is at a location, needs pick-up, drive(s), and a drop
                h += dist + 2

        return h
