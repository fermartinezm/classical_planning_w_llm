import collections
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each package that is not yet at its goal
        location. For a package not at its goal, the estimated cost includes
        actions for picking it up, dropping it off, and the minimum driving
        cost for a vehicle to transport it from its current location to its
        goal location. The heuristic ignores vehicle capacity constraints and
        assumes a suitable vehicle is always available when needed.

    Assumptions:
        - The road network is static and provided in the static facts.
        - Each package that needs to be moved has a specific goal location
          defined in the task goals as an `(at ?package ?location)` fact.
        - The cost of each action (drive, pick-up, drop) is 1.
        - The heuristic does not need to be admissible.
        - The heuristic assumes any vehicle can transport any package (ignoring
          specific capacity requirements beyond the general need for capacity
          which is simplified away).
        - The heuristic assumes a vehicle is available at the required location
          for pick-up or drop-off (cost of moving a vehicle to the pick-up
          location is not explicitly included unless the package is already
          in a vehicle).
        - States are well-formed according to the domain (e.g., a package is
          either at a location or in a vehicle, not both or neither; a vehicle
          carrying a package is at a location). Inconsistent states are assigned
          an infinite heuristic value.
        - Objects appearing in `(at ...)` facts that are not packages with
          defined goals are vehicles.

    Heuristic Initialization:
        1. Parse the 'road' facts from the static information to build an
           adjacency list representation of the road network graph.
        2. Collect all unique location names mentioned in the initial state,
           goals, and static facts. This ensures all relevant locations are
           considered for distance calculations.
        3. Compute the shortest path distance between all pairs of these
           locations using Breadth-First Search (BFS) on the road network graph.
           Store these distances in a dictionary of dictionaries. Unreachable
           locations will have a distance of infinity (`math.inf`).
        4. Parse the goal facts from the task goals to create a mapping from
           each package that needs to be moved to its goal location.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Create dictionaries to store the current location of each package
           (`package_location`) and vehicle (`vehicle_location`), and which
           vehicle is carrying which package (`package_in_vehicle`).
        3. Iterate through the facts in the current state:
           - If a fact is `'(at ?obj ?loc)'`:
             - Parse `?obj` and `?loc`.
             - If `?obj` is a package (check if it's a key in the `goal_location` map),
               record `package_location[?obj] = ?loc`.
             - Otherwise (assume it's a vehicle based on domain structure and assumptions),
               record `vehicle_location[?obj] = ?loc`.
           - If a fact is `'(in ?pkg ?veh)'`:
             - Parse `?pkg` and `?veh`.
             - If `?pkg` is a package with a goal (check if it's a key in `goal_location`),
               record `package_in_vehicle[?pkg] = ?veh`.
        4. Iterate through each package `p` and its goal location `l_goal`
           stored during initialization (only packages that need to be moved).
        5. For package `p`:
           - Check if package `p` is currently at its goal location `l_goal`. This happens
             if `p` is in `package_location` and `package_location[p] == l_goal`.
             If so, the cost for this package is 0, and we continue to the next package.
           - If package `p` is not at its goal location, determine its current status:
             - If `p` is in `package_location` (meaning it's at a location `l_current`
               and `l_current != l_goal`):
               - The estimated cost for this package is 2 (for pick-up and drop actions)
                 plus the shortest path distance from `l_current` to `l_goal`
                 (representing the minimum driving cost). Add `2 + dist[l_current][l_goal]` to `h`.
                 If `l_goal` is unreachable from `l_current`, `dist` will be `math.inf`,
                 and `h` will become `math.inf`.
             - If `p` is in `package_in_vehicle` (meaning it's inside a vehicle `v`):
               - Find the vehicle's current location `l_v` from `vehicle_location`.
               - If `l_v` is not found (inconsistent state), return `math.inf` immediately.
               - Otherwise, the estimated cost for this package is 1 (for the drop action)
                 plus the shortest path distance from `l_v` to `l_goal` (representing
                 the minimum driving cost). Add `1 + dist[l_v][l_goal]` to `h`.
                 If `l_goal` is unreachable from `l_v`, `dist` will be `math.inf`,
                 and `h` will become `math.inf`.
             - If package `p` is neither in `package_location` nor `package_in_vehicle`
               (and it has a goal), this indicates an inconsistent state. Return `math.inf`
               immediately.
           - If at any point `h` becomes `math.inf` during the loop, we can stop
             early and return `math.inf`.
        6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.task = task
        self.goal_location = self._parse_goal_locations(task.goals)
        self.all_locations = self._get_all_locations(task.initial_state, task.goals, task.static)
        self.road_graph = self._build_road_graph(task.static)
        self.location_distances = self._compute_all_pairs_shortest_paths(self.road_graph, self.all_locations)

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding brackets and split by space
        # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
        # Example: '(capacity v1 c1)' -> ['capacity', 'v1', 'c1']
        return fact_string[1:-1].split()

    def _parse_goal_locations(self, goals):
        """
        Parses the goal facts to find the target location for each package.

        Args:
            goals: A frozenset of goal fact strings.

        Returns:
            A dictionary mapping package names (str) to goal location names (str).
            Only includes packages mentioned in `(at ?package ?location)` goals.
        """
        goal_locs = {}
        for goal in goals:
            parsed_goal = self._parse_fact(goal)
            if parsed_goal[0] == 'at' and len(parsed_goal) == 3:
                # Goal is (at ?package ?location)
                package = parsed_goal[1]
                location = parsed_goal[2]
                goal_locs[package] = location
        return goal_locs

    def _get_all_locations(self, initial_state, goals, static_facts):
        """Collects all unique location names from initial state, goals, and static facts."""
        locations = set()
        # From initial state 'at' facts
        for fact in initial_state:
             parsed = self._parse_fact(fact)
             if parsed[0] == 'at' and len(parsed) == 3:
                 locations.add(parsed[2])
        # From goal 'at' facts
        for fact in goals:
             parsed = self._parse_fact(fact)
             if parsed[0] == 'at' and len(parsed) == 3:
                 locations.add(parsed[2])
        # From road facts
        for fact in static_facts:
            parsed = self._parse_fact(fact)
            if parsed[0] == 'road' and len(parsed) == 3:
                locations.add(parsed[1])
                locations.add(parsed[2])
        return list(locations) # Return as list for consistent iteration order

    def _build_road_graph(self, static_facts):
        """
        Builds an adjacency list representation of the road network.

        Args:
            static_facts: A frozenset of static fact strings.

        Returns:
            A dictionary where keys are location names (str) and values are
            sets of connected location names (str).
        """
        graph = collections.defaultdict(set)
        for fact in static_facts:
            parsed_fact = self._parse_fact(fact)
            if parsed_fact[0] == 'road' and len(parsed_fact) == 3:
                loc1 = parsed_fact[1]
                loc2 = parsed_fact[2]
                graph[loc1].add(loc2)
                graph[loc2].add(loc1) # Roads are typically bidirectional
        return graph

    def _compute_all_pairs_shortest_paths(self, graph, all_locations):
        """
        Computes shortest path distances between all pairs of locations
        using BFS.

        Args:
            graph: The road network graph as an adjacency list.
            all_locations: A list or set of all location names in the problem.

        Returns:
            A dictionary of dictionaries, dist[l1][l2] gives the shortest
            distance from l1 to l2. Returns math.inf for unreachable pairs.
        """
        distances = {}

        for start_loc in all_locations:
            distances[start_loc] = {}
            # Initialize all distances to infinity
            for loc in all_locations:
                 distances[start_loc][loc] = math.inf

            # BFS from start_loc
            queue = collections.deque([(start_loc, 0)])
            distances[start_loc][start_loc] = 0 # Distance to self is 0
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                # Ensure current_loc is in the graph keys before accessing neighbors
                # A location might exist in initial/goal state but not in any road fact
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                # If current_loc is not in graph, it has no neighbors, BFS from here stops.

        return distances


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of fact strings representing the current state.

        Returns:
            An integer or float representing the estimated cost to reach the goal.
            Returns math.inf if the state is detected as inconsistent or leads
            to an apparently unreachable goal location for a package.
        """
        h = 0
        package_location = {}
        vehicle_location = {}
        package_in_vehicle = {}

        # Parse current state to find locations/status of packages and vehicles
        for fact in state:
            parsed_fact = self._parse_fact(fact)
            predicate = parsed_fact[0]

            if predicate == 'at' and len(parsed_fact) == 3:
                obj = parsed_fact[1]
                loc = parsed_fact[2]
                # Check if the object is a package we care about (i.e., has a goal)
                if obj in self.goal_location:
                    package_location[obj] = loc
                else:
                    # Assume any other 'at' object is a vehicle for this domain
                    vehicle_location[obj] = loc

            elif predicate == 'in' and len(parsed_fact) == 3:
                pkg = parsed_fact[1]
                veh = parsed_fact[2]
                # Only track packages that have a goal
                if pkg in self.goal_location:
                    package_in_vehicle[pkg] = veh

        # Compute heuristic for each package with a goal
        for package, goal_loc in self.goal_location.items():
            # Check if package is at its goal location
            if package in package_location and package_location[package] == goal_loc:
                # Package is at goal, cost is 0 for this package
                continue

            # Package is not at goal. It's either at some other location or in a vehicle.
            # Note: A package cannot be both at a location and in a vehicle simultaneously
            # in a valid state according to the domain effects.

            if package in package_location:
                # Package is at a location != goal_loc
                current_loc = package_location[package]
                # Cost = pick-up (1) + drop (1) + drive (shortest path from current_loc to goal_loc)
                drive_cost = self.location_distances.get(current_loc, {}).get(goal_loc, math.inf)
                h += 2 + drive_cost

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find vehicle's location
                vehicle_loc = vehicle_location.get(vehicle)
                if vehicle_loc is None:
                     # This state is inconsistent (package in vehicle, but vehicle not at location)
                     # or vehicle is not tracked (e.g. vehicle had no initial 'at' fact).
                     # This shouldn't happen in valid states derived from the initial state.
                     # Treat as unsolvable from here.
                     return math.inf # Return infinity immediately

                # Cost = drop (1) + drive (shortest path from vehicle_loc to goal_loc)
                drive_cost = self.location_distances.get(vehicle_loc, {}).get(goal_loc, math.inf)
                h += 1 + drive_cost

            else:
                 # Package has a goal but is neither at a location nor in a vehicle.
                 # This indicates an inconsistent state.
                 return math.inf # Return infinity immediately

            # If at any point h becomes infinity, we can stop and return infinity
            if h == math.inf:
                 return math.inf

        return h
