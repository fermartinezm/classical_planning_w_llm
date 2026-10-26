import collections
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing up
        the minimum estimated actions required for each package that is not
        yet at its goal location. For a package not at its goal, the estimated
        cost includes actions for pickup (if needed), driving the required
        distance, and dropping the package. The driving distance is the
        shortest path distance in the road network graph.

    Assumptions:
        - The road network is static and provided in the static facts.
        - Package goal locations are specified by (at ?p ?l) facts in the goal state.
        - Vehicle capacity constraints are ignored for simplicity and efficiency
          in this non-admissible heuristic.
        - The heuristic assumes a dedicated vehicle is available for each package
          if needed, ignoring potential conflicts or shared trips.
        - All locations are reachable from each other if a path exists in the
          road network. Unreachable goals result in an infinite heuristic value.
        - Objects appearing in '(at ...)' facts that are not packages with goals
          are assumed to be vehicles.

    Heuristic Initialization:
        During initialization, the heuristic precomputes static information:
        1.  The road network graph is built from the (road ?l1 ?l2) facts.
        2.  All-pairs shortest path distances between locations are computed
            using Breadth-First Search (BFS) starting from each location.
            These distances represent the minimum number of 'drive' actions.
        3.  The goal location for each package is extracted from the task's
            goal state facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1.  Initialize the total heuristic value `h` to 0.
        2.  Parse the current state to determine:
            -   The current location of each package (if 'at' a location).
            -   The vehicle containing each package (if 'in' a vehicle).
            -   The current location of each vehicle.
            (Note: Objects in 'at' facts that are not packages with goals are assumed to be vehicles).
        3.  For each package `p` that has a goal location `goal_l` (identified during initialization):
            a.  Check if `p` is currently 'at' a location `l`.
                -   If `l == goal_l`: The package is at its goal. Add 0 to `h`.
                -   If `l != goal_l`: The package needs to be moved. The estimated cost for this package is:
                    -   1 (for pick-up action)
                    -   + shortest distance from `l` to `goal_l` (number of drive actions)
                    -   + 1 (for drop action)
                    -   Add `2 + distances[(l, goal_l)]` to `h`. If `goal_l` is unreachable from `l`, add infinity.
            b.  Check if `p` is currently 'in' a vehicle `v`.
                -   Find the current location `l` of vehicle `v`.
                -   If `l == goal_l`: The package is at the goal location but inside a vehicle. It needs to be dropped. Add 1 (for drop action) to `h`.
                -   If `l != goal_l`: The package needs to be transported and dropped. The estimated cost for this package is:
                    -   shortest distance from `l` to `goal_l` (number of drive actions)
                    -   + 1 (for drop action)
                    -   Add `1 + distances[(l, goal_l)]` to `h`. If `goal_l` is unreachable from `l`, add infinity.
            c.  If the package is neither 'at' a location nor 'in' a vehicle (should not happen in valid states), treat as unreachable.
        4.  Return the total heuristic value `h`. If any required location was unreachable, return infinity.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object containing initial state, goals,
                     operators, and static facts.
        """
        self.task = task
        self.package_goals = self._extract_package_goals(task.goals)
        self.road_graph = self._build_road_graph(task.static)
        self.distances = self._compute_all_pairs_shortest_paths(self.road_graph)

    def _parse_fact(self, fact_str):
        """Parses a fact string into a list of strings [predicate, obj1, obj2, ...]."""
        # Remove surrounding brackets and split by space
        parts = fact_str.strip("()").split()
        return parts

    def _extract_package_goals(self, goals):
        """Extracts the goal location for each package from the goal facts."""
        package_goals = {}
        for goal_fact_str in goals:
            parts = self._parse_fact(goal_fact_str)
            # Goal facts are typically (at ?p ?l) for packages
            if parts[0] == 'at' and len(parts) == 3:
                # Assuming the second part is the package and the third is the location
                package = parts[1]
                location = parts[2]
                package_goals[package] = location
            # Ignore other potential goal facts if any (e.g., capacity goals, though unlikely)
        return package_goals

    def _build_road_graph(self, static_facts):
        """Builds the road network graph from static facts."""
        road_graph = collections.defaultdict(list)
        locations = set()
        for fact_str in static_facts:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                road_graph[loc1].append(loc2)
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all locations mentioned in roads are keys in the graph,
        # even if they have no outgoing roads defined explicitly.
        # This is important for BFS starting points.
        for loc in locations:
             if loc not in road_graph:
                 road_graph[loc] = []

        return road_graph

    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        locations = list(graph.keys()) # Get all locations from the graph

        for start_node in locations:
            # BFS from start_node
            queue = collections.deque([(start_node, 0)])
            visited = {start_node: 0}

            while queue:
                current_node, dist = queue.popleft()
                distances[(start_node, current_node)] = dist

                # Check if current_node has outgoing roads in the graph
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal, or float('inf') if likely unreachable.
        """
        h = 0
        package_locations = {}
        package_in_vehicle = {}
        vehicle_locations = {}

        # Parse current state facts to find locations/containment of packages and vehicles
        packages_with_goals = set(self.package_goals.keys())

        # First pass: identify packages and their locations/vehicles
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in packages_with_goals:
                     package_locations[obj] = loc

            elif predicate == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in packages_with_goals:
                    package_in_vehicle[package] = vehicle

        # Second pass: identify vehicle locations. Assume anything 'at' a location
        # that is not a package we are tracking is a vehicle.
        for fact_str in state:
             parts = self._parse_fact(fact_str)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # If the object is 'at' a location and is not a package
                 # that we found in 'at' or 'in' facts, assume it's a vehicle.
                 # This covers vehicles that might not be carrying a package we track.
                 if obj not in package_locations and obj not in package_in_vehicle:
                     vehicle_locations[obj] = loc


        # Compute heuristic based on packages not at their goals
        for package, goal_l in self.package_goals.items():
            package_cost = 0
            is_at_location = package in package_locations
            is_in_vehicle = package in package_in_vehicle

            if is_at_location:
                current_l = package_locations[package]
                if current_l != goal_l:
                    # Needs pickup (1), drive, drop (1)
                    dist = self.distances.get((current_l, goal_l), math.inf)
                    if dist == math.inf:
                        return math.inf # Goal is unreachable from here
                    package_cost = 1 + dist + 1
                else:
                    # Already at goal location (at p goal_l) is satisfied
                    package_cost = 0

            elif is_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Vehicle must be at some location if it contains a package
                # If vehicle location is not in state, something is wrong, treat as unreachable
                if vehicle not in vehicle_locations:
                     # This indicates an inconsistent state, which shouldn't happen
                     # in a valid planning problem/state representation.
                     # Return infinity as a safe default for likely unsolvable path.
                     return math.inf

                current_l = vehicle_locations[vehicle]

                if current_l != goal_l:
                    # Needs drive, drop (1)
                    dist = self.distances.get((current_l, goal_l), math.inf)
                    if dist == math.inf:
                        return math.inf # Goal is unreachable from here
                    package_cost = dist + 1
                else:
                    # At goal location but inside vehicle, needs drop (1)
                    # The goal is (at p goal_l), not (not (in p v)).
                    # So if package is in vehicle at goal location, it's NOT at goal yet.
                    package_cost = 1

            else:
                 # Package is neither at a location nor in a vehicle.
                 # This state should not be reachable in a valid problem execution.
                 # Treat as unreachable goal.
                 return math.inf

            h += package_cost

        return h
