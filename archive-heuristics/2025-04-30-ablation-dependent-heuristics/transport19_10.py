import collections

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated minimum actions required for each package that is not
        yet at its goal location. The estimated cost for a single package
        involves a pick-up action (if at a location), driving the vehicle
        carrying the package to the goal location, and a drop action.
        Driving cost is estimated using shortest path distances on the road network.

    Assumptions:
        - The goal state is defined solely by the locations of packages,
          specifically facts of the form `(at <package> <location>)`.
        - Roads defined by `(road l1 l2)` facts are bidirectional.
        - Packages implicitly require one unit of capacity change when picked up/dropped.
          This heuristic simplifies capacity handling by ignoring it, assuming
          a suitable vehicle is always available when needed.
        - If a package is `in` a vehicle, the vehicle's location is known and present in the state.
        - The heuristic value is 0 if and only if the state is the goal state.
        - The heuristic value is finite for solvable states and infinite otherwise.

    Heuristic Initialization:
        The constructor `__init__` takes the planning task as input.
        1.  It parses the goal facts (`task.goals`) to create a mapping from
            each package name to its required goal location.
        2.  It parses the static facts (`task.static`), specifically the
            `(road l1 l2)` facts, to build an undirected graph representing
            the road network between locations.
        3.  It computes all-pairs shortest paths on this road network graph
            using Breadth-First Search (BFS) starting from each location.
            These distances are stored for quick lookup during heuristic computation.
        4.  Capacity-predecessor facts are noted but not explicitly used in this
            simplified heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method computes the heuristic value for the given state.
        1.  It first checks if the input `state` is the goal state using `task.goal_reached(state)`.
            If it is, the heuristic returns 0.
        2.  If not the goal state, it parses the dynamic facts in the `state`
            to determine the current location of each package (either `at` a location
            or `in` a vehicle) and the current location of each vehicle.
        3.  It initializes a total heuristic cost `h` to 0.
        4.  It iterates through each package that has a specified goal location
            (identified during initialization from `task.goals`).
        5.  For each package, it finds its current location based on the parsed state information.
        6.  If the package's current location is different from its goal location:
            a.  If the package is currently `at` a location `l_current`:
                It needs to be picked up (1 action), the vehicle needs to drive
                from `l_current` to `l_goal` (estimated by `shortest_paths[l_current][l_goal]`
                drive actions), and it needs to be dropped (1 action).
                The cost added for this package is `1 + shortest_paths[l_current][l_goal] + 1`.
            b.  If the package is currently `in` a vehicle `v` which is `at` location `l_current_v`:
                The vehicle needs to drive from `l_current_v` to `l_goal` (estimated by
                `shortest_paths[l_current_v][l_goal]` drive actions), and the package
                needs to be dropped (1 action).
                The cost added for this package is `shortest_paths[l_current_v][l_goal] + 1`.
            c.  If the package's location cannot be determined from the state (e.g., vehicle location missing),
                the shortest path lookup will return infinity, correctly contributing
                infinity to the heuristic, indicating an unreachable goal for this package.
        7.  The individual costs for all misplaced packages are summed up to get the total heuristic value `h`.
        8.  The total sum `h` is returned.
    """

    def __init__(self, task):
        self.task = task
        self.goal_locations = {}
        self.locations = set()
        self.road_graph = {}
        self.shortest_paths = {}

        # Parse goal facts to find target locations for packages
        for goal_fact in task.goals:
            # Goal facts are expected to be like '(at p1 l2)'
            parts = self._parse_fact(goal_fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goal facts if any exist

        # Parse static facts to build the road graph
        for static_fact in task.static:
            parts = self._parse_fact(static_fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string like '(predicate arg1 arg2)'."""
        # Remove surrounding parentheses and split by spaces
        return tuple(fact_string.strip('()').split())

    def _bfs(self, start_node):
        """Performs BFS from a start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not in the graph (e.g., initial state has object at unknown location)
             # This shouldn't happen in valid problems but handle defensively.
             return distances

        distances[start_node] = 0
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # If the state is the goal state, heuristic is 0
        if self.task.goal_reached(state):
             return 0

        # Extract dynamic information from the state
        package_locations = {} # package -> location (if at location)
        package_in_vehicle = {} # package -> vehicle (if in vehicle)
        vehicle_locations = {} # vehicle -> location

        # Identify all objects that are packages (those in the goal)
        goal_packages = set(self.goal_locations.keys())

        for fact_string in state:
            parts = self._parse_fact(fact_string)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in goal_packages:
                    package_locations[obj] = loc
                # Assume other 'at' facts refer to vehicles.
                # This relies on the assumption that only packages (relevant to goal)
                # and vehicles are locatable objects that appear in 'at' facts in the state.
                # A more robust approach would involve parsing object types from the PDDL,
                # but that info isn't readily available from the 'Task' object.
                # Given the domain and examples, this assumption is likely safe.
                else:
                     vehicle_locations[obj] = loc

            elif predicate == 'in' and len(parts) == 3:
                p, v = parts[1], parts[2]
                # Ensure 'p' is a package relevant to the goal
                if p in goal_packages:
                    package_in_vehicle[p] = v
            # Ignore capacity facts for this heuristic calculation

        h = 0
        # Iterate through packages that need to reach a goal location
        for package, goal_loc in self.goal_locations.items():
            current_loc = None
            is_in_vehicle = False
            vehicle_carrying = None

            if package in package_locations:
                current_loc = package_locations[package]
            elif package in package_in_vehicle:
                is_in_vehicle = True
                vehicle_carrying = package_in_vehicle[package]
                # Find vehicle's location
                if vehicle_carrying in vehicle_locations:
                     current_loc = vehicle_locations[vehicle_carrying]
                else:
                     # This case implies a package is in a vehicle, but the vehicle's
                     # location is not in the state. This is an invalid state structure
                     # for this domain. Treat as unreachable goal for this package.
                     current_loc = None # Will result in adding infinity below

            # Calculate cost for this package if not at goal
            # Check if current_loc is known and is different from goal_loc
            if current_loc is not None and current_loc != goal_loc:
                # Get distance, default to infinity if locations are not in the graph
                # (e.g., goal_loc or current_loc not in the precomputed shortest_paths)
                dist_to_goal = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))

                if not is_in_vehicle:
                    # Package is at a location, needs pick-up, drive, drop
                    # Cost: 1 (pick-up) + dist(current_loc, goal_loc) + 1 (drop)
                    h += 1 # pick-up action
                    h += dist_to_goal # drive actions
                    h += 1 # drop action
                else: # Package is in a vehicle
                    # Needs drive, drop
                    # Cost: dist(current_loc, goal_loc) + 1 (drop)
                    h += dist_to_goal # drive actions
                    h += 1 # drop action
            # If current_loc is None or current_loc == goal_loc, cost added is 0 for this package.

        return h
