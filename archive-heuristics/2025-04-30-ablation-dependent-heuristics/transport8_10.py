import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)"""
    # Remove parentheses and split by space
    parts = fact_string.strip('()').split()
    return tuple(parts)

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum number of actions required for each package that is not
        at its goal location. For a package currently at a location, the
        estimated cost is 2 (pickup + drop) plus the shortest distance
        from its current location to its goal location. For a package
        currently inside a vehicle, the estimated cost is 1 (drop) plus
        the shortest distance from the vehicle's current location to the
        package's goal location. The shortest distances between locations
        are precomputed using BFS on the road network. This heuristic ignores
        vehicle capacity constraints and assumes a suitable vehicle is always
        available when needed at a location.

    Assumptions:
        - The goal only consists of (at ?p ?l) facts for packages.
        - The road network is static and provided in the static facts.
        - Locations, vehicles, and packages can be identified by their
          appearance in specific predicates across the task's facts (static,
          initial, goal, operator definitions).
        - The shortest path between any two relevant locations is computable
          (i.e., relevant parts of the graph are connected). Unreachable goals
          result in an infinite heuristic value.
        - The state representation is complete for packages and vehicles
          mentioned in the goals or initial state.

    Heuristic Initialization:
        The constructor takes the Task object. It first identifies all objects
        that are locations, vehicles, and packages based on their usage in
        static, initial, goal, and operator facts. It extracts the goal
        locations for each package from the task's goal facts. It then builds
        the road network graph from the static facts and computes the shortest
        distances between all pairs of identified locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
        1. For a given state (a frozenset of facts):
        2. Check if the state is a goal state using `self.task.goal_reached(state)`.
           If it is, return 0.
        3. Initialize the total heuristic value `h` to 0.
        4. Create dictionaries to store the current state of packages
           (`package_current_state`: maps package name to its location string
           or the vehicle name it is in) and vehicles (`vehicle_locations`:
           maps vehicle name to its location string).
        5. Iterate through the facts in the current state. Parse each fact.
           If the fact is `(at obj loc)`:
             If `obj` is identified as a package, record `package_current_state[obj] = loc`.
             If `obj` is identified as a vehicle, record `vehicle_locations[obj] = loc`.
           If the fact is `(in pkg veh)`:
             If `pkg` is identified as a package and `veh` as a vehicle, record
             `package_current_state[pkg] = veh`.
           Ignore other fact types (like capacity).
        6. Iterate through each package `pkg` and its goal location `goal_loc`
           stored in `self.goal_locations`.
        7. Get the current state of `pkg` from `package_current_state`. If the
           package is not found in the state dictionary (which implies an issue
           with state representation or object identification), return `float('inf')`.
        8. If the `current_state` of the package is equal to its `goal_loc`,
           this package goal is satisfied in this state; continue to the next package.
        9. If the package is not at its goal:
           a. If `current_state` is a location (check if it's in `self.locations`):
              - This means the package is `(at pkg current_state)`.
              - The estimated cost for this package is 2 (for pickup and drop actions)
                plus the precomputed shortest distance from `current_state` to `goal_loc`.
              - Retrieve the distance using `self._get_distance(current_state, goal_loc)`.
              - If the distance is `None` (indicating unreachable), return `float('inf')`
                for the total heuristic.
              - Otherwise, add `2 + distance` to `h`.
           b. If `current_state` is a vehicle (check if it's in `self.vehicles`):
              - This means the package is `(in pkg current_state)`.
              - Get the current location `veh_loc` of the vehicle `current_state`
                from `vehicle_locations`. If the vehicle's location is not found
                (implies state issue), return `float('inf')`.

              - The estimated cost for this package is 1 (for the drop action)
                plus the precomputed shortest distance from `veh_loc` to `goal_loc`.
              - Retrieve the distance using `self._get_distance(veh_loc, goal_loc)`.
              - If the distance is `None` (indicating unreachable), return `float('inf')`.
              - Otherwise, add `1 + distance` to `h`.
           c. If `current_state` is neither a known location nor a known vehicle,
              return `float('inf')` (indicates state parsing/identification issue).
        10. After processing all packages with goals, return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing shortest paths and goal locations.

        Args:
            task: The planning Task object.
        """
        self.task = task
        self.goal_locations = self._extract_goal_locations(task.goals)

        # Identify object types based on predicate usage across all facts
        self.locations = set()
        self.vehicles = set()
        self.packages = set()
        all_objects = set()

        # Collect all object names from all facts in the task
        # task.facts contains all possible ground facts from the domain and problem
        for fact_string in task.facts:
             fact = parse_fact(fact_string)
             all_objects.update(fact[1:]) # Add all arguments except predicate name

        # Categorize objects based on predicate usage
        for obj in all_objects:
            # An object is a vehicle if it appears as the first arg of 'capacity'
            # or the second arg of 'in'.
            is_vehicle = any(parse_fact(f)[0] == 'capacity' and parse_fact(f)[1] == obj for f in task.facts) or \
                         any(parse_fact(f)[0] == 'in' and parse_fact(f)[2] == obj for f in task.facts)

            # An object is a package if it appears as the first arg of 'in'
            # or the first arg of a goal 'at' fact.
            is_package = any(parse_fact(f)[0] == 'in' and parse_fact(f)[1] == obj for f in task.facts) or \
                         any(parse_fact(f)[0] == 'at' and parse_fact(f)[1] == obj for f in task.goals)

            # An object is a location if it appears as an arg in a 'road' fact
            # or the second arg of an 'at' fact.
            is_location = any(parse_fact(f)[0] == 'road' and (parse_fact(f)[1] == obj or parse_fact(f)[2] == obj) for f in task.static) or \
                          any(parse_fact(f)[0] == 'at' and parse_fact(f)[2] == obj for f in task.facts)

            if is_location: self.locations.add(obj)
            if is_vehicle: self.vehicles.add(obj)
            if is_package: self.packages.add(obj)

        # Refine sets: ensure objects belong to only one category if possible
        # Assuming no object is both a location and a locatable, or both package/vehicle
        # The checks above should be specific enough, but set differences add robustness
        self.packages = self.packages - self.vehicles - self.locations
        self.vehicles = self.vehicles - self.packages - self.locations
        self.locations = self.locations - self.packages - self.vehicles


        # Precompute distances using the identified locations and graph from static facts.
        self.shortest_distances = self._precompute_shortest_distances(task.static, self.locations)

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach a goal state, or float('inf')
            if the goal is unreachable from the current state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0
        package_current_state = {} # Map package -> location or vehicle
        vehicle_locations = {} # Map vehicle -> location

        # Extract current state of relevant objects from the state
        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at':
                obj, loc = fact[1], fact[2]
                if obj in self.packages:
                     package_current_state[obj] = loc
                elif obj in self.vehicles:
                     vehicle_locations[obj] = loc
                # else: ignore other 'at' facts if any

            elif fact[0] == 'in':
                pkg, veh = fact[1], fact[2]
                if pkg in self.packages and veh in self.vehicles:
                    package_current_state[pkg] = veh
                # else: ignore other 'in' facts if any

            # Ignore capacity facts for this heuristic

        # Calculate cost for each package that has a goal
        for pkg, goal_loc in self.goal_locations.items():
            current_state = package_current_state.get(pkg)

            # If package is not in the state, it cannot contribute to reaching its goal
            # This case should ideally not happen in valid planning states.
            # Returning inf is a safe default for unreachable goals.
            if current_state is None:
                 return float('inf')

            # If package is at its goal location, cost is 0 for this package
            if current_state == goal_loc:
                 continue # Goal reached for this package

            # Package is not at its goal
            if current_state in self.locations: # Package is at a location
                current_loc = current_state
                # Needs pickup (1) + drive (dist) + drop (1)
                dist = self._get_distance(current_loc, goal_loc)
                if dist is None:
                    # Goal is unreachable from current location
                    return float('inf') # Unsolvable state
                h += 2 + dist

            elif current_state in self.vehicles: # Package is inside a vehicle
                 veh = current_state
                 veh_loc = vehicle_locations.get(veh)
                 if veh_loc is None:
                     # Vehicle location unknown (shouldn't happen in valid states)
                     return float('inf') # Invalid state representation?

                 # Needs drive (dist) + drop (1)
                 dist = self._get_distance(veh_loc, goal_loc)
                 if dist is None:
                     # Goal is unreachable from vehicle location
                     return float('inf') # Unsolvable state
                 h += 1 + dist
            else:
                 # current_state is neither a known location nor a known vehicle
                 # This indicates an issue with state parsing or object identification
                 return float('inf')


        return h

    def _extract_goal_locations(self, goals):
        """Extracts goal locations for packages from goal facts."""
        goal_locs = {}
        # Goals can be conjunctions, iterate through the set of goal facts
        for goal_fact_string in goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact[0] == 'at':
                # Assuming goal facts are (at package location)
                pkg, loc = goal_fact[1], goal_fact[2]
                goal_locs[pkg] = loc
            # Ignore other potential goal types if any
        return goal_locs

    def _precompute_shortest_distances(self, static_facts, all_locations):
        """
        Builds the road network graph and computes shortest distances
        between all pairs of locations using BFS.
        """
        graph = collections.defaultdict(set)

        for fact_string in static_facts:
            fact = parse_fact(fact_string)
            if fact[0] == 'road':
                l1, l2 = fact[1], fact[2]
                graph[l1].add(l2)
                # Locations are collected in __init__

        distances = {}
        # Compute distances from every known location
        for start_node in all_locations:
            distances[start_node] = self._bfs(graph, start_node, all_locations)

        return distances

    def _bfs(self, graph, start_node, all_locations):
        """Performs BFS to find shortest distances from start_node."""
        # Initialize dist for all known locations
        dist = {node: float('inf') for node in all_locations}
        if start_node in dist: # Ensure start_node is one of the known locations
            dist[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                u = queue.popleft()
                # Only explore if u is in the graph (has outgoing roads)
                if u in graph:
                    for v in graph[u]:
                        # Ensure v is a known location before updating distance
                        if v in dist and dist[v] == float('inf'):
                            dist[v] = dist[u] + 1
                            queue.append(v)
        # If start_node was not in all_locations, dist remains all inf, which is correct.
        return dist

    def _get_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest distance."""
        # Check if loc1 is a valid start node and loc2 is a valid end node
        if loc1 not in self.shortest_distances or loc2 not in self.shortest_distances[loc1]:
             # This means loc1 is not a known location or loc2 is not a known location
             return None # Indicate unreachable

        distance = self.shortest_distances[loc1][loc2]

        # If distance is still infinity, it means loc2 is unreachable from loc1
        if distance == float('inf'):
            return None

        return distance
