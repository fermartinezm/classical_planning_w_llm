import collections

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated costs for each package that is not yet at its goal location.
    For a package not at its goal, the estimated cost depends on whether
    the package is currently on the ground or inside a vehicle.
    - If the package is on the ground at location L_p, the estimated cost is
      the minimum distance from any vehicle's current location to L_p,
      plus 1 (for pickup), plus the shortest distance from L_p to the
      package's goal location L_goal, plus 1 (for drop).
    - If the package is inside a vehicle V, and V is at location L_v, the
      estimated cost is the shortest distance from L_v to the package's
      goal location L_goal, plus 1 (for drop).
    The shortest distances between locations are precomputed using BFS on
    the road network.

    Assumptions:
    - The heuristic assumes that any vehicle can eventually pick up any package
      and deliver it, ignoring capacity constraints.
    - It assumes the road network is connected enough for vehicles to reach
      package locations and goal locations from their current positions.
      If a required location is unreachable, the heuristic returns infinity.
    - It assumes valid states where packages with goals are either on the
      ground at a location or inside a vehicle that is at a location.
    - Object names follow conventions: vehicles start with 'v', packages with 'p'.

    Heuristic Initialization:
    1. Parse the static facts from the task (Task.static).
    2. Build the road network graph from '(road l1 l2)' facts. Identify all locations.
    3. Compute all-pairs shortest paths between all identified locations using BFS. Store these distances.
    4. Parse the goal facts from the task (Task.goals) to identify packages and their goal locations.
    5. Identify all vehicles and packages present in the initial state (Task.initial_state)
       based on naming conventions ('v' for vehicle, 'p' for package).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value h to 0.
    2. Parse the current state to determine the current location of each vehicle
       and the current status of each package (at a location or in a vehicle).
    3. Iterate through each package that has a goal location defined in the task.
    4. For a package P with goal location L_goal:
       a. Check if P is already at L_goal in the current state. If yes, the cost for this package is 0, continue to the next package.
       b. If P is not at L_goal, determine its current status:
          i. If P is on the ground at location L_current:
             - Find the minimum shortest distance from any vehicle's current location
               to L_current. Let this be min_dist_v_to_p.
             - If no vehicle is located or no vehicle can reach the package location, return infinity.
             - Add min_dist_v_to_p + self.dist(L_current, L_goal) + 2 to h.
               (Cost = travel vehicle to package + pickup + travel package to goal + drop)
          ii. If P is inside vehicle V:
              - Find the current location L_v of vehicle V.
              - If V is not located, return infinity.
              - Add self.dist(L_v, L_goal) + 1 to h.
                (Cost = travel vehicle to goal + drop)
          iii. If P is neither on the ground nor in a vehicle (and has a goal):
              - This indicates an invalid state. Return infinity.
    5. Return the total heuristic value h.
    """

    def __init__(self, task):
        self.task = task
        self._parse_static_info()
        self._compute_shortest_paths()
        self._parse_goals()
        self._parse_initial_state_objects() # To identify all vehicles/packages

    def _parse_static_info(self):
        self.locations = set()
        self.road_graph = collections.defaultdict(list)
        # capacity_predecessors is not used in this heuristic, but parsed for completeness
        self.capacity_predecessors = {} # s1 -> s2 mapping

        for fact_str in self.task.static:
            fact = fact_str.strip('()').split()
            predicate = fact[0]
            if predicate == 'road' and len(fact) == 3:
                l1, l2 = fact[1], fact[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph[l1].append(l2)
                # Assuming roads are bidirectional if listed explicitly in static facts
                # The example files list both directions, so we don't add the reverse here.

            elif predicate == 'capacity-predecessor' and len(fact) == 3:
                s1, s2 = fact[1], fact[2]
                self.capacity_predecessors[s1] = s2 # s1 is smaller than s2

        self.locations = sorted(list(self.locations)) # Consistent order


    def _compute_shortest_paths(self):
        self.distances = {} # {start_loc: {end_loc: dist}}

        for start_loc in self.locations:
            distances_from_start = {loc: float('inf') for loc in self.locations}
            distances_from_start[start_loc] = 0
            queue = collections.deque([start_loc])

            while queue:
                u = queue.popleft()
                current_dist = distances_from_start[u]

                # Ensure u is in road_graph keys before iterating
                if u in self.road_graph:
                    for v in self.road_graph[u]:
                        if distances_from_start[v] == float('inf'):
                            distances_from_start[v] = current_dist + 1
                            queue.append(v)
            self.distances[start_loc] = distances_from_start

    def dist(self, l1, l2):
        """Returns the shortest distance between two locations."""
        # Check if locations exist and distance was computed
        if l1 not in self.distances or l2 not in self.distances[l1]:
             # This case should ideally not be reached if locations are correctly identified
             # from static facts and BFS covers all of them.
             # However, returning inf is safe.
             return float('inf')
        return self.distances[l1][l2]

    def _parse_goals(self):
        self.package_goals = {} # {package_name: goal_location_name}
        # Store goal facts as strings for quick lookup in __call__
        self.goal_facts_set = set(self.task.goals)

        for fact_str in self.task.goals:
            fact = fact_str.strip('()').split()
            # Assuming goal facts are only (at ?p ?l) for packages
            if fact[0] == 'at' and len(fact) == 3:
                p, l = fact[1], fact[2]
                self.package_goals[p] = l

    def _parse_initial_state_objects(self):
        # Identify all potential packages and vehicles from initial state and goals
        self.all_packages = set(self.package_goals.keys())
        self.all_vehicles = set()

        # Add objects from initial state based on naming convention
        for fact_str in self.task.initial_state:
             fact = fact_str.strip('()').split()
             predicate = fact[0]
             if predicate == 'at' and len(fact) == 3:
                 obj, loc = fact[1], fact[2]
                 if obj.startswith('v'):
                     self.all_vehicles.add(obj)
                 elif obj.startswith('p'):
                     self.all_packages.add(obj)
             elif predicate == 'in' and len(fact) == 3:
                 p, v = fact[1], fact[2]
                 if p.startswith('p'):
                     self.all_packages.add(p)
                 if v.startswith('v'):
                     self.all_vehicles.add(v)

        # Ensure all locations mentioned in initial state are added to self.locations
        # This handles cases where a location exists but has no roads connected in static facts
        # (though such locations are usually not useful).
        # We rely on locations from road facts for the graph. If an object is at a location
        # not in the road graph, dist() will return inf, correctly indicating potential issue.
        # We could add locations from initial state 'at' facts to self.locations, but
        # they wouldn't be part of the road graph anyway.

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of facts representing the current state.
        @return: The estimated number of actions to reach a goal state,
                 or float('inf') if the state is likely unsolvable.
        """
        # Check if goal is reached
        # This check is slightly redundant with the loop below, but explicit and clear.
        # The loop below will result in h=0 if all package goals are met.
        # if self.task.goal_reached(state):
        #     return 0

        vehicle_location = {} # {vehicle_name: current_location_name}
        package_location = {} # {package_name: current_location_name}
        package_in_vehicle = {} # {package_name: vehicle_name}

        # Parse current state facts to get dynamic object locations/containment
        for fact_str in state:
            fact = fact_str.strip('()').split()
            predicate = fact[0]
            if predicate == 'at' and len(fact) == 3:
                obj, loc = fact[1], fact[2]
                if obj in self.all_vehicles:
                    vehicle_location[obj] = loc
                elif obj in self.all_packages:
                    package_location[obj] = loc
            elif predicate == 'in' and len(fact) == 3:
                p, v = fact[1], fact[2]
                if p in self.all_packages and v in self.all_vehicles:
                     package_in_vehicle[p] = v

        h = 0
        # Consider only packages that have a goal location
        for p, l_goal in self.package_goals.items():
            # Check if the goal fact for this package is already true
            if f'(at {p} {l_goal})' in state:
                 continue # This package is already at its goal

            # Package is not at its goal, calculate its contribution
            if p in package_location:
                # Package is on the ground at l_current
                l_current = package_location[p]

                # Find the minimum distance from any vehicle to l_current
                min_dist_v_to_p = float('inf')
                found_vehicle_location = False
                for v in self.all_vehicles:
                    if v in vehicle_location:
                        found_vehicle_location = True
                        l_v = vehicle_location[v]
                        dist_v_to_p = self.dist(l_v, l_current)
                        min_dist_v_to_p = min(min_dist_v_to_p, dist_v_to_p)

                # If no vehicle is located or no vehicle can reach the package
                if not found_vehicle_location or min_dist_v_to_p == float('inf'):
                     # This state might be unsolvable or require complex steps
                     return float('inf')

                # Distance from package's current location to its goal
                dist_p_to_goal = self.dist(l_current, l_goal)
                if dist_p_to_goal == float('inf'):
                     # Package goal location is unreachable from its current location
                     return float('inf')

                # Cost = travel vehicle to package + pickup + travel package to goal + drop
                h += min_dist_v_to_p + 1 + dist_p_to_goal + 1

            elif p in package_in_vehicle:
                # Package is inside vehicle v
                v = package_in_vehicle[p]
                if v not in vehicle_location:
                    # Vehicle containing package is not located? Invalid state.
                    return float('inf')
                l_v = vehicle_location[v] # Vehicle's current location

                # Distance from vehicle's current location to package's goal
                dist_v_to_goal = self.dist(l_v, l_goal)
                if dist_v_to_goal == float('inf'):
                     # Package goal location is unreachable from vehicle's current location
                     return float('inf')

                # Cost = travel vehicle to goal + drop
                h += dist_v_to_goal + 1

            else:
                # Package with a goal is neither at a location nor in a vehicle.
                # This indicates an invalid state representation according to the domain.
                return float('inf')

        return h
