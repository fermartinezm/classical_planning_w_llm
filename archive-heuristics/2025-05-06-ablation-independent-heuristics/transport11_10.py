import heapq
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Task


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum estimated
        actions required for each package that is not yet at its goal location.
        The estimated cost for a package is calculated based on its current
        state (at a location or inside a vehicle) and the shortest path
        distance on the road network to its goal location.

    Assumptions:
        - The heuristic ignores vehicle capacity constraints.
        - The heuristic ignores the specific assignment of packages to vehicles;
          it assumes any package can be picked up by some vehicle.
        - The heuristic ignores the current location of vehicles relative to
          packages that are at locations; it only considers the package's
          location.
        - The heuristic treats each package's delivery independently, not
          considering potential optimizations from transporting multiple
          packages together.
        - Road network is undirected (if road l1 l2 exists, road l2 l1 exists).
        - All locations and relevant objects (packages, vehicles) appear in
          the task.facts (which includes initial state, goals, and static facts).

    Heuristic Initialization:
        In the constructor, the heuristic processes the task facts to:
        1. Identify all locations involved in the problem.
        2. Identify all packages and vehicles based on predicate arguments in
           'in' and 'capacity' facts.
        3. Build an undirected graph representing the road network from static
           '(road ...)' facts.
        4. Compute the shortest path distance between all pairs of locations
           using Breadth-First Search (BFS), storing these distances for quick
           lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Initialize the total heuristic value to 0.
        2. Determine the current location or state (at a location or in a vehicle)
           for every package and vehicle present in the current state facts. This involves
           iterating through the facts in the state and parsing predicates
           like '(at ...)' and '(in ...)'. Store these in dictionaries mapping
           object names to their state/location.
        3. Iterate through each goal fact defined in the task.
        4. If a goal fact is of the form '(at ?p ?l_goal)' (meaning package ?p
           needs to be at location ?l_goal):
            a. Check the current state of package ?p using the information gathered
               in step 2.
            b. If ?p is currently at ?l_goal, this goal is satisfied for this
               package, and it contributes 0 to the heuristic.
            c. If ?p is currently at location ?l_current (?l_current != ?l_goal):
               Estimate the cost as 1 (pick-up action) + shortest_path_distance(?l_current, ?l_goal)
               (minimum drive actions) + 1 (drop action). Add this to the total
               heuristic value. The shortest path distance is looked up from the
               precomputed table. If the goal location is unreachable from the
               current location, the distance is considered infinite.
            d. If ?p is currently inside a vehicle ?v, and ?v is at location ?l_current
               (vehicle location is also found in step 2):
               If ?l_current == ?l_goal: Estimate the cost as 1 (drop action). Add this
               to the total heuristic value.
               If ?l_current != ?l_goal: Estimate the cost as shortest_path_distance(?l_current, ?l_goal)
               (minimum drive actions) + 1 (drop action). Add this to the total
               heuristic value. If the goal location is unreachable from the
               vehicle's current location, the distance is considered infinite.
            e. If the package's state cannot be determined from the current state
               facts (e.g., the package is not mentioned in any relevant 'at' or 'in'
               fact), this indicates an invalid or unreachable state. In this case,
               the heuristic returns infinity.
        5. If at any point an infinite distance is encountered (meaning a required
           location is unreachable), the total heuristic value is considered infinity.
        6. Return the total accumulated heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.locations = set()
        self.packages = set()
        self.vehicles = set()
        self.road_graph = {}
        self.shortest_paths = {} # Stores (loc1, loc2) -> distance

        # Collect objects, locations, and build road graph from all facts
        # Using task.facts covers all symbols used in the domain/problem
        for fact_str in task.facts:
            parsed = self._parse_fact(fact_str)
            predicate = parsed[0]
            args = parsed[1:]

            if predicate == 'road' and len(args) == 2:
                l1, l2 = args[0], args[1]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Road is undirected
            elif predicate == 'at' and len(args) == 2:
                 obj, loc = args[0], args[1]
                 self.locations.add(loc)
                 # Object type inferred below from 'in'/'capacity'
            elif predicate == 'in' and len(args) == 2:
                 pkg, veh = args[0], args[1]
                 self.packages.add(pkg)
                 self.vehicles.add(veh)
            elif predicate == 'capacity' and len(args) == 2:
                 veh, size = args[0], args[1]
                 self.vehicles.add(veh)
            # Ignore capacity-predecessor and other potential predicates

        # Ensure all locations mentioned in goals are included, even if not in roads/init
        for goal_fact_str in self.goals:
             parsed_goal = self._parse_fact(goal_fact_str)
             if parsed_goal[0] == 'at' and len(parsed_goal) == 3:
                  self.locations.add(parsed_goal[2])


        # Compute shortest paths between all pairs of locations
        self._compute_shortest_paths()

    def _parse_fact(self, fact_str):
        """Parses a fact string into a list of strings."""
        # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
        # Handles potential extra spaces
        return fact_str.strip('()').split()

    def _compute_shortest_paths(self):
        """Computes all-pairs shortest paths using BFS."""
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.shortest_paths[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Ensure current_loc is in road_graph keys, handle isolated locations
                # Isolated locations have no roads, BFS from them only finds themselves
                neighbors = self.road_graph.get(current_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.shortest_paths[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        h_value = 0

        # Determine current locations of packages and vehicles
        package_current_state = {} # pkg_name -> loc_name or ('in', veh_name)
        vehicle_current_location = {} # veh_name -> loc_name

        for fact_str in state:
            parsed = self._parse_fact(fact_str)
            predicate = parsed[0]
            args = parsed[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_location[obj] = loc
                # Ignore 'at' facts for other types if any
            elif predicate == 'in' and len(args) == 2:
                pkg, veh = args[0], args[1]
                if pkg in self.packages and veh in self.vehicles:
                    package_current_state[pkg] = ('in', veh)
                # Ignore 'in' facts for other types if any
            # Ignore other predicates in state like 'capacity'

        # Calculate heuristic based on package goals
        for goal_fact_str in self.goals:
            parsed_goal = self._parse_fact(goal_fact_str)
            # Only consider (at ?p ?l) goals for packages
            if parsed_goal[0] == 'at' and len(parsed_goal) == 3:
                pkg_name, goal_loc = parsed_goal[1], parsed_goal[2]

                # Ensure the goal is about a package we identified
                if pkg_name not in self.packages:
                     # This goal is not about a package object we know.
                     # This might indicate an issue with the domain/problem definition
                     # or our object identification. For safety, treat as unsolvable.
                     return float('inf')

                # Check if the package is in the current state representation
                if pkg_name not in package_current_state:
                     # The package exists (identified in __init__), but its location/state
                     # is not described by any 'at' or 'in' fact in the current state.
                     # This indicates an invalid state.
                     return float('inf')


                current_state_of_pkg = package_current_state[pkg_name]

                if current_state_of_pkg == goal_loc:
                    # Package is already at the goal location
                    continue # Cost is 0 for this package

                if isinstance(current_state_of_pkg, str):
                    # Package is at a location (not the goal location)
                    current_loc = current_state_of_pkg
                    # Cost = pick-up (1) + drive + drop (1)
                    # Lookup shortest path distance. Use .get for safety if goal_loc is somehow not in self.locations
                    drive_cost = self.shortest_paths.get((current_loc, goal_loc), float('inf'))

                    if drive_cost == float('inf'):
                         # Goal location unreachable from current location
                         return float('inf')

                    h_value += 1 + drive_cost + 1

                elif isinstance(current_state_of_pkg, tuple) and current_state_of_pkg[0] == 'in':
                    # Package is inside a vehicle
                    veh_name = current_state_of_pkg[1]

                    # Find vehicle's location
                    if veh_name not in vehicle_current_location:
                        # Vehicle carrying the package is not at any location? Invalid state.
                        return float('inf')

                    current_loc = vehicle_current_location[veh_name]

                    if current_loc == goal_loc:
                        # Package is in vehicle at goal location
                        # Cost = drop (1)
                        h_value += 1
                    else:
                        # Package is in vehicle, vehicle is not at goal location
                        # Cost = drive + drop (1)
                        drive_cost = self.shortest_paths.get((current_loc, goal_loc), float('inf'))

                        if drive_cost == float('inf'):
                             # Goal location unreachable from vehicle's current location
                             return float('inf')

                        h_value += drive_cost + 1
                # No else needed, as package_current_state should only contain str or ('in', veh)

        return h_value
