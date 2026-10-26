from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming task.py is available and defines Task

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum actions
        required for each package that is not yet at its goal location.
        The minimum actions for a package involve picking it up (if at a location),
        driving it to the goal location, and dropping it. The driving cost is
        estimated by the shortest path distance in the road network.

    Assumptions:
        - The goal only consists of (at ?p ?l) facts for packages.
        - Any vehicle can transport any package (ignores capacity constraints).
        - Vehicles are always available to perform pick-up/drop-off/drive actions
          as needed for each package independently.
        - The road network is static and defined by (road ?l1 ?l2) facts.
        - Locations and packages are correctly identified in facts.

    Heuristic Initialization:
        1. Parses static facts to build the road network graph.
        2. Identifies all unique locations from the road network and initial/goal states.
        3. Computes all-pairs shortest paths between locations using BFS on the road graph.
           This precomputation is done once during the heuristic's initialization.
        4. Identifies the set of packages that are part of the goal state.

    Step-By-Step Thinking for Computing Heuristic:
        1. For a given state, build dictionaries to quickly look up the current
           location of packages (that are at a location), the vehicle carrying
           a package (if it's in a vehicle), and the location of vehicles.
           Only consider packages that are part of the goal.
        2. Initialize the total heuristic value (h) to 0.
        3. Iterate through each goal fact.
        4. If the goal fact is (at ?p ?goal_l) and it is not already true in the current state:
           a. Find the current status of package ?p using the lookups created in step 1:
              - If ?p is found in the package_location dictionary at ?current_l:
                - The cost for this package is 1 (pick-up) + shortest_path(?current_l, ?goal_l) (drive) + 1 (drop).
              - If ?p is found in the package_in_vehicle dictionary, carried by ?v:
                - Look up the location of vehicle ?v in the vehicle_location dictionary, say ?vehicle_l.
                - The cost for this package is shortest_path(?vehicle_l, ?goal_l) (drive) + 1 (drop).
              - If ?p's status is not found in either lookup (meaning it's not at a location or in a vehicle, which implies an invalid state), the cost is infinity.
           b. If the calculated cost for this package is infinity (e.g., due to unreachable locations), the total heuristic for the state is infinity, and we can return immediately.
           c. Otherwise, add the calculated cost for this package to the total heuristic value h.
        5. Return the total heuristic value h.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # Build the road graph and find all locations
        self.road_graph = {}
        locations = set()

        # Collect locations from road facts
        for fact_string in self.static_facts:
            pred, args = self._parse_fact(fact_string)
            if pred == 'road':
                l1, l2 = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph:
                    self.road_graph[l2] = []
                self.road_graph[l1].append(l2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph[l2].append(l1)

        # Collect locations from initial state and goals
        # This ensures we include locations that might not be part of any road
        # but where objects start or need to end up.
        for fact_string in self.task.initial_state:
             pred, args = self._parse_fact(fact_string)
             if pred == 'at':
                 locatable, loc = args
                 locations.add(loc)
                 if loc not in self.road_graph:
                     self.road_graph[loc] = [] # Add isolated locations

        for goal_fact_string in self.task.goals:
             pred, args = self._parse_fact(goal_fact_string)
             if pred == 'at':
                 locatable, loc = args
                 locations.add(loc)
                 if loc not in self.road_graph:
                     self.road_graph[loc] = [] # Add isolated locations

        self.locations = list(locations)

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify packages based on goal facts once during init
        self.goal_packages = {self._parse_fact(g)[1][0] for g in self.goals if self._parse_fact(g)[0] == 'at'}


    def _parse_fact(self, fact_string):
        # Removes surrounding parentheses and splits by space
        # Example: '(at p1 l1)' -> ('at', ['p1', 'l1'])
        parts = fact_string[1:-1].split()
        if not parts:
             # Should not happen with valid PDDL facts
             return None, [] # Or raise an error
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _compute_all_pairs_shortest_paths(self):
        distances = {}
        for start_node in self.locations:
            distances[start_node] = self._bfs(start_node)
        return distances

    def _bfs(self, start_node):
        # Performs BFS from a start_node to find distances to all other nodes
        dist = {node: float('inf') for node in self.locations}
        dist[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is in graph keys (handles locations with no roads but present in state/goals)
            if u in self.road_graph:
                for v in self.road_graph[u]:
                    if dist[v] == float('inf'):
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # Build quick lookups for current state
        package_location = {} # package -> location
        package_in_vehicle = {} # package -> vehicle
        vehicle_location = {} # vehicle -> location

        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at':
                item, loc = args
                # Check if the item is one of the packages we care about (those in goals)
                if item in self.goal_packages:
                     package_location[item] = loc
                else:
                     # Assume other 'at' items are vehicles or other locatables not in goals
                     vehicle_location[item] = loc
            elif pred == 'in':
                pkg, veh = args
                # Only track packages that are goals
                if pkg in self.goal_packages:
                    package_in_vehicle[pkg] = veh
            # Ignore other predicates like capacity, capacity-predecessor

        h_value = 0

        # Calculate heuristic based on goals
        for goal_fact_string in self.goals:
            g_pred, g_args = self._parse_fact(goal_fact_string)
            if g_pred == 'at':
                package, goal_l = g_args

                # If goal is already satisfied, cost is 0 for this package
                if goal_fact_string in state:
                    continue

                # Package is not at goal, calculate cost
                cost_for_package = float('inf') # Default to infinity

                # Case 1: Package is at a location
                if package in package_location:
                    current_l = package_location[package]
                    # Need to pick up (1), drive (dist), drop (1)
                    if current_l in self.distances and goal_l in self.distances[current_l]:
                         drive_cost = self.distances[current_l][goal_l]
                         if drive_cost != float('inf'):
                             cost_for_package = 1 + drive_cost + 1 # pick-up + drive + drop
                         # else: cost_for_package remains float('inf')
                    # else: cost_for_package remains float('inf') (location not found in distances)


                # Case 2: Package is in a vehicle
                elif package in package_in_vehicle:
                    vehicle = package_in_vehicle[package]
                    # Find vehicle location
                    if vehicle in vehicle_location:
                        vehicle_l = vehicle_location[vehicle]
                        # Need to drive (dist), drop (1)
                        if vehicle_l in self.distances and goal_l in self.distances[vehicle_l]:
                            drive_cost = self.distances[vehicle_l][goal_l]
                            if drive_cost != float('inf'):
                                cost_for_package = drive_cost + 1 # drive + drop
                            # else: cost_for_package remains float('inf')
                        # else: cost_for_package remains float('inf') (location not found in distances)
                    # else: cost_for_package remains float('inf') (vehicle location unknown)

                # Case 3: Package status unknown (not at location, not in vehicle)
                # This state is likely invalid or a dead end. cost_for_package is already float('inf')

                # If any package requires infinite cost, the total heuristic is infinite
                if cost_for_package == float('inf'):
                    return float('inf')

                h_value += cost_for_package

        return h_value
