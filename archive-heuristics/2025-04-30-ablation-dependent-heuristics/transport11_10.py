import sys
from collections import deque

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function to build road graph
def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network."""
    graph = {}
    locations = set()
    for fact_string in static_facts:
        if fact_string.startswith('(road '):
            _, l1, l2 = parse_fact(fact_string)
            locations.add(l1)
            locations.add(l2)
            if l1 not in graph:
                graph[l1] = set()
            graph[l1].add(l2)
    return graph, list(locations)

# Helper function to compute all-pairs shortest paths using BFS
def compute_shortest_paths(graph, locations):
    """Computes shortest path distances between all pairs of locations using BFS."""
    shortest_paths = {}
    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node}
        shortest_paths[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        shortest_paths[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))
    return shortest_paths

# Helper function to build size hierarchy (parsed but not used in this simple heuristic)
def build_size_hierarchy(static_facts):
    """
    Builds the size hierarchy mapping size strings to integer levels.
    (Not used in the current heuristic calculation but parsed from static info).
    """
    size_levels = {}
    successors = {} # size_smaller -> size_larger

    pred_facts = []
    for fact_string in static_facts:
        if fact_string.startswith('(capacity-predecessor '):
            _, s1, s2 = parse_fact(fact_string)
            pred_facts.append((s1, s2)) # s1 is predecessor of s2

    if not pred_facts:
         return size_levels, successors

    predecessors = {} # size_larger -> size_smaller
    all_sizes = set()
    sizes_with_predecessor = set()
    for s1, s2 in pred_facts:
        predecessors[s2] = s1
        successors[s1] = s2
        all_sizes.add(s1)
        all_sizes.add(s2)
        sizes_with_predecessor.add(s2)

    root_size = None
    for size in all_sizes:
        if size not in sizes_with_predecessor:
            root_size = size
            break

    if root_size is not None:
        level = 0
        current_size = root_size
        size_levels[current_size] = level
        while current_size in successors:
            level += 1
            current_size = successors[current_size]
            size_levels[current_size] = level

    return size_levels, successors


class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum required actions for each package that is not yet at its
        goal location. For a package on the ground, the estimated cost is
        1 (pick-up) + shortest_path(current_location, goal_location) + 1 (drop).
        For a package inside a vehicle, the estimated cost is
        shortest_path(vehicle_location, goal_location) + 1 (drop).
        The shortest paths are precomputed using BFS on the road network.
        This heuristic is not admissible as it ignores vehicle capacity
        constraints and potential synergies (multiple packages per vehicle),
        and the cost of moving a vehicle to a package's location if needed.

    Assumptions:
        - The road network defined by `(road l1 l2)` facts is static.
        - The size hierarchy defined by `(capacity-predecessor s1 s2)` facts
          forms a simple chain (e.g., c0 < c1 < c2 ...). This hierarchy is
          parsed but not directly used in the heuristic calculation itself,
          as the heuristic simplifies capacity effects.
        - All packages implicitly require one unit of capacity for pick-up.
        - The problem is solvable (goal locations are reachable from initial
          package/vehicle locations via the road network). Unreachable goals
          will result in infinite heuristic value.
        - Every package is either `at` a location or `in` a vehicle in any
          valid state reachable from the initial state.
        - Objects appearing in goal `(at ...)` facts are packages. Objects
          appearing in state `(at ...)` facts that are not goal objects are
          vehicles or other locatables (vehicles are the only other locatables
          in this domain with 'at' facts).

    Heuristic Initialization:
        The constructor `__init__` receives the `Task` object. It extracts
        static information:
        1.  The road network graph from `(road l1 l2)` facts.
        2.  All locations from the road network facts.
        3.  All-pairs shortest paths between locations using BFS.
        4.  The goal location for each package from the `goals` set.
        5.  The size hierarchy from `(capacity-predecessor s1 s2)` facts,
            mapping size strings to integer levels (c0=0, c1=1, ...). This
            is parsed but not used in the heuristic calculation logic.

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method computes the heuristic value for a given state.
        1.  Initialize total heuristic `h = 0`.
        2.  Parse the current `state` to determine:
            -   The location of each package that is `(at p l)`.
            -   Which package is `(in p v)` and the location of vehicle `v`.
            -   The location of each vehicle `(at v l)`.
        3.  Iterate through each package `p` that is listed in the `goals`.
        4.  Retrieve the goal location `l_goal` for package `p` from the
            precomputed goal locations (`self.goal_locations`).
        5.  Determine the current status and location of package `p`:
            -   If `p` is found in `package_locations`: Its current location is `l_current = package_locations[p]`, and it is on the ground (`is_in_vehicle = False`).
            -   If `p` is found in `package_in_vehicle`: The vehicle is `v = package_in_vehicle[p]`. Find the vehicle's location `l_v = vehicle_locations.get(v)`. The package's current location is effectively `l_v`, and it is inside a vehicle (`is_in_vehicle = True`). If the vehicle's location is not found (`l_v is None`), the state is inconsistent, return `float('inf')`.
            -   If `p` is not found in either `package_locations` or `package_in_vehicle`, the state is inconsistent, return `float('inf')`.
        6.  If the package is on the ground (`not is_in_vehicle`) and its current location (`current_location`) is the same as `l_goal`, the package is already at its destination. Its cost contribution is 0. Continue to the next package.
        7.  If the package is not yet at the goal location (either on the ground elsewhere or in a vehicle):
            -   Retrieve the shortest path distance `dist` from the package's current effective location (`current_location`) to `l_goal` from the precomputed table (`self.shortest_paths`).
            -   If the path is not found (`dist is None`), the goal location is unreachable from the package's current location/vehicle's location. Return `float('inf')` to indicate an unsolvable state.
            -   Calculate the cost contribution for this package:
                -   If the package is `is_in_vehicle`: Cost = `dist (drive actions) + 1 (drop action)`.
                -   If the package is on the ground (`not is_in_vehicle`): Cost = `1 (pick-up action) + dist (drive actions) + 1 (drop action)`.
            -   Add this cost contribution to the total heuristic `h`.
        8.  After iterating through all goal packages, return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.static_facts = task.static
        self.goals = task.goals

        # 1-3: Build road graph and compute shortest paths
        self.road_graph, self.locations = build_road_graph(self.static_facts)
        self.shortest_paths = compute_shortest_paths(self.road_graph, self.locations)

        # 4: Extract goal locations for packages
        self.goal_locations = {}
        for goal_fact_string in self.goals:
            if goal_fact_string.startswith('(at '):
                _, package, location = parse_fact(goal_fact_string)
                self.goal_locations[package] = location
            # Ignore other goal types if any (domain only has 'at' goals for packages)

        # 5: Build size hierarchy (parsed but not used in this simple heuristic)
        # This is included as per the requirement to extract static info.
        self.size_levels, self.size_successors = build_size_hierarchy(self.static_facts)

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated cost (integer or float('inf')).
        """
        h = 0
        package_locations = {} # {package_name: location_name} if at location
        package_in_vehicle = {} # {package_name: vehicle_name} if in vehicle
        vehicle_locations = {} # {vehicle_name: location_name}

        # Parse current state facts
        for fact_string in state:
            parsed = parse_fact(fact_string)
            predicate = parsed[0]
            if predicate == 'at':
                _, obj, loc = parsed
                # Infer type: Assume objects in goals are packages, others with 'at' are vehicles.
                if obj in self.goal_locations:
                     package_locations[obj] = loc
                else:
                     vehicle_locations[obj] = loc
            elif predicate == 'in':
                _, package, vehicle = parsed
                package_in_vehicle[package] = vehicle
            # Ignore 'capacity' facts for this simple heuristic calculation

        # Calculate cost for each package that needs to reach a goal location
        for package, l_goal in self.goal_locations.items():
            current_location = None
            is_in_vehicle = False

            if package in package_locations:
                current_location = package_locations[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                current_location = vehicle_locations.get(vehicle) # Use .get for safety
                is_in_vehicle = True
                if current_location is None:
                    # Vehicle location unknown - inconsistent state?
                    # print(f"Warning: Vehicle {vehicle} containing package {package} has no location in state.", file=sys.stderr)
                    return float('inf') # Indicate unsolvable from this state?
            else:
                 # Package not found in 'at' or 'in' facts. Inconsistent state.
                 # print(f"Warning: Package {package} not found in 'at' or 'in' facts in state.", file=sys.stderr)
                 return float('inf') # Indicate unsolvable?

            # If package is AT the goal location, cost is 0 for this package
            if not is_in_vehicle and current_location == l_goal:
                 continue # Package is at goal, no cost needed for this package

            # If package is not at the goal location (either on ground elsewhere or in vehicle)
            dist = self.shortest_paths.get((current_location, l_goal))

            if dist is None:
                # Goal location is unreachable from the package's current location/vehicle's location
                return float('inf') # Indicate unsolvable from this state

            if is_in_vehicle:
                # Package is in a vehicle at current_location, needs to go to l_goal
                # Cost = drive from current_location to l_goal + drop at l_goal
                h += dist + 1 # 1 for drop action
            else:
                # Package is on the ground at current_location, needs to go to l_goal
                # Cost = pick-up at current_location + drive from current_location to l_goal + drop at l_goal
                h += 1 + dist + 1 # 1 for pick-up, 1 for drop

        return h
