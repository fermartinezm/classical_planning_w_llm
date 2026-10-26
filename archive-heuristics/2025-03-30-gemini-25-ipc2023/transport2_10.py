import sys
from collections import deque
# Ensure Heuristic base class is available. Adjust path as needed.
try:
    # Attempt to import from the standard location within the planner framework
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the standard import fails (e.g., running standalone)
    try:
        # Assumes heuristic_base.py is in the same directory or Python path
        from heuristic_base import Heuristic
    except ImportError:
        # If it's truly unavailable, define a dummy base class to allow script execution
        print("Warning: Heuristic base class not found. Using dummy base.", file=sys.stderr)
        class Heuristic:
            def __init__(self, task): pass
            def __call__(self, node): return 0


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Removes surrounding parentheses and splits by space.
    Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    """
    # Return an empty list if fact is not a valid string or is empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove leading '(' and trailing ')' before splitting
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from the current state in the transport domain. It calculates an estimated cost
    for each package that is not yet at its goal location and sums these costs.
    The cost for a single package considers the minimum number of 'drive' actions
    needed (based on shortest paths in the road network) and the necessary
    'pick-up' and 'drop' actions. It is designed for use with Greedy Best-First Search
    and therefore does not need to be admissible, prioritizing informativeness and
    computational efficiency over admissibility.

    # Assumptions
    - The road network defined by `(road l1 l2)` predicates potentially forms
      a connected graph for relevant locations. If locations required to solve the
      problem are disconnected, the heuristic might return infinity.
    - Roads are assumed to be bidirectional, as is common in PDDL benchmarks
      and suggested by the example instances (e.g., both `(road l1 l2)` and
      `(road l2 l1)` are present). The implementation reflects this.
    - The heuristic completely ignores vehicle capacity constraints (`capacity`
      and `capacity-predecessor` predicates). It assumes any vehicle can pick
      up any package if they are at the same location.
    - It assumes an "ideal" scenario for vehicle availability:
        - When a package needs picking up from a location, the cost calculation
          does not include the actions needed to drive a vehicle *to* that location first.
        - It focuses purely on the travel distance from the package's current effective location
          (either its 'at' location or the location of the vehicle it's 'in')
          to its goal location.
    - The heuristic sums costs independently for each package. This means it does not
      account for potential efficiencies like multiple packages sharing a single
      vehicle trip. This simplification makes it non-admissible but easier and faster
      to compute.

    # Heuristic Initialization
    - The constructor (`__init__`) preprocesses information from the task definition (`task`):
        - It parses `task.goals` to identify the target location specified by `(at package location)`
          facts for each package. This mapping is stored in `self.goal_locations`. It also
          collects the set of all package names involved in the goals (`self.packages`).
        - It parses `task.static` facts to build an adjacency list representation
          of the road network (`self.adj`). It collects all unique location names
          involved in roads, goals, or initial 'at' predicates into `self.locations`.
        - It computes all-pairs shortest path distances between all known locations
          using Breadth-First Search (BFS) starting from each location. The
          distances (representing the minimum number of 'drive' actions) are stored
          in `self.dist`. A distance of `float('inf')` indicates unreachability between two locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** The `__call__` method receives a `node` (representing a state in the search).
        It first checks if the current state (`node.state`) already satisfies all goal conditions
        using `self.task.goal_reached(node.state)`. If yes, the goal is reached, and the
        heuristic value is 0.
    2.  **State Parsing:** If the current state is not a goal state, the heuristic parses the set of
        facts in `node.state` to determine the current status and location of relevant objects:
        - It identifies the location of each object currently specified by an `(at ...)` fact.
        - It identifies which packages are inside which vehicles using `(in ...)` facts.
        - It implicitly identifies vehicles as objects that are 'at' a location but are not packages,
          or objects that appear as the second argument in an `(in ...)` fact.
        - It stores the location of each vehicle (`vehicle_location`) and the status of each package
          (`package_info`, indicating if it's 'at' a location or 'in' a vehicle).
    3.  **Package Cost Calculation:** The heuristic iterates through all packages `p` that have a
        defined goal location (`self.packages`).
        - For each package `p`, it retrieves its required `goal_loc` from `self.goal_locations`.
        - It looks up the current status of `p` (either `('at', current_loc)` or `('in', vehicle)`)
          from the parsed state information (`package_info`). If a goal package is missing from
          the current state, it indicates an anomaly, and the heuristic returns infinity.
        - **Case A: Package `p` is `at current_loc`:**
            - If `current_loc` is the same as `goal_loc`, this package is already in place,
              contributing 0 to the total heuristic cost.
            - If `current_loc` is different from `goal_loc`, the estimated cost for this package is:
              `1` (for the `pick-up` action) + `self.dist[current_loc][goal_loc]` (for `drive` actions)
              + `1` (for the `drop` action). If `goal_loc` is unreachable from `current_loc`
              (distance is infinity), the heuristic returns `float('inf')`.
        - **Case B: Package `p` is `in vehicle v`:**
            - It finds the current location of the vehicle `v`, denoted `vehicle_loc`, using the
              parsed state information. If the vehicle's location cannot be determined, it indicates
              an invalid state, and the heuristic returns infinity.
            - If `vehicle_loc` is the same as `goal_loc`, the package only needs to be dropped.
              The estimated cost for this package is `1` (for the `drop` action).
            - If `vehicle_loc` is different from `goal_loc`, the estimated cost is:
              `self.dist[vehicle_loc][goal_loc]` (for `drive` actions) + `1` (for the `drop` action).
              If `goal_loc` is unreachable from `vehicle_loc`, the heuristic returns `float('inf')`.
    4.  **Summation:** The costs estimated for each individual package that is not yet at its
        goal location are summed up to get the total heuristic value `h_cost`.
    5.  **Return Value:** The total sum `h_cost` is returned. If any package's goal was found
        to be unreachable during the calculation, `float('inf')` is returned, signaling
        that the current state might be a dead end or on an unsolvable path according to
        this heuristic's estimation.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations and package names from goal facts
        self.goal_locations = {} # Stores goal location for each package
        self.packages = set()    # Stores names of all packages mentioned in goals
        for goal in self.goals:
            parts = get_parts(goal)
            # Expect goals like '(at package location)'
            if len(parts) == 3 and parts[0] == "at":
                package, location = parts[1], parts[2]
                # Assume the first argument of 'at' in a goal is a package
                self.goal_locations[package] = location
                self.packages.add(package)

        # 2. Build road graph and identify all locations involved
        self.locations = set() # Stores names of all known locations
        self.adj = {}          # Adjacency list for the road graph (location -> set of neighbors)
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "road":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Store neighbors assuming roads are bidirectional
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # Add locations mentioned in goals or initial state 'at' predicates
        # This ensures locations without explicit roads are included in distance calculations
        # (distance will be 0 to self, infinity to others unless connected)
        for loc in self.goal_locations.values():
            self.locations.add(loc)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == "at":
                 # Add the location part (argument 2) if it's a known type or looks like one
                 # For simplicity, add all second arguments of 'at' predicates as potential locations
                 self.locations.add(parts[2])

        # 3. Compute all-pairs shortest paths using BFS
        # Initialize distances: inf everywhere except 0 from a node to itself
        self.dist = {loc: {other_loc: float('inf') for other_loc in self.locations} for loc in self.locations}

        for start_node in self.locations:
            # Basic check: ensure start_node is in the keys (it should be)
            if start_node not in self.dist: continue

            self.dist[start_node][start_node] = 0
            queue = deque([(start_node, 0)]) # Queue stores (node, distance_from_start)
            # Visited dictionary stores the shortest distance found so far to avoid cycles and redundant work
            visited = {start_node: 0}

            while queue:
                u, d = queue.popleft()

                # Explore neighbors only if the location has outgoing roads defined in adj
                if u in self.adj:
                    for v in self.adj[u]:
                        # Process neighbor v only if it hasn't been visited yet
                        if v not in visited:
                            visited[v] = d + 1
                            self.dist[start_node][v] = d + 1
                            queue.append((v, d + 1))

    def __call__(self, node):
        state = node.state
        # Check if goal is already reached
        if self.task.goal_reached(state):
            return 0

        h_cost = 0

        # Parse current state to find locations of packages and vehicles
        package_info = {} # Maps package -> ('at', location) or ('in', vehicle)
        vehicle_location = {} # Maps vehicle -> location

        # Identify objects that are 'at' a location
        objects_at_location = {} # obj -> loc
        # Identify packages inside vehicles
        packages_in_vehicle = {} # pkg -> vehicle

        for fact in state:
            parts = get_parts(fact)
            # Skip if parsing failed or fact is not of expected structure
            if not parts:
                continue
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                objects_at_location[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, vehicle = parts[1], parts[2]
                # Ensure the first argument is indeed a package we care about (i.e., has a goal)
                if pkg in self.packages:
                    packages_in_vehicle[pkg] = vehicle

        # Determine vehicle locations and package status
        # Assume anything 'at' a location that isn't a package relevant to the goal is a vehicle
        for obj, loc in objects_at_location.items():
            if obj not in self.packages:
                vehicle_location[obj] = loc # Tentatively mark as vehicle

        # Populate package_info based on 'at' or 'in' status for goal packages
        for package in self.packages:
            if package in packages_in_vehicle:
                package_info[package] = ('in', packages_in_vehicle[package])
            elif package in objects_at_location:
                package_info[package] = ('at', objects_at_location[package])
            else:
                # If a package required for the goal is not found 'at' or 'in' anything
                # in the current state, this indicates a potential problem (e.g., invalid state).
                # Returning infinity signals this branch is likely unproductive or invalid.
                # print(f"Warning: Goal package {package} not found in state {state}. Assigning inf cost.", file=sys.stderr)
                return float('inf')

        # Calculate cost for each package towards its goal
        for package in self.packages:
            goal_loc = self.goal_locations[package]
            # We already checked package exists in package_info above
            status, location_or_vehicle = package_info[package]

            if status == 'at':
                current_loc = location_or_vehicle
                if current_loc != goal_loc:
                    # Cost: pickup(1) + drive(dist) + drop(1)
                    # Use .get for safer dictionary access, defaulting to infinity if a location is somehow unknown
                    distance = self.dist.get(current_loc, {}).get(goal_loc, float('inf'))
                    if distance == float('inf'):
                        # Goal is unreachable for this package from its current location
                        return float('inf')
                    h_cost += (1 + distance + 1)

            elif status == 'in':
                vehicle = location_or_vehicle
                # Find the location of the vehicle this package is in
                if vehicle not in vehicle_location:
                    # The vehicle carrying the package doesn't have an 'at' predicate? Invalid state.
                    # print(f"Error: Location of vehicle {vehicle} carrying package {package} not found in state {state}. Assigning inf cost.", file=sys.stderr)
                    return float('inf')

                vehicle_loc = vehicle_location[vehicle]

                # If vehicle is already at the goal, only drop is needed.
                if vehicle_loc == goal_loc:
                    h_cost += 1 # Cost: drop(1)
                else:
                    # Cost: drive(dist) + drop(1)
                    distance = self.dist.get(vehicle_loc, {}).get(goal_loc, float('inf'))
                    if distance == float('inf'):
                        # Goal is unreachable via this vehicle from its current location
                        return float('inf')
                    h_cost += (distance + 1)

        # Return the total estimated cost
        # The cost calculation naturally results in non-negative values or infinity.
        return h_cost
