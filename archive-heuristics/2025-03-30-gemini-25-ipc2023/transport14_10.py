import collections
import math
from fnmatch import fnmatch
# Assuming Heuristic base class is available in the specified path
# from heuristics.heuristic_base import Heuristic
# If the Heuristic base class is not available, this class can stand alone
# but won't inherit from a base class. For the purpose of this example,
# we assume it's available.
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts safely
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space. Handles potential errors.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    Returns an empty list if the format is invalid.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    Estimates the cost to reach the goal state by summing the estimated costs
    for moving each package to its target location specified in the goal. The cost
    for a package is calculated based on its current state (at a location or in a
    vehicle). It includes the necessary pick-up (if applicable), drive, and drop
    actions. The driving cost is based on the shortest path distance in the road
    network graph. This heuristic is designed for greedy best-first search and is
    not necessarily admissible.

    # Assumptions
    - The goal is specified solely by `(at package location)` predicates for a subset of packages.
    - The road network defined by `(road l1 l2)` facts is static and undirected.
      If roads are directed, the graph construction in `__init__` needs adjustment.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored.
    - The cost of moving a vehicle *to* a package's location for pickup is ignored.
    - Vehicle contention or optimal assignment is not modeled; the heuristic assumes
      an abstract transport capability is always available when needed.
    - All locations involved in the problem are connected via the road network if
      the problem instance is solvable. Disconnected components are handled (infinite distance).
    - Objects found in `(at ...)` facts that are not packages listed in the goal
      are assumed to be vehicles for the purpose of locating packages `(in vehicle)`.

    # Heuristic Initialization
    - Parses the task's goal conditions (`task.goals`) to identify the target location
      for each package specified in the goal. Stores these in `self.goal_locations`.
    - Stores the set of all packages that have a defined goal location in `self.packages`.
    - Parses the task's static facts (`task.static`) to build an undirected graph
      representation of the road network (locations and connections).
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS) for
      all locations identified in the network. Unreachable pairs have infinite distance.
    - Stores the computed distances in a nested dictionary `self.dist[loc1][loc2]`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state (`node.state`) already satisfies all goal conditions
       (`self.goals`). If yes, the heuristic value is 0, indicating the goal is reached.
    2. Initialize the total heuristic estimate `h` to 0.0 (using float for infinity).
    3. Parse the current state (`node.state`) to determine the status of each package
       and the location of each vehicle:
       - Create `current_package_pos`: maps package `p` -> `('at', l)` or `('in', v)`.
       - Create `vehicle_location`: maps vehicle `v` -> location `l`.
       - Iterate through facts in the state:
         - If `(at obj loc)`: If `obj` is in `self.packages` (a package with a goal),
           store `('at', loc)` in `current_package_pos[obj]`. Otherwise, assume `obj`
           is a vehicle and store its location in `vehicle_location[obj]`.
         - If `(in p v)`: If `p` is in `self.packages`, store `('in', v)` in
           `current_package_pos[p]`.
    4. Iterate through each package `p` in `self.packages` (packages with goals):
       a. Retrieve the goal location `g_loc` for `p` from `self.goal_locations`.
       b. Get the current status `(type, value)` of package `p` from `current_package_pos`.
          If `p` is not found in the current state representation (e.g., missing `at` or `in` fact),
          consider the state inconsistent or the goal unreachable; return infinity.
       c. **Case 1: Package `p` is already at its goal location `g_loc`.**
          - If status is `('at', g_loc)`, this package contributes 0 to `h`.
       d. **Case 2: Package `p` is at location `l` (`l != g_loc`).**
          - Status is `('at', l)`.
          - Add 1.0 to `h` (cost for `pick-up`).
          - Find shortest path distance `d = self.dist[l].get(g_loc, math.inf)`.
            The `.get` handles cases where `l` or `g_loc` might not have been in the original graph,
            defaulting to infinity. If `d` is infinity, the goal is unreachable; return infinity.
          - Add `d` to `h` (cost for `drive`).
          - Add 1.0 to `h` (cost for `drop`).
       e. **Case 3: Package `p` is inside vehicle `v`.**
          - Status is `('in', v)`.
          - Find the location `l_v` of vehicle `v` from `vehicle_location`. If `v` or
            its location is unknown (not found in `vehicle_location`), return infinity
            (inconsistent state).
          - Find shortest path distance `d = self.dist[l_v].get(g_loc, math.inf)`.
            If `d` is infinity, return infinity.
          - Add `d` to `h` (cost for `drive`).
          - Add 1.0 to `h` (cost for `drop`).
    5. After iterating through all packages, if `h` reached infinity at any point, return infinity.
    6. If `h` is 0 but the state is not a goal state (checked in step 1), return 1.
       This ensures non-goal states always have a positive heuristic value, preventing
       the search from getting stuck if the heuristic incorrectly estimates 0 for a
       non-goal state (e.g., if `self.packages` was empty).
    7. Otherwise, return the calculated heuristic value rounded to the nearest integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic: finds package goals, builds road graph, computes APSP.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {} # package_name -> goal_location_name
        self.packages = set()      # Set of package names mentioned in goals
        locations = set()          # Set of all location names found
        adj = collections.defaultdict(set) # Adjacency list for location graph

        # Parse goals to find package destinations and identify relevant packages/locations
        for goal in self.goals:
            parts = get_parts(goal)
            # Expect goals like: (at package location)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, loc = parts[1], parts[2]
                self.goal_locations[package] = loc
                self.packages.add(package)
                locations.add(loc)

        # Parse static facts for road network and identify all locations involved
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            # Expect roads like: (road loc1 loc2)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                adj[l1].add(l2)
                adj[l2].add(l1) # Assuming undirected roads based on domain structure/examples
                locations.add(l1)
                locations.add(l2)
            # Add locations mentioned in other static facts if necessary (e.g., capacity predicates mention vehicles, but not locations)
            # If locations can exist without being in roads or goals, they might be missed here.
            # A more robust approach might involve parsing task.objects if available.

        # Compute All-Pairs Shortest Paths (APSP) using BFS
        self.dist = collections.defaultdict(lambda: collections.defaultdict(lambda: math.inf))

        # Initialize distances for all known locations before running BFS
        for loc in locations:
            self.dist[loc][loc] = 0

        # Run BFS from each location to compute shortest path distances
        for start_node in locations:
            # Check if start_node exists in adj, otherwise it's isolated
            if start_node not in adj and not any(start_node in neighbors for neighbors in adj.values()):
                 continue # Isolated node, distances already set to infinity except for self.dist[start_node][start_node] = 0

            queue = collections.deque([(start_node, 0)])
            # visited_bfs stores nodes visited *in this specific BFS run* to avoid cycles
            visited_bfs = {start_node}

            while queue:
                current_node, d = queue.popleft()

                # Explore neighbors
                for neighbor in adj.get(current_node, set()): # Use .get for safety
                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        # Update distance from start_node to neighbor
                        self.dist[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        h_value = 0.0 # Use float for calculations involving infinity

        # Check if the state is a goal state first for efficiency
        # A state is a goal state if all goal predicates hold true in the state set
        is_goal_state = self.goals <= state
        if is_goal_state:
            return 0

        # Parse current state to find package and vehicle locations
        current_package_pos = {} # p -> ('at', l) or ('in', v)
        vehicle_location = {}    # v -> l

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages: # Check if the object is a package with a goal
                    current_package_pos[obj] = ('at', loc)
                else:
                    # Assume other objects at locations are vehicles relevant to the problem
                    vehicle_location[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.packages: # Check if the package is one we track
                    current_package_pos[package] = ('in', vehicle)

        # Calculate heuristic sum based on each package's state
        for p in self.packages:
            # Ensure package has a goal location (should always be true based on init)
            if p not in self.goal_locations:
                 # This case should ideally not happen if initialization is correct
                 continue
            goal_loc = self.goal_locations[p]

            # Ensure package current position is known from the state parsing
            if p not in current_package_pos:
                # If a package listed in goals is not found in the state ('at' or 'in'),
                # the state might be invalid or the problem definition inconsistent.
                # Returning infinity signals an issue or unreachability.
                return math.inf

            p_state_type, p_state_val = current_package_pos[p]

            # Case 1: Package is already at its goal location
            if p_state_type == 'at' and p_state_val == goal_loc:
                # This package's goal is met, contribution is 0
                continue

            # Case 2: Package is at a non-goal location 'l'
            elif p_state_type == 'at':
                current_loc = p_state_val
                # Retrieve distance, default to infinity if locations are unknown or disconnected
                distance = self.dist[current_loc].get(goal_loc, math.inf)
                if distance == math.inf:
                    # If distance is infinite, the goal is unreachable for this package
                    return math.inf
                # Cost = pickup(1) + drive(distance) + drop(1)
                h_value += (1.0 + distance + 1.0)

            # Case 3: Package is inside vehicle 'v'
            elif p_state_type == 'in':
                vehicle = p_state_val
                if vehicle not in vehicle_location:
                    # The vehicle carrying the package doesn't have a known location in the state.
                    return math.inf
                current_vehicle_loc = vehicle_location[vehicle]
                # Retrieve distance from vehicle's location to package's goal
                distance = self.dist[current_vehicle_loc].get(goal_loc, math.inf)
                if distance == math.inf:
                    # Goal is unreachable from the vehicle's current location
                    return math.inf
                # Cost = drive(distance) + drop(1)
                h_value += (distance + 1.0)

        # Final checks and return value
        if h_value == math.inf:
            # Return infinity if any package goal was deemed unreachable
            return math.inf
        else:
            # If h_value is 0 but it's not a goal state (checked at the beginning),
            # return 1. This ensures search makes progress even if the heuristic
            # estimate is 0 for a non-goal state (e.g., if no packages had goals).
            if h_value == 0 and not is_goal_state:
                 return 1
            # Otherwise, return the calculated cost, rounded to the nearest integer.
            return int(round(h_value))

