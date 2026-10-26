from collections import deque, defaultdict
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    by summing the estimated costs for each package that is not yet at its
    goal location. For each such package, the cost is estimated as:
    - If the package is currently at a location (not in a vehicle):
      1 (pick-up) + shortest_path_distance(current_location, goal_location) + 1 (drop)
    - If the package is currently in a vehicle:
      shortest_path_distance(current_location_of_vehicle, goal_location) + 1 (drop)
    This heuristic ignores vehicle capacity constraints and assumes any vehicle
    can be used to transport any package. It also assumes a vehicle is always
    available or can reach the package's location. The shortest path distances
    between locations are precomputed using BFS on the road network.

    Assumptions:
    - The input state and task objects conform to the structure described
      in the problem description (frozenset of strings for state/static,
      Task object with goals as frozenset of strings).
    - Goal facts for packages are always of the form '(at package location)'.
    - Road network is static and provided in the static facts.
    - Capacity and capacity-predecessor facts are present but ignored by
      this specific heuristic calculation.
    - Objects starting with 'p' are packages and objects starting with 'v'
      are vehicles. This convention is used to distinguish object types
      from fact strings without full PDDL parsing.
    - All locations mentioned in road facts, goal facts, and initial state
      'at' facts are considered in the road network graph for distance
      calculations.

    Heuristic Initialization:
    During initialization, the heuristic processes the static information
    and the goal state from the Task object.
    1. It identifies the goal location for each package from the goal facts
       '(at package goal_location)'. These packages are the only ones
       considered by the heuristic.
    2. It builds an adjacency list representation of the road network graph
       from the '(road l1 l2)' facts. It also collects all unique locations
       mentioned in road facts, goal facts, and initial state 'at' facts.
    3. It computes the shortest path distance between every pair of identified
       locations using Breadth-First Search (BFS) starting from each location.
       These distances are stored in a nested dictionary `distances[l1][l2]`
       for quick lookup. Unreachable locations will not have finite distances.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Initialize the total heuristic value `h` to 0.
    2. Create temporary dictionaries to store the current location of packages
       and vehicles, and which package is in which vehicle, by iterating
       through the state facts. This is done efficiently in a single pass
       over the state facts.
    3. For each package `p` that has a goal location specified in the task's goals:
        a. Determine the current location `l_current` of package `p`. This is
           either directly from an `(at p l_current)` fact or indirectly
           from an `(in p v)` fact combined with an `(at v l_current)` fact.
           Keep track if the package is currently inside a vehicle.
        b. If the package `p` is already at its goal location (i.e., `l_current`
           is the goal location), the cost for this package is 0. Continue
           to the next package.
        c. If the package `p` is not at its goal location and its current
           location `l_current` was successfully determined:
           - Retrieve the goal location `l_goal` for package `p` (precomputed
             during initialization).
           - Look up the shortest path distance `dist` from `l_current` to
             `l_goal` using the precomputed distances. If the goal is
             unreachable from the current location, `dist` will be effectively
             infinity (represented by a large number).
           - If `dist` is infinite, add a large penalty to `h` as this state
             likely leads to an unsolvable path for this package.
           - If `dist` is finite:
             - If `p` is currently in a vehicle: The estimated cost for this
               package is `dist` (drive) + 1 (drop). Add this cost to `h`.
             - If `p` is currently at a location (not in a vehicle): The
               estimated cost for this package is 1 (pick-up) + `dist` (drive)
               + 1 (drop). Add this cost to `h`.
        d. If the package's location could not be determined from the state
           facts (e.g., it's not 'at' any location and not 'in' any vehicle),
           this indicates an unexpected state structure. The heuristic
           currently does not add cost for such packages, assuming valid states.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.package_goal_location = {}
        self.road_graph = defaultdict(list)
        self.locations = set()
        self.distances = {} # Stores shortest path distances: l1 -> {l2: dist}

        # Parse goal facts to find package goal locations FIRST
        # Add goal locations to the set of known locations
        for goal_fact_str in task.goals:
            predicate, args = self._parse_fact(goal_fact_str)
            if predicate == 'at' and len(args) == 2: # Expecting (at package location)
                package, goal_loc = args
                # Assuming objects starting with 'p' are packages
                if package.startswith('p'):
                    self.package_goal_location[package] = goal_loc
                    self.locations.add(goal_loc)
            # Ignore other types of goal facts if any

        # Parse static facts to build road graph and add locations
        for fact_str in task.static:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'road' and len(args) == 2: # Expecting (road l1 l2)
                l1, l2 = args
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)
            # Ignore capacity-predecessor and other static facts for this heuristic

        # Add locations from initial state 'at' facts
        # This ensures BFS is run from all starting locations of locatables
        for fact_str in task.initial_state:
             predicate, args = self._parse_fact(fact_str)
             if predicate == 'at' and len(args) == 2: # Expecting (at obj loc)
                 obj, loc = args
                 self.locations.add(loc)

        # Compute all-pairs shortest paths for all identified locations
        # Only compute if there are locations
        if self.locations:
            for start_loc in self.locations:
                self.distances[start_loc] = self._bfs(start_loc)

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove leading/trailing parentheses and split by space
        # Handles cases like '(at p1 l1)' -> ['at', 'p1', 'l1']
        # and '(capacity-predecessor c0 c1)' -> ['capacity-predecessor', 'c0', 'c1']
        parts = fact_str.strip("()").split()
        if not parts:
            return None, [] # Handle empty string case, though unlikely
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _bfs(self, start_node):
        """Helper function to perform BFS from a start node."""
        distances = {start_node: 0}
        queue = deque([start_node])
        # Only explore if the start_node is actually a known location
        if start_node not in self.locations:
             # This node is not part of the relevant locations set
             return {start_node: 0}

        known_locations = self.locations # Use the set of all identified locations

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check neighbors only if current_node has outgoing edges in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Only process neighbors that are known locations
                    if neighbor in known_locations and neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        h = 0
        package_current_location = {} # Map package to its current location string
        package_in_vehicle = {} # Map package to vehicle string if in vehicle
        vehicle_locations = {} # Map vehicle to its current location string

        # Find current status of locatables in the state in a single pass
        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at' and len(args) == 2:
                obj, loc = args
                # Check if obj is a package we care about or a vehicle
                if obj in self.package_goal_location:
                     package_current_location[obj] = loc
                # Assuming objects starting with 'v' are vehicles
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc

            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args
                # Only track packages we care about (those in the goal)
                if package in self.package_goal_location:
                    package_in_vehicle[package] = vehicle

        # Calculate heuristic for each package that needs to be moved
        for package, goal_loc in self.package_goal_location.items():
            current_loc = None
            is_in_vehicle = False

            # Determine package's current location and status (in vehicle or at location)
            if package in package_current_location:
                current_loc = package_current_location[package]
                is_in_vehicle = False # Explicitly state not in vehicle
            elif package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Need vehicle's location to know package's location
                if vehicle in vehicle_locations:
                    current_loc = vehicle_locations[vehicle]
                    is_in_vehicle = True
                # else: Vehicle location unknown. current_loc remains None.

            # If package is already at goal, cost is 0 for this package
            if current_loc == goal_loc:
                continue

            # If package needs to be moved and its location is known
            if current_loc is not None:
                # Get shortest distance from current_loc to goal_loc
                # Use get with default dict for robustness if current_loc wasn't a BFS start node
                # or if goal_loc wasn't reached from current_loc's BFS.
                dist_map = self.distances.get(current_loc, {})
                dist = dist_map.get(goal_loc, math.inf) # Default to inf if path not found

                if dist == math.inf:
                     # Goal location unreachable from current location in the road network
                     # This state is likely a dead end or part of an unsolvable path.
                     # Assign a large penalty.
                     h += 1000000
                else:
                    # Finite distance found
                    if is_in_vehicle:
                        # Package is in a vehicle, needs drive + drop
                        h += dist + 1
                    else:
                        # Package is at a location, needs pick-up + drive + drop
                        h += 1 + dist + 1
            # else: Package location could not be determined from the state.
            # This shouldn't happen in valid states generated by the planner.
            # We add 0 cost for this package in this case, assuming it's a valid state
            # and the package will eventually appear at a known location.
            # A more robust heuristic might add a penalty here too.

        return h
