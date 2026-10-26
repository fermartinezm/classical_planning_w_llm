from collections import defaultdict, deque

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...).
    Assumes fact string is in the format '(predicate arg1 arg2 ...)'.
    """
    # Remove surrounding parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function to build the road network graph
def build_road_graph(static_facts):
    """
    Builds a directed graph representing the road network from static facts.
    Nodes are locations, edges are roads.
    Returns the graph as a defaultdict(list) and a list of all locations.
    """
    graph = defaultdict(list)
    locations = set()
    for fact_string in static_facts:
        fact = parse_fact(fact_string)
        if fact[0] == 'road':
            l1, l2 = fact[1], fact[2]
            graph[l1].append(l2)
            locations.add(l1)
            locations.add(l2)
    return graph, list(locations)

# Helper function to compute all-pairs shortest paths using BFS
def compute_all_pairs_shortest_paths(graph, locations):
    """
    Computes shortest path distances between all pairs of locations
    in the road graph using BFS.
    Returns a dictionary distances[start_loc][end_loc] = distance.
    Assigns a large value (1000) for unreachable locations.
    """
    distances = {}
    # Use a large value for unreachable locations
    unreachable_dist = 1000 # Large penalty per unreachable package

    for start_node in locations:
        distances[start_node] = {}
        q = deque([(start_node, 0)])
        visited = {start_node}
        distances[start_node][start_node] = 0 # Distance to self is 0

        while q:
            current_node, dist = q.popleft()

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start_node][neighbor] = dist + 1
                    q.append((neighbor, dist + 1))

        # Fill in unreachable locations with the large value
        for loc in locations:
             if loc not in distances[start_node]:
                 distances[start_node][loc] = unreachable_dist

    return distances

# Helper function to infer object types from static and goal facts
def infer_object_types(static_facts, goal_facts):
    """
    Infers object types (locations, sizes, vehicles, packages)
    based on the predicates they appear in within static and goal facts.
    """
    locations = set()
    sizes = set()
    vehicles = set()
    locatables = set() # Can be vehicles or packages

    for fact_string in static_facts:
        fact = parse_fact(fact_string)
        if fact[0] == 'road':
            locations.add(fact[1])
            locations.add(fact[2])
        elif fact[0] == 'capacity-predecessor':
            sizes.add(fact[1])
            sizes.add(fact[2])
        elif fact[0] == 'capacity':
            vehicles.add(fact[1])
            sizes.add(fact[2])

    for fact_string in goal_facts:
         fact = parse_fact(fact_string)
         if fact[0] == 'at':
             locatables.add(fact[1])
             locations.add(fact[2]) # Goal locations are locations
         elif fact[0] == 'in':
             locatables.add(fact[1]) # Package
             vehicles.add(fact[2]) # Vehicle

    # Packages are locatables that are not vehicles
    packages = locatables - vehicles

    return locations, sizes, vehicles, packages

# Helper function to extract goal locations for packages
def extract_goal_locations(goal_facts, packages):
    """
    Extracts the target location for each package from the goal facts.
    Only considers 'at' goals for packages.
    Returns a dictionary mapping package name to goal location name.
    """
    goal_locs = {}
    for fact_string in goal_facts:
        fact = parse_fact(fact_string)
        if fact[0] == 'at' and fact[1] in packages:
            goal_locs[fact[1]] = fact[2]
        # We only care about 'at' goals for packages in this heuristic
    return goal_locs


class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        estimated costs for each package that is not yet at its goal location.
        The estimated cost for a package is calculated based on a relaxation
        of the problem where vehicle capacity is ignored, and vehicles are
        assumed to be available wherever needed for pickup or drop-off.
        The cost for a package needing transport is estimated as:
        1 (pickup) + shortest_path_distance (drive) + 1 (drop).
        If the package is already in a vehicle, the pickup cost is omitted.
        If the package is in a vehicle at the goal location, only the drop cost is counted.
        The shortest path distances between locations are pre-calculated using BFS
        on the static road network.

    Assumptions:
        - The goal is always to have packages at specific locations, represented
          by '(at package location)' facts in the goal state.
        - The road network is static and defined by 'road' facts.
        - Objects can be identified as packages, vehicles, locations, or sizes
          by examining the predicates they appear in within the static facts
          and goal facts. Specifically, objects appearing in 'road' facts are
          locations, objects appearing in 'capacity' facts are vehicles (first arg)
          and sizes (second arg), objects appearing in 'capacity-predecessor'
          facts are sizes, and objects appearing as the first argument of 'at'
          or 'in' facts in the goal that are not vehicles (identified via capacity)
          are considered packages.
        - The road network is reasonably connected such that relevant locations
          are reachable from each other. Unreachable locations are assigned a
          large penalty in the distance calculation.
        - Valid states derived from the initial state and operators will always
          contain either an '(at package location)' or '(in package vehicle)'
          fact for every package, and an '(at vehicle location)' fact for every vehicle.

    Heuristic Initialization:
        1. Parse static facts to build the road network graph (adjacency list).
        2. Identify all locations from the road graph.
        3. Compute all-pairs shortest path distances between locations using BFS.
           Store these distances in a dictionary `self.distances`. Unreachable
           locations are assigned a large cost (1000).
        4. Infer object types (locations, sizes, vehicles, packages) from static
           and goal facts using the logic described in Assumptions.
        5. Extract the goal location for each package from the goal facts.
           Store these in a dictionary `self.goal_locations`. Only packages
           with an '(at package location)' goal fact are considered.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Create dictionaries to track the current status of each package
           (`package_current_info`: {package_name: {'status': 'at'/'in', 'loc': loc, 'veh': veh}})
           and the current location of each vehicle (`vehicle_locations`: {vehicle_name: loc})
           in the given state.
        3. Iterate through each fact string in the current state:
           - Parse the fact string.
           - If the fact is `(at ?x ?l)`:
             - If `?x` is identified as a package during initialization, record its location in `package_current_info`.
             - If `?x` is identified as a vehicle during initialization, record its location in `vehicle_locations`.
           - If the fact is `(in ?p ?v)`:
             - If `?p` is identified as a package and `?v` as a vehicle, record that `?p` is in `?v` in `package_current_info`.
        4. Iterate through each package `p` that has an '(at p l_goal)' goal fact (i.e., each package in `self.goal_locations`).
           - Get the goal location `l_goal` for package `p` from `self.goal_locations`.
           - Construct the goal fact string `'(at p l_goal)'`.
           - If this goal fact string is present in the current state, this package is already at its goal, so its contribution to the heuristic is 0. Continue to the next package.
           - If the goal fact is NOT in the state, this package needs to be moved. Determine its current status from `package_current_info`.
           - If package `p` is currently `(at p l_current)` (found in `package_current_info`):
             - This package needs to be picked up, transported, and dropped.
             - The estimated cost is 1 (pickup) + shortest_path_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
             - Look up the distance `dist` from `self.distances[l_current][l_goal]`.
             - If `dist` is the large unreachable value (1000), add 1000 to `h` (penalty for unreachable goal).
             - Otherwise, add `1 + dist + 1` to `h`.
           - If package `p` is currently `(in p v)` (found in `package_current_info`):
             - This package needs to be transported (if not already at goal location) and dropped.
             - Find the current location `l_current` of vehicle `v` from `vehicle_locations`.
             - If `l_current` is not found (should not happen in valid states), add a large penalty (1000) to `h` and continue.
             - If `l_current` is not `l_goal`:
               - The estimated cost is shortest_path_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
               - Look up the distance `dist` from `self.distances[l_current][l_goal]`.
               - If `dist` is the large unreachable value (1000), add 1000 to `h` (penalty).
               - Otherwise, add `dist + 1` to `h`.
             - If `l_current` is `l_goal`:
               - The estimated cost is 1 (drop). Add 1 to `h`.
           - If a package from `self.goal_locations` is not found in `package_current_info` (i.e., not `at` or `in` in the state), this indicates an invalid state or unhandled case. Add a large penalty (1000) to `h`.
        5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        # 1. Build road graph and get locations
        self.road_graph, self.locations = build_road_graph(task.static)

        # 2. Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.road_graph, self.locations)
        self.unreachable_penalty = 1000 # Use the same penalty value

        # 3. Infer object types
        self.locations, self.sizes, self.vehicles, self.packages = infer_object_types(task.static, task.goals)

        # 4. Extract goal locations for packages
        self.goal_locations = extract_goal_locations(task.goals, self.packages)

    def __call__(self, state):
        h = 0
        package_current_info = {} # {package_name: {'status': 'at'/'in', 'loc': loc, 'veh': veh}}
        vehicle_locations = {} # {vehicle_name: loc}

        # Populate current info from state
        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at':
                obj_name, loc_name = fact[1], fact[2]
                if obj_name in self.packages:
                    package_current_info[obj_name] = {'status': 'at', 'loc': loc_name}
                elif obj_name in self.vehicles:
                    vehicle_locations[obj_name] = loc_name
            elif fact[0] == 'in':
                pkg_name, veh_name = fact[1], fact[2]
                # Ensure both package and vehicle are known types before adding
                if pkg_name in self.packages and veh_name in self.vehicles:
                     package_current_info[pkg_name] = {'status': 'in', 'veh': veh_name}

        # Calculate cost for packages NOT at their goal location
        for pkg, goal_loc in self.goal_locations.items():
            # Check if the goal fact (at pkg goal_loc) is already true
            goal_fact_string = f"'(at {pkg} {goal_loc})'"
            if goal_fact_string in state:
                continue # This package is already at its goal location

            # If not at goal, calculate cost
            current_info = package_current_info.get(pkg)

            if current_info is None:
                 # This package is not found in the state (neither at nor in).
                 # This shouldn't happen in valid states derived from initial state.
                 # Assign a large penalty as it's likely an unsolvable path or error.
                 h += self.unreachable_penalty
                 continue

            current_status = current_info['status']

            if current_status == 'at':
                current_loc = current_info['loc']
                # Needs pickup (1) + drive (dist) + drop (1)
                dist = self.distances.get(current_loc, {}).get(goal_loc, self.unreachable_penalty)
                if dist == self.unreachable_penalty:
                     h += self.unreachable_penalty # Unreachable goal location from current location
                else:
                     h += 1 + dist + 1

            elif current_status == 'in':
                veh = current_info['veh']
                current_loc = vehicle_locations.get(veh)

                if current_loc is None:
                    # Vehicle location unknown (shouldn't happen in valid states)
                    h += self.unreachable_penalty
                    continue

                if current_loc != goal_loc:
                    # Needs drive (dist) + drop (1)
                    dist = self.distances.get(current_loc, {}).get(goal_loc, self.unreachable_penalty)
                    if dist == self.unreachable_penalty:
                         h += self.unreachable_penalty # Unreachable goal location from current location
                    else:
                         h += dist + 1
                else: # current_loc == goal_loc
                    # Needs drop (1)
                    h += 1

        return h
