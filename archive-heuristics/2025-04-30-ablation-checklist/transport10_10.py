import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_road_graph(static_facts):
    """
    Build an adjacency list representation of the road network.

    Args:
        static_facts: A frozenset of static PDDL facts (strings).

    Returns:
        A tuple: (graph_adj_list, all_locations_set)
        graph_adj_list: dict mapping location -> set of connected locations.
        all_locations_set: set of all unique locations in the graph.
    """
    graph = collections.defaultdict(set)
    locations = set()
    for fact in static_facts:
        if match(fact, "road", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            graph[loc1].add(loc2)
            graph[loc2].add(loc1) # Assuming roads are bidirectional
            locations.add(loc1)
            locations.add(loc2)
    return dict(graph), locations

def bfs(start_loc, graph, all_locations):
    """
    Perform Breadth-First Search to find shortest distances from a start location.

    Args:
        start_loc: The starting location.
        graph: Adjacency list representation of the road network.
        all_locations: Set of all locations in the graph.

    Returns:
        A dictionary mapping location -> shortest distance from start_loc.
        Returns float('inf') for unreachable locations.
    """
    distances = {loc: float('inf') for loc in all_locations}
    if start_loc not in all_locations:
         # If start_loc isn't even in the known locations, it's unreachable from anywhere in the graph
         # This might happen if a package/vehicle is at a location not mentioned in road facts
         # In a valid PDDL, this shouldn't happen if all locations are connected or listed.
         # We'll return inf for all distances if the start is invalid.
         return distances

    distances[start_loc] = 0
    queue = collections.deque([start_loc])

    while queue:
        current_loc = queue.popleft()

        # Check if current_loc is in the graph keys before iterating
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
    return distances

def precompute_shortest_paths(locations, graph):
    """
    Compute shortest paths between all pairs of locations using BFS.

    Args:
        locations: Set of all unique locations.
        graph: Adjacency list representation of the road network.

    Returns:
        A dictionary mapping (loc1, loc2) -> shortest distance.
    """
    shortest_paths = {}
    for start_loc in locations:
        distances_from_start = bfs(start_loc, graph, locations)
        for end_loc in locations:
            shortest_paths[(start_loc, end_loc)] = distances_from_start[end_loc]
    return shortest_paths

def parse_capacity_predecessors(static_facts):
    """
    Parse capacity-predecessor facts to map size strings to integers.

    Args:
        static_facts: A frozenset of static PDDL facts (strings).

    Returns:
        A dictionary mapping size string -> integer representation.
        e.g., {'c0': 0, 'c1': 1, 'c2': 2, ...}
    """
    size_map = {}
    predecessors = {} # Map s2 -> s1
    for fact in static_facts:
        if match(fact, "capacity-predecessor", "*", "*"):
            _, s1, s2 = get_parts(fact)
            predecessors[s2] = s1

    # Find the smallest size (c0) - it has no predecessor
    smallest_size = None
    all_sizes = set(predecessors.keys()) | set(predecessors.values())
    for size in all_sizes:
        if size not in predecessors.values():
            smallest_size = size
            break

    if smallest_size is None and all_sizes:
         # This case might happen if there's a cycle or no clear smallest element
         # For transport, c0 is usually the smallest. Assume c0 if found.
         if 'c0' in all_sizes:
             smallest_size = 'c0'
         else:
             # Fallback: just assign arbitrary order if no c0 and no clear smallest
             # This is less ideal but prevents crash. Assign 0 to the first size found.
             smallest_size = next(iter(all_sizes))


    if smallest_size:
        size_map[smallest_size] = 0
        current_size = smallest_size
        current_value = 0
        # Build the map by following the predecessor chain backwards
        # Map s1 -> s2 (size s1 is capacity for size s2 package)
        successors = {v: k for k, v in predecessors.items()}
        while current_size in successors:
             next_size = successors[current_size]
             current_value += 1
             size_map[next_size] = current_value
             current_size = next_size

    return size_map


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It considers whether
    a package is on the ground or in a vehicle and uses precomputed shortest
    path distances for drive actions.

    # Assumptions:
    - Each package needs to reach a specific goal location.
    - The cost for a package on the ground is 1 (pick) + shortest_path(current, goal) + 1 (drop).
    - The cost for a package in a vehicle is shortest_path(vehicle_loc, goal) + 1 (drop).
    - Capacity constraints and vehicle availability for pickup are ignored for simplicity
      and computational efficiency.
    - Roads are bidirectional.

    # Heuristic Initialization
    - Extracts goal locations for packages.
    - Builds the road graph from static facts.
    - Precomputes all-pairs shortest paths on the road graph.
    - Parses capacity-predecessor facts (though not strictly used in the simple cost calculation).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each package.
    2. For each package that is NOT at its goal location:
       a. Determine its current status: on the ground at a location, or inside a vehicle.
       b. If on the ground at `loc_current`:
          - Estimate cost as 1 (pick-up) + shortest_path(loc_current, goal_loc) + 1 (drop).
       c. If inside vehicle `v`, which is at `loc_vehicle`:
          - Estimate cost as shortest_path(loc_vehicle, goal_loc) + 1 (drop).
    3. Sum the estimated costs for all packages not at their goal.
    4. If any required shortest path is infinite (locations disconnected), the heuristic is infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task) # Call base class constructor

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in task.goals:
            # Goal facts are typically (at ?p ?l)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
            # Add other goal types if necessary, but (at ?p ?l) is standard for transport

        # Build road graph and get all locations
        self.road_graph, self.all_locations = build_road_graph(task.static)

        # Precompute all-pairs shortest paths
        self.shortest_paths = precompute_shortest_paths(self.all_locations, self.road_graph)

        # Parse capacity predecessors (useful for understanding domain, but not used in simple h)
        self.size_map = parse_capacity_predecessors(task.static)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Map current locations of packages and vehicles
        package_location = {} # package -> location (if on ground)
        vehicle_location = {} # vehicle -> location
        package_in_vehicle = {} # package -> vehicle (if in vehicle)

        # Populate current state mappings
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj_name, loc_name = parts[1], parts[2]
                # Determine if the object is a package or vehicle based on goals/init
                # A simple way is to check if it's a package in goal_locations
                if obj_name in self.goal_locations or any(match(f, "at", obj_name, "*") for f in self.task.initial_state): # Check if it's a package or vehicle
                     # This is a bit hacky; a proper parser would know types.
                     # Assume anything in goal_locations is a package.
                     # Assume anything else with 'at' is a vehicle if not a package.
                     if obj_name in self.goal_locations:
                         package_location[obj_name] = loc_name
                     else: # Assume it's a vehicle
                         vehicle_location[obj_name] = loc_name

            elif predicate == "in" and len(parts) == 3:
                 pkg_name, veh_name = parts[1], parts[2]
                 package_in_vehicle[pkg_name] = veh_name


        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location
            if package in package_location and package_location[package] == goal_location:
                continue # Package is already at the goal, no cost for this package

            # Package is not at its goal. Calculate cost based on its current state.
            package_cost = 0

            if package in package_location:
                # Package is on the ground
                current_location = package_location[package]
                # Cost: pick-up + drive + drop
                # Need to drive from current_location to goal_location
                drive_cost = self.shortest_paths.get((current_location, goal_location), float('inf'))

                if drive_cost == float('inf'):
                    # If the goal location is unreachable from the package's current location,
                    # this state is likely a dead end or requires complex coordination not
                    # captured by this simple sum. Return infinity.
                    return float('inf')

                package_cost = 1 + drive_cost + 1 # pick + drive + drop

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle_name = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle_name not in vehicle_location:
                    # This shouldn't happen in a valid state, but handle defensively
                    # If vehicle location is unknown, package cannot be moved.
                    return float('inf') # Cannot reach goal

                vehicle_current_location = vehicle_location[vehicle_name]
                # Cost: drive + drop
                # Need to drive vehicle from its current_location to package's goal_location
                drive_cost = self.shortest_paths.get((vehicle_current_location, goal_location), float('inf'))

                if drive_cost == float('inf'):
                     # If the goal location is unreachable from the vehicle's current location,
                     # this state is likely a dead end. Return infinity.
                     return float('inf')

                package_cost = drive_cost + 1 # drive + drop
            else:
                 # Package location is unknown (not at, not in). This indicates an invalid state.
                 return float('inf') # Cannot reach goal

            total_cost += package_cost

        return total_cost

