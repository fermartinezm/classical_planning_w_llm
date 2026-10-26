import collections

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_str.strip('()').split()
    if not parts:
        return None, [] # Should not happen with valid PDDL facts
    return parts[0], parts[1:]

def bfs(graph, start_node):
    """Computes shortest distances from start_node to all reachable nodes in a graph."""
    # Collect all unique nodes in the graph (keys and values)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Initialize distances to infinity for all nodes
    distances = {node: float('inf') for node in all_nodes}

    # If start_node is not in the graph (e.g., an isolated location not in road facts),
    # distances to other nodes remain infinity.
    if start_node not in all_nodes:
         return distances

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Iterate over neighbors. Check if current_node is a key first.
        # This handles locations with no outgoing roads.
        if current_node in graph:
            for neighbor in graph[current_node]:
                # If the neighbor hasn't been visited yet (distance is inf)
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Collect all unique nodes in the graph (keys and values)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    for start_node in all_nodes:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the minimum
    actions required for each package that is not yet at its goal location.
    For a package currently at a location, the estimated cost is 2 (pick-up + drop)
    plus the shortest path distance (number of drive actions) from its current
    location to its goal location.
    For a package currently inside a vehicle, the estimated cost is 1 (drop)
    plus the shortest path distance (number of drive actions) from the vehicle's
    current location to the package's goal location.
    This heuristic ignores vehicle capacity constraints and assumes any vehicle
    can transport any package and is available when needed. It also assumes
    vehicles can magically appear at the package's location if needed (by adding
    the drive cost for the package itself, not for a specific vehicle).

    Assumptions:
    - The road network is static and provided in the static facts.
    - Package goal locations are specified by (at ?p ?l) facts in the goal state.
    - The heuristic does not consider vehicle capacity or specific vehicle assignments.
    - The heuristic assumes the shortest path in the road network is always usable
      by a vehicle.
    - The input state and task are valid according to the domain definition.

    Heuristic Initialization:
    1. Parse the goal facts (`task.goals`) to create a mapping from each package
       to its goal location (`self.package_goals`). Only `(at ?p ?l)` goal facts
       are considered for packages.
    2. Parse the static facts (`task.static`) to build the road network graph
       (`self.road_graph`) as an adjacency list. Only `(road ?l1 ?l2)` facts
       are considered.
    3. Compute all-pairs shortest paths on the road network graph using BFS.
       This precomputes the minimum number of drive actions between any two
       locations. The results are stored in `self.distances`, a dictionary
       where `self.distances[l1][l2]` is the shortest distance from `l1` to `l2`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Create temporary mappings from the current state for quick lookups:
       - `at_map`: maps locatable objects (packages, vehicles) to their current location.
       - `in_map`: maps packages to the vehicle they are currently inside.
    3. Iterate through each package `p` that has a goal location defined in
       `self.package_goals`.
    4. Get the goal location `loc_p_goal` for package `p`.
    5. Determine the current status of package `p` using `at_map` and `in_map`:
       a. If `p` is at a location `loc_p_current` (i.e., `at_map.get(p)` is not None):
          - If `loc_p_current` is the same as `loc_p_goal`, the package is at its goal.
            Add 0 to `h`.
          - If `loc_p_current` is different from `loc_p_goal`:
            - The package needs to be picked up, transported, and dropped.
            - Estimated actions: 1 (pick-up) + shortest_distance(`loc_p_current`, `loc_p_goal`) (drive) + 1 (drop).
            - Look up the distance `dist` from `self.distances.get(loc_p_current, {}).get(loc_p_goal, float('inf'))`.
            - If `dist` is infinity (goal unreachable), the state is likely unsolvable; return `float('inf')`.
            - Otherwise, add `2 + dist` to `h`.
       b. If `p` is inside a vehicle `v` (i.e., `in_map.get(p)` is not None):
          - Find the current location `loc_v_current` of vehicle `v` using `at_map.get(v)`.
          - If `loc_v_current` is None:
             # Vehicle location not found - invalid state?
             # Treat as unreachable for safety in heuristic
             return float('inf')

          if vehicle_location == goal_location:
              # Package is in a vehicle that is at the goal location
              # Needs only drop
              cost_p = 1
          else:
              # Package is in a vehicle, vehicle is not at goal
              # Needs drive (by vehicle) and drop
              # Cost = distance(vehicle_location, goal_location) + 1 (drop)
              dist = self.distances.get(vehicle_location, {}).get(goal_location, float('inf'))
              if dist == float('inf'):
                   # Goal is unreachable from vehicle's current location via road network
                   return float('inf')
              cost_p = 1 + dist
       c. If the package's status is not found in `at_map` or `in_map` (e.g., it's not at a location and not in a vehicle): This indicates an invalid state representation; return `float('inf')`.
    6. After iterating through all packages with goals, the total value `h` is the heuristic estimate. Return `h`.
    """
    def __init__(self, task):
        self.package_goals = self._parse_goals(task.goals)
        self.road_graph = self._build_road_graph(task.static)
        self.distances = self._compute_distances(self.road_graph)
        # Capacity info is ignored for this heuristic's calculation

    def _parse_goals(self, goals):
        """Parses goal facts to find package goal locations."""
        package_goals = {}
        for goal_fact in goals:
            pred, args = parse_fact(goal_fact)
            # Assuming goals only specify final location for packages using (at ?p ?l)
            # and that the first argument of 'at' in goals is always a package.
            if pred == 'at' and len(args) == 2:
                item, location = args
                # We could potentially check if 'item' is a package type if type info was readily available here.
                # For now, assume any 'at' goal fact refers to a package.
                package_goals[item] = location
        return package_goals

    def _build_road_graph(self, static_facts):
        """Builds the road network graph from static facts."""
        graph = collections.defaultdict(list)
        locations = set()
        for fact_str in static_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'road' and len(args) == 2:
                l1, l2 = args
                graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Ensure all locations mentioned are keys in the graph, even if they have no outgoing roads
        for loc in locations:
             if loc not in graph:
                 graph[loc] = []

        return graph

    def _compute_distances(self, road_graph):
        """Computes all-pairs shortest paths using BFS."""
        return compute_all_pairs_shortest_paths(road_graph)

    def __call__(self, state):
        """Computes the heuristic value for the given state."""
        h = 0

        # Map current locations/vehicles for all locatables in the state for quick lookup
        at_map = {} # item -> location
        in_map = {} # package -> vehicle
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                item, location = args
                at_map[item] = location
            elif pred == 'in' and len(args) == 2:
                package, vehicle = args
                in_map[package] = vehicle

        # Calculate cost for each package that has a goal
        for package, goal_location in self.package_goals.items():
            current_location = at_map.get(package)
            current_vehicle = in_map.get(package)

            if current_location == goal_location:
                # Package is already at its goal location
                cost_p = 0
            elif current_location is not None:
                # Package is at a location, but not the goal
                # Needs pickup, drive, drop
                # Cost = 1 (pickup) + distance(current_location, goal_location) + 1 (drop)
                # Check if current_location and goal_location are in the computed distances
                dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))

                if dist == float('inf'):
                    # Goal is unreachable from current location via road network
                    return float('inf')
                cost_p = 2 + dist
            elif current_vehicle is not None:
                # Package is inside a vehicle
                vehicle_location = at_map.get(current_vehicle)
                if vehicle_location is None:
                     # Vehicle location not found - invalid state?
                     # Treat as unreachable for safety in heuristic
                     return float('inf')

                if vehicle_location == goal_location:
                    # Package is in a vehicle that is at the goal location
                    # Needs only drop
                    cost_p = 1
                else:
                    # Package is in a vehicle, vehicle is not at goal
                    # Needs drive (by vehicle) and drop
                    # Cost = distance(vehicle_location, goal_location) + 1 (drop)
                    dist = self.distances.get(vehicle_location, {}).get(goal_location, float('inf'))
                    if dist == float('inf'):
                         # Goal is unreachable from vehicle's current location via road network
                         return float('inf')
                    cost_p = 1 + dist
            else:
                # Package status not found (not at, not in) - invalid state?
                # Treat as unreachable
                return float('inf')

            h += cost_p

        return h
