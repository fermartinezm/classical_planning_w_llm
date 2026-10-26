import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Perform Breadth-First Search to find shortest distances from a start_node
    to all reachable nodes in a graph.

    Args:
        graph: A dictionary representing the graph (adjacency list),
               e.g., {location1: [location2, location3], ...}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its shortest distance
        from the start_node, e.g., {locationX: distance, ...}.
        Returns an empty dictionary if start_node is not in the graph.
    """
    distances = {}
    if start_node not in graph:
        return distances

    queue = collections.deque([start_node])
    distances[start_node] = 0
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Ensure the node has neighbors defined
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity
    and vehicle availability constraints. It sums the estimated costs for each
    package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Packages can only move inside vehicles.
    - Vehicles move between locations connected by roads.
    - Vehicle capacity is ignored.
    - Vehicle availability at package locations is ignored (assumed a vehicle is
      available when needed).
    - Roads are bidirectional (if (road l1 l2) exists, assume (road l2 l1) is possible).

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on `road` facts.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status:
       - Is it on the ground at some location `l_current`? (`(at package l_current)`)
       - Is it inside a vehicle `v`? (`(in package v)`)
    2. If the package is on the ground at `l_current` (and `l_current` is not the goal):
       - It needs to be picked up (1 action).
       - The vehicle needs to drive from `l_current` to the goal location `l_goal`
         (shortest path distance `dist(l_current, l_goal)` actions).
       - It needs to be dropped (1 action).
       - Estimated cost for this package: `1 + dist(l_current, l_goal) + 1 = 2 + dist(l_current, l_goal)`.
    3. If the package is inside a vehicle `v`:
       - Find the vehicle's current location `l_v` (`(at v l_v)`).
       - The vehicle needs to drive from `l_v` to the goal location `l_goal`
         (shortest path distance `dist(l_v, l_goal)` actions).
       - It needs to be dropped (1 action).
       - Estimated cost for this package: `dist(l_v, l_goal) + 1`.
    4. If the package is already at its goal location:
       - Estimated cost for this package: 0.
    5. The total heuristic value is the sum of the estimated costs for all packages.
    6. If any required shortest path is infinite (locations are disconnected), the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road graph from static facts.
        # The graph is represented as an adjacency list: {location: [neighbor1, neighbor2, ...]}
        self.road_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional based on domain examples
                self.road_graph[loc2].append(loc1)
                locations.add(loc1)
                locations.add(loc2)

        # Precompute shortest path distances between all pairs of locations.
        # We can optimize by only computing distances *to* goal locations.
        # Since the graph is symmetric (bidirectional roads), dist(A, B) == dist(B, A).
        # So, running BFS from each goal location gives us distances *from* any location *to* that goal.
        self.distances = {} # distances[from_loc][to_loc] = distance
        goal_locations_set = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is always (at package location)
                package, location = args
                goal_locations_set.add(location)

        # Ensure all locations from the graph are considered as potential start nodes for BFS
        # if they are also goal locations.
        all_relevant_locations = locations.union(goal_locations_set)

        for goal_loc in goal_locations_set:
             # Run BFS from the goal location to find distances *to* it from all other locations
             # (due to symmetry of the graph).
             distances_from_goal = bfs(self.road_graph, goal_loc)
             for from_loc, dist in distances_from_goal.items():
                 if from_loc not in self.distances:
                     self.distances[from_loc] = {}
                 self.distances[from_loc][goal_loc] = dist


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
            # Note: This heuristic only considers (at ?p ?l) goals for packages.
            # Other goal types (like vehicle location or capacity) are ignored.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Create quick lookups for current locations of locatables (packages and vehicles)
        # and which packages are inside which vehicles.
        locatables_at = {} # {object_name: location_name} for objects on the ground
        packages_in = {}   # {package_name: vehicle_name} for packages inside vehicles

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                locatables_at[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                packages_in[pkg] = veh

        total_cost = 0.0  # Use float for potential infinity

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location.
            if package in locatables_at and locatables_at[package] == goal_location:
                # Package is on the ground at the goal. Cost is 0 for this package.
                continue

            # Determine the package's current physical location.
            current_physical_location = None
            is_in_vehicle = False

            if package in locatables_at:
                # Package is on the ground.
                current_physical_location = locatables_at[package]
                is_in_vehicle = False
            elif package in packages_in:
                # Package is inside a vehicle. Find the vehicle's location.
                vehicle = packages_in[package]
                if vehicle in locatables_at:
                    current_physical_location = locatables_at[vehicle]
                    is_in_vehicle = True
                else:
                    # This state shouldn't happen in a valid domain/problem,
                    # but if a package is 'in' a vehicle that isn't 'at' a location,
                    # the package is effectively unreachable.
                    return float('inf')
            else:
                 # Package location is unknown (not at, not in). Should not happen.
                 # Treat as unreachable for safety.
                 return float('inf')


            # If the package is not at the goal, calculate the cost to move it.
            # Get the shortest distance from the current physical location to the goal location.
            # Use .get() with float('inf') to handle cases where the goal location
            # is unreachable from the current location in the road network.
            distance_to_goal = self.distances.get(current_physical_location, {}).get(goal_location, float('inf'))

            if distance_to_goal == float('inf'):
                # If the goal location is unreachable from the package's current location,
                # the state is likely a dead end or the goal is impossible.
                return float('inf')

            # Calculate cost based on package's current state (on ground or in vehicle).
            if is_in_vehicle:
                # Package is in a vehicle. Needs drive + drop.
                # Cost = drive actions + 1 (drop)
                cost_for_package = distance_to_goal + 1
            else:
                # Package is on the ground. Needs pick-up + drive + drop.
                # Cost = 1 (pick-up) + drive actions + 1 (drop)
                cost_for_package = 1 + distance_to_goal + 1

            total_cost += cost_for_package

        return total_cost

