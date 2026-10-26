from collections import deque
# from fnmatch import fnmatch # Using custom match function

# Define a dummy Heuristic base class if running standalone for testing
# In the actual planner environment, this import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError

# Helper functions (defined outside the class for clarity)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (strings or '*').
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(part == arg or arg == '*' for part, arg in zip(parts, args))

# BFS function for shortest paths
def bfs(start_node, graph):
    """
    Performs Breadth-First Search to find shortest distances from a start node.
    Args:
        start_node: The starting node (location string).
        graph: Adjacency list representation of the graph {node: [neighbor1, neighbor2, ...]}
    Returns:
        A dictionary {node: distance} containing shortest distances from start_node.
    """
    # Collect all nodes potentially reachable or in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    all_nodes.add(start_node) # Ensure start node is included even if isolated

    distances = {node: float('inf') for node in all_nodes}
    if start_node in distances: # Ensure start_node is one of the known locations
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the estimates for all packages not yet at their goal.
    For a package on the ground, the estimate includes pick-up, driving the shortest
    path, and drop-off. For a package inside a vehicle, the estimate includes
    driving the vehicle the shortest path to the goal location and drop-off.
    This heuristic ignores vehicle capacity and the cost of moving a vehicle
    to a package's location if the package is on the ground.

    # Assumptions
    - All locations relevant to the problem (initial package/vehicle locations, goal locations,
      and locations mentioned in road facts) are part of the road network graph,
      even if some might be isolated.
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity is not explicitly considered in the cost calculation.
    - The cost of moving a vehicle to a package's location (if the package is on the ground)
      is not included in the estimate for that package.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representing the road network from static facts.
    - Identifies all unique locations mentioned in road facts and goal locations.
    - Computes all-pairs shortest paths between these locations using BFS and stores them.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of every locatable object (packages and vehicles).
       This involves checking `(at ?x ?l)` facts and `(in ?p ?v)` facts (and then finding where `?v` is).
    3. For each package `p` that has a goal location `l_goal`:
       a. Check if `p` is already at `l_goal` on the ground (i.e., `(at p l_goal)` is in the state). If yes, this package contributes 0 to the heuristic.
       b. If `p` is not at `l_goal`, find its current status:
          - If `p` is on the ground at `l_current` (i.e., `(at p l_current)` is in the state, where `l_current != l_goal`):
            - Estimate the cost for this package as 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
          - If `p` is inside a vehicle `v` (i.e., `(in p v)` is in the state):
            - Find the current location `l_v` of vehicle `v` (i.e., `(at v l_v)` is in the state).
            - Estimate the cost for this package as shortest_distance(`l_v`, `l_goal`) (drive) + 1 (drop).
       c. If the calculated distance is infinity (meaning the goal is unreachable from the current location/vehicle location), the state is likely unsolvable or very far; return infinity for the heuristic.
       d. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road network graph and collect all relevant locations.
        self.road_graph = {}
        all_locations = set()

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                all_locations.add(l1)
                all_locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Roads are bidirectional

        # Add goal locations to the set of all locations, even if isolated in the road network
        for loc in self.goal_locations.values():
             all_locations.add(loc)
             # Ensure goal locations are keys in the graph dictionary, even if they have no roads
             self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths.
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = bfs(start_loc, self.road_graph)

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if loc1 is a known start node for BFS and loc2 is in its distance map
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             dist = self.distances[loc1][loc2]
             return dist # Will be float('inf') if unreachable
        # If loc1 or loc2 was not a known location during initialization,
        # or if loc2 is not in the distance map from loc1 (shouldn't happen
        # if BFS was run from all_locations to all_locations), treat as unreachable.
        return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles.
        # A package can be (at p l) or (in p v).
        # A vehicle can be (at v l).
        current_locations = {} # {object_name: location_name or vehicle_name}
        vehicle_locations = {} # {vehicle_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == "at":
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_locations[obj] = loc
                    # Simple check if the object is a vehicle based on name prefix 'v'
                    # This assumes a naming convention from the problem instances.
                    # A more robust way would involve parsing types from the domain,
                    # but this is often sufficient for domain-dependent heuristics.
                    if obj.startswith('v'):
                         vehicle_locations[obj] = loc
            elif predicate == "in":
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_locations[package] = vehicle # Package is inside this vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # If the goal is (at p l_goal), the package is done if (at p l_goal) is true.
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at its goal location on the ground. Cost is 0 for this package.

            # Package is not at its goal location on the ground.
            # Find its current status.
            current_status = current_locations.get(package)

            if current_status is None:
                 # Package exists and has a goal, but is neither (at p l) nor (in p v).
                 # This indicates an invalid state according to the domain predicates.
                 # Treat as unreachable/infinite cost.
                 return float('inf')

            elif current_status.startswith('v'): # Package is inside a vehicle
                vehicle = current_status
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                    # Vehicle exists and contains a package, but its own location is unknown.
                    # Invalid state. Treat as unreachable/infinite cost.
                    return float('inf')

                # Package is in vehicle at vehicle_location, needs to reach goal_location
                # Cost = drive vehicle from vehicle_location to goal_location + drop
                drive_cost = self.get_distance(vehicle_location, goal_location)
                if drive_cost == float('inf'):
                    # Goal location is unreachable from the vehicle's current location.
                    return float('inf')

                total_cost += drive_cost + 1 # 1 for drop action

            else: # Package is on the ground at current_status (which is a location)
                current_location = current_status
                # Package is at current_location, needs to reach goal_location
                # Cost = pick-up + drive from current_location to goal_location + drop
                # Note: This ignores the cost of getting a vehicle to current_location.
                drive_cost = self.get_distance(current_location, goal_location)
                if drive_cost == float('inf'):
                    # Goal location is unreachable from the package's current location.
                    return float('inf')

                total_cost += 1 + drive_cost + 1 # 1 for pick-up, 1 for drop

        return total_cost
