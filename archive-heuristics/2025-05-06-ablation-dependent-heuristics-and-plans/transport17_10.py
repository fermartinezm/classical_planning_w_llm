from fnmatch import fnmatch
from collections import deque
# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# This is just for standalone testing or if the base class is simple.
# In a real environment, the import should work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a minimal dummy class that matches the expected interface
    class Heuristic:
        def __init__(self, task):
            # Dummy initialization
            self.goals = task.goals if hasattr(task, 'goals') else set()
            self.static = task.static if hasattr(task, 'static') else set()
            # print("Warning: Using dummy Heuristic base class.") # Suppress dummy warning in final output

        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not implemented.")


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not isinstance(fact, str) or not fact or fact[0] != '(' or fact[-1] != ')':
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

def bfs_distance(start_node, end_node, graph):
    """
    Calculates the shortest path distance between two nodes in an unweighted graph using BFS.

    Args:
        start_node: The starting location (string).
        end_node: The target location (string).
        graph: Adjacency dictionary representing the road network {location: [connected_locations]}.

    Returns:
        The shortest distance (number of drive actions) or float('inf') if unreachable.
    """
    if start_node == end_node:
        return 0
    # Ensure start and end nodes are valid locations in the graph
    # If a location exists but has no roads, it won't be a key in the graph.
    # If start_node is not in graph, it's isolated, cannot reach end_node unless end_node is start_node.
    # If end_node is not in graph, it's isolated, cannot be reached from start_node unless start_node is end_node.
    if start_node not in graph or end_node not in graph:
         # If start_node == end_node, we already returned 0.
         # If they are different and one is not in the graph, they are unreachable from each other via roads.
         return float('inf')


    queue = deque([(start_node, 0)]) # (current_location, distance)
    visited = {start_node}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end_node:
            return dist

        # Check if current_loc has neighbors defined in the graph
        # This check is technically redundant because we already checked if current_loc is in graph keys
        # at the beginning, and we only add nodes that are keys to the queue.
        # However, keeping it doesn't hurt.
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Target not reachable

# The main heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    package independently, ignoring vehicle capacity and potential conflicts
    or synergies between package movements.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Any vehicle can transport any package (capacity is ignored).
    - Roads are bidirectional (derived from the PDDL example).
    - The cost of each action (pick-up, drop, drive) is 1.
    - The shortest path between locations in the road network is the minimum
      number of drive actions required.
    - Objects in 'at' facts are either packages with goals or vehicles.
    - Objects in 'in' facts are packages (first arg) and vehicles (second arg).
    - Every location mentioned in a 'road' fact is a valid location node.
    - Every object mentioned in an 'at' or 'in' fact that is relevant (package with goal or vehicle)
      will have its status/location findable in the state facts.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph representing the road network from the static facts. This
      graph is used to calculate shortest path distances between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to determine:
       - The current status (location on ground or vehicle inside) for each package that has a goal.
       - The current location for each vehicle.
       This is done by iterating through the state facts. Facts like `(in package vehicle)`
       indicate a package is inside a vehicle. Facts like `(at object location)` indicate
       an object is on the ground. Objects in `at` facts that are not packages with goals
       are assumed to be vehicles.
    3. For each package that has a goal location specified in the task goals:
       a. Check if the package is already on the ground at its goal location in the current state.
          This is verified by checking if the specific goal fact `(at package goal_location)` is present in the state.
          If yes, continue to the next package (cost for this package is 0).
       b. If the package is not at its goal location on the ground:
          i. Get the package's current status (location or vehicle) as determined in step 2.
          ii. If the package's status indicates it is on the ground at `current_l`:
              - Calculate the shortest distance `dist` (number of drive actions) between `current_l` and the package's goal location `goal_l` using the road network graph.
              - If `goal_l` is unreachable from `current_l` via the road network, the problem is likely unsolvable from this state, return `float('inf')`.
              - Otherwise, the estimated cost for this package is 1 (pick-up action) + `dist` (drive actions) + 1 (drop action).
          iii. If the package's status indicates it is inside a vehicle `v`:
              - Find the current location `current_l_v` of vehicle `v` from the state parsing in step 2.
              - If the vehicle's location is unknown (e.g., vehicle not found in any `at` fact), return `float('inf')`.
              - Calculate the shortest distance `dist` between `current_l_v` and the package's goal location `goal_l` using the road network graph.
              - If `goal_l` is unreachable from `current_l_v` via the road network, return `float('inf')`.
              - Otherwise, the estimated cost for this package is `dist` (drive actions) + 1 (drop action). (The pick-up action has already occurred).
       c. Add the estimated cost for this package to the total heuristic cost.
       d. If at any point the cost becomes `float('inf`'), propagate this and return `float('inf')` immediately.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the road graph.

        Args:
            task: The planning task object containing initial state, goals, operators, and static facts.
        """
        super().__init__(task) # Call base class constructor if needed

        self.goals = task.goals  # Goal conditions (frozenset of fact strings).
        static_facts = task.static  # Static facts (frozenset of fact strings).

        # Store goal locations for each package.
        # We only care about (at ?p ?l) goals for packages.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically simple predicates like (at obj loc)
            if parts and parts[0] == "at" and len(parts) == 3:
                 # Assuming the first argument of 'at' in a goal is always a package
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road graph from static facts.
        self.road_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Add bidirectional roads
                self.road_graph.setdefault(loc1, []).append(loc2)
                self.road_graph.setdefault(loc2, []).append(loc1)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            The estimated cost (non-negative integer or float('inf')).
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Check if the goal is already reached.
        # The goal is a conjunction of facts. Check if all goal facts are in the state.
        if self.goals <= state:
            return 0

        # --- State Parsing ---
        # Track current location/status of packages and vehicles.
        # package -> location (if on ground) or vehicle (if in vehicle)
        package_status = {}
        # vehicle -> location
        vehicle_locations = {}
        # Temporary storage for objects found at locations
        object_locations_on_ground = {}

        # First pass: Collect all 'at' and 'in' facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Only track packages that have a goal
                if package in self.goal_locations:
                    package_status[package] = vehicle # Package is inside this vehicle
            elif predicate == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 object_locations_on_ground[obj] = loc # Object obj is at location loc

        # Second pass: Refine package_status and populate vehicle_locations
        for package in self.goal_locations:
             # If package_status was set in the first pass, it means the package is 'in' a vehicle.
             # If not, check if it's 'at' a location.
             if package not in package_status:
                 if package in object_locations_on_ground:
                     package_status[package] = object_locations_on_ground[package]
             # Note: A package should be either 'in' a vehicle or 'at' a location, not both.
             # The logic prioritizes 'in' if both were somehow present.

        # Populate vehicle_locations - assume any object found 'at' a location
        # that is NOT a package with a goal is a vehicle.
        for obj, loc in object_locations_on_ground.items():
             if obj not in self.goal_locations:
                  # Assume it's a vehicle
                  vehicle_locations[obj] = loc

        # --- Heuristic Calculation ---
        total_cost = 0

        # Calculate cost for each package whose individual goal fact is not met.
        for package, goal_location in self.goal_locations.items():
            # Check if the specific goal fact for this package is met
            if f"(at {package} {goal_location})" in state:
                 continue # This package is already at its goal location on the ground.

            # Package is not at its goal location on the ground.
            current_status = package_status.get(package)

            if current_status is None:
                 # Package is not found in 'at' or 'in' facts. Invalid state?
                 # Treat as unreachable.
                 # print(f"Warning: Package {package} not found in state facts.")
                 return float('inf')

            # Check if current_status is a location (package is on the ground)
            # A location must be a node in our road graph to be reachable/meaningful for distance.
            if current_status in self.road_graph:
                current_l = current_status
                dist = bfs_distance(current_l, goal_location, self.road_graph)
                if dist == float('inf'):
                    # Goal location unreachable from package's current location on the ground
                    return float('inf')
                # Cost: pick-up (1) + drive (dist) + drop (1)
                total_cost += 1 + dist + 1

            # Check if current_status is a vehicle (package is inside)
            elif current_status in vehicle_locations:
                vehicle = current_status
                current_l_v = vehicle_locations.get(vehicle)

                if current_l_v is None:
                    # Vehicle location not found in state facts. Invalid state?
                    # print(f"Warning: Location for vehicle {vehicle} not found in state facts.")
                    return float('inf') # Cannot determine vehicle location

                dist = bfs_distance(current_l_v, goal_location, self.road_graph)
                if dist == float('inf'):
                    # Goal location unreachable from vehicle's current location
                    return float('inf')
                # Cost: drive (dist) + drop (1)
                total_cost += dist + 1
            else:
                 # current_status is neither a known location node nor a known vehicle.
                 # This should not happen in a valid state given the domain structure and assumptions.
                 # It might happen if an object type assumption is wrong.
                 # For robustness, return infinity.
                 # print(f"Warning: Unhandled package status type for {package}: {current_status}")
                 return float('inf')


        return total_cost
