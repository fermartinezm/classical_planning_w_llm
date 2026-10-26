from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, though PDDL facts should be strings
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# BFS function to find shortest path distance in the road network
def bfs_distance(graph, start, end):
    """
    Find the shortest path distance between start and end in the graph using BFS.
    Returns the number of edges in the shortest path. Returns float('inf') if unreachable.
    """
    if start == end:
        return 0

    # Handle cases where start or end location might not be in the graph (e.g., isolated)
    # If start is not in graph, it cannot move. If end is not in graph, it cannot be reached.
    if start not in graph or end not in graph:
         return float('inf')


    queue = deque([(start, 0)]) # (location, distance)
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc == end:
            return dist

        # Ensure current_loc is a valid key in the graph before accessing neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Not reachable

class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, ignoring vehicle capacity constraints
    and potential conflicts (like multiple packages needing the same vehicle
    or path). It sums the estimated costs for each package independently.

    # Assumptions
    - The road network is static and provides connections between locations.
    - Any vehicle can carry any package (capacity is ignored).
    - A vehicle is always available when needed at a package's location.
    - The cost of each action (drive, pick-up, drop) is 1.
    - Package names start with 'p', vehicle names start with 'v'. Location names do not start with 'p' or 'v'. This is used to distinguish object types from fact structure.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from the static `road` facts to enable
      shortest path calculations (distances between locations). Assumes roads are bidirectional based on example instance format.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that is not yet at its goal location `goal_l`:

    1. Determine the package's current state:
       - Is it at a location `current_l` (using `(at p current_l)`)?
       - Is it inside a vehicle `v` (using `(in p v)`)? If so, find the vehicle's
         location `vehicle_l` (using `(at v vehicle_l)`) from the current state.

    2. Calculate the estimated cost for this package based on its state:
       - If `p` is at `current_l` (`current_l != goal_l`):
         - It needs to be picked up (1 action).
         - A vehicle needs to drive from `current_l` to `goal_l` (distance(`current_l`, `goal_l`) actions).
         - It needs to be dropped off (1 action).
         - Estimated cost for this package = 1 (pick) + distance(`current_l`, `goal_l`) + 1 (drop).
       - If `p` is in vehicle `v`, and `v` is at `vehicle_l`:
         - The vehicle needs to drive from `vehicle_l` to `goal_l` (distance(`vehicle_l`, `goal_l`) actions).
         - It needs to be dropped off (1 action).
         - Estimated cost for this package = distance(`vehicle_l`, `goal_l`) + 1 (drop).
       - If `p` is at `goal_l`, the cost for this package is 0.

    3. If at any point a required location is unreachable via the road network (e.g., package is at a location from which goal is unreachable, or package is in a vehicle whose current location cannot reach the goal), the heuristic for this package (and thus the total heuristic) is considered infinite.

    4. The total heuristic value is the sum of the estimated costs for all
       packages that are not yet at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2: # Ensure correct number of arguments
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road network graph from static 'road' facts.
        self.road_graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Add bidirectional edges
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = set()
                self.road_graph[loc1].add(loc2)
                if loc2 not in self.road_graph:
                     self.road_graph[loc2] = set()
                self.road_graph[loc2].add(loc1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current location or containment for packages
        package_positions = {} # Maps package name to its location or the vehicle it's in
        vehicle_locations = {} # Maps vehicle name to its location

        # First pass: find all vehicle locations and package positions
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                 obj, loc = args
                 # Simple heuristic assumption: vehicle names start with 'v', package names with 'p'
                 # This is a heuristic assumption based on example data, not guaranteed by PDDL types alone.
                 if obj.startswith('v'):
                     vehicle_locations[obj] = loc
                 elif obj.startswith('p'):
                     package_positions[obj] = loc
            elif predicate == "in" and len(args) == 2:
                 package, vehicle = args
                 # Assuming the first arg is package, second is vehicle based on domain definition
                 # Check if they follow the naming convention assumption
                 if package.startswith('p') and vehicle.startswith('v'):
                     package_positions[package] = vehicle # Store the vehicle name

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If the package is not found in the current state, it's an issue.
            # For heuristic, assume unreachable.
            if package not in package_positions:
                 # This package is not 'at' any location and not 'in' any vehicle.
                 # This state is likely invalid or the package is lost.
                 # Returning infinity is appropriate as the goal for this package is unreachable.
                 return float('inf')

            current_pos = package_positions[package]

            # Check if the package is already at its goal location
            if current_pos == goal_location:
                continue # Package is already at goal, cost is 0 for this package

            # Determine if current_pos is a location or a vehicle name
            # If current_pos is a key in vehicle_locations, it must be a vehicle name.
            # Otherwise, assume it's a location name. This relies on vehicle_locations
            # containing all vehicles that are 'at' a location.
            if current_pos in vehicle_locations: # Case: Package is inside a vehicle
                vehicle_name = current_pos
                current_vehicle_location = vehicle_locations[vehicle_name]

                # Cost: drive (distance) + drop (1)
                drive_cost = bfs_distance(self.road_graph, current_vehicle_location, goal_location)
                if drive_cost == float('inf'):
                     # If the goal location is unreachable from the vehicle's current location,
                     # this package cannot reach its goal via this vehicle.
                     # Assume this package makes the state unreachable for its goal.
                     return float('inf')
                total_cost += drive_cost + 1 # drive + drop

            else: # Case: Package is at a location current_pos != goal_location
                current_package_location = current_pos
                # Cost: pick-up (1) + drive (distance) + drop (1)
                drive_cost = bfs_distance(self.road_graph, current_package_location, goal_location)
                if drive_cost == float('inf'):
                    # If the goal location is unreachable from the package's current location,
                    # this package cannot reach its goal.
                    return float('inf')
                total_cost += 1 + drive_cost + 1 # pick + drive + drop

        return total_cost
