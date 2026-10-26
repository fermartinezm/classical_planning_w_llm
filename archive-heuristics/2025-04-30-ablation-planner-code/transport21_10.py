# Assume Heuristic base class exists and is imported like this:
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list or handle error as appropriate for expected input
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity and
    vehicle availability constraints. It sums the estimated costs for all packages
    not yet at their destination.

    # Assumptions
    - Each package can be moved independently.
    - Vehicle capacity is unlimited for the purpose of this heuristic calculation.
    - A suitable vehicle is always available at the package's location for pickup (if needed).
    - The cost of moving a package involves pickup (if on ground), driving, and dropping.
    - Driving between connected locations costs 1 action.
    - Pickup costs 1 action.
    - Drop costs 1 action.
    - Goal conditions only specify the final location of packages.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package present in the state.
    2. Identify the current location for every vehicle present in the state.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a specified goal location:
       a. Check if the package is currently at its goal location on the ground (i.e., the state contains `(at package goal_location)`). If yes, the cost for this package is 0, continue to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current physical location.
             - If the package is currently `at` a location `current_loc`, its physical location is `current_loc`.
             - If the package is currently `in` a vehicle `vehicle_name`, its physical location is the location where `vehicle_name` is currently `at` (`vehicle_loc`). If the vehicle's location is unknown, the state is potentially malformed or unreachable, return infinity.
          ii. Calculate the estimated cost for this package:
              - If the package was `at current_loc` (on the ground): Cost = 1 (pick-up) + shortest_distance(`current_loc`, `goal_location`) (drive) + 1 (drop). If the goal is unreachable from `current_loc`, return infinity.
              - If the package was `in vehicle_name` located at `vehicle_loc`:
                  - If `vehicle_loc` is the same as `goal_location`: Cost = 1 (drop).
                  - If `vehicle_loc` is different from `goal_location`: Cost = shortest_distance(`vehicle_loc`, `goal_location`) (drive) + 1 (drop). If the goal is unreachable from `vehicle_loc`, return infinity.
          iii. Add the calculated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing location distances.
        """
        # Assuming task object has attributes: goals, static, objects
        self.goals = task.goals
        static_facts = task.static
        objects = task.objects # Assuming task.objects is a dict like {'type': [obj1, ...]}

        # Get lists of objects by type
        self.packages = set(objects.get('package', []))
        self.vehicles = set(objects.get('vehicle', []))
        self.locations = set(objects.get('location', []))
        self.sizes = set(objects.get('size', [])) # Not strictly needed for this heuristic

        # Map package to its goal location
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Only consider goals for packages and valid locations
                if obj in self.packages and loc in self.locations:
                     self.package_goals[obj] = loc
                # Ignore other types of goals if any exist

        # Build road network graph
        self.road_graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Ensure locations are valid objects defined in the problem
                if l1 in self.locations and l2 in self.locations:
                    self.road_graph[l1].append(l2)
                # Note: PDDL road predicates define directed edges.
                # If roads are bidirectional, the instance file must contain both (road l1 l2) and (road l2 l1).
                # BFS on this directed graph correctly finds shortest paths respecting road directions.


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances_from_start = {loc: float('inf') for loc in self.locations}
        distances_from_start[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is a valid location and has neighbors in the graph
            if u in self.road_graph:
                for v in self.road_graph.get(u, []): # Use .get for safety, though keys should be in self.locations
                    if distances_from_start[v] == float('inf'):
                        distances_from_start[v] = distances_from_start[u] + 1
                        queue.append(v)
        return distances_from_start

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Track current location/container for packages and vehicles
        current_package_status = {} # Maps package -> location or vehicle
        current_vehicle_locations = {} # Maps vehicle -> location

        # Populate status dictionaries from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages and loc in self.locations:
                    current_package_status[obj] = loc
                elif obj in self.vehicles and loc in self.locations:
                    current_vehicle_locations[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 if pkg in self.packages and veh in self.vehicles:
                     current_package_status[pkg] = veh
            # Ignore other predicates like capacity, capacity-predecessor, road

        total_cost = 0

        # Calculate cost for each package that needs to reach a goal
        for package, goal_location in self.package_goals.items():
            # A package is at its goal if the fact (at package goal_location) is in the state.
            # We check this explicitly rather than relying solely on current_package_status
            # because current_package_status might say the package is 'in' a vehicle
            # even if that vehicle is at the goal location. The goal requires the package
            # to be *at* the location, not just inside a vehicle there.
            if f'(at {package} {goal_location})' in state:
                 continue # Package is already at its goal location on the ground, cost is 0 for this package

            # Package is not at its goal location on the ground. Calculate cost.
            package_cost = 0

            # Find the package's current status (at a location or in a vehicle)
            current_status = current_package_status.get(package)

            if current_status is None:
                 # Package is not mentioned in 'at' or 'in' facts in the state.
                 # This shouldn't happen in a well-formed state for objects that exist.
                 # Treat as unreachable or assign high cost.
                 # Returning infinity is safer for search completeness if it indicates an impossible state.
                 return float('inf')

            if current_status in self.locations: # Package is on the ground at current_status
                current_loc = current_status
                # Cost: pick-up + drive + drop
                # We need to drive from current_loc to goal_location
                drive_cost = self.distances.get(current_loc, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location via road network
                    return float('inf') # Problem is likely unsolvable from here

                package_cost = 1 + drive_cost + 1 # pick + drive + drop

            elif current_status in self.vehicles: # Package is in a vehicle
                vehicle = current_status
                # Find vehicle's location
                vehicle_loc = current_vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle location unknown - indicates state representation issue
                    # Or vehicle is not 'at' any location (e.g., initial state might not list all vehicle locations?)
                    # Assuming vehicle location is always present if package is in it.
                    return float('inf') # Cannot proceed if vehicle location is unknown

                # Cost: drive (if needed) + drop
                if vehicle_loc == goal_location:
                    # Vehicle is already at the goal location
                    package_cost = 1 # drop
                else:
                    # Vehicle needs to drive from vehicle_loc to goal_location
                    drive_cost = self.distances.get(vehicle_loc, {}).get(goal_location, float('inf'))

                    if drive_cost == float('inf'):
                        # Goal is unreachable from vehicle's current location via road network
                        return float('inf') # Problem is likely unsolvable from here

                    package_cost = drive_cost + 1 # drive + drop
            else:
                 # current_status is neither a location nor a vehicle? Malformed state.
                 # This case should ideally not be reached with valid PDDL states.
                 return float('inf') # Indicate error/unsolvable path

            total_cost += package_cost

        return total_cost
