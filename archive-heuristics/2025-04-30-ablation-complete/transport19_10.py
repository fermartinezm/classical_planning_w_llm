# from heuristics.heuristic_base import Heuristic # Assuming this is available

# Helper functions
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Basic check for expected format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # In a real system, might log an error or raise an exception
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_path(graph, start, end):
    """
    Find the shortest path distance between start and end in an unweighted graph.
    Returns distance (number of edges) or float('inf') if no path exists.
    Handles cases where start or end might not be in the graph (or have no edges).
    """
    if start == end:
        return 0
    # Ensure start and end are valid locations known in the graph structure
    if start not in graph or end not in graph:
         return float('inf')

    queue = deque([(start, 0)]) # (location, distance)
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        # Check if current_loc has neighbors defined (should be true if current_loc in graph keys)
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # No path found

# Define the heuristic class
# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# Uncomment the following line in the actual environment if needed
# from heuristics.heuristic_base import Heuristic

# If the base class needs to be defined for the code to be syntactically complete in isolation:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# Assuming the environment provides the Heuristic base class and handles its import.
# The class definition should inherit from it.
class transportHeuristic(Heuristic): # Assuming 'Heuristic' is defined/imported elsewhere
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    to its goal location independently. It considers the steps required for
    picking up, driving, and dropping, using shortest path distances on the
    road network for driving costs.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability at package locations for pick-up is not explicitly modeled;
      it's assumed a vehicle can eventually reach the package.
    - The road network is static and bidirectional (if road l1 l2 exists, road l2 l1 also exists,
      or the graph construction handles both directions). The PDDL shows bidirectional roads.
    - Goal predicates are only of the form `(at package location)`.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts to enable shortest path calculations.
    - Extracts capacity predecessor relationships (though not used in this simple version).

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current state: Is it on the ground at a location `l_p`, or is it inside a vehicle `v`?
    2. If the package is on the ground at `l_p` (and `l_p` is not the goal):
       - It needs to be picked up (1 action).
       - The vehicle carrying it needs to drive from `l_p` to the goal location `l_goal`. The cost is the shortest path distance between `l_p` and `l_goal` on the road network.
       - It needs to be dropped at `l_goal` (1 action).
       - Total cost for this package: 1 (pick-up) + distance(`l_p`, `l_goal`) + 1 (drop).
    3. If the package is inside a vehicle `v`:
       - Find the current location `l_v` of vehicle `v`.
       - If `l_v` is the goal location `l_goal`:
         - It needs to be dropped (1 action).
         - Total cost for this package: 1 (drop).
       - If `l_v` is not the goal location `l_goal`:
         - The vehicle needs to drive from `l_v` to `l_goal`. The cost is the shortest path distance between `l_v` and `l_goal`.
         - It needs to be dropped at `l_goal` (1 action).
         - Total cost for this package: distance(`l_v`, `l_goal`) + 1 (drop).
    4. The total heuristic value is the sum of the costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the road graph.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Build the road network graph
        self.road_graph = {}
        # Collect all locations first to ensure all nodes are in the graph, even if isolated
        all_locations = set()
        for fact in static_facts:
             if match(fact, "road", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, loc1, loc2 = parts
                     all_locations.add(loc1)
                     all_locations.add(loc2)
        # Initialize graph with all locations
        for loc in all_locations:
            self.road_graph[loc] = []
        # Add edges
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, loc1, loc2 = parts
                    self.road_graph[loc1].append(loc2)
                    # Assuming roads are bidirectional based on example instance
                    self.road_graph[loc2].append(loc1)


        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Ensure the goal fact has the expected number of arguments
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location
                # else: Handle unexpected goal format? Assuming valid PDDL goals.
            # Handle potential 'in' goals? The domain doesn't seem to have them.
            # If it did, we'd need to decide how to handle them. Assuming only 'at' goals for packages.

        # Store capacity predecessors (not used in this simple heuristic, but extracted)
        self.capacity_predecessors = {}
        for fact in static_facts:
             if match(fact, "capacity-predecessor", "*", "*"):
                 # Ensure the fact has the expected number of arguments
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, s1, s2 = parts
                     self.capacity_predecessors[s2] = s1 # s1 is predecessor of s2
                 # else: Handle unexpected fact format? Assuming valid PDDL facts.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Track current locations of packages and vehicles
        current_locations = {} # Maps locatable object (package or vehicle) to its location string
        package_in_vehicle = {} # Maps package to vehicle it's in, if any

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                # Ensure the fact has the expected number of arguments
                if len(args) == 2:
                    obj, location = args # obj is either vehicle or package
                    current_locations[obj] = location
                # else: Handle unexpected fact format? Assuming valid PDDL facts.
            elif predicate == "in":
                 # Ensure the fact has the expected number of arguments
                if len(args) == 2:
                    package, vehicle = args
                    package_in_vehicle[package] = vehicle
                    # The package's "location" is the vehicle it's in for lookup purposes
                    current_locations[package] = vehicle # Store vehicle name as package location
                # else: Handle unexpected fact format? Assuming valid PDDL facts.


        total_cost = 0

        # Iterate through packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # Find the package's current state (location or vehicle)
            current_state_loc = current_locations.get(package)

            # If a package listed in the goal is not found in the current state,
            # it implies an invalid state or the package was never in the initial state.
            # For a heuristic, we can treat this as an unreachable goal for this package.
            if current_state_loc is None:
                 return float('inf') # Goal is unreachable from this state

            # Check if package is already at goal
            # A package is at goal if it's on the ground at the goal location
            # Check if the package is in the package_in_vehicle map. If not, it's on the ground.
            is_on_ground = package not in package_in_vehicle
            is_at_goal = is_on_ground and (current_state_loc == goal_location)

            if is_at_goal:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal. Calculate cost to get it there.
            if is_on_ground:
                # Package is on the ground at current_state_loc (which is a location string)
                current_package_location = current_state_loc
                # Cost: pick-up (1) + drive (dist) + drop (1)
                drive_cost = bfs_shortest_path(self.road_graph, current_package_location, goal_location)
                if drive_cost == float('inf'):
                     # If goal location is unreachable from package's current location
                     return float('inf')
                total_cost += 1 + drive_cost + 1 # pick + drive + drop
            else:
                # Package is in a vehicle (current_state_loc is the vehicle name)
                vehicle_name = current_state_loc
                current_vehicle_location = current_locations.get(vehicle_name) # Get vehicle's location

                if current_vehicle_location is None:
                    # This implies the vehicle carrying the package has no location.
                    # Treat as unreachable for this package.
                     return float('inf')

                # Cost: drive (dist) + drop (1)
                drive_cost = bfs_shortest_path(self.road_graph, current_vehicle_location, goal_location)
                if drive_cost == float('inf'):
                     # If goal location is unreachable from vehicle's current location
                     return float('inf')
                total_cost += drive_cost + 1 # drive + drop

        return total_cost
