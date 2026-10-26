# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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


# Helper function for BFS shortest paths
def bfs_shortest_paths(locations, road_facts):
    """
    Computes shortest path distances between all pairs of locations
    based on road connections.

    Args:
        locations: A set of all location object names (e.g., {'l1', 'l2', ...}).
        road_facts: A set of static facts like '(road l1 l2)'.

    Returns:
        A dictionary where dist[l1][l2] is the shortest distance (number of roads)
        from l1 to l2. Returns float('inf') if no path exists.
    """
    graph = {loc: set() for loc in locations}
    for fact in road_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'road':
            # road facts are (road l1 l2)
            if len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Ensure locations exist in the graph dictionary before adding edges
                if l1 in graph and l2 in graph:
                    graph[l1].add(l2)
                    graph[l2].add(l1) # Assuming roads are bidirectional

    distances = {}
    for start_loc in locations:
        distances[start_loc] = {loc: float('inf') for loc in locations}
        distances[start_loc][start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[start_loc][current_loc]

            # Ensure current_loc is in graph before accessing neighbors
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if distances[start_loc][neighbor] == float('inf'):
                        distances[start_loc][neighbor] = current_dist + 1
                        queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location.
    It calculates this estimate for each package independently and sums the results.

    # Assumptions
    - The cost of each action (pick-up, drive, drop) is 1.
    - Roads are bidirectional.
    - Shortest path distances between locations can be precomputed.
    - Vehicle availability and capacity constraints are ignored for simplicity.
    - A vehicle is assumed to be available at the package's location (or can reach it instantly)
      and has sufficient capacity when needed.
    - All relevant locations are mentioned in the static `road` facts.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Identifies all locations from the static `road` facts.
    - Builds a graph of the road network using these locations and `road` facts.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. For each package that has a goal location defined in the task:
       a. Determine the package's current status:
          - Is it at a location `(at package location)`?
          - Is it inside a vehicle `(in package vehicle)`?
       b. Check if the package is already at its goal location *on the ground*. If yes, it contributes 0 to the heuristic.
       c. If the package is not at its goal location on the ground, determine its current physical location (where it is, or where the vehicle it's in is).
       d. Calculate the shortest distance from the package's current physical location to its goal location using the precomputed distances. If unreachable, the state is likely unsolvable, return infinity.
       e. Estimate the actions needed:
          - If the package is currently in a vehicle: Cost is `distance_to_goal + 1` (for the drop action).
          - If the package is currently on the ground: Cost is `1` (for pick-up) + `distance_to_goal` + `1` (for drop).
       f. Add this estimated cost for the package to the total heuristic.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location

        # Identify all locations from static road facts.
        locations = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'road':
                 # road facts are (road l1 l2)
                 if len(parts) == 3:
                     locations.add(parts[1])
                     locations.add(parts[2])

        # Precompute shortest path distances between all locations.
        self.distances = bfs_shortest_paths(locations, static_facts)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # This dictionary will map object names (packages, vehicles) to their location or container.
        current_status = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                predicate = parts[0]
                if predicate == "at":
                    # (at obj location)
                    if len(parts) == 3:
                        obj, location = parts[1], parts[2]
                        current_status[obj] = location # obj is at location
                elif predicate == "in":
                    # (in package vehicle)
                    if len(parts) == 3:
                        package, vehicle = parts[1], parts[2]
                        current_status[package] = vehicle # package is in vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # This is the exact goal condition for this package.
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at the goal on the ground. Cost is 0 for this package.

            # Package is not yet satisfying the goal predicate (at package goal_location).
            # It needs to reach goal_location and be on the ground there.

            # Find the package's current status (location or vehicle).
            current_status_of_package = current_status.get(package)

            if current_status_of_package is None:
                 # Package not found in state facts (neither 'at' nor 'in').
                 # This indicates an unexpected state or an object not relevant to goals.
                 # If it's a goal package, this state is likely invalid or unreachable.
                 return float('inf')

            current_physical_location = None
            is_in_vehicle = False

            # Determine current physical location and if it's in a vehicle
            # Check if the status is a known location (i.e., package is on the ground).
            if current_status_of_package in self.distances: # Check if the status string is a known location
                 current_physical_location = current_status_of_package
                 is_in_vehicle = False
            # Check if the status is a vehicle (i.e., package is inside a vehicle).
            # We check if the status string is a key in our current_status map,
            # and if that key (the vehicle) has a location.
            elif current_status_of_package in current_status: # Check if the status string is a known object (likely a vehicle)
                 vehicle = current_status_of_package
                 vehicle_location = current_status.get(vehicle)
                 if vehicle_location is not None and vehicle_location in self.distances:
                     current_physical_location = vehicle_location
                     is_in_vehicle = True
                 else:
                     # Vehicle location unknown or not a valid location.
                     return float('inf')
            else:
                 # The package's status is neither a known location nor a vehicle with a known location.
                 # This indicates an unexpected state representation.
                 return float('inf')

            # Now we have the package's current physical location (current_physical_location)
            # and know if it's inside a vehicle.

            # Calculate the shortest distance from the package's current physical location to its goal location.
            dist_to_goal_location = self.distances.get(current_physical_location, {}).get(goal_location, float('inf'))

            if dist_to_goal_location == float('inf'):
                 # Goal location is unreachable from the package's current physical location via roads.
                 return float('inf') # Unreachable goal location

            # Estimate actions needed:
            # If package is currently in a vehicle:
            #   - Needs `dist_to_goal_location` drive actions by the vehicle.
            #   - Needs 1 drop action at the goal location.
            #   Total cost = dist_to_goal_location + 1
            # If package is currently on the ground:
            #   - Needs 1 pick-up action by a vehicle at its current location.
            #   - Needs `dist_to_goal_location` drive actions by the vehicle.
            #   - Needs 1 drop action at the goal location.
            #   Total cost = 1 + dist_to_goal_location + 1

            if is_in_vehicle:
                 total_cost += dist_to_goal_location + 1
            else:
                 total_cost += 1 + dist_to_goal_location + 1

        # The total_cost is the sum of estimated costs for each package
        # that is not yet at its goal location on the ground.
        # If all packages are at their goal location on the ground, the loop
        # continues for all, and total_cost remains 0. This is correct.

        return total_cost
