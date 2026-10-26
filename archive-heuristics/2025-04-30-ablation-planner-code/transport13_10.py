from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 locationA)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each package
    to its goal location, considering pick-up, drop, and drive actions. It calculates
    the shortest path distance between locations using BFS on the road network.
    The heuristic is the sum of the estimated costs for each package that is not yet
    at its goal location on the ground.

    # Assumptions
    - Each package needs to reach a specific goal location specified in the task goals.
    - Vehicle capacity and availability are ignored in the distance calculation. This is a relaxation.
    - The cost of moving a package from its current effective location (ground or vehicle's location)
      to its goal location is the shortest path distance (number of drive actions) plus
      1 action for dropping the package.
    - If the package is currently on the ground, an additional 1 action is added for picking it up.
    - All locations mentioned in the problem (initial state, goals, roads) are part of the road network graph.
    - Roads are bidirectional.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goal conditions.
    - Identifies objects by type (packages, vehicles, locations) from the task's objects dictionary.
    - Builds a graph of locations based on 'road' facts found in the static information.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS)
      on the constructed road graph. These distances represent the minimum number of 'drive' actions
      between any two locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (represented as a frozenset of fact strings):
    1. Parse the state to determine the current status of each package (on the ground at a location, or inside a vehicle) and the current location of each vehicle.
    2. Initialize the total heuristic cost `h` to 0.
    3. Iterate through each package that has a specified goal location:
       a. Get the package's current status (location or vehicle).
       b. Determine the package's *effective current location* for transport. If the package is on the ground, this is its current location. If it's in a vehicle, this is the vehicle's current location.
       c. If the package is on the ground *and* its current location is the goal location, add 0 to `h` for this package and continue to the next package.
       d. If the package is not yet at its goal location on the ground:
          - Calculate the minimum number of drive actions required to move from the package's effective current location to its goal location. This is the shortest path distance obtained from the precomputed distances. If no path exists, the distance is considered infinite, and the heuristic should return infinity as the state is likely unsolvable.
          - Add this drive distance to the cost for the current package.
          - Add 1 to the cost for the necessary 'drop' action at the goal location.
          - If the package was initially on the ground (not in a vehicle), add an additional 1 to the cost for the necessary 'pick-up' action.
       e. Add the calculated cost for this package to the total heuristic cost `h`.
    4. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the road graph,
        and computing shortest path distances.
        """
        # Call the base class constructor
        super().__init__(task)

        # Identify object types from task.objects (assuming it's a dict {name: type})
        self.packages = {obj for obj, obj_type in self.objects.items() if obj_type == 'package'}
        self.vehicles = {obj for obj, obj_type in self.objects.items() if obj_type == 'vehicle'}
        self.locations = {obj for obj, obj_type in self.objects.items() if obj_type == 'location'}
        self.sizes = {obj for obj, obj_type in self.objects.items() if obj_type == 'size'} # Not strictly needed for this heuristic

        # Store goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            if match(goal, "at", "*", "*"):
                predicate, package, location = get_parts(goal)
                # Ensure the object is a package and the location is a valid location object
                if package in self.packages and location in self.locations:
                     self.package_goals[package] = location
            # Ignore other goal types for this heuristic

        # Build the location graph from road facts
        self.location_graph = {}
        # Initialize graph with all known locations, even if isolated
        for loc in self.locations:
             self.location_graph[loc] = []

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                predicate, loc1, loc2 = get_parts(fact)
                # Ensure loc1 and loc2 are valid location objects
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph.setdefault(loc1, []).append(loc2)
                    self.location_graph.setdefault(loc2, []).append(loc1) # Roads are typically bidirectional


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.location_graph:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start_node to compute distances to all reachable nodes."""
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, current_dist = q.popleft()

            # Ensure current_node is in the graph (should be if from self.location_graph keys)
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = current_dist + 1
                        # Store bidirectional distance as roads are assumed bidirectional
                        self.distances[(neighbor, start_node)] = current_dist + 1
                        q.append((neighbor, current_dist + 1))

        # Locations not visited from start_node remain unreachable, distance is implicitly infinity
        # when looked up using .get() with a default value.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Track current status of packages and vehicles
        # package_current_status: {package_name: location_name or vehicle_name}
        # vehicle_current_locations: {vehicle_name: location_name}
        package_current_status = {}
        vehicle_current_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages and loc in self.locations:
                    package_current_status[obj] = loc
                elif obj in self.vehicles and loc in self.locations:
                    vehicle_current_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles:
                     package_current_status[package] = vehicle
            # Ignore other predicates like capacity, capacity-predecessor, road

        total_cost = 0

        # Calculate cost for each package that needs to reach a goal
        for package, goal_location in self.package_goals.items():
            # If package is not in the current state (e.g., problem setup issue), skip it.
            # In valid states, all objects should have a location or be in a vehicle.
            if package not in package_current_status:
                 # This indicates an unexpected state representation or problem definition.
                 # For robustness, we could return infinity or log a warning.
                 # Assuming valid states, this case shouldn't be reached for packages in goals.
                 continue

            current_status = package_current_status[package]

            # Determine the package's effective current location for driving
            effective_current_location = None
            is_on_ground = False

            if current_status in self.locations: # Package is on the ground
                effective_current_location = current_status
                is_on_ground = True
            elif current_status in self.vehicles: # Package is in a vehicle
                vehicle = current_status
                # Find the vehicle's location
                if vehicle in vehicle_current_locations:
                    effective_current_location = vehicle_current_locations[vehicle]
                else:
                    # Vehicle location unknown - problem state is likely invalid or unsolvable
                    # The heuristic cannot proceed meaningfully.
                    return float('inf')
            else:
                 # Current status is neither a known location nor a known vehicle - invalid state
                 return float('inf')

            # If the package is already at its goal location (on the ground), cost is 0 for this package
            if is_on_ground and effective_current_location == goal_location:
                 continue

            # Calculate drive cost from effective current location to goal location
            # Use .get() with infinity default for unreachable locations
            drive_cost = self.distances.get((effective_current_location, goal_location), float('inf'))

            # If goal is unreachable from the package's current effective location, return infinity
            if drive_cost == float('inf'):
                return float('inf')

            # Calculate total cost for this package
            # Cost = drive actions + drop action
            package_cost = drive_cost + 1 # +1 for drop action

            # If the package was on the ground, add pick-up action
            if is_on_ground:
                package_cost += 1 # +1 for pick-up action
            # Note: If package is in a vehicle *at* the goal location, drive_cost is 0,
            # package_cost is 0 + 1 (drop) = 1. This is correct.

            total_cost += package_cost

        return total_cost
