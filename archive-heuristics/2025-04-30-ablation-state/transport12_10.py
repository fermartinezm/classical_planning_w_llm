# Assuming the Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args for a valid match attempt
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a minimal Heuristic base class if it's not provided externally
# This is just a placeholder if the import fails.
# In a real environment, the actual base class would be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently. The cost for a package includes:
    1. Picking up the package (if it's on the ground).
    2. Driving the vehicle carrying the package to the goal location (shortest path).
    3. Dropping the package at the goal location.

    # Assumptions
    - Capacity constraints are ignored. Any vehicle can pick up any package.
    - A suitable vehicle is always available at the package's current location
      or the package is already in a vehicle.
    - The road network is static and bidirectional (if road l1 l2 exists, road l2 l1 exists).
    - All goal locations are reachable from current package locations via the road network
      in solvable problems.

    # Heuristic Initialization
    - Infers object types (packages, vehicles, locations, sizes) from predicate usage
      in the initial state, static facts, and goals. This is a heuristic approach
      as the PDDL object types are not directly parsed.
    - Extracts the road network from static facts to build an adjacency list graph
      for shortest path calculations.
    - Extracts the goal location for each package that is specified in the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every locatable object (packages and vehicles)
       by examining the `(at ?x ?l)` and `(in ?p ?v)` facts in the state.
       Store the immediate container/location for each object (either a location name or a vehicle name).
       Store the physical location for each vehicle.
       If any package or vehicle's location/containment is not defined in the state,
       or if a package is in a vehicle whose location is unknown, the state is considered
       invalid and the heuristic returns infinity.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined in the problem:
       a. Check if the package is already at its goal location on the ground. This is done
          by checking if the exact goal fact `(at package goal_location)` is present in the state.
          If yes, this package contributes 0 to the heuristic, continue to the next package.
       b. If the package is not at its goal location on the ground:
          i. Determine the package's current physical location. If it's inside a vehicle,
             find the vehicle's physical location.
          ii. Add 1 to the cost for this package if it is currently on the ground
              (representing the pick-up action needed).
          iii. Calculate the shortest path distance (number of drive actions) between the package's
               current physical location and its goal location using BFS on the road graph.
               If the goal is unreachable from the current location, return infinity for the total heuristic.
          iv. Add this shortest path distance to the cost for this package.
          v. Add 1 to the cost for this package (representing the drop action needed).
       c. Add the calculated cost for this package to the total heuristic cost.
    4. The total sum is the heuristic value for the state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Infer object types from predicate usage in initial state, static facts, and goals
        all_facts = set(initial_state) | set(static_facts) | set(self.goals)

        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()

        # Collect all objects mentioned in relevant predicates
        potential_packages = set()
        potential_vehicles = set()
        potential_locations = set()
        potential_sizes = set()

        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts if any
             pred = parts[0]
             if pred == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 potential_packages.add(obj) # Assume locatable is package until proven vehicle
                 potential_locations.add(loc)
             elif pred == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 potential_packages.add(pkg)
                 potential_vehicles.add(veh)
             elif pred == "capacity" and len(parts) == 3:
                 veh, size = parts[1], parts[2]
                 potential_vehicles.add(veh)
                 potential_sizes.add(size)
             elif pred == "capacity-predecessor" and len(parts) == 3:
                 s1, s2 = parts[1], parts[2]
                 potential_sizes.add(s1)
                 potential_sizes.add(s2)
             elif pred == "road" and len(parts) == 3:
                 loc1, loc2 = parts[1], parts[2]
                 potential_locations.add(loc1)
                 potential_locations.add(loc2)

        # Refine types based on typical usage patterns in the domain
        # An object is a vehicle if it appears in 'capacity' or as the second arg of 'in'.
        self.vehicles = potential_vehicles.copy()
        # An object is a package if it appears as the first arg of 'in' or has a goal location,
        # AND is not a vehicle.
        packages_from_in = {get_parts(fact)[1] for fact in all_facts if get_parts(fact)[0] == "in" and len(get_parts(fact)) == 3}
        packages_from_goals = {args[0] for goal in self.goals for predicate, *args in [get_parts(goal)] if predicate == "at" and len(get_parts(goal)) == 3}
        self.packages = (packages_from_in | packages_from_goals) - self.vehicles
        # Locations are objects appearing in 'road' or as the second arg of 'at'.
        locations_from_road = {arg for fact in all_facts if get_parts(fact)[0] == "road" and len(get_parts(fact)) == 3 for arg in get_parts(fact)[1:]}
        locations_from_at = {get_parts(fact)[2] for fact in all_facts if get_parts(fact)[0] == "at" and len(get_parts(fact)) == 3}
        self.locations = locations_from_road | locations_from_at
        # Sizes are objects appearing in 'capacity' or 'capacity-predecessor'.
        sizes_from_capacity = {get_parts(fact)[2] for fact in all_facts if get_parts(fact)[0] == "capacity" and len(get_parts(fact)) == 3}
        sizes_from_predecessor = {arg for fact in all_facts if get_parts(fact)[0] == "capacity-predecessor" and len(get_parts(fact)) == 3 for arg in get_parts(fact)[1:]}
        self.sizes = sizes_from_capacity | sizes_from_predecessor


        # Build the road network graph (adjacency list).
        self.road_graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Only store goals for objects identified as packages
                if package in self.packages:
                    self.goal_locations[package] = location

    def _shortest_path_distance(self, start_loc, end_loc):
        """
        Computes the shortest path distance (number of drive actions)
        between two locations using BFS on the road graph.
        Returns infinity if end_loc is unreachable from start_loc.
        """
        if start_loc == end_loc:
            return 0

        # Handle cases where start or end location is not in the graph (e.g., invalid state)
        if start_loc not in self.road_graph or end_loc not in self.road_graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc is still valid in the graph during traversal
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Goal is unreachable from start_loc in this graph.
        return float('inf')

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track where packages and vehicles are currently located.
        # Maps locatable object name -> its immediate container/location name
        current_pos_map = {}
        # Maps vehicle name -> its physical location
        vehicle_physical_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_pos_map[obj] = loc
                if obj in self.vehicles:
                    vehicle_physical_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                current_pos_map[pkg] = veh

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # This is the goal condition we care about.
            if f"(at {package} {goal_location})" in state:
                 # Package is at the goal location on the ground, cost is 0 for this package
                 continue

            # Package is not yet at its goal location on the ground. Calculate its cost.

            # Find the package's current position (location or vehicle)
            pkg_current_pos = current_pos_map.get(package)

            # If package is not found in the state, it's an invalid state.
            # This shouldn't happen in valid states generated by the planner.
            # Return infinity to prune.
            if pkg_current_pos is None:
                 return float('inf')

            # Determine the package's physical location and if it's on the ground
            pkg_physical_location = None
            is_on_ground = False

            if pkg_current_pos in self.locations:
                 pkg_physical_location = pkg_current_pos
                 is_on_ground = True
            elif pkg_current_pos in self.vehicles:
                 vehicle = pkg_current_pos
                 pkg_physical_location = vehicle_physical_locations.get(vehicle)
                 is_on_ground = False
            else:
                 # pkg_current_pos is neither a known location nor a known vehicle.
                 # This state is invalid according to the domain predicates.
                 # Return infinity to prune this branch.
                 return float('inf')

            # If the package's physical location couldn't be determined (e.g., vehicle location unknown)
            if pkg_physical_location is None:
                 # This happens if a package is 'in' a vehicle, but the vehicle's 'at' location is missing.
                 # Invalid state. Return infinity.
                 return float('inf')

            # Calculate cost for this package
            package_cost = 0

            # Cost for pick-up if on the ground
            if is_on_ground:
                 package_cost += 1

            # Cost for driving
            drive_cost = self._shortest_path_distance(pkg_physical_location, goal_location)
            if drive_cost == float('inf'):
                 # Goal is unreachable from current location. This state is likely
                 # a dead end or indicates an unsolvable problem.
                 # For a non-admissible heuristic, return a very large value.
                 return float('inf')

            package_cost += drive_cost

            # Cost for drop
            package_cost += 1

            total_cost += package_cost

        # The heuristic is 0 only if all packages are at their goal locations.
        # The loop above only adds cost for packages *not* at their goal.
        # If all packages are at their goal, the loop is skipped, and total_cost remains 0.
        # This satisfies the requirement that h=0 only for goal states.

        return total_cost
