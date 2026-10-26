# Required imports
import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (adapted from provided examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
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

# The heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package from its current location to its goal location, ignoring
    vehicle capacity constraints and potential conflicts when multiple
    packages need the same vehicle or path. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Each package needs to be loaded into a vehicle, transported, and unloaded.
    - The cost of loading and unloading is 1 action each.
    - The cost of transporting a package between two locations is the shortest
      path distance (number of drive actions) between those locations for a vehicle.
    - Vehicle capacity and availability are not considered in the cost calculation.
    - If a package is already in a vehicle, the load action is not needed.
    - If a package is already at its goal location on the ground, no actions
      are needed for that package.
    - The road network is static and bidirectional.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Identifies all locations and builds the road network graph from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies packages from goals and vehicles from initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package. A package can be on the
       ground at a location `L` (`(at P L)`) or inside a vehicle `V` (`(in P V)`).
       If inside a vehicle, find the vehicle's current location `L_V` (`(at V L_V)`).
    2. For each package `P` that is not yet at its goal location `Goal_P` on the ground:
       a. If `P` is on the ground at `Loc_P`:
          - The estimated cost for this package is 1 (load) + shortest_distance(`Loc_P`, `Goal_P`) + 1 (unload).
       b. If `P` is inside vehicle `V` which is at `Loc_V`:
          - The estimated cost for this package is shortest_distance(`Loc_V`, `Goal_P`) + 1 (unload).
    3. The total heuristic value for the state is the sum of the estimated costs
       for all packages that are not yet at their goal location on the ground.
    4. If a package is already at its goal location on the ground, its cost is 0.
    5. If a package is at its goal location but inside a vehicle, its cost is 1 (for unload).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and building the road network graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package and identify packages
        self.goal_locations = {}
        self.packages = set()
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        # Identify all locations from road facts
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Build the road graph (adjacency list)
        self.graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Roads are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Identify vehicles from initial state facts
        self.vehicles = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts:
                 # Vehicles have capacity
                 if parts[0] == "capacity" and len(parts) == 3:
                     self.vehicles.add(parts[1])
                 # Vehicles are at locations and are not packages
                 elif parts[0] == "at" and len(parts) == 3 and parts[2] in self.locations:
                     obj = parts[1]
                     if obj not in self.packages:
                          self.vehicles.add(obj)
                 # Vehicles can contain packages
                 elif parts[0] == "in" and len(parts) == 3:
                     # The second argument of 'in' is the container, which must be a vehicle
                     obj = parts[2]
                     if obj not in self.packages: # Ensure it's not a package inside another package (invalid)
                         self.vehicles.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles
        package_current_loc = {}
        package_current_vehicle = {}
        vehicle_current_loc = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_loc[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_loc[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                 obj, vehicle = parts[1], parts[2]
                 if obj in self.packages and vehicle in self.vehicles:
                     package_current_vehicle[obj] = vehicle


        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if package in package_current_loc and package_current_loc[package] == goal_location:
                 # Goal reached for this package on the ground
                 continue

            # Package is not yet at its goal location on the ground. Calculate cost.
            cost_for_package = 0

            if package in package_current_loc:
                # Package is on the ground at current_loc
                current_loc = package_current_loc[package]
                # Cost = Load + Drive + Unload
                # Load cost = 1
                # Drive cost = distance from current_loc to goal_location
                # Unload cost = 1
                drive_cost = self.distances.get((current_loc, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # If goal is unreachable, the problem is likely unsolvable from here
                    return float('inf')
                cost_for_package = 1 + drive_cost + 1

            elif package in package_current_vehicle:
                # Package is inside a vehicle
                vehicle = package_current_vehicle[package]
                vehicle_loc = vehicle_current_loc.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle location unknown - indicates an invalid state or parsing issue
                    # Treat as unreachable for safety
                    return float('inf')

                # Cost = Drive + Unload
                # Drive cost = distance from vehicle_loc to goal_location
                # Unload cost = 1
                drive_cost = self.distances.get((vehicle_loc, goal_location), float('inf'))
                if drive_cost == float('inf'):
                     # If goal is unreachable, the problem is likely unsolvable from here
                    return float('inf')
                cost_for_package = drive_cost + 1
            else:
                 # Package exists but is neither at a location nor in a vehicle? Invalid state.
                 # This could happen if a package is 'in' a vehicle, but the vehicle's location
                 # is not asserted in the state. Assuming valid states, this branch shouldn't be hit.
                 # Return inf to signal an issue or unsolvability from this state.
                 return float('inf')

            total_cost += cost_for_package

        return total_cost
