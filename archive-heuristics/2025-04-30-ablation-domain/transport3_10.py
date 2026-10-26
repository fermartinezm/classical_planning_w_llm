from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections
import math

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

# Heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, considering load, unload, and drive actions.
    It sums the estimated costs for each package independently, ignoring vehicle
    capacity constraints and shared vehicle trips.

    # Assumptions
    - The cost of each action (load, unload, drive) is 1.
    - Packages can be transported independently.
    - A suitable vehicle is always available at the package's current location
      or can reach it with minimum drive actions (which are accounted for
      in the package's movement cost).
    - Vehicle capacity is not a bottleneck in this estimate.
    - Objects starting with 'p' are packages, objects starting with 'v' are vehicles.

    # Heuristic Initialization
    - Parses static facts to build a road network graph and compute all-pairs
      shortest path distances between locations.
    - Parses goal facts to identify the target location for each package.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the current location or containment status for every package
       and the current location for every vehicle by iterating through the state facts.
    3. Initialize total estimated cost to 0.
    4. For each package that has a goal location defined:
       a. Determine the package's current physical location (either on the ground
          at a location or inside a vehicle that is at a location). If the package
          or its carrying vehicle's location is unknown, the state is likely invalid;
          return infinity.
       b. Get the package's goal location (pre-calculated during initialization).
       c. If the package is already at its goal location (and on the ground), add 0 to the total cost for this package.
       d. If the package is currently inside a vehicle that is already at the goal location:
          - Add 1 (unload) to the total cost.
       e. If the package is currently on the ground at `current_loc` (which is not the goal):
          - Find the shortest distance `dist` from `current_loc` to `goal_loc` using the pre-calculated distances.
          - If `goal_loc` is unreachable from `current_loc`, the state is likely unsolvable; return infinity.
          - Add 1 (load) + `dist` (drive actions) + 1 (unload) to the total cost.
       f. If the package is currently inside a vehicle at `current_loc` (which is not the goal vehicle location):
          - Find the shortest distance `dist` from `current_loc` to `goal_loc` using the pre-calculated distances.
          - If `goal_loc` is unreachable from `current_loc`, the state is likely unsolvable; return infinity.
          - Add `dist` (drive actions) + 1 (unload) to the total cost.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and initial state information.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?p ?l)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Assuming task.goals is a set of ground facts that must be true.

        # Build the road network graph and find all locations.
        self.road_graph = collections.defaultdict(set)
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Roads are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Compute all-pairs shortest path distances using BFS from each location.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            # Initialize distances
            for loc in self.locations:
                self.distances[start_loc][loc] = math.inf
            self.distances[start_loc][start_loc] = 0

            # BFS
            queue = collections.deque([start_loc])
            while queue:
                current_loc = queue.popleft()
                current_dist = self.distances[start_loc][current_loc]

                if current_loc in self.road_graph: # Check if current_loc has roads
                    for neighbor in self.road_graph[current_loc]:
                        if self.distances[start_loc][neighbor] == math.inf:
                            self.distances[start_loc][neighbor] = current_dist + 1
                            queue.append(neighbor)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Track current locations/status of packages and vehicles
        package_status = {} # {package_name: {'type': 'at', 'loc': loc} or {'type': 'in', 'vehicle': vehicle_name}}
        vehicle_locations = {} # {vehicle_name: loc}

        # Iterate through state facts to populate status dictionaries
        for fact in state:
            parts = get_parts(fact)
            if len(parts) < 2: continue # Skip malformed facts

            predicate = parts[0]
            obj1 = parts[1]

            if predicate == "at" and len(parts) == 3:
                 loc = parts[2]
                 if obj1.startswith('p'): # Assuming 'p' objects are packages based on examples
                     package_status[obj1] = {'type': 'at', 'loc': loc}
                 elif obj1.startswith('v'): # Assuming 'v' objects are vehicles based on examples
                     vehicle_locations[obj1] = loc
                 # Add other object types if necessary based on domain analysis

            elif predicate == "in" and len(parts) == 3:
                 package, vehicle = obj1, parts[2]
                 if package.startswith('p'): # Assuming 'p' objects are packages
                     package_status[package] = {'type': 'in', 'vehicle': vehicle}
                 # Add other containment types if necessary

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that has a goal
        for package, goal_location in self.goal_locations.items():
            # If package is not mentioned in the current state facts relevant to location/containment,
            # its status is unknown. This shouldn't happen in valid states.
            if package not in package_status:
                 # State is likely invalid if a goal package's location is unknown
                 return math.inf

            current_status = package_status[package]

            # Determine the package's current physical location
            current_physical_location = None
            if current_status['type'] == 'at':
                current_physical_location = current_status['loc']
            elif current_status['type'] == 'in':
                vehicle = current_status['vehicle']
                # The physical location of the package is the location of the vehicle
                if vehicle in vehicle_locations:
                    current_physical_location = vehicle_locations[vehicle]
                else:
                    # Vehicle location is unknown - this indicates an invalid state
                    return math.inf # Cannot determine package location, state is likely invalid/unsolvable

            # If we couldn't determine the physical location (shouldn't happen with checks above)
            if current_physical_location is None:
                 return math.inf # State is likely invalid/unsolvable


            # Check if the package is already at the goal location (on the ground)
            if current_status['type'] == 'at' and current_physical_location == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # If the package is inside a vehicle already at the goal location, cost is 1 (unload)
            if current_status['type'] == 'in' and current_physical_location == goal_location:
                 total_cost += 1
                 continue # Done with this package

            # Package is not at goal location (either on ground or in vehicle elsewhere)
            # Calculate distance from current physical location to goal location
            dist = self.distances.get(current_physical_location, {}).get(goal_location, math.inf)

            if dist == math.inf:
                # Goal location is unreachable from current physical location
                return math.inf # This state is likely unsolvable

            # Add cost based on status:
            if current_status['type'] == 'at':
                # Package on ground, not at goal. Needs Load + Drive + Unload.
                total_cost += 1 + dist + 1
            elif current_status['type'] == 'in':
                # Package in vehicle, not at goal vehicle location. Needs Drive + Unload.
                total_cost += dist + 1

        return total_cost
