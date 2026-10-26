import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, summing the costs for all packages.
    It relaxes the problem by ignoring vehicle capacity constraints and
    assuming a suitable vehicle is available when needed for pick-up.
    The cost for a package is estimated based on its current state (on ground
    or in a vehicle) and the shortest path distance between locations.

    # Assumptions
    - The primary goal is to move packages to specific locations.
    - Vehicles are the only means of transport for packages.
    - Vehicles move between locations connected by roads.
    - Packages must be picked up by a vehicle and dropped at the destination.
    - Vehicle capacity is ignored when estimating the cost for a single package.
    - Vehicle availability at a package's location for pick-up is ignored.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build a graph of locations based on `road` predicates.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state, the heuristic calculates the cost as follows:

    1. Identify the current location or containment status for all packages and vehicles.
    2. For each package that is not yet at its goal location:
       a. Determine the package's current state:
          - Is it on the ground at some location `current_loc`?
          - Is it inside a vehicle `v`?
       b. If the package is on the ground at `current_loc` (and `current_loc` is not the goal):
          - It needs to be picked up (1 action).
          - It needs to be transported by a vehicle from `current_loc` to its `goal_loc`. The minimum number of drive actions is the shortest path distance `dist(current_loc, goal_loc)`.
          - It needs to be dropped at `goal_loc` (1 action).
          - Estimated cost for this package: 1 (pick-up) + `dist(current_loc, goal_loc)` (drive) + 1 (drop).
       c. If the package is inside a vehicle `v`:
          - Find the current location `vehicle_loc` of vehicle `v`.
          - If `vehicle_loc` is not the `goal_loc`:
            - The vehicle needs to transport the package from `vehicle_loc` to `goal_loc`. Minimum drive actions: `dist(vehicle_loc, goal_loc)`.
            - It needs to be dropped at `goal_loc` (1 action).
            - Estimated cost for this package: `dist(vehicle_loc, goal_loc)` (drive) + 1 (drop).
          - If `vehicle_loc` is the `goal_loc`:
            - It needs to be dropped at `goal_loc` (1 action).
            - Estimated cost for this package: 1 (drop).
    3. The total heuristic value is the sum of the estimated costs for all packages not at their goal.
    4. If any required location is unreachable (distance is infinity), the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is always (at package location)
                package, location = args
                self.package_goals[package] = location

        # Build the road graph and compute all-pairs shortest paths.
        self.distances = self._compute_all_pairs_shortest_paths(static_facts)

    def _compute_all_pairs_shortest_paths(self, static_facts):
        """
        Builds the road graph from static facts and computes shortest path
        distances between all pairs of locations using BFS.
        """
        locations = set()
        graph = {}

        # Build the graph from road facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in graph:
                    graph[loc1] = []
                graph[loc1].append(loc2)

        # Ensure all locations mentioned in goals or potentially in state are in the graph
        # This is important if a location exists but has no roads connected in static facts
        for goal_loc in self.package_goals.values():
             locations.add(goal_loc)
        # We don't have access to all possible locations from the domain definition here,
        # but locations in static facts and goals should cover most cases in typical problems.
        # A more robust approach would parse all objects of type 'location'.
        # For this implementation, we rely on 'road' facts and goal locations.

        # Initialize distances dictionary
        distances = {loc: {other_loc: math.inf for other_loc in locations} for loc in locations}
        for loc in locations:
            distances[loc][loc] = 0 # Distance from a location to itself is 0

        # Compute shortest paths using BFS from each location
        for start_node in locations:
            q = deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_loc, dist = q.popleft()
                distances[start_node][current_loc] = dist

                # Get neighbors, handling locations with no outgoing roads
                neighbors = graph.get(current_loc, [])
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # Maps object name (package or vehicle) to its location string or the vehicle it's in.
        current_status = {}
        # Need vehicle locations to determine where packages inside them are.
        vehicle_locations = {}

        # First pass: find locations of everything (especially vehicles)
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, loc = args
                current_status[obj] = loc
                # Also store vehicle locations separately for quick lookup
                # We need to know which objects are vehicles. The domain file
                # tells us types, but the state facts don't explicitly.
                # We can infer vehicles from (at ?v ?l) where ?v is not a package.
                # A simpler approach for this heuristic is to assume anything
                # appearing as the second argument of 'in' is a vehicle.
                # Or, we can check if the object appears in a capacity predicate.
                # Let's assume anything with a capacity predicate is a vehicle.
                # A more robust way would be to parse types from the domain,
                # but we only have state and static facts here.
                # Let's infer vehicles from 'capacity' facts in the static info
                # or state. Or, more simply, just assume the second arg of 'in' is a vehicle.
                # Let's refine: vehicles are objects that appear in 'at' facts AND can carry packages ('in' facts).
                # Or, even simpler, just look up the location of the object a package is 'in'.
                # Let's use the 'at' facts for vehicles directly.
                # We need to know which objects are vehicles. We can get this from the state
                # by looking for objects in 'at' facts that are not packages (packages are
                # objects that appear in goal 'at' facts or 'in' facts).
                # Let's assume objects in 'capacity' facts are vehicles.
                # A more general approach: objects in 'at' facts that are NOT in the package_goals keys are vehicles.
                # This is still not perfect if a vehicle is empty and not in goals.
                # Let's assume for simplicity that any object appearing in an (at ?obj ?loc) fact
                # that is NOT a package (i.e., not in self.package_goals keys) is a vehicle.
                if obj not in self.package_goals: # Heuristic assumption: objects in 'at' not in goals are vehicles
                     vehicle_locations[obj] = loc


        # Second pass: find packages inside vehicles
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "in":
                 package, vehicle = args
                 current_status[package] = vehicle # Store the vehicle name as the package's status

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal
            if package in current_status and current_status[package] == goal_location:
                 # Check if the package is on the ground at the goal location
                 if f"(at {package} {goal_location})" in state:
                    continue # Package is already at the goal location on the ground. Cost is 0 for this package.
                 # If package is IN a vehicle at the goal location, it still needs to be dropped.
                 # This case is handled below.

            # Package is not yet satisfied at the goal location on the ground.
            # Determine package's current status
            current_status_val = current_status.get(package)

            if current_status_val is None:
                 # This package doesn't appear in 'at' or 'in' facts. This shouldn't happen
                 # in a valid state if the package exists, but handle defensively.
                 # Assume it's unreachable or some error state.
                 return math.inf # Or a large number

            # Case 1: Package is on the ground at current_loc
            if f"(at {package} {current_status_val})" in state:
                current_loc = current_status_val
                # We already checked if current_loc == goal_location above.
                # If we are here, current_loc != goal_location.
                
                # Cost: pick-up + drive + drop
                # Need distance from current_loc to goal_location
                if current_loc not in self.distances or goal_location not in self.distances[current_loc]:
                    return math.inf # Goal location unreachable from package's current location

                drive_cost = self.distances[current_loc][goal_location]
                if drive_cost == math.inf:
                     return math.inf # Unreachable

                total_cost += 1 # pick-up
                total_cost += drive_cost # drive actions
                total_cost += 1 # drop

            # Case 2: Package is inside a vehicle
            elif f"(in {package} {current_status_val})" in state:
                vehicle = current_status_val
                # Find the vehicle's location
                vehicle_loc = vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle location unknown. Should not happen in a valid state.
                    return math.inf # Or a large number

                # If vehicle is already at the goal location, only drop is needed
                if vehicle_loc == goal_location:
                    total_cost += 1 # drop
                else:
                    # Vehicle needs to drive to the goal location, then drop
                    if vehicle_loc not in self.distances or goal_location not in self.distances[vehicle_loc]:
                         return math.inf # Goal location unreachable from vehicle's current location

                    drive_cost = self.distances[vehicle_loc][goal_location]
                    if drive_cost == math.inf:
                         return math.inf # Unreachable

                    total_cost += drive_cost # drive actions
                    total_cost += 1 # drop

            # Else: Package status is something unexpected.
            else:
                 # This case should not be reached if state facts are well-formed
                 # and cover all packages mentioned in goals.
                 return math.inf # Or a large number


        return total_cost

