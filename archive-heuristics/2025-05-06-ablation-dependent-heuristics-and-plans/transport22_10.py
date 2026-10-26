# from heuristics.heuristic_base import Heuristic # Assuming this is provided

# Define a dummy Heuristic base class if not provided by the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package to its goal location, assuming vehicles are
    available when needed and have sufficient capacity. The total heuristic
    value is the sum of the estimated costs for each package that is not yet
    at its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location defined by an `(at ?p ?l)` goal fact.
    - Vehicles are assumed to be available at the required locations
      (package's current location if on the ground, or vehicle's current location
      if the package is inside a vehicle) and have sufficient capacity to pick up
      the package. The cost of moving a vehicle to a package's location if it's
      not already there is ignored.
    - The cost of each action (drive, pick-up, drop) is 1.
    - The road network is static and defines possible movements for vehicles.
    - Objects starting with 'v' are assumed to be vehicles. Objects in goal facts
      with predicate 'at' are assumed to be packages.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph representation of the road network from the static facts.
    - Compute the shortest path distance between all pairs of locations using BFS.
      These distances represent the minimum number of 'drive' actions needed
      to travel between two locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic value is 0.
    2. Identify the current status (location or inside which vehicle) for every package
       and the current location for every vehicle by parsing the state facts.
       Packages are identified by being present in the goal facts. Vehicles are
       identified by starting with 'v'.
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `goal_l` (as identified during initialization):
       a. Check if the package is currently at its goal location on the ground `(at p goal_l)`. If yes, continue to the next package (cost for this package is 0).
       b. If the package is not at its goal, determine its current status from the parsed state information (`current_package_status`).
       c. If the package's status cannot be determined (e.g., not 'at' a location and not 'in' a vehicle), return infinity as the state is unexpected or unsolvable.
       d. If the package is on the ground at `current_l_p` (i.e., `p`'s status is a location string `current_l_p`):
          - Calculate the shortest path distance `drive_cost` from `current_l_p` to `goal_l` using the precomputed distances (`self.shortest_paths`).
          - If `goal_l` is unreachable from `current_l_p` (distance is None), the state is likely unsolvable or problematic; return infinity.
          - The estimated cost for this package is 1 (pick-up) + `drive_cost` (drive) + 1 (drop).
       e. If the package is inside a vehicle `v` (i.e., `p`'s status is a vehicle name `v`):
          - Find the current location `current_l_v` of vehicle `v` from `current_vehicle_locations`.
          - If vehicle location is not found, return infinity (unexpected state).
          - If `current_l_v` is the same as `goal_l`: The estimated cost is 1 (drop).
          - If `current_l_v` is different from `goal_l`:
            - Calculate the shortest path distance `drive_cost` from `current_l_v` to `goal_l`.
            - If `goal_l` is unreachable from `current_l_v`, return infinity.
            - The estimated cost is `drive_cost` (drive) + 1 (drop).
       f. Add the estimated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph for shortest path calculations.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically '(at package location)'
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph (adjacency list).
        self.locations = set()
        self.road_graph = {} # location -> set of connected locations

        for fact in self.static:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2:
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are typically bidirectional

        # Compute all-pairs shortest paths using BFS from each location.
        self.shortest_paths = {} # (start_l, end_l) -> distance

        for start_l in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_l] = 0
            queue = deque([start_l])

            while queue:
                current_l = queue.popleft()

                # If current_l is not in road_graph, it's an isolated location
                # with no outgoing roads. BFS stops here for this path.
                # Also handle cases where a location might be in self.locations
                # but not have any roads defined in self.road_graph.
                if current_l not in self.road_graph:
                    continue

                for neighbor_l in self.road_graph[current_l]:
                    if distances[neighbor_l] == float('inf'):
                        distances[neighbor_l] = distances[current_l] + 1
                        queue.append(neighbor_l)

            # Store distances from start_l to all reachable locations
            for end_l, dist in distances.items():
                 if dist != float('inf'):
                    self.shortest_paths[(start_l, end_l)] = dist
                 # Note: Unreachable pairs (dist is inf) are not stored,
                 # lookup will return None or require .get(..., float('inf'))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Check if the goal is already reached
        if self.task.goal_reached(state):
             return 0

        # Track current locations of packages and vehicles.
        # package -> location (if on ground) or vehicle (if in vehicle)
        current_package_status = {}
        # vehicle -> location
        current_vehicle_locations = {}

        # Populate status maps by parsing the current state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                # Check if the object is one of the packages we care about (those in goals)
                if obj in self.goal_locations:
                     current_package_status[obj] = loc
                # Assume objects starting with 'v' are vehicles
                elif obj.startswith('v'):
                     current_vehicle_locations[obj] = loc

            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                # Assume the first arg of 'in' is always a package based on domain
                # Check if this package is one we care about
                if package in self.goal_locations:
                    current_package_status[package] = vehicle
                # Note: Vehicles can also be 'in' other containers in some domains,
                # but not in 'transport'. The second arg of 'in' is the container (vehicle).


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # This check is already covered by self.task.goal_reached(state)
            # but checking per package allows summing costs only for unreached goals.
            # Let's explicitly check the goal fact for this package.
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal, calculate its cost
            package_cost = 0

            # Find the package's current status
            current_status = current_package_status.get(package)

            # If package status is not found, it's an unexpected state.
            # This could happen if a package is not 'at' any location and not 'in' any vehicle.
            # Return infinity to indicate a potentially unsolvable or malformed state.
            if current_status is None:
                 # print(f"Debug: Package {package} status not found in state.")
                 return float('inf')


            # Case 1: Package is on the ground at current_l_p
            # Check if the status string is a known location
            if current_status in self.locations:
                current_l_p = current_status
                # Needs pick-up, drive, drop
                # Cost = 1 (pick) + drive_cost + 1 (drop)
                drive_cost = self.shortest_paths.get((current_l_p, goal_location))

                if drive_cost is None: # No path found
                    # Goal location is unreachable from package's current location
                    # print(f"Debug: Goal {goal_location} unreachable from {current_l_p} for package {package}.")
                    return float('inf') # Unsolvable state or unreachable goal

                package_cost = 1 + drive_cost + 1 # pick + drive + drop

            # Case 2: Package is inside a vehicle v
            # Check if the status string is a known vehicle (present in vehicle locations map)
            elif current_status in current_vehicle_locations:
                vehicle = current_status
                current_l_v = current_vehicle_locations.get(vehicle)

                # If vehicle location is not found, it's an unexpected state.
                # A package is in a vehicle, but the vehicle's location is unknown.
                if current_l_v is None:
                     # print(f"Debug: Location of vehicle {vehicle} (carrying {package}) not found in state.")
                     return float('inf') # Indicate a potentially problematic state


                # Needs drive (if vehicle not at goal), drop
                if current_l_v == goal_location:
                    # Vehicle is already at the goal location, just need to drop
                    package_cost = 1 # drop
                else:
                    # Vehicle needs to drive to the goal location, then drop
                    drive_cost = self.shortest_paths.get((current_l_v, goal_location))

                    if drive_cost is None: # No path found
                         # Goal location is unreachable from vehicle's current location
                         # print(f"Debug: Goal {goal_location} unreachable from vehicle location {current_l_v} for package {package}.")
                         return float('inf') # Unsolvable state or unreachable goal

                    package_cost = drive_cost + 1 # drive + drop
            else:
                 # Status is neither a known location nor a known vehicle.
                 # This indicates an unexpected state fact or object type for a package.
                 # print(f"Debug: Unexpected package status for {package}: {current_status}")
                 return float('inf') # Indicate a potentially problematic state

            total_cost += package_cost

        return total_cost
