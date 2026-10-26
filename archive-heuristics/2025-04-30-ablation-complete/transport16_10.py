from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the total number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity constraints and assuming
    vehicles are available when needed.

    # Assumptions
    - The goal state is defined solely by the locations of specific packages.
    - Roads are bidirectional (inferred from example instances).
    - Vehicle capacity constraints are ignored for the cost estimation.
    - Any vehicle can transport any package (size constraints ignored).
    - Vehicles are identified by the presence of a `(capacity ?v ?s)` fact in the initial state or static facts.
    - Packages are identified by their presence in the goal conditions `(at ?p ?l)`.

    # Heuristic Initialization
    - Extracts the goal location for each package specified in the task goals.
    - Builds the road network graph from static `(road ?l1 ?l2)` facts.
    - Computes all-pairs shortest path distances between all locations using BFS.
    - Identifies vehicles based on `(capacity ?v ?s)` facts in the initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the state to determine the current location of each package (either on the ground at a location or inside a vehicle) and the current location of each vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
        a. If the package is already at its goal location, add 0 to the total cost.
        b. If the package is on the ground at a location `l_current`:
           - Estimate the cost as 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop). Add this to the total cost.
           - If the goal location is unreachable from `l_current`, the state is likely unsolvable (or requires reaching a different part of the graph first, which this simple heuristic doesn't model), return infinity.
        c. If the package is inside a vehicle `v`, and the vehicle is at location `l_v`:
           - If `l_v` is the package's goal location `l_goal`, estimate the cost as 1 (drop). Add this to the total cost.
           - If `l_v` is not the package's goal location `l_goal`, estimate the cost as shortest_distance(`l_v`, `l_goal`) (drive) + 1 (drop). Add this to the total cost.
           - If the goal location is unreachable from `l_v`, return infinity.
        d. If the package's status is unknown or invalid, return infinity.
    4. The total heuristic value is the sum of the estimated costs for all misplaced packages. If this sum is 0, it implies all tracked packages are at their goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Needed to find vehicles with capacity

        # Extract package goal locations
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Identify vehicles based on capacity facts in initial state or static facts
        self.vehicles = set()
        # Check both initial state and static facts for capacity predicates
        for fact in self.initial_state | self.static:
             if match(fact, "capacity", "*", "*"):
                 vehicle = get_parts(fact)[1]
                 self.vehicles.add(vehicle)

        # Build road network graph and compute distances
        self.road_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional roads

        self.locations = list(locations) # Store locations
        self.distances = {loc: {other_loc: float('inf') for other_loc in self.locations} for loc in self.locations}

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self.distances[start_loc][start_loc] = 0
            queue = deque([start_loc])
            visited = {start_loc}

            while queue:
                u = queue.popleft()
                current_dist = self.distances[start_loc][u]

                # Handle locations with no outgoing roads gracefully
                if u in self.road_graph:
                    for v in self.road_graph[u]:
                        if v not in visited:
                            visited.add(v)
                            self.distances[start_loc][v] = current_dist + 1
                            queue.append(v)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Collect current locations/containers for all locatables mentioned in state
        locatable_status = {} # locatable -> location (if at ground) or vehicle (if in vehicle)
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                locatable_status[obj] = loc
                if obj in self.vehicles: # Use the set of vehicles identified in __init__
                    vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1:]
                # Only track packages that are in our goals
                if package in self.package_goals:
                    locatable_status[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that need to reach a goal
        for package, goal_location in self.package_goals.items():
            current_status = locatable_status.get(package)

            # If package is not found in the state facts, it's an issue.
            # Assuming valid states always list package location/container.
            if current_status is None:
                 # This indicates an issue with state representation or parsing, or an unreachable state.
                 return float('inf')

            # Check if the package is already at its goal
            if current_status == goal_location:
                continue # Cost is 0 for this package

            # Case 1: Package is on the ground at current_location
            # Check if the status is a known location (not a vehicle name)
            if current_status in self.locations:
                current_location = current_status
                # Needs pick-up (1), drive (distance), drop (1)
                # Ensure locations are in our distance map (should be if parsed correctly)
                if current_location in self.distances and goal_location in self.distances[current_location]:
                     dist = self.distances[current_location][goal_location]
                     if dist != float('inf'): # Check if reachable
                         total_cost += 1 # pick-up
                         total_cost += dist # drive
                         total_cost += 1 # drop
                     else:
                         # Goal is unreachable from current location
                         return float('inf')
                else:
                    # Should not happen if parsing is correct and all locations are connected
                    # or if goal location is one of the known locations.
                    return float('inf')


            # Case 2: Package is inside a vehicle (current_status is the vehicle name)
            elif current_status in self.vehicles: # Check if the status is a known vehicle
                vehicle = current_status
                current_vehicle_location = vehicle_locations.get(vehicle)

                if current_vehicle_location is None:
                    # Vehicle carrying package is not at any location? Invalid state?
                    return float('inf')

                # Needs drive (distance), drop (1)
                # Ensure locations are in our distance map
                if current_vehicle_location in self.distances and goal_location in self.distances[current_vehicle_location]:
                    dist = self.distances[current_vehicle_location][goal_location]
                    if dist != float('inf'):
                        total_cost += dist # drive
                        total_cost += 1 # drop
                    else:
                        # Goal is unreachable from current vehicle location
                        return float('inf')
                else:
                     return float('inf') # Invalid vehicle location or goal
            else:
                 # current_status is neither a known location nor a known vehicle
                 # Indicates unexpected state fact or object naming convention.
                 return float('inf')

        # The sum is 0 if and only if all packages in self.package_goals are at their goal locations.
        # Assuming the goal state is reached if and only if all packages in self.package_goals
        # are at their respective goal locations.
        return total_cost

