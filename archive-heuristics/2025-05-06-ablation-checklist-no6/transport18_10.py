# Required imports
from collections import defaultdict, deque
import math
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string before slicing
    if not isinstance(fact, str):
        # This case indicates an unexpected input format.
        # Depending on robustness requirements, could log a warning or raise an error.
        # For a heuristic operating on valid states, this shouldn't be reached.
        return []
    # Basic check for fact format (starts with '(', ends with ')')
    if not fact.startswith('(') or not fact.endswith(')'):
         # Not a valid PDDL fact string format
         return []
    # Handle potential empty fact like "()"
    if len(fact) <= 2:
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move packages to their goal locations.
    It sums the minimum actions required for each package independently, considering
    whether the package is on the ground or in a vehicle, and the shortest path
    distances between locations. It ignores vehicle capacity constraints and
    potential conflicts.

    Heuristic value is 0 if and only if the state is a goal state.
    Heuristic value is math.inf if the goal is unreachable from the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph to compute shortest path distances.
        """
        # The task object contains initial_state, goals, operators, static facts.
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        # We only care about (at ?p ?l) goals for packages.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Assuming objects starting with 'p' are packages based on examples.
                # A more robust approach would parse types from the domain definition.
                if package.startswith('p'):
                    self.goal_locations[package] = location
            # Assuming goals are only (at ?p ?l) for packages.

        # Build the road network graph and compute all-pairs shortest paths.
        self.road_graph = defaultdict(set)
        locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].add(l2)
                # Assuming roads are bidirectional unless specified otherwise
                # The example instance files show bidirectional roads.
                self.road_graph[l2].add(l1)
                locations.add(l1)
                locations.add(l2)

        self.distances = {}
        # Compute shortest path from every location to every other location
        # using BFS.
        for start_loc in locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc is a valid key in the graph
                # This check is technically redundant if locations set is built correctly
                # from road facts, but adds robustness.
                if current_loc not in self.road_graph and current_loc != start_loc:
                     continue

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def get_distance(self, l1, l2):
        """Returns the shortest distance between two locations, or infinity if unreachable."""
        # If either location is not in the graph built from static facts, they are unreachable
        # within the defined road network.
        if l1 not in self.road_graph or l2 not in self.road_graph:
             return math.inf
        # The distances dictionary stores computed shortest paths.
        # If a pair is not in the dictionary after BFS from all nodes, it's unreachable.
        return self.distances.get((l1, l2), math.inf)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of strings).

        # Map locatable objects (packages, vehicles) to their current location or vehicle.
        package_locations = {} # Maps package -> location (if on ground)
        vehicle_locations = {} # Maps vehicle -> location
        package_in_vehicle = {} # Maps package -> vehicle

        # Identify all vehicles present in the state that have a location
        vehicles_with_location = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects starting with 'p' are packages and 'v' are vehicles.
                if obj.startswith('p'):
                    package_locations[obj] = loc
                elif obj.startswith('v'):
                    vehicle_locations[obj] = loc
                    vehicles_with_location.add(obj)
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Assuming objects starting with 'p' are packages and 'v' are vehicles.
                if package.startswith('p') and vehicle.startswith('v'):
                    package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location defined in the problem.
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at the goal location on the ground.
            # The goal is (at ?p ?l). If (at p l) is in the state and l is the goal, this package is done.
            if package in package_locations and package_locations[package] == goal_location:
                continue # Package is at goal, cost is 0 for this package.

            # Package is not at the goal location on the ground.
            # Find its actual current status (on ground or in vehicle) and location.

            actual_package_location = None
            is_in_vehicle = False

            if package in package_locations:
                # Package is on the ground at package_locations[package]
                actual_package_location = package_locations[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle in vehicle_locations:
                    actual_package_location = vehicle_locations[vehicle]
                    is_in_vehicle = True
                else:
                    # Vehicle carrying package has no location? This implies an invalid state
                    # or the vehicle is not present in the state facts (unlikely for vehicles carrying packages).
                    # Treat as unreachable.
                    return math.inf
            else:
                # Package is not 'at' any location and not 'in' any vehicle.
                # This should only happen if the package is already at its goal and the state
                # explicitly contains the goal fact (at package goal_location).
                # The first check `if package in package_locations and package_locations[package] == goal_location:`
                # handles this case. If we reach this 'else' block, it means the package is in self.goal_locations
                # but is not found in package_locations (meaning (at package l) is not in state, or l != goal_location)
                # and not found in package_in_vehicle (meaning (in package v) is not in state).
                # This implies the package is not located in the state facts. For a package that is a goal,
                # this suggests an issue with the state representation or problem definition.
                # Returning inf is a safe way to indicate an unresolvable situation for this package.
                return math.inf


            # Package is not at its goal location. Calculate cost.
            if is_in_vehicle:
                # Package is in vehicle at actual_package_location
                # Cost = 1 (drop) + distance(actual_package_location, goal_location) (drive)
                dist_vehicle_to_goal = self.get_distance(actual_package_location, goal_location)
                if dist_vehicle_to_goal == math.inf:
                    return math.inf # Unreachable goal location from vehicle's current location
                cost_for_package = 1 + dist_vehicle_to_goal
            else:
                # Package is at actual_package_location on the ground
                # Cost = 1 (pick-up) + 1 (drop) + distance(actual_package_location, goal_location) (drive)
                # Plus cost for a vehicle to reach actual_package_location
                dist_package_to_goal = self.get_distance(actual_package_location, goal_location)
                if dist_package_to_goal == math.inf:
                    return math.inf # Unreachable goal location from package's current location

                min_dist_vehicle_to_package = math.inf
                if not vehicles_with_location: # No vehicles exist in the state with a location
                     return math.inf # Cannot move package

                # Find the closest vehicle to the package's current location
                for vehicle in vehicles_with_location:
                    v_loc = vehicle_locations.get(vehicle) # v_loc is guaranteed to exist by vehicles_with_location set
                    min_dist_vehicle_to_package = min(min_dist_vehicle_to_package, self.get_distance(v_loc, actual_package_location))

                if min_dist_vehicle_to_package == math.inf:
                     return math.inf # No vehicle can reach the package's current location

                # Cost = (drive vehicle to package) + (pick-up) + (drive package to goal) + (drop)
                cost_for_package = min_dist_vehicle_to_package + 1 + dist_package_to_goal + 1

            total_cost += cost_for_package

        return total_cost
