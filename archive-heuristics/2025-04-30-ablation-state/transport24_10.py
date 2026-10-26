# Assuming Heuristic base class is available from heuristics.heuristic_base import Heuristic
# If running standalone or in a different environment, you might need to define
# a compatible base class or adapt the code.
# Example:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError # To be implemented by subclasses

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated costs for each package
    independently, based on its current location (on the ground or in a vehicle)
    and the shortest path distance on the road network.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability at pick-up locations is ignored.
    - Shortest path distances on the road network represent the minimum drive actions.
    - Packages are either on the ground at a location or inside a vehicle.
    - Only packages explicitly listed in the goal condition `(at ?p ?l)` are considered.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Identify all packages that have a goal.
    - Build the road network graph from static 'road' facts.
    - Identify all relevant locations mentioned in the problem.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Index the current locations of packages (on ground) and vehicles from `(at ?x ?l)` facts in the state.
    2. Index which packages are inside which vehicles from `(in ?p ?v)` facts in the state.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a goal location (identified during initialization):
        a. Check if the package is currently on the ground at its goal location. If yes, the cost for this package is 0, continue to the next package.
        b. If the package is on the ground at a different location `loc_current`:
            - Look up the shortest distance `d` from `loc_current` to the package's goal location.
            - If `d` is infinite (unreachable), the total heuristic is infinite.
            - Otherwise, add `1` (pick-up) + `d` (drive actions) + `1` (drop) to the total cost.
        c. If the package is inside a vehicle `v`:
            - Find the current location `loc_v` of vehicle `v`. If the vehicle's location is unknown, the total heuristic is infinite.
            - Look up the shortest distance `d` from `loc_v` to the package's goal location.
            - If `d` is infinite (unreachable), the total heuristic is infinite.
            - Otherwise, add `d` (drive actions) + `1` (drop) to the total cost.
        d. If the package is neither on the ground nor in a vehicle (and not at the goal), this indicates an invalid state, return infinite cost.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest path distances.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        self.goal_packages = set() # Store just the package names that have goals

        # Collect all locations mentioned in the problem
        all_locations = set()

        # Process goal state to find packages and goal locations
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.goal_packages.add(package) # These are the packages we care about
                all_locations.add(location)

        # Process initial state to collect locations
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                all_locations.add(loc)
            elif predicate == "in":
                 # Package and vehicle names are in initial state, but locations are what matter for graph
                 pass


        # Process static facts to build the road graph and collect all locations
        self.road_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "road":
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, []).append(l2)
                all_locations.add(l1)
                all_locations.add(l2)
            # We ignore capacity-predecessor for this simple heuristic

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, dist = q.popleft()
                self.distances[start_loc][current_loc] = dist

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

            # Ensure all locations have a distance entry (even if unreachable)
            # This is important so that lookup self.distances.get(l1, {}).get(l2, float('inf')) works correctly
            for loc in all_locations:
                 if loc not in self.distances[start_loc]:
                     self.distances[start_loc][loc] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Index current locations for quick lookup
        package_locations = {} # {package: location} if on ground
        package_in_vehicle = {} # {package: vehicle} if in vehicle
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                # We don't have type info directly. Infer based on whether the object
                # is one of the goal packages. This is a simplification.
                if obj in self.goal_packages:
                     package_locations[obj] = loc
                else: # Assume anything else 'at' a location is a vehicle
                     vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                # Only track packages that have a goal
                if package in self.goal_packages:
                    package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if package in package_locations and package_locations[package] == goal_location:
                continue # Package is already at its goal

            # Package is not at goal, estimate cost
            package_cost = 0

            if package in package_locations: # Package is on the ground elsewhere
                current_package_loc = package_locations[package]
                # Cost includes pick-up, drive, drop
                package_cost += 1 # Pick-up

                # Add drive cost
                drive_cost = self.distances.get(current_package_loc, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # Goal is unreachable from current location on the ground
                    return float('inf')
                package_cost += drive_cost

                package_cost += 1 # Drop

            elif package in package_in_vehicle: # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                current_vehicle_loc = vehicle_locations.get(vehicle)

                if current_vehicle_loc is None:
                     # Vehicle location is unknown - should not happen in valid states
                     return float('inf')

                # Cost includes drive (by vehicle) and drop
                drive_cost = self.distances.get(current_vehicle_loc, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # Goal is unreachable from vehicle's current location
                    return float('inf')
                package_cost += drive_cost

                package_cost += 1 # Drop
            else:
                 # Package is not at a location and not in a vehicle.
                 # This should only happen if the package was already at the goal
                 # and the goal fact was the only one mentioning its location,
                 # which is handled by the initial check. If it's a package
                 # with a goal but no location fact in the state, it's an invalid state.
                 return float('inf')

            total_cost += package_cost

        return total_cost
