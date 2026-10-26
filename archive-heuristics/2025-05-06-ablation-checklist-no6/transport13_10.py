# from heuristics.heuristic_base import Heuristic # Assume this is available
from collections import deque
from fnmatch import fnmatch

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic: # Assume this implicitly inherits from Heuristic or similar base
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions required to move all packages to their
    goal locations. It sums the estimated cost for each package independently.
    The estimated cost for a package includes pick-up, drop, and driving actions.
    Driving cost is estimated using precomputed shortest path distances
    in the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages,
        identifying locations and vehicles, and precomputing shortest path
        distances between locations.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all objects/locations

        # Extract goal locations for each package
        self.package_goals = {}
        packages_from_goals = set()
        for goal in self.goals:
            # Goal is typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location
                packages_from_goals.add(package)

        # Identify all locations, vehicles, and packages from initial state and static facts
        self.locations = set()
        self.vehicles = set()
        self.packages = set(packages_from_goals) # Start with packages from goals
        self.roads = []

        # Collect objects and locations mentioned in initial state, static, and goals
        objects_in_capacity = set() # Potential vehicles
        objects_in_in_package = set() # Potential packages
        objects_in_in_vehicle = set() # Potential vehicles
        objects_in_at_locatable = set() # Potential locatables (vehicles or packages)

        for fact in self.initial_state | self.static | self.goals:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts

             predicate = parts[0]
             args = parts[1:]

             # Collect objects and locations based on predicate structure
             if predicate == "capacity" and len(args) == 2:
                 objects_in_capacity.add(args[0]) # ?v - vehicle
             elif predicate == "in" and len(args) == 2:
                 objects_in_in_package.add(args[0]) # ?x - package
                 objects_in_in_vehicle.add(args[1]) # ?v - vehicle
             elif predicate == "at" and len(args) == 2:
                 objects_in_at_locatable.add(args[0]) # ?x - locatable
                 self.locations.add(args[1]) # ?v - location
             elif predicate == "road" and len(args) == 2:
                 self.locations.add(args[0]) # ?l1 - location
                 self.locations.add(args[1]) # ?l2 - location
                 self.roads.append((args[0], args[1]))


        # Refined type inference:
        # Vehicles are objects appearing in capacity facts or as the second arg of 'in'.
        self.vehicles = objects_in_capacity | objects_in_in_vehicle
        # Packages are objects appearing as the first arg of 'in' or in goal 'at' facts.
        self.packages = objects_in_in_package | packages_from_goals
        # Any object in 'at' that is not a package must be a vehicle (assuming only vehicles and packages are locatable)
        self.vehicles.update(obj for obj in objects_in_at_locatable if obj not in self.packages)
        # Ensure packages and vehicles sets are disjoint if necessary, though not strictly needed for this heuristic logic.

        # Build the road network graph and precompute shortest paths
        self.location_distances = self._precompute_distances()


    def _precompute_distances(self):
        """
        Builds the road network graph and computes all-pairs shortest paths
        using BFS.
        """
        locations = list(self.locations) # Use a list for consistent indexing if needed, or just iterate set
        adj = {loc: set() for loc in locations}
        for l1, l2 in self.roads:
            # Ensure locations from roads were added to self.locations
            if l1 in adj and l2 in adj:
                adj[l1].add(l2)
                adj[l2].add(l1) # Assuming roads are bidirectional
            # else: print(f"Warning: Road {l1}-{l2} involves unknown location.")


        # Compute shortest paths from each location using BFS
        distances = {}
        # Use a large value for unreachable locations. A value larger than any possible path.
        # Max possible path length in a graph with V nodes is V-1. Let's use V * 2 for safety.
        max_dist = len(locations) * 2 if locations else 0 # Avoid multiplying by zero if no locations

        for start_loc in locations:
            distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()
                distances[start_loc][current_loc] = dist

                for neighbor in adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            # Fill in unreachable locations with max_dist
            for loc in locations:
                 if loc not in distances[start_loc]:
                     distances[start_loc][loc] = max_dist

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Find current location/state for all packages and vehicles present in the state
        package_current_state = {} # Map package -> location or vehicle
        vehicle_locations = {} # Map vehicle -> location

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                 _, package, vehicle = get_parts(fact)
                 if package in self.packages and vehicle in self.vehicles:
                     package_current_state[package] = vehicle


        total_cost = 0

        # Calculate cost for each package that has a goal
        for package, goal_location in self.package_goals.items():

            # Check if the goal is already met for this package
            if f"(at {package} {goal_location})" in state:
                 continue # Goal reached for this package

            # Package is not at goal. Find its current location/vehicle.
            current_state = package_current_state.get(package)

            if current_state is None:
                 # This package is not mentioned in 'at' or 'in' facts in the state.
                 # This indicates an inconsistent state representation for a package with a goal.
                 # Return infinity as this state is likely unsolvable or invalid.
                 return float('inf')

            # Ensure goal_location is a known location
            if goal_location not in self.locations:
                 # Goal location is not in the set of known locations.
                 # This indicates an invalid problem definition or state.
                 return float('inf')


            if current_state in self.vehicles:
                # Package is in a vehicle
                vehicle = current_state
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                    # Vehicle location unknown? Should not happen if package is in vehicle.
                    # This indicates an inconsistent state.
                    return float('inf') # Problematic state

                # Ensure vehicle_location is a known location
                if vehicle_location not in self.locations:
                     # Vehicle location is not in the set of known locations.
                     # This indicates an invalid state.
                     return float('inf')


                # Cost: drive from vehicle_location to goal_location + drop
                # Distance lookup handles unreachable locations by returning max_dist
                drive_cost = self.location_distances[vehicle_location][goal_location]
                total_cost += drive_cost + 1 # 1 for drop action

            elif current_state in self.locations: # Package is at a location (on the ground)
                current_location = current_state

                # Ensure current_location is a known location
                if current_location not in self.locations:
                     # Package location is not in the set of known locations.
                     # This indicates an invalid state.
                     return float('inf')

                # Cost: pick-up + drive from current_location to goal_location + drop
                # Distance lookup handles unreachable locations by returning max_dist
                drive_cost = self.location_distances[current_location][goal_location]
                total_cost += 1 + drive_cost + 1 # 1 for pick-up, 1 for drop

            else:
                 # current_state is neither a vehicle nor a location? Invalid state.
                 # print(f"Warning: Package {package} is in an invalid state: {current_state}")
                 return float('inf')


        # The heuristic is 0 only if all packages with goals are at their goal locations.
        # This is handled by the loop and the initial check.

        return total_cost
