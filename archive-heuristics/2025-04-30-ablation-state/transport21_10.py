from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to compare parts and args element-wise. fnmatch handles wildcards.
    # This assumes the pattern has the correct number of arguments for the predicate.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current location to its goal location.
    It sums the estimated costs for each package independently.

    # Assumptions
    - Any package can be picked up by any vehicle at the same location. Vehicle capacity
      constraints are ignored.
    - Vehicles can move directly between any two locations connected by a path of roads.
      The cost of driving is the shortest path distance in the road network.
    - The heuristic treats each package's transport problem independently, ignoring
      potential synergies (e.g., one vehicle carrying multiple packages) or conflicts
      (e.g., multiple packages needing the same vehicle).
    - The road network is undirected (if road A-B exists, road B-A also exists).

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on the static `road` facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. For each package `p` that has a goal location `goal_l` (extracted during initialization):
        a. Check if `p` is already at `goal_l` (i.e., `(at p goal_l)` is in the state).
           If yes, the cost for this package is 0. Continue to the next package.
        b. If `p` is not at `goal_l`, find its current status:
           - If `(at p current_l)` is in the state: The package is on the ground at `current_l`.
             The estimated cost for this package is 1 (pick-up) + `dist(current_l, goal_l)` (drive) + 1 (drop).
           - If `(in p v)` is in the state: The package is inside vehicle `v`. Find the location of `v`
             by looking for `(at v current_v_l)` in the state. The estimated cost for this package is
             `dist(current_v_l, goal_l)` (drive) + 1 (drop).
           - If the package's location cannot be determined (neither 'at' nor 'in' facts for the package exist),
             this indicates an invalid state representation; assign a very high cost.
           - If the package is 'in' a vehicle but the vehicle's location ('at' fact for the vehicle) is missing,
             this also indicates an invalid state; assign a very high cost.
        c. Add the estimated cost for package `p` to the total heuristic cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                package, location = args
                self.goal_locations[package] = location
            # Ignore other types of goals if any exist in this domain (though none shown)

        # Build the location graph from road facts.
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        self.locations = list(locations) # Store locations found in road facts

        # Define a large number for unreachable locations
        self.UNREACHABLE_DISTANCE = 1000000

        # Compute all-pairs shortest paths using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        Returns a dictionary where distances[start_loc][end_loc] is the distance.
        """
        distances = {}
        # Iterate over all known locations (from road facts)
        for start_node in self.locations:
            distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()
                distances[start_node][current_node] = dist

                # Get neighbors from the graph, handle locations with no roads
                neighbors = self.location_graph.get(current_node, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Note: If a location exists in the problem (e.g., in init or goal) but is not
        # mentioned in any road fact, it won't be in self.locations and thus won't
        # be a key in the top-level distances dictionary. Accessing distances[unknown_loc]
        # will raise a KeyError. The __call__ method needs to handle this using .get().
        # The .get() calls in __call__ handle cases where start_node or goal_node might
        # not be in the distances dictionary (e.g., if they weren't in the road facts).
        # We don't need to explicitly fill UNREACHABLE_DISTANCE here, as the .get()
        # with a default value in __call__ handles unreachable pairs correctly.

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track where packages and vehicles are currently located or contained.
        package_locations = {} # package -> location (if at)
        package_in_vehicle = {} # package -> vehicle (if in)
        vehicle_locations = {} # vehicle -> location (if at)

        # Populate the location/containment dictionaries from the current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]

            if predicate == "at":
                # Fact is (at obj loc)
                obj, loc = parts[1], parts[2]
                # Assume objects with goals are packages, others are vehicles/locatables
                if obj in self.goal_locations:
                     package_locations[obj] = loc
                else:
                     # Could be a vehicle or another locatable type. We only need vehicle locations.
                     # Let's assume anything 'at' a location that isn't a package with a goal is a vehicle we might care about.
                     # This is a heuristic simplification.
                     vehicle_locations[obj] = loc

            elif predicate == "in":
                # Fact is (in package vehicle)
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
                # Vehicle location will be found from an 'at' fact for the vehicle.

        total_cost = 0

        # Iterate through packages that have goals
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if package in package_locations and package_locations[package] == goal_location:
                continue # Package is already at its goal

            # Package is not at goal, calculate cost
            cost_for_package = 0

            current_location = None # Physical location of the package or its container

            if package in package_locations:
                # Package is on the ground at package_locations[package]
                current_location = package_locations[package]
                # Cost = pick-up + drive + drop
                # Pick-up cost = 1
                # Drive cost = distance from current_location to goal_location
                # Drop cost = 1
                # Use .get() with default to handle cases where current_location or goal_location
                # were not part of the road network used for BFS.
                drive_cost = self.distances.get(current_location, {}).get(goal_location, self.UNREACHABLE_DISTANCE)
                cost_for_package = 1 + drive_cost + 1

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle in vehicle_locations:
                    current_location = vehicle_locations[vehicle]
                    # Cost = drive + drop
                    # Drive cost = distance from current_vehicle_location to goal_location
                    # Drop cost = 1
                    # Use .get() with default to handle cases where current_location or goal_location
                    # were not part of the road network used for BFS.
                    drive_cost = self.distances.get(current_location, {}).get(goal_location, self.UNREACHABLE_DISTANCE)
                    cost_for_package = drive_cost + 1
                else:
                    # Invalid state: package is in vehicle, but vehicle location is unknown.
                    # Assign a very high cost.
                    cost_for_package = self.UNREACHABLE_DISTANCE
                    # print(f"Warning: Vehicle {vehicle} containing package {package} has no 'at' fact.") # Debugging

            else:
                # Invalid state: package location is unknown (neither 'at' nor 'in').
                # Assign a very high cost.
                cost_for_package = self.UNREACHABLE_DISTANCE
                # print(f"Warning: Location of package {package} is unknown.") # Debugging


            total_cost += cost_for_package

        # The heuristic is 0 if and only if all packages are at their goal locations,
        # because the loop only adds cost for packages not at their goal.
        # If any package is not at its goal, and its location is valid and reachable,
        # the cost will be at least 1 (drop) or 3 (pick+drive+drop).
        # If unreachable or state is invalid, cost is UNREACHABLE_DISTANCE.
        # So, h=0 implies goal state.

        return total_cost
