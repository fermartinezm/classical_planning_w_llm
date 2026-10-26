from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.
    Wildcards `*` are allowed in `args`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_road_graph_and_distances(static_facts, initial_state, goals):
    """
    Builds the road network graph and computes all-pairs shortest paths.
    Infers locations from road facts, initial state, and goals.
    """
    graph = defaultdict(list)
    locations = set()

    # Add locations from roads
    for fact in static_facts:
        if match(fact, "road", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            graph[loc1].append(loc2)
            graph[loc2].append(loc1) # Assuming bidirectional roads based on examples
            locations.add(loc1)
            locations.add(loc2)

    # Add locations from initial state and goals to ensure all relevant locations are included
    for state_facts in [initial_state, goals]:
        for fact in state_facts:
            # Locations appear in 'at' predicates
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                locations.add(loc)
            # 'in' facts refer to objects inside other objects, not locations directly.

    # Ensure all identified locations are keys in the graph dictionary, even if isolated
    for loc in locations:
        if loc not in graph:
            graph[loc] = []

    # Compute all-pairs shortest paths using BFS from each location
    distances = {}
    for start_node in locations:
        distances[start_node] = {}
        q = deque([(start_node, 0)])
        visited = {start_node}
        while q:
            current_node, dist = q.popleft()
            distances[start_node][current_node] = dist

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))

    return distances, locations

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the estimates for
    all packages not yet at their goal. It considers whether a package is on the
    ground or inside a vehicle.

    # Assumptions
    - Packages need to reach specific goal locations.
    - Roads are bidirectional (inferred from examples).
    - Vehicle capacity (number of packages) is not a limiting factor for the heuristic calculation.
    - Any vehicle can carry any package (ignoring size constraints for simplicity in the heuristic).
    - A vehicle is available at the package's location when needed for loading/transport.
    - The cost of each action (load, unload, drive) is 1.
    - Vehicles can be identified by predicate arguments starting with 'v'.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph representing the road network from static facts and initial/goal locations.
    - Computes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Pre-process the state to quickly look up object locations and package containment.
       - Create dictionaries mapping objects ('at' predicate) to locations, packages ('in' predicate) to vehicles, and vehicles ('at' predicate, inferred by name) to locations.
    3. For each package that has a goal location defined in the task:
        a. Check if the package is already at its goal location based on the state facts `(at ?p goal_loc)`. If yes, cost for this package is 0, continue to the next package.
        b. If not at the goal, determine the package's current effective location and status (on_ground or in_vehicle):
           - If the package is found as being `in` a vehicle in the pre-processed containment map, its effective location is the location of that vehicle (looked up in the vehicle location map). The status is `in_vehicle`.
           - If the package is not `in` a vehicle but is found as being `at` a location in the pre-processed location map, its effective location is that location. The status is `on_ground`.
           - If the package's location cannot be determined (neither `at` nor `in` facts found), it indicates an invalid state or an unreachable goal from this state. Return a large penalty.
        c. Get the package's goal location.
        d. If the package's current effective location or goal location is not found among the locations identified during initialization (meaning they are in disconnected components or locations not seen), return a large penalty.
        e. Get the shortest distance `d` between the current effective location and the goal location from the pre-computed distances. If no path exists (distance is None), return a large penalty.
        f. Calculate the cost for this package:
           - If the package is currently `on_ground`: Cost = 1 (load) + d (drives) + 1 (unload).
           - If the package is currently `in_vehicle`: Cost = d (drives) + 1 (unload).
        g. Add the calculated cost for this package to the total heuristic cost.
    4. Return the total sum.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph and pre-computing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
            # Assuming only 'at' goals for packages are relevant for the heuristic

        # Build road graph and compute distances
        self.distances, self.all_locations = build_road_graph_and_distances(
            self.static_facts, self.initial_state, self.goals
        )

        # Define a large number for unreachable locations
        # A value larger than any possible path length in a connected graph of |L| nodes
        # is |L|-1. Doubling it plus a buffer is safe.
        self.unreachable_penalty = len(self.all_locations) * 2 + 100 if self.all_locations else 1000

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        total_cost = 0

        # Pre-process state to quickly find locations/containment
        current_locations = {} # {obj: loc} for objects 'at' a location
        package_in_vehicle = {} # {package: vehicle} for packages 'in' a vehicle
        vehicle_location = {} # {vehicle: loc} for vehicles 'at' a location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Simple check if it's a vehicle based on naming convention 'v*'
                if obj.startswith('v'): # Domain-specific assumption
                     vehicle_location[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at its goal

            # Package is not at goal, calculate cost
            cost_for_package = 0

            # Determine package's current effective location and status (on_ground or in_vehicle)
            current_effective_location = None
            on_ground = False # Flag to track if package is on the ground

            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_location:
                    current_effective_location = vehicle_location[vehicle]
                # else: vehicle is 'in' something but the vehicle itself isn't 'at' a location? Invalid state.
                # This case should ideally not happen in valid states generated by the planner.
                # If it does, the unreachable_penalty will be returned below.
            elif package in current_locations:
                 current_effective_location = current_locations[package]
                 on_ground = True
            # else: package is neither 'at' a location nor 'in' a vehicle? Invalid state.
            # This case should ideally not happen.

            # If we couldn't determine the location, it's an issue.
            if current_effective_location is None:
                 # This state is likely problematic or represents an unhandled case.
                 # Return a high cost to discourage search towards this state.
                 return self.unreachable_penalty

            # Check if the locations are known from initialization
            if current_effective_location not in self.all_locations or goal_location not in self.all_locations:
                 return self.unreachable_penalty

            # Get distance from current effective location to goal location
            # Use .get() with a default to handle cases where goal is unreachable from current loc
            distance = self.distances.get(current_effective_location, {}).get(goal_location)

            if distance is None:
                # Goal location is unreachable from current location in the road network
                return self.unreachable_penalty

            # Calculate cost based on package's current status
            if on_ground:
                # Package is on the ground at current_effective_location
                cost_for_package = 1 + distance + 1 # load + drive(s) + unload
            else: # Package is in a vehicle
                # Package is in a vehicle which is at current_effective_location
                cost_for_package = distance + 1 # drive(s) + unload

            total_cost += cost_for_package

        return total_cost
