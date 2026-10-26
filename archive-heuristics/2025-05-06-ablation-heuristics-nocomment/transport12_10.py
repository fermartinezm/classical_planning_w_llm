# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the environment

from collections import deque

def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network."""
    graph = {}
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'road':
            l1, l2 = parts[1], parts[2]
            graph.setdefault(l1, set()).add(l2)
            graph.setdefault(l2, set()).add(l1) # Roads are bidirectional
            locations.add(l1)
            locations.add(l2)
    return graph, list(locations) # Return graph and list of all locations found

def precompute_distances(graph, locations):
    """Computes shortest path distances between all pairs of locations using BFS."""
    distances = {}

    for start_loc in locations:
        distances[start_loc] = {}
        # Distance to self is 0
        distances[start_loc][start_loc] = 0

        # Only proceed if the location is in the graph (has roads connected)
        if start_loc in graph:
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                # Ensure current_loc is a key in the graph before iterating neighbors
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    # For locations not reachable from start_loc, their distance remains undefined in distances[start_loc].
    # This is handled in the heuristic calculation by checking for existence.

    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing up the estimated costs
    for each package that is not yet at its goal location. The cost for a
    misplaced package is estimated as the sum of:
    1. A pick-up action (if the package is currently at a location).
    2. The shortest path distance (number of drive actions) for a vehicle
       to move from the package's current location (or the vehicle's location
       if the package is in a vehicle) to the package's goal location.
    3. A drop action (if the package needs to be moved).

    Assumptions:
    - Roads are bidirectional.
    - Any vehicle can be used to transport any package (ignores capacity constraints).
    - The cost of pick-up, drop, and drive actions is 1.
    - The shortest path distance in the road network is a reasonable estimate
      for the number of drive actions required.
    - All locations relevant to package movement (initial, current, goal) are
      part of the road network graph, or we handle isolated locations by assigning
      a large penalty for unreachable destinations.
    - Goal facts are exclusively of the form `(at ?p ?l)`.

    Heuristic Initialization:
    - Parses the goal facts to store the target location for each package in `self.goal_locations`.
    - Parses the static facts to build the road network graph (`self.road_graph`) and
      list all known locations (`self.locations`).
    - Precomputes all-pairs shortest path distances between locations using BFS,
      storing results in `self.distances`.
    - Defines a large penalty value (`self.unreachable_penalty`) for cases where
      a goal location is unreachable from the current location/vehicle location.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Create dictionaries to track the current location of all objects (`current_locations`)
       and which package is inside which vehicle (`package_in_vehicle`) by iterating
       through the facts in the current state (`node.state`).
    3. Iterate through each package `p` and its goal location `goal_l` stored
       in `self.goal_locations`.
    4. For the current package `p`, check if it exists in the `current_locations` mapping.
       If not, the package is not in a known state (neither at a location nor in a vehicle),
       which is unexpected for a solvable problem. Add `self.unreachable_penalty` to `h`
       and continue to the next package.
    5. Determine the current status of package `p` (`current_status = current_locations[p]`).
       This status is either a location string (if `(at p location)` is true) or a vehicle
       name (if `(in p vehicle)` is true).
    6. If `current_status` is a known location (i.e., `current_status in self.locations`):
       - This means the package is currently at `current_status`.
       - If `current_status` is equal to the `goal_location`, the package is already
         at its goal, so it contributes 0 to the heuristic for this package. Continue
         to the next package.
       - If `current_status` is not the `goal_location`, the package needs to be moved.
         This requires a pick-up action (cost 1) and a drop action (cost 1). Add 2 to `h`.
       - A vehicle is needed to transport the package from `current_status` to `goal_location`.
         Find the precomputed shortest distance `dist(current_status, goal_location)`
         from `self.distances`. If `current_status` is not in `self.distances` or
         `goal_location` is not a key in `self.distances[current_status]`, add
         `self.unreachable_penalty` to `h`. Otherwise, add the distance to `h`.
    7. If `current_status` is not a known location, check if the package is recorded
       as being in a vehicle (`package in package_in_vehicle and package_in_vehicle[package] == current_status`).
       - If true, `vehicle = current_status`.
       - Find the current location of this vehicle from `current_locations[vehicle]`.
         If the vehicle's location is not known (`vehicle not in current_locations`),
         add `self.unreachable_penalty` to `h` and continue.
       - The package needs to be dropped from the vehicle at the `goal_location`. Add 1 to `h`.
       - The vehicle needs to travel from its current location (`current_vehicle_location`)
         to the `goal_location`. Find the precomputed shortest distance
         `dist(current_vehicle_location, goal_location)`. If `current_vehicle_location`
         is not in `self.distances` or `goal_location` is not a key in
         `self.distances[current_vehicle_location]`, add `self.unreachable_penalty` to `h`.
         Otherwise, add the distance to `h`.
    8. If `current_status` is neither a known location nor a vehicle the package is
       recorded as being in, this indicates an unexpected state. Add `self.unreachable_penalty`
       to `h`.
    9. After iterating through all packages with defined goals, the final value of `h`
       is the heuristic estimate for the current state.
    10. Return `h`.
    """
    def __init__(self, task):
        # Assuming Heuristic base class is initialized correctly
        # super().__init__(task)
        self.task = task # Manual assignment if super().__init__ is not used
        self.goals = task.goals
        self.static = task.static


        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are only (at ?p ?l)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build road graph and precompute distances
        self.road_graph, self.locations = build_road_graph(self.static)
        self.distances = precompute_distances(self.road_graph, self.locations)

        # Define a large value for unreachable locations
        self.unreachable_penalty = 1000000 # Use a large number for unreachable locations

    def __call__(self, node):
        state = node.state
        h = 0

        # Find current locations of all locatables (packages and vehicles)
        current_locations = {} # Maps object name to its location string or vehicle name
        package_in_vehicle = {} # Maps package name to vehicle name if in vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
                # Store vehicle name as location for packages inside vehicles
                current_locations[package] = vehicle # This helps distinguish 'at' vs 'in'

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is in the state (it should be if solvable)
            if package not in current_locations:
                 # This package is missing from the state, likely an unsolvable state
                 # or an invalid state representation. Assign a high cost.
                 h += self.unreachable_penalty
                 continue

            current_status = current_locations[package] # This is either a location string or a vehicle name

            # Case 1: Package is at a location
            if current_status in self.locations: # Check if the status is a known location
                current_location = current_status
                # If package is already at goal, cost is 0 for this package
                if current_location == goal_location:
                    continue

                # Package is at a location but not the goal
                # Needs pick-up (1), drive (dist), drop (1)
                h += 2 # Pick-up + Drop

                # Add drive cost
                if current_location in self.distances and goal_location in self.distances[current_location]:
                    h += self.distances[current_location][goal_location]
                else:
                    # Goal location is unreachable from current location
                    h += self.unreachable_penalty

            # Case 2: Package is in a vehicle
            # Check if the package is recorded as being in the vehicle indicated by current_status
            elif package in package_in_vehicle and package_in_vehicle[package] == current_status:
                 vehicle = current_status
                 # Find vehicle's location
                 if vehicle not in current_locations:
                     # Vehicle is missing from state, likely unsolvable
                     h += self.unreachable_penalty
                     continue

                 current_vehicle_location = current_locations[vehicle]

                 # Needs drive (dist), drop (1)
                 h += 1 # Drop

                 # Add drive cost
                 if current_vehicle_location in self.distances and goal_location in self.distances[current_vehicle_location]:
                     h += self.distances[current_vehicle_location][goal_location]
                 else:
                     # Goal location is unreachable from current vehicle location
                     h += self.unreachable_penalty
            else:
                 # This case should ideally not be reached in a valid state
                 # for a package that has a goal. It means the package's
                 # current_locations entry is neither a known location nor
                 # a vehicle it is recorded as being 'in'.
                 # Treat as unreachable.
                 h += self.unreachable_penalty


        # The heuristic is 0 if and only if all packages with goals are at their goal locations.
        # Based on the example goal definitions, this is equivalent to the goal state.

        return h
