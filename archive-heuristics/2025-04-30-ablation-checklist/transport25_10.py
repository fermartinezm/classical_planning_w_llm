from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    to their goal locations. It sums the estimated cost for each package
    that is not yet at its goal. The cost for a package is estimated based on
    whether it's on the ground or in a vehicle, and the shortest path distance
    from its current location (or its vehicle's location) to its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost of moving a package involves picking it up (if on the ground),
      driving it to the destination, and dropping it.
    - The shortest path distance on the road network is a reasonable estimate
      for the number of drive actions needed.
    - Capacity constraints and vehicle availability/sharing are ignored for
      simplicity and efficiency in this greedy heuristic.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on `road` predicates.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location of every package and vehicle.
    2. For each package that has a goal location and is not currently at that goal:
       a. Determine if the package is on the ground at some location L_p, or
          if it is inside a vehicle V which is at location L_v.
       b. Find the package's goal location L_goal.
       c. If the package is on the ground at L_p (L_p != L_goal):
          - It needs to be picked up (1 action).
          - The vehicle needs to drive from L_p to L_goal. Estimate this cost
            as the shortest path distance `dist(L_p, L_goal)`.
          - It needs to be dropped (1 action).
          - Estimated cost for this package: 1 (pick-up) + dist(L_p, L_goal) + 1 (drop).
       d. If the package is in a vehicle V at L_v:
          - The vehicle needs to drive from L_v to L_goal. Estimate this cost
            as the shortest path distance `dist(L_v, L_goal)`.
          - It needs to be dropped (1 action).
          - Estimated cost for this package: dist(L_v, L_goal) + 1 (drop).
    3. The total heuristic value is the sum of the estimated costs for all
       packages not yet at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Build the location graph from road facts.
        self.location_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                self.location_graph[loc1].append(loc2)

        self.locations = list(locations) # Store list of all locations

        # Compute all-pairs shortest paths using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances from every location to every other
        location using BFS.
        Returns a dictionary distances[start_loc][end_loc] = distance.
        """
        distances = {}
        for start_node in self.locations:
            distances[start_node] = {}
            # Initialize distances to infinity, start node to 0
            for loc in self.locations:
                 distances[start_node][loc] = float('inf')
            distances[start_node][start_node] = 0

            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Get neighbors from the graph (handle locations with no outgoing roads)
                neighbors = self.location_graph.get(current_node, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Check if the goal is already reached
        if self.goals.issubset(state):
            return 0

        # Track current locations of packages and vehicles
        current_locations = {} # Maps object name (package or vehicle) to its location (location name)
        package_in_vehicle = {} # Maps package name to vehicle name if inside

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # We need to know if obj is a package or vehicle.
                # A simple way is to check if it's in the goal_locations keys (must be package)
                # or if it appears as the second argument in an 'in' predicate (must be vehicle).
                # Or, more robustly, check against known objects from the task, but we don't have that here.
                # Let's assume anything in goal_locations is a package, others are vehicles if they appear with 'at'.
                if obj in self.goal_locations:
                     current_locations[obj] = loc # Package on the ground
                # Check if it's a vehicle by seeing if it appears as the second arg in any 'in' fact in the state
                # This is inefficient. A better way is to parse object types in __init__ if available.
                # For now, let's assume anything with 'at' that isn't a goal package is a vehicle.
                # This is a simplification based on typical transport problems.
                else:
                     # Heuristic needs vehicle locations to calculate distances for packages in vehicles
                     current_locations[obj] = loc # Vehicle location

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
                # The package's "location" is the vehicle it's in
                current_locations[package] = vehicle


        total_cost = 0

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})" in state):
                 continue # Package is already at goal, no cost

            # Find the package's current status and location
            # It must be either at a location or in a vehicle if not at goal on ground
            if package not in current_locations:
                 # This case shouldn't happen in a valid state if the package exists,
                 # but as a fallback, assume it needs full transport.
                 # Or, more likely, it's in a vehicle whose location isn't tracked yet.
                 # Let's refine state parsing to handle vehicles first.
                 pass # Will handle below based on package_in_vehicle or current_locations

            # If package is in a vehicle
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle in current_locations:
                    vehicle_location = current_locations[vehicle]
                    # Cost is distance from vehicle's location to package's goal + 1 (drop)
                    # Ensure locations exist in our distance map
                    if vehicle_location in self.distances and goal_location in self.distances[vehicle_location]:
                         distance = self.distances[vehicle_location][goal_location]
                         if distance != float('inf'): # Check if reachable
                             total_cost += distance + 1 # Drive + Drop
                         else:
                             # If goal is unreachable from vehicle location, assign a large cost
                             total_cost += 1000 # Arbitrary large number for unreachable goals
                    else:
                         # Locations not found in graph, assign a large cost
                         total_cost += 1000

                else:
                    # Vehicle location not found in state (shouldn't happen in valid states)
                    total_cost += 1000 # Assign a large cost

            # If package is on the ground (and not at its goal, handled by initial check)
            elif package in current_locations: # current_locations[package] holds the ground location
                 package_location = current_locations[package]
                 # Cost is 1 (pick-up) + distance from package's location to goal + 1 (drop)
                 # Ensure locations exist in our distance map
                 if package_location in self.distances and goal_location in self.distances[package_location]:
                     distance = self.distances[package_location][goal_location]
                     if distance != float('inf'): # Check if reachable
                         total_cost += 1 + distance + 1 # Pick-up + Drive + Drop
                     else:
                         # If goal is unreachable from package location, assign a large cost
                         total_cost += 1000 # Arbitrary large number for unreachable goals
                 else:
                     # Locations not found in graph, assign a large cost
                     total_cost += 1000

            # If package is not found anywhere (should not happen in valid states)
            else:
                 total_cost += 1000 # Assign a large cost


        return total_cost

