from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle potential length differences gracefully with fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    that is not yet at its goal. The cost for a package is estimated based on
    whether it needs to be picked up, driven, and dropped, using shortest path
    distances in the road network for the drive actions.

    # Assumptions
    - The road network is static and can be precomputed.
    - Shortest path distances in the road network represent the minimum number of drive actions.
    - Vehicle availability and capacity constraints are ignored for heuristic calculation;
      it assumes a suitable vehicle is always available when needed.
    - The cost of pick-up, drop, and drive actions is 1.
    - All locations relevant to package goals are connected in the road network.
      If a goal location is unreachable from a package's current location,
      the heuristic returns infinity.
    - Objects appearing as the first argument of an 'at' predicate in the goal
      are considered packages whose goal location needs to be reached.
      Other objects appearing as the first argument of 'at' or 'in' predicates
      in the state are considered vehicles.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the current location or containing vehicle for every package and vehicle by iterating through the state facts. Store these in lookup dictionaries (`package_locations`, `package_in_vehicle`, `vehicle_locations`). Packages are identified based on the goal facts; other relevant objects are assumed to be vehicles.
    3. Initialize the total heuristic cost to 0.
    4. For each package whose goal location is known (`self.goal_locations`):
       a. If the package is already at its goal location in the current state (check for the exact `(at package goal_loc)` fact), the cost for this package is 0. Continue to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current status: Is it at a location (on the ground) or inside a vehicle?
          ii. If the package is at a location `current_loc` (found in `package_locations`):
              - Get the shortest distance from `current_loc` to `goal_loc` from the precomputed `self.distances`.
              - If the distance is not found (indicating unreachable), return `float('inf')`.
              - The estimated cost for this package is 1 (pick-up) + distance(`current_loc`, `goal_loc`) (drive) + 1 (drop).
          iii. If the package is inside a vehicle `veh` (found in `package_in_vehicle`), find the vehicle's location `current_loc` (from `vehicle_locations`):
              - If the vehicle's location is not found (invalid state or unreachable vehicle), return `float('inf')`.
              - Get the shortest distance from `current_loc` to `goal_loc` from `self.distances`.
              - If the distance is not found (unreachable), return `float('inf')`.
              - The estimated cost for this package is distance(`current_loc`, `goal_loc`) (drive) + 1 (drop).
          iv. If the package's status is not found (neither at a location nor in a vehicle), return `float('inf')` (invalid state).
          v. Add the estimated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances in the road network.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        # Identify package objects from goal facts like (at p1 l2)
        package_objects = set()
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 predicate, obj, location = get_parts(goal)
                 # Assuming the first argument of 'at' in a goal is always a package
                 package_objects.add(obj)
                 self.goal_locations[obj] = location

        # Build the road network graph and collect all locations
        self.graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                predicate, l1, l2 = get_parts(fact)
                all_locations.add(l1)
                all_locations.add(l2)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = self._bfs(start_loc) # BFS only needs the graph

    def _bfs(self, start_loc):
        """
        Performs BFS from a start location to find distances to all reachable locations.
        Uses the precomputed self.graph.
        Returns a dictionary {location: distance}.
        """
        distances_from_start = {start_loc: 0}
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()
            dist = distances_from_start[current_loc]

            # Get neighbors from the graph, handle locations with no roads (shouldn't be in graph keys if from road facts)
            neighbors = self.graph.get(current_loc, set())

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances_from_start[neighbor] = dist + 1
                    queue.append(neighbor)

        return distances_from_start

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the state is a goal state
        if self.goals <= state:
             return 0

        # Efficiently look up current locations/status
        package_locations = {} # {package: location} if at location
        package_in_vehicle = {} # {package: vehicle} if in vehicle
        vehicle_locations = {} # {vehicle: location}

        # Build lookup dicts for objects in the current state
        # First, identify all objects that are packages based on goals
        package_objects = set(self.goal_locations.keys())

        # Now iterate state facts to populate location/in dicts
        for fact_str in state:
             parts = get_parts(fact_str)
             if not parts: continue # Skip empty facts if any

             predicate = parts[0]
             args = parts[1:]

             if predicate == "at" and len(args) == 2:
                 obj, loc = args
                 if obj in package_objects:
                     package_locations[obj] = loc
                 else: # Assume it's a vehicle if not a package we care about
                     vehicle_locations[obj] = loc
             elif predicate == "in" and len(args) == 2:
                 pkg, veh = args
                 # Assuming pkg is always a package and veh is always a vehicle based on domain structure
                 package_in_vehicle[pkg] = veh
             # Ignore other predicates like capacity, capacity-predecessor, road in state

        total_cost = 0

        for package, goal_loc in self.goal_locations.items():
            # Check if package is already at its goal location
            # Goal fact is (at package goal_loc)
            goal_fact_str = f"(at {package} {goal_loc})"
            if goal_fact_str in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at its goal. Estimate cost.
            cost_for_package = 0

            if package in package_locations:
                # Package is on the ground at current_loc
                current_loc = package_locations[package]
                # Cost: pick-up + drive + drop
                # Need distance from current_loc to goal_loc
                # Check if current_loc is in our distance map (i.e., connected to road network)
                if current_loc not in self.distances:
                     # Package is at a location not connected to the road network
                     return float('inf')

                drive_cost = self.distances[current_loc].get(goal_loc)

                if drive_cost is None:
                     # Goal location is unreachable from current location
                     return float('inf')

                cost_for_package = 1 + drive_cost + 1 # pick + drive + drop

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                veh = package_in_vehicle[package]
                # Need vehicle's location
                if veh in vehicle_locations:
                    current_loc = vehicle_locations[veh]
                    # Cost: drive + drop
                    # Check if vehicle's current_loc is in our distance map
                    if current_loc not in self.distances:
                         # Vehicle is at a location not connected to the road network
                         return float('inf')

                    drive_cost = self.distances[current_loc].get(goal_loc)

                    if drive_cost is None:
                        # Goal location is unreachable from vehicle's current location
                        return float('inf')

                    cost_for_package = drive_cost + 1 # drive + drop
                else:
                     # Vehicle containing package has no location fact - invalid state?
                     # Or vehicle is not a 'locatable' type? Domain says vehicle is locatable.
                     # This case should ideally not happen in a valid state.
                     # Treat as unreachable.
                     return float('inf')
            else:
                 # Package is neither at a location nor in a vehicle. Invalid state?
                 # PDDL states list all *true* facts. If a package exists, it must be somewhere.
                 # Treat as unreachable.
                 return float('inf')


            # Add cost for this package to total
            total_cost += cost_for_package

        return total_cost
