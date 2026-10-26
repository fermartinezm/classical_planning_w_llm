from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the minimum actions
    needed for each package independently, ignoring vehicle capacity constraints
    and vehicle availability beyond the package's current state (i.e., if a
    package is on the ground, it assumes a vehicle will become available; if it's
    in a vehicle, it assumes that vehicle will be used).

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity is not a bottleneck (ignored).
    - Any location is reachable from any other location via roads (graph is connected).
    - The shortest path distance between locations represents the minimum number of drive actions.
    - Package goals are always of the form (at ?p ?l).
    - Objects appearing as the first argument of an 'at' predicate that are not packages with goals are vehicles.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph of locations based on the 'road' predicates in the static facts.
    - Compute the shortest path distance between all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the overall task goal is met. If yes, the heuristic is 0.
    2. Identify the current location of every package that has a goal. A package can be on the ground
       at a location (`at ?p ?l`) or inside a vehicle (`in ?p ?v`). If inside a vehicle,
       find the current location of that vehicle (`at ?v ?l`).
    3. For each package `p` with goal location `l_goal`:
       a. If the goal `(at p l_goal)` is already true in the state, the cost for this package is 0.
       b. If not at the goal location:
          i. Determine the package's current physical location (`l_current_physical`). This is either the location from `at ?p l_current_physical` or the location of the vehicle `v` from `at v l_current_physical` if `in ?p v`. Handle cases where the package or its vehicle is not found in the state (return infinity).
          ii. Determine if the package is currently inside a vehicle (`is_in_vehicle`).
          iii. Calculate the cost for this package:
              - If `is_in_vehicle`: The package is in a vehicle at `l_current_physical`. It needs to be transported to `l_goal` and dropped. Cost = shortest_distance(`l_current_physical`, `l_goal`) + 1 (drop).
              - If not `is_in_vehicle`: The package is on the ground at `l_current_physical`. It needs to be picked up, transported to `l_goal`, and dropped. Cost = 1 (pick-up) + shortest_distance(`l_current_physical`, `l_goal`) + 1 (drop).
          iv. Handle cases where the goal location is unreachable from the current physical location (return infinity).
    4. The total heuristic value for the state is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.goal_locations = {}
        # Store goal facts for quick lookup
        self.goal_facts = set()
        for goal in self.goals:
            self.goal_facts.add(goal)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # Assuming only (at ?p ?l) goals for packages.

        # Build the location graph from 'road' predicates.
        self.location_graph = {}
        locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Roads are typically bidirectional

        self.locations = list(locations) # Store locations for easy iteration

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """
        Performs BFS starting from start_loc to find shortest distances to all
        other locations.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc exists as a key in the graph before accessing neighbors
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if distances[neighbor] == float('inf'): # If not visited
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Quick check if overall goal is reached (heuristic is 0)
        if self.goal_facts <= state:
             return 0

        # Track current locations of packages and vehicles, and packages inside vehicles.
        package_state = {} # Maps package -> {'type': 'at'/'in', 'location'/'vehicle'}
        vehicle_location = {} # Maps vehicle -> location

        # Populate package_state and vehicle_location
        # Iterate through the state facts once
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we care about
                     package_state[obj] = {'type': 'at', 'location': loc}
                # Assume any other object 'at' a location is a vehicle
                # This is a simplification based on domain structure
                # Add to vehicle_location only if it's not a package we are tracking
                elif obj not in self.goal_locations:
                     vehicle_location[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Assume the first argument of 'in' is always a package
                if package in self.goal_locations: # It's a package we care about
                    package_state[package] = {'type': 'in', 'vehicle': vehicle}

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If the goal (at package goal_location) is already true, cost is 0 for this package.
            goal_fact_str = f"(at {package} {goal_location})"
            if goal_fact_str in state:
                 continue # Package is already at the goal location on the ground. Cost is 0.

            # Package is not yet at the goal location on the ground. Calculate cost to get it there.

            # Check if package state is known
            if package not in package_state:
                 # This case indicates the package is not represented in the state facts
                 # as either 'at' or 'in'. This shouldn't happen in a valid STRIPS state
                 # representation for objects that exist. Return infinity.
                 # print(f"Error: Package {package} not found in state.")
                 return float('inf')

            p_state = package_state[package]

            # Determine the package's current physical location
            current_physical_location = None
            is_in_vehicle = False

            if p_state['type'] == 'at':
                current_physical_location = p_state['location']
                is_in_vehicle = False
            elif p_state['type'] == 'in':
                is_in_vehicle = True
                vehicle = p_state['vehicle']
                # Find the vehicle's location
                if vehicle in vehicle_location:
                    current_physical_location = vehicle_location[vehicle]
                else:
                    # Vehicle location unknown - this state might be unreachable or malformed.
                    # If a package is in a vehicle, the vehicle must have an 'at' predicate.
                    # If not found, return infinity.
                    # print(f"Error: Vehicle {vehicle} carrying {package} has no known location in state.")
                    return float('inf') # Indicate a problematic state

            # Need shortest distance from current physical location to goal location
            if current_physical_location not in self.shortest_paths or goal_location not in self.shortest_paths[current_physical_location]:
                 # Goal location or current location not in graph (shouldn't happen if parsed correctly)
                 # Or graph is disconnected.
                 # print(f"Warning: Cannot find path from {current_physical_location} to {goal_location}")
                 return float('inf') # Indicate unreachable goal

            dist = self.shortest_paths[current_physical_location][goal_location]

            if dist == float('inf'):
                 # print(f"Warning: Path from {current_physical_location} to {goal_location} is infinite.")
                 return float('inf') # Indicate unreachable goal

            # Cost calculation based on current state and distance
            if is_in_vehicle:
                # Package is in a vehicle at current_physical_location.
                # Vehicle needs to drive from current_physical_location to goal_location (dist actions).
                # Then package needs to be dropped at goal_location (1 action).
                total_cost += dist + 1
            else:
                # Package is on the ground at current_physical_location.
                # Needs pick-up (1 action).
                # Vehicle needs to drive from current_physical_location to goal_location (dist actions).
                # Then package needs to be dropped at goal_location (1 action).
                total_cost += 1 + dist + 1 # = 2 + dist

        return total_cost
