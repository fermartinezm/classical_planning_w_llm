from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS implementation
def compute_distances(locations, graph):
    """Computes shortest path distances between all pairs of locations using BFS."""
    distances = {l1: {l2: float('inf') for l2 in locations} for l1 in locations}
    for start_loc in locations:
        distances[start_loc][start_loc] = 0
        queue = deque([(start_loc, 0)])

        while queue:
            current_loc, dist = queue.popleft()

            # If we found a shorter path later, ignore this older entry
            if dist > distances[start_loc][current_loc]:
                continue

            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if distances[start_loc][neighbor] == float('inf'): # Found first path to neighbor
                        distances[start_loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    that is not yet at its goal location. It sums the estimated costs for each
    such package independently. The cost for a package includes pickup (if on
    the ground), transport (drive actions based on shortest path distance),
    and dropoff.

    # Assumptions
    - Any vehicle can carry any package (capacity constraints are ignored).
    - Vehicles are available when needed (vehicle availability and contention
      are ignored).
    - The road network allows travel between necessary locations. If a goal
      location is unreachable from a package's or its vehicle's current
      location, the heuristic returns infinity.
    - Action costs are uniform (1).
    - Object names for locations are distinct from object names for vehicles
      and packages.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goal state.
    - Identifies all unique locations present in the initial state, goal state,
      and static road facts.
    - Builds a directed graph representing the road network based on static
      `road` facts.
    - Computes all-pairs shortest path distances between all identified locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Create a mapping `current_location_or_vehicle` to track where each
       locatable object (package or vehicle) is. Iterate through the state's
       facts: if `(at ?obj ?loc)` is found, map `?obj` to `?loc`; if
       `(in ?pkg ?veh)` is found, map `?pkg` to `?veh`.
    2. Initialize the total heuristic cost `total_cost` to 0.
    3. For each package `package` and its goal location `goal_location` stored
       during initialization (`self.goal_locations`):
       a. Check if the fact `(at package goal_location)` is present in the
          current state. If it is, the package is already delivered to its
          final location on the ground; add 0 cost for this package and
          continue to the next package.
       b. If the package is not at its goal location, retrieve its current
          status (`current_status`) from the map created in step 1. This
          `current_status` will be either a location name (if the package is
          on the ground) or a vehicle name (if the package is inside a vehicle).
       c. Determine if the package is on the ground or in a vehicle by checking
          if `current_status` is present in the set of known location names
          (`self.all_locations`).
       d. If the package is on the ground at `loc_p_current` (`current_status`
          is a location):
          - Look up the shortest distance `dist` from `loc_p_current` to
            `goal_location` in the precomputed distances (`self.distances`).
          - If `dist` is `float('inf')`, it means the goal location is
            unreachable from the package's current position via the road
            network. In this case, the problem is likely unsolvable from
            this state, so return `float('inf')` immediately.
          - Otherwise, add `dist + 2` to `total_cost`. This represents the
            cost of picking up the package (1 action), driving the vehicle
            `dist` times, and dropping the package (1 action).
       e. If the package is inside a vehicle `v` (`current_status` is a vehicle):
          - Find the current location `loc_v_current` of vehicle `v` from the
            map created in step 1. If the vehicle's location is not found (which
            should not happen in a valid state), return `float('inf')`.
          - Look up the shortest distance `dist` from `loc_v_current` to
            `goal_location` in `self.distances`.
          - If `dist` is `float('inf')`, return `float('inf')`.
          - Otherwise, add `dist + 1` to `total_cost`. This represents the
            cost of driving the vehicle `dist` times and dropping the package
            (1 action).
    4. After iterating through all goal packages, return the accumulated
       `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and extracting goals."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # 2. Identify all unique locations
        all_locations = set()
        road_facts = set()

        # Locations from static road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)
                road_facts.add(fact) # Store road facts for graph building

        # Locations from initial 'at' facts
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 # obj = parts[1] # obj can be vehicle or package
                 loc = parts[2]
                 all_locations.add(loc)

        # Locations from goal 'at' facts
        for fact in self.goals:
             parts = get_parts(fact)
             if parts[0] == "at":
                 # obj = parts[1] # obj is package
                 loc = parts[2]
                 all_locations.add(loc)

        self.all_locations = all_locations # Keep as set for O(1) lookup

        # 3. Build the road graph
        self.graph = {loc: [] for loc in self.all_locations}
        for fact in road_facts:
            parts = get_parts(fact)
            l1, l2 = parts[1], parts[2]
            # Ensure l1 is a known location before adding to graph (should be if collected correctly)
            if l1 in self.graph:
                self.graph[l1].append(l2)
            # else: print(f"Warning: Road fact {fact} involves unknown location {l1}") # Optional warning


        # 4. Compute all-pairs shortest path distances
        # Pass list(self.all_locations) because compute_distances expects an iterable
        self.distances = compute_distances(list(self.all_locations), self.graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current locations/vehicles for locatables
        current_location_or_vehicle = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_location_or_vehicle[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_location_or_vehicle[package] = vehicle

        # 2. Initialize total cost
        total_cost = 0

        # 3. Iterate through packages and their goals
        for package, goal_location in self.goal_locations.items():
            # 3a. Check if package is already at goal
            # Check for the exact goal fact string in the state frozenset
            if "(at {} {})".format(package, goal_location) in state:
                continue # Package is already at its goal location on the ground

            # 3b. Package is not at goal, determine current status
            current_status = current_location_or_vehicle.get(package)

            if current_status is None:
                 # This package is a goal package but is not found in the state
                 # (neither at a location nor in a vehicle). This implies an
                 # invalid state or the package was never initialized correctly.
                 # Treat as unreachable goal.
                 return float('inf')

            # 3c/3d. Check if current_status is a location (on ground) or a vehicle (in vehicle)
            if current_status in self.all_locations: # Package is on the ground
                loc_p_current = current_status
                # Cost = distance(loc_p_current, goal_location) + 2 (pickup, drive, drop)
                dist = self.distances.get(loc_p_current, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from package's current location
                total_cost += dist + 2

            else: # current_status is a vehicle (package is inside a vehicle)
                vehicle = current_status
                loc_v_current = current_location_or_vehicle.get(vehicle)

                if loc_v_current is None:
                    # Vehicle exists and contains a package, but the vehicle's
                    # location is unknown. Invalid state? Treat as unreachable.
                    return float('inf')

                # Cost = distance(loc_v_current, goal_location) + 1 (drive, drop)
                dist = self.distances.get(loc_v_current, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from vehicle's current location
                total_cost += dist + 1

        # 4. Return total sum
        return total_cost
