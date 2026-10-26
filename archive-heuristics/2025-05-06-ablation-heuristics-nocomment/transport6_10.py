import collections
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Helper to parse PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class transportHeuristic: # Inherit from Heuristic if available, otherwise standalone
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    up the estimated costs for each package that is not currently at its
    final goal location. The cost for a single package is estimated based
    on whether it needs to be picked up (if on the ground), the shortest
    path distance it needs to travel, and the cost to drop it at the
    destination.

    Assumptions:
    - Roads are represented by `(road l1 l2)` facts and are directed. The graph
      is built exactly as specified by these facts.
    - Vehicle capacity is ignored. Any vehicle is assumed capable of carrying
      any package needed.
    - Vehicle availability is simplified. The heuristic assumes a vehicle is
      eventually available to perform necessary pick-up and transport for
      each package.
    - The shortest path distance between locations is a reasonable estimate
      of the minimum number of drive actions required to move between them.
    - Objects mentioned in goal facts like `(at package location)` are packages.
    - Objects mentioned in static facts like `(capacity vehicle size)` are vehicles.

    Heuristic Initialization:
    1. Parses the goal facts to create a mapping from each package to its
       required goal location (`self.goal_locations`).
    2. Identifies all vehicles based on static `(capacity ...)` facts (`self.vehicles`).
    3. Builds a graph representation of the road network from the static
       `(road l1 l2)` facts (`self.road_graph`). Collects all locations mentioned
       in road facts or goal facts.
    4. Computes the shortest path distance between all pairs of collected
       locations using Breadth-First Search (BFS) and stores them in
       `self.distances`. Unreachable locations have an effective distance
       of infinity.

    Step-By-Step Thinking for Computing Heuristic:
    1. For a given state, identify the current location or containing vehicle
       for every package (that is a goal package) and the location for every vehicle.
       Store this in `current_package_status` and `vehicle_locations`.
    2. Initialize the total heuristic cost `total_cost` to 0.
    3. Iterate through each package `p` that has a goal location `l_goal`
       defined in the problem's goal state (`self.goal_locations`).
    4. Determine the package's actual current location (`l_current`) and
       whether it is currently inside a vehicle (`is_in_vehicle`). This
       involves looking up the package in `current_package_status` and,
       if it's in a vehicle, looking up the vehicle's location in
       `vehicle_locations`. If a package or its containing vehicle is not
       found, the goal is considered unreachable (heuristic infinity).
    5. If the package is already at its goal location on the ground
       (`l_current == l_goal` and not `is_in_vehicle`), it contributes 0
       to the heuristic for this package, and we move to the next package.
    6. If the package is not at its goal location on the ground:
       a. Initialize `package_cost` to 0.
       b. If the package is currently on the ground (`not is_in_vehicle`),
          add 1 to `package_cost` for the required `pick-up` action.
       c. Calculate the shortest path distance from `l_current` to `l_goal`
          using the pre-calculated distances (`self.get_distance`). Add this
          distance to `package_cost` for the required `drive` actions. If the
          goal is unreachable (`get_distance` returns infinity), the total
          heuristic is infinity.
       d. Add 1 to `package_cost` for the required `drop` action at the goal
          location.
       e. Add `package_cost` to the `total_cost`.
    7. The final `total_cost` is the heuristic value for the state.
    The heuristic is 0 if and only if all goal packages are at their
    respective goal locations on the ground.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating static information.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal facts to get package -> goal_location mapping
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure it's (at ?pkg ?loc)
                    _, package, location = parts
                    self.goal_locations[package] = location

        # 2. Identify vehicles from static capacity facts
        self.vehicles = set()
        for fact in static_facts:
             if match(fact, "capacity", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure it's (capacity ?veh ?size)
                    _, vehicle, _ = parts
                    self.vehicles.add(vehicle)


        # 3. Build the road network graph and collect all relevant locations
        self.road_graph = collections.defaultdict(set)
        all_potential_locations = set() # Collect all locations mentioned

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (road ?l1 ?l2)
                    l1, l2 = parts[1], parts[2]
                    self.road_graph[l1].add(l2)
                    all_potential_locations.add(l1)
                    all_potential_locations.add(l2)

        # Add any goal locations not in road_graph keys/values (isolated locations)
        for goal_loc in self.goal_locations.values():
             all_potential_locations.add(goal_loc)

        # 4. Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        self._infinity = 1000000 # A large number representing unreachable

        # Ensure BFS runs from/to all relevant locations
        all_locations_list = list(all_potential_locations)

        for start_loc in all_locations_list:
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Store distance
                self.distances[(start_loc, current_loc)] = dist

                # Explore neighbors
                if current_loc in self.road_graph: # Handle locations with no outgoing roads
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

        # For pairs not found by BFS, distance remains implicitly infinity (KeyError or handled by get_distance)

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        # If BFS found a path, the distance is stored.
        # If BFS didn't find a path, the pair won't be in self.distances.
        # get() method handles missing keys by returning the default value (_infinity)
        return self.distances.get((loc1, loc2), self._infinity)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated number of actions
            to reach a goal state. Returns infinity if a goal is unreachable.
        """
        state = node.state

        # Map current package locations and vehicle locations
        current_package_status = {} # package -> location (if at) or vehicle (if in)
        vehicle_locations = {} # vehicle -> location

        # First pass to find vehicle locations and package statuses
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc
                elif obj in self.goal_locations: # Assume objects in goals are packages
                    current_package_status[obj] = loc
            elif parts[0] == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # Only track packages relevant to goals
                    current_package_status[package] = vehicle # Store vehicle name for package

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state status (e.g., missing), treat as unreachable goal
            if package not in current_package_status:
                 return self._infinity

            current_loc_or_vehicle = current_package_status[package]

            # Determine the package's actual current location and if it's in a vehicle
            is_in_vehicle = current_loc_or_vehicle in self.vehicles # Check if the status is a vehicle name

            if is_in_vehicle:
                vehicle_name = current_loc_or_vehicle
                # Need vehicle's location. If vehicle not found in state, something is wrong.
                if vehicle_name not in vehicle_locations:
                    # This implies an inconsistent state (package in vehicle, but vehicle not at a location)
                    # For heuristic, treat as unreachable or high cost.
                    return self._infinity # Or a very large number

                current_location = vehicle_locations[vehicle_name]
            else: # It's a location name
                current_location = current_loc_or_vehicle


            # If package is already at its goal on the ground, cost is 0 for this package
            if current_location == goal_location and not is_in_vehicle:
                continue # Goal achieved for this package

            # Calculate cost for this package
            package_cost = 0

            if not is_in_vehicle:
                # Package is on the ground, needs pick-up
                package_cost += 1 # Pick-up action

            # Needs to reach goal location
            dist = self.get_distance(current_location, goal_location)

            if dist == self._infinity:
                 # If goal is unreachable, return infinity for the heuristic
                 return self._infinity

            package_cost += dist # Drive actions

            # Needs drop action at goal location
            package_cost += 1 # Drop action

            total_cost += package_cost

        # The heuristic is 0 if and only if all goal conditions are met.
        # Goal conditions are (at package goal_location).
        # Our loop only adds cost if package is NOT at goal_location (and not in vehicle).
        # If a package is (in p v) and (at v goal_location), it still needs a drop.
        # Our logic handles this: current_location == goal_location but is_in_vehicle is True.
        # It will calculate cost += dist (0) + 1 (drop). This is correct.
        # If all packages are (at p goal_location), the loop continues, current_location == goal_location and not is_in_vehicle is true, so continue is hit, total_cost remains 0.
        # So, h=0 iff all packages are at their goal locations on the ground.

        return total_cost
