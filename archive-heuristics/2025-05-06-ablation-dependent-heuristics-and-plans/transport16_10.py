from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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
    # Pattern must not be longer than the fact parts
    if len(args) > len(parts):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location or vehicle to its goal location. It sums the
    estimated costs for all packages that are not yet at their goal. The cost
    includes actions for picking up (if on the ground), driving (estimated by
    shortest path distance), and dropping off.

    # Assumptions
    - Each package needs a sequence of actions: potentially pick-up, one or more
      drives, and drop-off.
    - Vehicle capacity is simplified: the heuristic does not explicitly model
      vehicle capacity constraints or the availability of a vehicle for pick-up.
      It assumes a vehicle is available when a package needs to be picked up.
    - The cost of transporting a package between two locations is approximated
      by the shortest path distance in the road network graph, where each edge
      represents one 'drive' action.
    - The heuristic ignores the specific vehicle used and its current location
      when estimating the cost for a package on the ground; it only considers
      the package's location and goal location. When a package is in a vehicle,
      it uses the vehicle's current location.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from the static 'road' facts.
    - Collect all unique locations mentioned in 'road' facts, initial state 'at'
      facts, and goal 'at' facts.
    - Precompute the shortest path distance between all pairs of these locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status (location on the ground or inside a vehicle)
       for every package and the current location for every vehicle by examining
       the 'at' and 'in' facts in the state.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined in the task:
       a. Check if the package is already at its goal location on the ground
          (i.e., the fact `(at package goal_location)` is in the state). If yes,
          the cost for this package is 0, and we move to the next package.
       b. If the package is not at its goal, determine its current status:
          - If the package is currently on the ground at `current_loc`
            (i.e., `(at package current_loc)` is in the state, and `current_loc`
            is not the goal):
            - Estimate the minimum actions needed for this package: 1 (pick-up)
              + shortest_distance(`current_loc`, `goal_location`) (drive actions)
              + 1 (drop).
            - Add this estimated cost to the total heuristic.
          - If the package is currently inside a vehicle `vehicle_name`
            (i.e., `(in package vehicle_name)` is in the state):
            - Find the vehicle's current location `vehicle_loc` from the state
              (i.e., look for `(at vehicle_name vehicle_loc)`).
            - If the vehicle's location is known:
              - Estimate the minimum actions needed for this package:
                shortest_distance(`vehicle_loc`, `goal_location`) (drive actions)
                + 1 (drop).
              - Add this estimated cost to the total heuristic.
            - If the vehicle's location is not found in the state (indicating a
              potentially invalid state or unreachable goal), add infinity to
              the total cost for this package.
          - If the package's status is neither 'at' a location nor 'in' a vehicle
            (indicating a potentially invalid state), add infinity to the total
            cost for this package.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road network, and precomputing shortest path distances.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Ensure the fact has enough parts before accessing args[1]
                if len(args) >= 2:
                    package, location = args
                    self.goal_locations[package] = location
                # else: Assume valid PDDL goal facts for simplicity.


        # Build the road network graph and collect all unique locations.
        self.road_network = {}
        all_locations = set()

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                # Ensure the fact has enough parts before accessing parts[1], parts[2]
                if len(parts) >= 3:
                    _, loc1, loc2 = parts
                    all_locations.add(loc1)
                    all_locations.add(loc2)
                    if loc1 not in self.road_network:
                        self.road_network[loc1] = []
                    self.road_network[loc1].append(loc2)
                # else: Assume valid PDDL static facts for simplicity.


        # Add locations from initial state 'at' facts and goal 'at' facts
        # that might not be explicitly mentioned in 'road' facts (e.g., isolated locations).
        # The Task object provides initial_state.
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                  parts = get_parts(fact)
                  if len(parts) >= 3:
                    _, obj, loc = parts
                    all_locations.add(loc)

        for loc in self.goal_locations.values():
             all_locations.add(loc)

        # Ensure all collected locations are keys in the network dict, even if they have no outgoing roads.
        # This is necessary so BFS can be run starting from any relevant location.
        for loc in all_locations:
             if loc not in self.road_network:
                 self.road_network[loc] = []


        # Precompute shortest path distances between all pairs of locations.
        self.shortest_distances = {}
        for start_loc in all_locations: # Run BFS from ALL unique locations
            self.shortest_distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """
        Perform BFS from a start location to find shortest distances to all
        reachable locations in the road network.
        """
        distances = {start_loc: 0}
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            # Get neighbors from the road network. Handle locations with no outgoing roads.
            # Use .get() with default empty list for robustness if current_loc somehow wasn't added to road_network keys
            neighbors = self.road_network.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

        return distances

    def get_distance(self, loc1, loc2):
        """
        Get the precomputed shortest distance between two locations.
        Returns float('inf') if loc2 is unreachable from loc1 or if loc1/loc2
        were not among the locations collected during initialization.
        """
        # Check if loc1 is a valid starting point in our precomputed map
        if loc1 not in self.shortest_distances:
             # This location wasn't seen in init facts (roads, initial 'at', goals).
             # It might be an intermediate location in a complex state, but if we can't
             # start a path from it, treat as unreachable for distance calculation.
             return float('inf')

        # Check if loc2 is reachable from loc1
        if loc2 not in self.shortest_distances[loc1]:
             # loc2 is not reachable from loc1
             return float('inf')

        return self.shortest_distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current location/status for all locatable objects in the state
        current_locations = {} # Maps locatable object (package or vehicle) to its location
        package_in_vehicle = {} # Maps package to the vehicle it's in

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                # Ensure fact has enough parts
                if len(args) >= 2:
                    obj, location = args
                    current_locations[obj] = location
            elif predicate == "in":
                 # Ensure fact has enough parts
                 if len(args) >= 2:
                    package, vehicle = args
                    package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            # We need to check if the exact goal fact exists in the state
            if f"(at {package} {goal_location})" in state:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal. Calculate cost based on its current status.
            cost_for_package = 0

            if package in package_in_vehicle:
                # Package is in a vehicle.
                vehicle = package_in_vehicle[package]
                vehicle_loc = current_locations.get(vehicle) # Get the vehicle's location

                if vehicle_loc is None:
                    # Vehicle location not found? This state is likely problematic
                    # or represents an unreachable goal from this point.
                    # Assign a very high cost.
                    cost_for_package = float('inf')
                else:
                    # Needs to drive from vehicle_loc to goal_location and then drop.
                    drive_cost = self.get_distance(vehicle_loc, goal_location)
                    drop_cost = 1
                    cost_for_package = drive_cost + drop_cost

            elif package in current_locations:
                # Package is on the ground at current_locations[package]
                current_loc = current_locations[package]

                # Needs to be picked up, driven from current_loc to goal_location, and dropped.
                pick_cost = 1
                drive_cost = self.get_distance(current_loc, goal_location)
                drop_cost = 1
                cost_for_package = pick_cost + drive_cost + drop_cost
            else:
                 # Package status is unknown (neither 'at' nor 'in'). Should not happen in valid PDDL.
                 # This state is likely problematic or represents an unreachable goal.
                 # Assign a very high cost.
                 cost_for_package = float('inf')

            total_cost += cost_for_package

        return total_cost
