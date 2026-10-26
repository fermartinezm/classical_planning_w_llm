import collections

# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It considers whether the package is on the ground or
    inside a vehicle and uses precomputed shortest path distances between locations
    to estimate drive costs. It sums the estimated costs for each package independently.

    # Assumptions
    - The primary goal is to move packages to their target locations.
    - Vehicle capacity and availability are simplified: it assumes a suitable vehicle
      will eventually be available to pick up a package or transport a package.
    - The cost of picking up and dropping a package is 1 action each.
    - The cost of driving between locations is the shortest path distance in the road network.
    - All packages are distinct and their goals are independent (cost is summed).
    - The road network is connected for all relevant locations.

    # Heuristic Initialization
    - Build a graph representing the road network from `road` facts.
    - Compute shortest path distances between all pairs of locations using BFS.
    - Extract the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every locatable object (packages and vehicles).
    2. Identify which packages are currently inside vehicles.
    3. Initialize total heuristic cost to 0.
    4. For each package `p` that has a goal location `l_goal` (extracted during initialization):
       a. Check if the package is already at `l_goal`. If `(at p l_goal)` is in the state, the cost for this package is 0. Continue to the next package.
       b. If the package is not at `l_goal`, determine its current status:
          - If `p` is currently on the ground at `l_current` (i.e., `(at p l_current)` is in the state and `l_current != l_goal`):
            - Estimate the cost to move this package: 1 (pick-up) + `dist(l_current, l_goal)` (drive) + 1 (drop). Total: `2 + dist(l_current, l_goal)`.
          - If `p` is currently inside vehicle `v` (i.e., `(in p v)` is in the state):
            - Find the current location of vehicle `v`, say `l_vehicle` (i.e., `(at v l_vehicle)` is in the state).
            - If `l_vehicle == l_goal`:
              - Estimate the cost: 1 (drop). Total: 1.
            - If `l_vehicle != l_goal`:
              - Estimate the cost: `dist(l_vehicle, l_goal)` (drive) + 1 (drop). Total: `1 + dist(l_vehicle, l_goal)`.
       c. Add the estimated cost for package `p` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing distances.
        """
        # Assuming task object has 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts)
        self.goals = task.goals
        static_facts = task.static

        # Build the road network graph.
        self.graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each package mentioned in the goals.
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are always (at package location) for packages
            if parts and parts[0] == "at" and len(parts) == 3:
                 # Check if the first argument is a package type.
                 # We don't have type info here, but typically goals are for packages.
                 # Let's assume any 'at' goal for a 'locatable' object is a package goal
                 # unless it's a vehicle goal (less common in transport problems).
                 # A safer approach would be to parse types from the domain or instance,
                 # but given the example goals, 'at' facts in goals are for packages.
                 package, location = parts[1], parts[2]
                 self.package_goals[package] = location

        # Capacity info is ignored in this basic heuristic.
        # self.capacity_predecessors = {}
        # for fact in static_facts:
        #     parts = get_parts(fact)
        #     if parts and parts[0] == "capacity-predecessor":
        #         s1, s2 = parts[1], parts[2]
        #         self.capacity_predecessors[s1] = s2 # s1 is smaller than s2

    def _bfs(self, start_location):
        """
        Perform BFS from a start location to find shortest distances to all other locations.
        Returns a dictionary mapping location to distance.
        Returns float('inf') for unreachable locations.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_location not in self.locations:
             # Start location is not in the graph (e.g., from a malformed state)
             return distances # All distances remain infinity

        distances[start_location] = 0
        queue = collections.deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc is a valid key in graph
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Map current locations of locatable objects (packages and vehicles)
        current_locations = {}
        # Map packages currently inside vehicles
        package_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                current_locations[obj] = location
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
                # Note: 'in' facts mean the package is *not* at a location on the ground.
                # We track its container (vehicle) separately.

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined in the task goals.
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground.
            # We need to check if the 'at' fact exists in the current state.
            if (package in current_locations and current_locations[package] == goal_location):
                 # Package is already at the goal location on the ground. Cost is 0 for this package.
                 continue

            # Check if the package is currently in a vehicle.
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find the location of the vehicle.
                vehicle_location = current_locations.get(vehicle)

                if vehicle_location is None:
                     # Vehicle location is unknown in the state. This shouldn't happen in valid states.
                     # Treat as a high cost or skip. Skipping this package's contribution.
                     # print(f"Warning: Vehicle {vehicle} location not found in state for package {package}.")
                     total_cost += 1000 # Add a penalty
                     continue

                # Get the distance from the vehicle's current location to the package's goal location.
                # Ensure both locations are in our precomputed distances graph.
                if vehicle_location in self.distances and goal_location in self.distances.get(vehicle_location, {}):
                    drive_cost = self.distances[vehicle_location][goal_location]

                    # If the vehicle is at the goal location, only a drop is needed.
                    if drive_cost == 0: # vehicle_location == goal_location
                        total_cost += 1 # Cost: drop

                    # If the vehicle is not at the goal location, it needs to drive and then drop.
                    else:
                        total_cost += drive_cost + 1 # Cost: drive + drop
                else:
                    # Cannot find distance (e.g., goal_location not in graph, or vehicle_location not in graph).
                    # This suggests an issue with the problem definition or state.
                    # Assume a high cost to discourage states with unreachable goals/vehicles.
                    # print(f"Warning: Cannot find distance from {vehicle_location} to {goal_location} for package {package}.")
                    total_cost += 1000 # Arbitrary large cost

            # Check if the package is on the ground at a non-goal location.
            elif package in current_locations: # package is at some location, and we already checked if it's the goal
                 current_location = current_locations[package]
                 # Get the distance from the package's current location to its goal location.
                 # Ensure both locations are in our precomputed distances graph.
                 if current_location in self.distances and goal_location in self.distances.get(current_location, {}):
                      drive_cost = self.distances[current_location][goal_location]
                      # Cost: pick-up + drive from current_location to goal_location + drop
                      total_cost += 1 + drive_cost + 1 # Cost: pick-up + drive + drop
                 else:
                      # Cannot find distance. Assume high cost.
                      # print(f"Warning: Cannot find distance from {current_location} to {goal_location} for package {package}.")
                      total_cost += 1000 # Arbitrary large cost

            # If a package with a goal is not found in 'at' or 'in' facts,
            # this indicates an invalid state representation.
            # We assume valid states where every locatable object has a status.
            # If it happens, the package's contribution is implicitly 0 in this loop,
            # which is incorrect if it needs to reach a goal.
            # However, the problem description implies valid states.

        # The heuristic is 0 if and only if all packages listed in self.package_goals
        # are currently located at their respective goal locations on the ground.
        # This aligns with the common goal structure for transport problems.

        return total_cost
