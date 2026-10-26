from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

# Helper function to parse PDDL facts string
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated minimum cost for each package that is not yet at its goal
    location. The cost for a package depends on its current state:
    - If the package is at a location different from its goal location,
      it needs to be picked up, transported, and dropped. The estimated cost
      is 1 (pick-up) + shortest_distance(current_location, goal_location)
      (drive actions) + 1 (drop).
    - If the package is inside a vehicle, it needs to be transported (if the
      vehicle is not already at the package's goal location) and dropped.
      The estimated cost is shortest_distance(vehicle_location, goal_location)
      (drive actions) + 1 (drop).
    Capacity constraints and shared vehicle trips are ignored, making the
    heuristic non-admissible but potentially effective for greedy search.

    Assumptions:
    - The state representation is a frozenset of strings, where each string
      is a PDDL fact like '(predicate arg1 arg2)'.
    - The goal is a frozenset of facts, typically '(at package location)'.
    - Static facts (like 'road' and 'capacity-predecessor') are provided
      during initialization.
    - Objects appearing in '(at ?p ?l)' goals are packages.
    - Objects appearing in '(capacity ?v ?s)' static facts are vehicles.
    - The road network is connected such that goal locations are reachable
      from relevant initial locations (or the heuristic returns infinity).
    - Action costs are uniform (1).

    Heuristic Initialization:
    - Parses the goal facts to create a mapping from package names to their
      target goal locations and identifies packages.
    - Parses static facts to identify vehicles and build an adjacency list
      representation of the road network graph, identifying locations.
    - Computes all-pairs shortest paths between identified locations using
      BFS starting from each location. Stores these distances in a dictionary.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h = 0`.
    2. Parse the current state facts to determine the current status of each
       package (location or vehicle) and the current location of each vehicle.
       Store this information in dictionaries (`package_current_status`,
       `vehicle_current_locations`) for quick lookup.
    3. Iterate through each package `p` and its goal location `loc_p_goal`
       as determined during initialization (from `self.goal_locations`).
    4. Check if the fact `(at p loc_p_goal)` is present in the current state.
       If it is, the package is already at its goal, and its contribution
       to the heuristic is 0. Continue to the next package.
    5. If the package `p` is not at its goal, find its current status from
       the parsed state information (`package_current_status.get(p)`).
    6. If the current status is a location `loc_p_current` (checked by seeing
       if the status string is in `self.locations`):
       - The package needs to be picked up (1 action), transported, and dropped
         (1 action). The transportation involves the vehicle driving from
         `loc_p_current` to `loc_p_goal`.
       - Look up the shortest distance `dist` from `loc_p_current` to `loc_p_goal`
         in the precomputed distances (`self.distances`).
       - If `dist` is found, add `dist + 2` to `h`.
       - If `dist` is not found (locations are disconnected), return `float('inf')`.
    7. If the current status is a vehicle `v` (checked by seeing if the status
       string is in `self.vehicles`):
       - Find the current location `loc_v_current` of vehicle `v` from
         `vehicle_current_locations.get(v)`.
       - If `loc_v_current` is not found (malformed state), return `float('inf')`.
       - The package needs to be transported (if the vehicle is not already
         at the goal) and dropped (1 action). The transportation involves
         the vehicle driving from `loc_v_current` to `loc_p_goal`.
       - Look up the shortest distance `dist` from `loc_v_current` to `loc_p_goal`
         in the precomputed distances (`self.distances`).
       - If `dist` is found, add `dist + 1` to `h`.
       - If `dist` is not found (locations are disconnected), return `float('inf')`.
    8. After iterating through all packages, the total value of `h` is the
       heuristic estimate for the current state.
    """
    def __init__(self, task):
        """
        Initializes the heuristic with task information.

        Args:
            task: The planning task object containing goals, initial state,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        road_graph = {} # adjacency list: location -> [neighbor_location, ...]

        # Identify objects and locations from goals and static facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                 package, location = parts[1], parts[2]
                 self.packages.add(package)
                 self.locations.add(location)
                 self.goal_locations[package] = location

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                road_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional based on example
                road_graph.setdefault(l2, []).append(l1)
            elif parts[0] == 'capacity': # Vehicles have capacity
                 vehicle = parts[1]
                 self.vehicles.add(vehicle)
            # capacity-predecessor facts are ignored for this heuristic

        # Ensure all locations mentioned in goals are in the graph structure
        # even if they have no roads connected (isolated locations might exist)
        for loc in self.goal_locations.values():
             self.locations.add(loc)
             road_graph.setdefault(loc, []) # Add location even if it has no roads

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in road_graph:
                    for neighbor in road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            The estimated cost (number of actions) to reach a goal state,
            or float('inf') if a goal location is unreachable.
        """
        state = node.state

        # Parse current state to find package statuses and vehicle locations
        package_current_status = {} # package -> location or vehicle
        vehicle_current_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_status[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles:
                     package_current_status[package] = vehicle

        total_cost = 0

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal
            if f'(at {package} {goal_location})' in state:
                continue # Package is already at goal, cost is 0 for this package

            # Find the package's current status (location or vehicle)
            current_status = package_current_status.get(package)

            # Handle potential missing package status (shouldn't happen in valid states)
            if current_status is None:
                 # This package from the goal is not found in the state.
                 # This indicates a potentially malformed state or unsolvable problem.
                 # Assign a very high cost.
                 return float('inf')

            # Case 1: Package is at a location
            if current_status in self.locations:
                loc_p_current = current_status
                # Estimated cost: pick-up (1) + drive (dist) + drop (1)
                # Need distance from loc_p_current to goal_location
                dist = self.distances.get(loc_p_current, {}).get(goal_location)

                if dist is None:
                    # Goal location is unreachable from the package's current location
                    return float('inf') # Problem likely unsolvable from this state

                total_cost += dist + 2

            # Case 2: Package is inside a vehicle
            elif current_status in self.vehicles:
                vehicle = current_status
                loc_v_current = vehicle_current_locations.get(vehicle)

                # Handle potential missing vehicle location (shouldn't happen in valid states)
                if loc_v_current is None:
                    # Vehicle containing the package is not found at any location.
                    return float('inf') # Malformed state

                # Estimated cost: drive (dist) + drop (1)
                # Need distance from vehicle's current location to goal_location
                dist = self.distances.get(loc_v_current, {}).get(goal_location)

                if dist is None:
                    # Goal location is unreachable from the vehicle's current location
                    return float('inf') # Problem likely unsolvable from this state

                total_cost += dist + 1

            # else: current_status is neither a location nor a vehicle? Malformed state.
            # We assume current_status is always one of these based on domain structure.

        return total_cost
