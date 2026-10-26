from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Removes leading/trailing parentheses and splits by space."""
    return fact[1:-1].split()

# Helper function to match fact parts with arguments (allowing wildcards)
def match(fact, *args):
    """Checks if fact parts match arguments (allowing wildcards)."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each package that is not yet at its final
    destination. The cost for a package is estimated based on its current
    status (at a location or in a vehicle) and the shortest path distance
    on the road network to its goal location. It simplifies the problem
    by ignoring vehicle capacity constraints and the availability of
    specific vehicles at specific locations, focusing primarily on the
    movement requirements for packages.

    Assumptions:
    - Roads are bidirectional (if a road exists from A to B, one exists from B to A).
    - The goal state only involves packages being at specific locations (facts like `(at ?p ?l)`).
    - Any package mentioned in a goal fact `(at ?p ?l)` is a package object.
    - Any object mentioned in an initial state fact `(at ?v ?l)` or `(in ?p ?v)`
      that is not identified as a package is a vehicle object.
    - Vehicle capacity constraints are ignored for heuristic calculation.
    - Vehicle availability at specific locations for pick-up is simplified;
      a pick-up action is assumed possible if the package is at a location.
    - The heuristic returns infinity if any package's goal location is
      unreachable via the road network from its current location or vehicle's location.

    Heuristic Initialization:
    1. Parses the goal facts to identify all packages that need to reach a specific
       location and stores their target locations in `self.package_goal_locations`.
    2. Identifies all package and vehicle objects based on goal facts and initial
       state facts.
    3. Builds a graph of locations based on the static `(road ?l1 ?l2)` facts.
       Assumes roads are bidirectional.
    4. Computes all-pairs shortest path distances between all locations using
       Breadth-First Search (BFS) and stores them in `self.shortest_paths`.

    Step-By-Step Thinking for Computing Heuristic:
    1. For a given state, parse the facts to determine the current location of
       each vehicle (`current_vehicle_locations`) and the current status of
       each package (`current_package_locations` if at a location,
       `packages_in_vehicles` if inside a vehicle).
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each package that has a goal location defined in the task.
    4. For the current package:
       a. Check if the package is already at its goal location (i.e., it's in
          `current_package_locations` and its location matches the goal location).
          If yes, this package contributes 0 to the heuristic.
       b. If the package is not at its goal:
          i. If the package is currently at a location (found in `current_package_locations`):
             - Estimate the cost as 1 (for pick-up) + shortest_distance(current_location, goal_location)
               (for driving) + 1 (for drop).
             - If the goal location is unreachable from the current location, the total
               heuristic for the state is infinity.
             - Add this estimated cost to the total heuristic.
          ii. If the package is currently inside a vehicle (found in `packages_in_vehicles`):
              - Find the current location of the vehicle carrying the package.
              - Estimate the cost as shortest_distance(vehicle_location, goal_location)
                (for driving) + 1 (for drop).
              - If the goal location is unreachable from the vehicle's location, the total
                heuristic for the state is infinity.
              - Add this estimated cost to the total heuristic.
          iii. If the package is neither at a location nor in a vehicle (implies an
               invalid or unreachable state for this package based on assumptions):
               - The total heuristic for the state is infinity.
    5. Return the calculated total heuristic cost.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Extract goal locations for packages and identify packages
        self.package_goal_locations = {}
        self.all_packages = set()
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goal_locations[package] = location
                self.all_packages.add(package)

        # 2. Identify vehicles and all locations
        self.all_vehicles = set()
        self.all_locations = set()

        # Identify vehicles and locations from initial state and static facts
        # Assuming objects in initial 'at' facts that are not goal packages are vehicles
        # Assuming objects in initial 'in' facts (second arg) are vehicles
        # Assuming locations in 'at' and 'road' facts are all locations
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj not in self.all_packages:
                     self.all_vehicles.add(obj)
                 self.all_locations.add(loc)
             elif parts[0] == 'in':
                 package, vehicle = parts[1], parts[2]
                 # Ensure package is known (might be in vehicle initially but not in goal)
                 self.all_packages.add(package)
                 self.all_vehicles.add(vehicle)
             elif parts[0] == 'road':
                 _, loc1, loc2 = parts[1], parts[2]
                 self.all_locations.add(loc1)
                 self.all_locations.add(loc2)

        # 3. Build the location graph
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Assuming bidirectional roads

        # 4. Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in self.all_locations:
             self.shortest_paths[start_loc] = self._bfs(start_loc, self.all_locations)


    def _bfs(self, start_loc, all_locations):
        """
        Performs Breadth-First Search to find shortest paths from start_loc
        to all other locations.

        Args:
            start_loc: The starting location.
            all_locations: A set of all locations in the domain.

        Returns:
            A dictionary mapping each location to its shortest distance from start_loc.
        """
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()
            # Check if u exists in graph (handle isolated locations)
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # 1. Identify current locations/statuses
        current_package_locations = {} # package -> location (if at location)
        packages_in_vehicles = {} # package -> vehicle (if in vehicle)
        current_vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.all_packages:
                    current_package_locations[obj] = loc
                elif obj in self.all_vehicles:
                    current_vehicle_locations[obj] = loc
                # Ignore 'at' facts for other types if any
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                if package in self.all_packages and vehicle in self.all_vehicles:
                     packages_in_vehicles[package] = vehicle
                # Ignore 'in' facts for other types if any

        total_cost = 0

        # 3. Iterate through packages that have a goal location
        for package, goal_location in self.package_goal_locations.items():
            # 4a. Check if the package is already at its goal location
            if package in current_package_locations and current_package_locations[package] == goal_location:
                continue # Package is already at the goal

            # 4b. Package is not at the goal. Calculate cost.
            if package in current_package_locations:
                # 4bi. Package is at a location, but not the goal
                current_location = current_package_locations[package]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                drive_cost = self.shortest_paths.get(current_location, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                     # Goal is unreachable for this package
                     return float('inf')
                total_cost += 1 + drive_cost + 1 # pick-up + drive + drop

            elif package in packages_in_vehicles:
                # 4bii. Package is inside a vehicle
                vehicle = packages_in_vehicles[package]
                # Need vehicle's current location
                if vehicle not in current_vehicle_locations:
                    # Inconsistent state or vehicle not identified correctly
                    # This case should ideally not happen in valid states explored by the planner
                    # if vehicles are correctly identified and always have an 'at' location.
                    # Returning inf indicates this path is likely invalid or leads to an unreachable state.
                    return float('inf')

                vehicle_location = current_vehicle_locations[vehicle]
                # Cost: drive (distance) + drop (1)
                drive_cost = self.shortest_paths.get(vehicle_location, {}).get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                     # Goal is unreachable for this package
                     return float('inf')
                total_cost += drive_cost + 1 # drive + drop
            else:
                 # 4biii. Package is not at a location and not in a vehicle.
                 # This package is lost or state is invalid.
                 # Returning inf indicates this path is likely invalid or leads to an unreachable state.
                 return float('inf')

        return total_cost
