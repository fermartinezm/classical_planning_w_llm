# Add necessary imports
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function (outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# The heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of necessary actions to move each package
    to its goal location. It sums the minimum actions required for each package
    independently, ignoring vehicle capacity and coordination.

    # Assumptions
    - Any vehicle can carry any package.
    - Vehicle capacity is not a limiting factor for the heuristic calculation.
    - Vehicle movement cost is the shortest path distance between locations.
    - The heuristic ignores the initial location of vehicles when calculating
      the cost for a package on the ground; it assumes a vehicle is available
      for loading at the package's current location.
    - The state representation passed to __call__ contains dynamic facts ('at', 'in').
    - Static facts ('road', 'capacity', 'capacity-predecessor') are available in task.static.
    - Packages are objects that appear in goal 'at' facts and are not vehicles.
    - Vehicles are objects that appear in static 'capacity' facts.

    # Heuristic Initialization
    - Build a graph of locations based on `road` facts from `task.static`.
    - Compute all-pairs shortest paths between locations using BFS on the graph.
    - Identify vehicles based on `capacity` facts from `task.static`.
    - Extract the goal location for each package from `task.goals`.
    - Identify packages as objects in the goals that are not vehicles.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is part of the goal:
    1. Check if the state is the goal state. If yes, return 0. (This check is done once at the start).
    2. Determine the package's current state by examining the 'at' and 'in' facts in the current state:
       Is it on the ground at a location `l_current`, or is it inside a vehicle `v`?
    3. If the package is on the ground at location `l_current`:
       - If `l_current` is the goal location `l_goal`, the cost for this package is 0.
       - If `l_current` is not `l_goal`, the package needs to be loaded (1 action),
         transported by a vehicle from `l_current` to `l_goal` (shortest_path(l_current, l_goal) actions),
         and then unloaded (1 action). The total cost for this package is
         2 + shortest_path(l_current, l_goal). If no path exists, the state is likely unsolvable, return infinity.
    4. If the package is inside a vehicle `v`:
       - Find the current location `l_v` of vehicle `v` by examining the 'at' facts for vehicles in the current state.
       - If `l_v` is the goal location `l_goal`, the package only needs to be unloaded (1 action).
       - If `l_v` is not `l_goal`, the package needs to be transported by vehicle `v`
         from `l_v` to `l_goal` (shortest_path(l_v, l_goal) actions), and then unloaded (1 action).
         The total cost for this package is shortest_path(l_v, l_goal) + 1. If no path exists, return infinity.
    5. If a goal package is not found in any 'at' or 'in' fact in the state, the state is invalid or unsolvable, return infinity.
    6. The total heuristic value is the sum of the costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the graph of locations based on road facts.
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                # Assuming roads are bidirectional based on example 1
                self.location_graph.setdefault(loc2, set()).add(loc1)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS from each location.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self._compute_shortest_paths_from(start_loc)

        # Identify vehicles from static facts (capacity predicate)
        self.vehicles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "capacity":
                self.vehicles.add(parts[1]) # The object with capacity is a vehicle

        # Store goal locations for each package.
        self.package_goals = {}
        # We need to identify packages. Packages are objects that appear in 'at' or 'in'
        # facts in the goal and are *not* vehicles.
        all_objects_in_goals = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location
                all_objects_in_goals.add(package)
            # Note: 'in' facts are not typically goal conditions in transport,
            # but if they were, we'd need to handle them. Assuming only 'at' goals for packages.

        # Infer packages: objects in goals that are not vehicles
        self.packages = all_objects_in_goals - self.vehicles

        # Note: We ignore capacity-predecessor and capacity facts for heuristic value calculation.

    def _compute_shortest_paths_from(self, start_loc):
        """
        Computes shortest path distances from a start location to all other
        locations using BFS. Stores results in self.shortest_paths.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Store the distance from start_loc to current_loc
            self.shortest_paths[(start_loc, current_loc)] = current_dist

            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

        # After BFS, any location still with distance float('inf') is unreachable.
        # These unreachable pairs will simply not be in the self.shortest_paths dictionary,
        # or their value will remain inf if we initialized all pairs.
        # Our lookup `self.shortest_paths.get((loc1, loc2))` will return None if no path was found,
        # which we handle by returning float('inf').

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (dynamic facts).

        # If the state is the goal state, the heuristic is 0.
        # This check is often done by the search algorithm, but including it here
        # ensures h(goal) = 0 regardless of the search implementation details.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        # Map object -> location or vehicle
        current_locations_or_vehicles = {}
        # Map vehicle -> location
        vehicle_locations = {}

        # Iterate through dynamic facts in the current state
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                current_locations_or_vehicles[obj] = loc
                if obj in self.vehicles:
                     vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                # Ensure the second arg is a known vehicle based on static facts
                if vehicle in self.vehicles:
                    current_locations_or_vehicles[package] = vehicle # Package is inside a vehicle
                # else: invalid state? package in non-vehicle? Assume valid states.


        total_cost = 0

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.package_goals.items():
            # We only care about packages that are part of the goal
            if package not in self.packages:
                 continue # Should not happen based on how self.packages is built

            current_loc_or_veh = current_locations_or_vehicles.get(package)

            # If a goal package is not found in any 'at' or 'in' fact in the state,
            # it's an invalid state for this heuristic or unsolvable. Return infinity.
            if current_loc_or_veh is None:
                 return float('inf')

            # Case 1: Package is on the ground at a location
            # Check if the value is a known location string (from self.locations)
            if current_loc_or_veh in self.locations:
                current_location = current_loc_or_veh
                # If package is already at the goal location on the ground, cost is 0 for this package.
                if current_location != goal_location:
                    # Needs load (1) + drive (shortest_path) + unload (1)
                    drive_cost = self.shortest_paths.get((current_location, goal_location))
                    if drive_cost is None: # No path found between current and goal location
                         return float('inf') # State is likely unsolvable
                    total_cost += 1 + drive_cost + 1 # Load + Drive + Unload

            # Case 2: Package is inside a vehicle
            # Check if the value is a known vehicle name (from self.vehicles)
            elif current_loc_or_veh in self.vehicles:
                vehicle = current_loc_or_veh
                # Find the location of the vehicle
                vehicle_location = vehicle_locations.get(vehicle)

                # If vehicle location is unknown, state is invalid.
                if vehicle_location is None:
                    return float('inf') # Should not happen in valid states

                # If vehicle is at the goal location, package only needs unload (1).
                if vehicle_location != goal_location:
                    # Needs drive (shortest_path) + unload (1)
                    drive_cost = self.shortest_paths.get((vehicle_location, goal_location))
                    if drive_cost is None: # No path found between vehicle's location and goal
                         return float('inf') # State is likely unsolvable
                    total_cost += drive_cost + 1 # Drive + Unload
                else: # vehicle_location == goal_location
                    # Needs only unload (1)
                    total_cost += 1 # Unload

            # If current_loc_or_veh is neither a known location nor a known vehicle,
            # the state representation is unexpected. Return infinity.
            else:
                 return float('inf')


        return total_cost
