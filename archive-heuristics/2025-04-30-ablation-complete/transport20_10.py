# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match a PDDL fact string against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args): # PDDL predicates have fixed arity
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class, potentially inheriting from a base class
# If running standalone, remove the inheritance.
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity
    and availability constraints. It sums the estimated costs for each package
    independently. The cost for a package includes pick-up, drop, and the
    shortest path distance (in drive actions) between the relevant locations.

    # Assumptions
    - Each package needs to reach a specific goal location specified by an `(at package location)` goal fact.
    - Vehicles can move between locations connected by roads.
    - The cost of pick-up, drop, and drive actions is 1.
    - Vehicle capacity is ignored for distance calculation, assuming a vehicle
      is always available to transport a package if needed.
    - The road network is static. Shortest path distances are precomputed.
    - All locations relevant to package movement (initial, goal, and vehicle locations)
      are part of the connected road network graph. Unreachable goals result in infinite heuristic cost.
    - The state representation is consistent: packages listed in goals are always
      either `(at location)` or `(in vehicle)` in the state facts, and vehicles
      carrying packages or present in `at` facts are locatable.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph of locations based on the 'road' predicates in static facts.
    - Compute all-pairs shortest paths on this location graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Collect all `(at obj loc)` and `(in package vehicle)` facts from the state.
    2. Identify objects that are packages (those in the goal list) and objects that are vehicles (those carrying packages or present in `at` facts but not packages).
    3. Record the location for each identified vehicle using the collected `(at ...)` facts.
    4. Initialize the total heuristic cost to 0.
    5. For each package that has a goal location (extracted during initialization):
       a. Determine the package's current effective location. This is its location
          if it's on the ground, or the vehicle's location if it's inside a vehicle.
          If the package is not found in either `in` or `at` facts, the goal is unreachable from this state; return infinity.
       b. Let the package's current effective location be `current_loc`. Let its goal location be `goal_loc`.
       c. If `current_loc` is the same as `goal_loc`:
          - If the package is on the ground (`status` was 'at'), it has reached its goal. Add 0 cost for this package.
          - If the package is inside a vehicle (`status` was 'in'), it still needs to be dropped. Add 1 (drop) to the total cost.
       d. If `current_loc` is different from `goal_loc`:
          - If the package is on the ground (`status` was 'at'): It needs pick-up, drive, drop. Add 1 (pick-up) + shortest_path(`current_loc`, `goal_loc`) (drive) + 1 (drop) to the total cost.
          - If the package is inside a vehicle (`status` was 'in'): It needs drive, drop. Add shortest_path(`current_loc`, `goal_loc`) (drive) + 1 (drop) to the total cost.
       e. If any required shortest path `shortest_path(current_loc, goal_loc)` is infinite (goal is unreachable), the total heuristic is infinite.
    6. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        shortest paths between locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3: # Ensure it's an (at ?obj ?loc) fact
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
            # Ignore other types of goal facts if any exist

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3: # Ensure it's a (road ?loc1 ?loc2) fact
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                self.road_graph[loc1].append(loc2)

        # Ensure all locations mentioned in roads are in the graph dictionary,
        # even if they have no outgoing roads defined explicitly.
        for loc in locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # Compute all-pairs shortest paths using BFS.
        self.all_pairs_distances = {}
        all_locations = list(locations) # Get a list of all unique locations

        for start_loc in all_locations:
            distances_from_start = self._bfs(start_loc)
            for end_loc, dist in distances_from_start.items():
                self.all_pairs_distances[(start_loc, end_loc)] = dist

        # Note: If the graph is disconnected, self.all_pairs_distances will not
        # contain entries for unreachable pairs. The .get() method with a default
        # of float('inf') handles this correctly during heuristic calculation.


    def _bfs(self, start_location):
        """
        Perform Breadth-First Search from a start location to find distances
        to all reachable locations.
        """
        distances = {start_location: 0}
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc has outgoing roads defined
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state  # Current world state as a frozenset of fact strings

        # Collect all 'at' and 'in' facts
        at_facts = {} # obj -> location
        in_facts = {} # package -> vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                at_facts[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                in_facts[package] = vehicle

        # Determine package status and vehicle locations
        package_status = {} # package -> {'status': 'at', 'location': loc} or {'status': 'in', 'vehicle': veh}
        vehicle_locations = {} # vehicle -> location

        # Identify packages that are in goals
        packages_in_goals = set(self.package_goals.keys())

        # Identify potential vehicles: anything in an 'at' fact that is not a package in goals
        # or anything that is the second argument of an 'in' fact.
        potential_vehicles = {obj for obj in at_facts.keys() if obj not in packages_in_goals}
        potential_vehicles.update(in_facts.values())

        for vehicle in potential_vehicles:
             if vehicle in at_facts:
                  vehicle_locations[vehicle] = at_facts[vehicle]

        # Determine status for packages in goals
        for package in packages_in_goals:
             if package in in_facts:
                  # Package is in a vehicle
                  vehicle = in_facts[package]
                  package_status[package] = {'status': 'in', 'vehicle': vehicle}
             elif package in at_facts:
                  # Package is on the ground
                  package_status[package] = {'status': 'at', 'location': at_facts[package]}
             # else: package is not located in the state facts. Treat as unreachable.


        total_cost = 0

        # Calculate cost for each package towards its goal
        for package, goal_location in self.package_goals.items():
            if package not in package_status:
                 # Package location/status unknown in state facts.
                 # This implies an inconsistent state or the package is not in the problem instance init.
                 # Assuming valid states where packages in goals are always located.
                 # If this happens, the goal is likely unreachable from this state.
                 return float('inf')

            status_info = package_status[package]

            if status_info['status'] == 'at':
                # Package is on the ground
                package_location = status_info['location']

                if package_location != goal_location:
                    # Needs pick-up, drive, drop
                    pick_up_cost = 1
                    drop_cost = 1
                    drive_cost = self.all_pairs_distances.get((package_location, goal_location), float('inf'))

                    if drive_cost == float('inf'):
                         return float('inf') # Unreachable

                    total_cost += pick_up_cost + drive_cost + drop_cost
                else:
                    # Package is on the ground at the goal. Cost is 0 for this package.
                    pass # Cost is 0

            elif status_info['status'] == 'in':
                # Package is in a vehicle
                vehicle = status_info['vehicle']

                if vehicle not in vehicle_locations:
                    # Vehicle location unknown. Assume unreachable.
                    return float('inf') # Unreachable

                vehicle_location = vehicle_locations[vehicle]

                if vehicle_location != goal_location:
                    # Package is in vehicle, vehicle is not at goal. Needs drive, drop.
                    drop_cost = 1
                    drive_cost = self.all_pairs_distances.get((vehicle_location, goal_location), float('inf'))

                    if drive_cost == float('inf'):
                         return float('inf') # Unreachable

                    total_cost += drive_cost + drop_cost
                else:
                    # Package is in vehicle, vehicle is at goal. Needs drop.
                    drop_cost = 1
                    total_cost += drop_cost

        return total_cost
