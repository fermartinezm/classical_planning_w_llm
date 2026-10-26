# Assuming Heuristic base class is available in heuristics.heuristic_base
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe return empty list or raise error
        # For PDDL facts from a parser, this format is expected.
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function for shortest paths
def bfs(graph, start_loc):
    """
    Performs Breadth-First Search to find shortest distances from start_loc
    to all reachable locations in the graph.
    """
    distances = {start_loc: 0}
    queue = [start_loc]
    visited = {start_loc}
    while queue:
        current_loc = queue.pop(0)
        # Ensure current_loc exists in graph keys
        for neighbor in graph.get(current_loc, []):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[current_loc] + 1
                queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It considers the actions
    needed to pick up, transport, and drop a package.

    # Assumptions
    - The cost of moving a vehicle between two locations is the shortest path
      distance in the road network. Each drive action covers one step in the path.
    - Picking up a package costs 1 action (`pick-up`).
    - Dropping a package costs 1 action (`drop`).
    - Vehicle capacity constraints are ignored.
    - The availability of a suitable vehicle at the required location for pickup
      is assumed. The cost of vehicle movement is estimated based on the package's
      current location (if on the ground) or the vehicle's current location
      (if inside a vehicle) relative to the package's goal location.
      Specifically, for a package on the ground at L_current needing to go to L_goal,
      we estimate the drive cost as distance(L_current, L_goal). For a package
      in a vehicle at L_vehicle needing to go to L_goal, we estimate the drive
      cost as distance(L_vehicle, L_goal). This simplifies the vehicle assignment
      problem.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on `road` predicates from static facts.
      Roads are assumed to be bidirectional.
    - Computes all-pairs shortest path distances between locations using BFS
      starting from every known location.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of estimated costs for each package
    that has a goal location and is not yet at that goal.

    For each package `p` with a goal location `l_goal(p)`:
    1. Check if `p` is currently at `l_goal(p)`. If yes, the cost for this package is 0.
    2. If `p` is not at `l_goal(p)`, determine its current status:
       a. If `p` is on the ground at `l_current` (i.e., `(at p l_current)` is true, and `l_current != l_goal(p)`):
          - This package needs to be picked up, transported, and dropped.
          - Estimate: 1 action for `pick-up` + 1 action for `drop`.
          - Estimate the vehicle movement cost as the shortest path distance from `l_current` to `l_goal(p)`.
          - Total cost for this package: 1 (pick) + distance(`l_current`, `l_goal(p)`) + 1 (drop).
       b. If `p` is inside a vehicle `v` (i.e., `(in p v)` is true):
          - Find the current location `l_vehicle` of vehicle `v` (i.e., `(at v l_vehicle)` is true).
          - This package needs to be transported further (if `l_vehicle != l_goal(p)`) and dropped.
          - Estimate: 1 action for `drop`.
          - Estimate the vehicle movement cost as the shortest path distance from `l_vehicle` to `l_goal(p)`.
          - Total cost for this package: distance(`l_vehicle`, `l_goal(p)`) + 1 (drop).
    3. Sum the costs calculated for each package to get the total heuristic value for the state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Extract goal locations for packages from the goal facts.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goals are always (at package location)
                package, location = args
                self.goal_locations[package] = location

        # Build the road graph from static 'road' facts.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assume roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        # Ensure all locations mentioned in goals or initial state are included
        # even if they have no roads connected (though problem instances usually connect them).
        all_locations_in_problem = set(locations)
        for goal_loc in self.goal_locations.values():
             all_locations_in_problem.add(goal_loc)
        # Add locations from initial state facts (at package/vehicle location)
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  all_locations_in_problem.add(loc)


        for start_loc in all_locations_in_problem:
             # BFS needs the graph, but also needs to know about isolated locations
             # that might be start/goal points. BFS from an isolated node will just
             # return distance 0 to itself.
             if start_loc not in self.road_graph:
                 self.road_graph[start_loc] = set() # Add isolated location to graph structure
             self.distances[start_loc] = bfs(self.road_graph, start_loc)


    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Look up distance in the precomputed table.
        # If loc1 or loc2 were not in the initial set of locations used for BFS,
        # or if loc2 is unreachable from loc1, the entry might be missing.
        # Returning infinity is appropriate for unreachable locations.
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state  # Current world state (frozenset of facts).

        # Track current locations/status of locatable objects (packages and vehicles).
        current_locations = {} # Maps locatable object name to its location or vehicle name
        vehicle_locations = {} # Maps vehicle name to its location name
        package_status = {} # Maps package name to 'at' or 'in'

        # Iterate through the state facts to populate location/status dictionaries.
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Distinguish packages from vehicles. A simple way is checking if the object
                # is a key in the goal_locations dictionary (only packages have goals here).
                if obj in self.goal_locations:
                    package_status[obj] = 'at'
                # Assume anything else with an 'at' predicate is a vehicle for this domain.
                # A more robust way might involve parsing types from the PDDL domain.
                # Based on the example, vehicle names start with 'v'.
                elif obj.startswith('v'): # Heuristic: check prefix 'v' for vehicles
                     vehicle_locations[obj] = loc
                # else: it's some other locatable object we might ignore for this heuristic

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Store the vehicle name as the package's "location" for now
                current_locations[package] = vehicle
                package_status[package] = 'in'
                # Note: We don't know the vehicle's location yet from this fact,
                # we'll look it up in vehicle_locations later.

        total_cost = 0  # Initialize total heuristic cost.

        # Iterate through all packages that have a defined goal location.
        for package, goal_location in self.goal_locations.items():
            # If the package is not present in the current state facts (e.g., not 'at' or 'in'),
            # it's an unexpected state, possibly unreachable or invalid.
            if package not in current_locations:
                 # This should ideally not happen in reachable states from a valid initial state.
                 # Penalize heavily to guide search away from such states.
                 # print(f"Warning: Package {package} not found in state {state}.")
                 return float('inf') # Use infinity to strongly penalize invalid states

            current_status = package_status.get(package) # 'at' or 'in'
            current_pos_or_vehicle = current_locations[package]

            # Check if the package is already at its goal location.
            # This is true if the package is 'at' the goal location in the current state.
            is_at_goal = (current_status == 'at' and current_pos_or_vehicle == goal_location)

            if is_at_goal:
                continue # Package is already at goal, cost is 0 for this package.

            # Package is not at the goal. Calculate the estimated cost for this package.
            package_cost = 0

            if current_status == 'at': # Package is on the ground at current_pos_or_vehicle
                current_package_location = current_pos_or_vehicle
                # Estimated actions: pick-up (1) + drive (?) + drop (1)
                package_cost += 2 # Cost for pick-up and drop actions.
                # Estimate drive cost as the shortest distance from the package's current location to its goal.
                # This assumes a vehicle can reach the package and then drive it to the goal.
                drive_cost = self.get_distance(current_package_location, goal_location)
                if drive_cost == float('inf'):
                     # Goal location is unreachable from the package's current location.
                     # This state is likely part of an unsolvable path.
                     return float('inf')
                package_cost += drive_cost

            elif current_status == 'in': # Package is inside a vehicle (current_pos_or_vehicle is the vehicle name)
                vehicle_name = current_pos_or_vehicle
                # Find the current location of the vehicle.
                current_vehicle_location = vehicle_locations.get(vehicle_name)

                if current_vehicle_location is None:
                    # This indicates an inconsistent state: package is in a vehicle, but the vehicle
                    # itself is not located anywhere according to the state facts.
                    # Penalize heavily.
                    # print(f"Warning: Vehicle {vehicle_name} carrying {package} not found at any location in state.")
                    return float('inf')

                # Estimated actions: drive (?) + drop (1)
                package_cost += 1 # Cost for the drop action.
                # Estimate drive cost as the shortest distance from the vehicle's current location to the package's goal.
                drive_cost = self.get_distance(current_vehicle_location, goal_location)
                if drive_cost == float('inf'):
                     # Goal location is unreachable from the vehicle's current location.
                     return float('inf')
                package_cost += drive_cost

            else:
                 # Package status is neither 'at' nor 'in'. This indicates an inconsistent state.
                 # print(f"Warning: Package {package} has unknown status '{current_status}' in state.")
                 return float('inf')

            total_cost += package_cost

        return total_cost
