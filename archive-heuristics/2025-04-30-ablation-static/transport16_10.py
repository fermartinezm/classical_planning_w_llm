from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity constraints and assuming vehicles
    are available when needed. The cost for a package depends on whether it's
    on the ground or in a vehicle, and the shortest path distance between its
    current effective location and its goal location.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped, unless it's
      already in a vehicle at the goal location (needs only drop) or already
      at the goal location on the ground (needs 0 actions).
    - Vehicle capacity constraints are ignored.
    - Vehicle availability at pick-up locations is ignored.
    - The cost of a 'drive' action is 1.
    - The cost of 'pick-up' and 'drop' actions is 1.
    - The shortest path distance between locations represents the minimum number
      of 'drive' actions needed.
    - For simplicity and efficiency, reachability between any two locations
      is assumed if the problem is solvable. Unreachable goal locations from
      a package's current location will result in a large heuristic penalty.
    - Packages are identified by being present in the goal conditions. Vehicles
      are identified by being 'at' a location and not being a package in the goal.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on 'road' predicates, including all
      locations mentioned in road facts, initial state 'at' facts, and goal 'at' facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or vehicle status for every package and vehicle
       relevant to the goals based on the state facts ('at' and 'in').
    2. Initialize total heuristic cost to 0.
    3. For each package specified in the goal:
        a. Check if the package is currently on the ground at its goal location. If yes, continue to the next package (cost is 0 for this package).
        b. If not at the goal location on the ground, determine the package's current effective location:
           - If the package is on the ground at `current_loc_p`, its effective location is `current_loc_p`.
           - If the package is inside a vehicle `v`, find the vehicle's location `current_loc_v`. The package's effective location is `current_loc_v`. If the vehicle's location is unknown, treat the package's goal as unreachable from its current state.
        c. Calculate the shortest path distance from the package's effective location to its goal location using the precomputed distances. If the goal is unreachable from the effective location via the road network, add a large penalty to the total cost and move to the next package.
        d. Estimate the actions needed for this package based on its status and the calculated distance:
           - If the package is on the ground at `current_loc_p` (`current_loc_p != goal_location`): Needs pick-up (1) + drive(s) (distance) + drop (1). Add `1 + distance + 1` to total cost.
           - If the package is in a vehicle `v` at `current_loc_v`:
             - If `current_loc_v != goal_location`: Needs drive(s) (distance) + drop (1). Add `distance + 1` to total cost.
             - If `current_loc_v == goal_location`: Needs drop (1). Add `1` to total cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road graph (adjacency list).
        self.road_graph = {}
        all_locations_set = set() # Collect all locations mentioned in roads

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                loc1, loc2 = parts[1], parts[2]
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                self.road_graph[loc1].append(loc2)

        # Add locations mentioned in goals or initial state that might not be in road facts
        all_potential_locations = set(all_locations_set)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  # (at obj loc)
                  all_potential_locations.add(parts[2])
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                  # (at obj loc)
                  all_potential_locations.add(parts[2])

        # Ensure all potential locations are in the graph keys, even if they have no outgoing roads
        for loc in all_potential_locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []


        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        all_locations = list(self.road_graph.keys()) # Use keys to get all known locations

        for start_node in all_locations:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_node][current_loc] = dist

                # Check if current_loc is in road_graph before iterating (should be, due to loop above)
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            # After BFS from start_node, any location in all_locations
            # that is NOT in self.distances[start_node] is unreachable from start_node.
            # Lookup will raise KeyError, which we handle in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of packages and vehicles.
        package_status = {} # Maps package -> ('at', location) or ('in', vehicle)
        vehicle_location = {} # Maps vehicle -> location

        # Identify packages we care about (those in the goal)
        goal_packages = set(self.goal_locations.keys())

        # Populate package_status and vehicle_location from the current state
        # First pass to get 'at' locations for all locatables
        at_locations = {}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at":
                  obj_name, loc_name = parts[1], parts[2]
                  at_locations[obj_name] = loc_name

        # Second pass to determine package status ('at' or 'in') and vehicle locations
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "in":
                  package_name, vehicle_name = parts[1], parts[2]
                  if package_name in goal_packages:
                       package_status[package_name] = ('in', vehicle_name)
                  # vehicle_name is a vehicle, its location is in at_locations

        # Now populate package_status for packages that are 'at' a location
        # and vehicle_location for objects that are 'at' a location and are vehicles.
        for obj_name, loc_name in at_locations.items():
             if obj_name in goal_packages:
                  # If a package is 'in' a vehicle, the 'in' fact takes precedence for status.
                  # If it's not in package_status yet, it must be 'at' this location.
                  if obj_name not in package_status:
                       package_status[obj_name] = ('at', loc_name)
             # If the object is not a goal package, assume it's a vehicle.
             # This is a simplification.
             elif obj_name not in goal_packages:
                  vehicle_location[obj_name] = loc_name


        total_cost = 0
        UNREACHABLE_PENALTY = 1000 # Large penalty for a package goal being unreachable

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not in package_status, it's an error in state representation
            # or it was never in the initial state. Assuming valid states.
            if package not in package_status:
                 # This shouldn't happen in a valid problem instance where goal objects exist.
                 # Treat as unreachable for safety.
                 print(f"Error: Package {package} from goal not found in state.")
                 total_cost += UNREACHABLE_PENALTY
                 continue

            status_type, current_where = package_status[package]

            # Check if the package is already at its goal location on the ground
            if status_type == 'at' and current_where == goal_location:
                continue # Package is already at the goal, cost is 0 for this package

            # Package is not at the goal location on the ground. Calculate cost.
            effective_location = None
            actions_needed_before_drive = 0 # Pick-up cost
            actions_needed_after_drive = 0 # Drop cost

            if status_type == 'at':
                # Package is on the ground, not at goal. Needs pick-up, drive, drop.
                effective_location = current_where
                actions_needed_before_drive = 1 # Pick-up
                actions_needed_after_drive = 1 # Drop
            elif status_type == 'in':
                # Package is in a vehicle. Needs drive, drop (unless already at goal loc).
                vehicle_name = current_where
                # Find the vehicle's location
                if vehicle_name in vehicle_location:
                    effective_location = vehicle_location[vehicle_name]
                    actions_needed_after_drive = 1 # Drop is always needed if package is in a vehicle and not at goal on ground
                else:
                    # Vehicle location not found - indicates an invalid state representation
                    # or an issue with parsing vehicle locations.
                    print(f"Error: Vehicle {vehicle_name} location not found for package {package} in state.")
                    total_cost += UNREACHABLE_PENALTY # Treat as unreachable
                    continue # Move to next package

            # If effective_location is None (due to errors above), skip distance calculation
            if effective_location is None:
                 continue

            # Calculate drive cost based on shortest path distance
            drive_cost = 0
            # Only need to drive if the effective location is not the goal location
            if effective_location != goal_location:
                 try:
                     drive_cost = self.distances[effective_location][goal_location]
                 except KeyError:
                     # This implies goal is unreachable from current effective location.
                     # Or effective_location/goal_location wasn't in the graph (e.g., isolated).
                     # Given how we build the graph, this should only happen if they are truly disconnected.
                     print(f"Warning: Goal location {goal_location} unreachable from {effective_location} for package {package}.")
                     total_cost += UNREACHABLE_PENALTY # Large penalty for apparent unreachability
                     continue # Move to next package

            # Sum up costs for this package
            # If package was 'at' not at goal: 1 (pick) + dist + 1 (drop)
            # If package was 'in' vehicle at goal: 0 (pick) + 0 (dist) + 1 (drop) = 1
            # If package was 'in' vehicle not at goal: 0 (pick) + dist + 1 (drop) = dist + 1
            cost_for_package = actions_needed_before_drive + drive_cost + actions_needed_after_drive

            total_cost += cost_for_package

        # The heuristic is 0 iff all packages are at their goal location on the ground.
        # This matches the PDDL goal definition (at pX lY).
        # If total_cost is 0, it means the loop completed without adding any cost,
        # which only happens if the 'continue' condition (at package goal_location)
        # was met for all packages in goal_locations.
        # If goal_locations is empty, the loop doesn't run, total_cost is 0.
        # An empty goal (true) should have h=0.
        # If a package is in a vehicle at the goal location, cost is 1, h > 0. Correct.

        return total_cost
