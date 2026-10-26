from fnmatch import fnmatch
from collections import deque
import sys
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    costs for each package that is not yet at its goal location. The cost for
    each package is estimated based on its current state (at a location or in a vehicle)
    and the shortest path distance in the road network.

    # Assumptions
    - The heuristic assumes that packages can be transported independently.
    - It assumes that a capable vehicle (one not at minimum capacity) is always
      available to pick up a package if needed, and it can reach the package's
      location via the shortest path.
    - It ignores vehicle capacity constraints beyond checking if a vehicle can
      pick up *any* package (i.e., its capacity is not the minimum).
    - It ignores potential conflicts when multiple packages require the same
      vehicle or road segments.
    - The cost of each action (drive, pick-up, drop) is assumed to be 1.

    # Heuristic Initialization
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest paths between locations using BFS from each node.
    - Identifies the minimum capacity size from `capacity-predecessor` facts.
    - Stores goal locations for each package.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that needs to be at goal location `goal_l`:

    1. If `p` is already at `goal_l`, the cost for this package is 0.

    2. If `p` is currently at location `current_l` (`current_l != goal_l`):
       - Find the set of vehicles that are capable of picking up a package
         (i.e., their current capacity is not the minimum capacity size).
       - Find the minimum shortest path distance from any capable vehicle's
         current location to `current_l`. If no capable vehicle exists or
         is reachable, this implies a very high cost.
       - The estimated cost for this package is:
         (shortest distance from a capable vehicle to `current_l`) + 1 (pick-up) +
         (shortest distance from `current_l` to `goal_l`) + 1 (drop).

    3. If `p` is currently inside a vehicle `v`:
       - Find the current location `v_loc` of vehicle `v`.
       - The estimated cost for this package is:
         (shortest distance from `v_loc` to `goal_l`) + 1 (drop).

    4. The total heuristic value is the sum of the estimated costs for all
       packages not yet at their goal location. If any required distance
       calculation results in infinity (locations are disconnected), the
       total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Build the road network graph
        self.graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)
        self.locations = list(locations) # Store locations for BFS

        # 2. Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.locations:
            self._bfs(start_node)

        # 3. Find the minimum capacity size
        predecessor_map = {} # s2 -> s1
        all_sizes = set()
        for fact in self.static:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                predecessor_map[s2] = s1
                all_sizes.add(s1)
                all_sizes.add(s2)

        # The minimum size is one that is in all_sizes but not a value in predecessor_map (i.e., not an s2)
        s2_values = set(predecessor_map.keys())
        min_sizes = all_sizes - s2_values
        # There should be exactly one minimum size for a valid capacity chain in a well-formed domain
        # Handle potential errors gracefully, though assertion is fine for development/testing
        if len(min_sizes) != 1:
             # This indicates an issue with the domain definition or static facts
             # For robustness, we could default or raise an error. Let's raise for clarity.
             raise ValueError(f"Could not determine unique minimum capacity size from capacity-predecessor facts. Found potential minimums: {min_sizes}")

        self.min_capacity_size = min_sizes.pop()


        # 4. Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                 _, package, location = get_parts(goal)
                 self.goal_locations[package] = location
            # Ignore other goal types if any exist and are not relevant to package location

    def _bfs(self, start_node):
        """Perform BFS from start_node to find distances to all other nodes."""
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            # Check if current_node exists in the graph keys before accessing neighbors
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # After BFS from start_node, any location in self.locations not in visited is unreachable
        # from start_node. Their distance will not be in self.distances, get_distance handles this.


    def get_distance(self, loc1, loc2):
        """Get the shortest distance between two locations."""
        # If locations are the same, distance is 0
        if loc1 == loc2:
            return 0
        # Look up pre-calculated distance. If not found, locations are disconnected.
        # Return sys.maxsize to represent infinity.
        return self.distances.get((loc1, loc2), sys.maxsize)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Track current locations of locatables (packages and vehicles)
        # A package is either 'at' a location or 'in' a vehicle.
        # A vehicle is always 'at' a location.
        current_locations = {} # obj -> location (if at a location) or vehicle (if in a vehicle)
        vehicle_locations = {} # vehicle -> location (subset of current_locations)

        # First pass to find all 'at' and 'in' facts
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                 _, package, vehicle = get_parts(fact)
                 current_locations[package] = vehicle # Store the vehicle name

        # Second pass to specifically identify vehicle locations using capacity info
        for fact in state:
             if match(fact, "capacity", "*", "*"):
                  _, vehicle, _ = get_parts(fact)
                  # Find where this vehicle is located from the first pass results
                  if vehicle in current_locations:
                       vehicle_locations[vehicle] = current_locations[vehicle]
                  # else: vehicle exists but is not 'at' a location? Invalid state.
                  # We assume valid states where vehicles are always at a location if they exist.


        # Track current capacities of vehicles
        current_capacities = {} # vehicle -> size
        for fact in state:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, size = get_parts(fact)
                current_capacities[vehicle] = size

        total_cost = 0

        # Identify capable vehicles (capacity is not the minimum size)
        capable_vehicles = {
            v for v, size in current_capacities.items()
            if size != self.min_capacity_size
        }

        # Iterate through each package goal
        for package, goal_location in self.goal_locations.items():
            # Find the package's current state
            # A package is either 'at' a location or 'in' a vehicle
            package_current_state = current_locations.get(package)

            if package_current_state is None:
                 # Package not found in state (neither at nor in). Should not happen in valid states.
                 # Treat as unreachable or add large penalty.
                 # print(f"Warning: Package {package} state not found.")
                 return sys.maxsize # Indicate potentially unsolvable or invalid state


            # Case 1: Package is already at the goal location
            if package_current_state == goal_location: # Note: if package_current_state is a vehicle, this check fails correctly
                cost_p = 0
            # Case 2: Package is in a vehicle
            elif package_current_state in vehicle_locations: # Check if the state is a known vehicle
                vehicle = package_current_state # This is the vehicle name
                vehicle_location = vehicle_locations[vehicle] # Get location from vehicle_locations dict

                # Cost = drive from vehicle_location to goal_location + drop
                drive_cost = self.get_distance(vehicle_location, goal_location)
                if drive_cost == sys.maxsize: return sys.maxsize # Goal location unreachable from vehicle location

                cost_p = drive_cost + 1 # 1 for drop action

            # Case 3: Package is at a location, but not the goal location
            else: # package_current_state must be a location string
                current_location = package_current_state # This is the location name

                # Cost = drive vehicle to package + pick-up + drive package to goal + drop
                # Find the closest capable vehicle to the package's current location
                min_v_dist_to_package = sys.maxsize
                found_reachable_capable_vehicle = False

                for vehicle in capable_vehicles:
                    if vehicle in vehicle_locations: # Ensure vehicle location is known
                        vehicle_location = vehicle_locations[vehicle]
                        dist = self.get_distance(vehicle_location, current_location)
                        if dist != sys.maxsize:
                             found_reachable_capable_vehicle = True
                             min_v_dist_to_package = min(min_v_dist_to_package, dist)

                if not found_reachable_capable_vehicle:
                     # No capable vehicle exists or is reachable from the package location.
                     # Problem might be unsolvable or require complex steps not modeled.
                     # print("Warning: No reachable capable vehicle found for package at its location.")
                     return sys.maxsize # Indicate unsolvable or very high cost

                drive_vehicle_to_package_cost = min_v_dist_to_package
                pick_cost = 1
                drive_package_to_goal_cost = self.get_distance(current_location, goal_location)
                if drive_package_to_goal_cost == sys.maxsize: return sys.maxsize # Goal location unreachable from package location

                drop_cost = 1

                cost_p = drive_vehicle_to_package_cost + pick_cost + drive_package_to_goal_cost + drop_cost

            total_cost += cost_p

        return total_cost
