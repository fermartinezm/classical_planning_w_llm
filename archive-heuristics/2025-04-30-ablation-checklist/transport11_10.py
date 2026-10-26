import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, considering its current location (ground or inside a vehicle)
    and the shortest path distance to its goal location.

    # Assumptions
    - The primary goal is to move packages to specific locations.
    - Vehicle capacity is not explicitly modeled in the heuristic cost,
      assuming sufficient capacity is eventually available.
    - The cost of getting a vehicle to a package's initial location (if on the ground)
      is not explicitly added, as this cost can be shared among multiple packages
      or might be covered by a vehicle already being there.
    - The heuristic counts 1 action for pick-up, 1 for drop-off, and the shortest
      path distance (number of drive actions) for transport.

    # Heuristic Initialization
    - Precomputes shortest path distances between all pairs of locations using BFS
      on the road network defined by `(road l1 l2)` facts.
    - Extracts the goal location for each package from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package specified in the goal:
    2. Check if the package is already at its goal location in the current state. If yes, cost for this package is 0.
    3. If the package is not at its goal location:
       a. Determine the package's current effective location. This is either its location if it's on the ground `(at p l)`, or the location of the vehicle it's inside `(in p v)` where the vehicle is at `(at v l)`.
       b. Calculate the shortest path distance (number of drive actions) from the package's current effective location to its goal location using the precomputed distances. If no path exists, the state is likely a dead end, return infinity.
       c. Estimate the actions needed for this package:
          - If the package is on the ground: 1 (pick-up) + distance (drive) + 1 (drop).
          - If the package is inside a vehicle: distance (drive) + 1 (drop).
    4. Sum the estimated costs for all packages that are not yet at their goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and precomputing
        shortest path distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Precompute shortest path distances between all locations
        self.distances = self._precompute_distances()

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at package location)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Note: Other goal types (like vehicle location) are ignored by this heuristic

    def _precompute_distances(self):
        """
        Build a graph from road facts and compute all-pairs shortest paths using BFS.
        Assumes roads are potentially directed.
        """
        graph = {}
        locations = set()

        # Build the graph from road facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                graph[l1].append(l2)

        # Ensure all locations are in the graph even if they have no outgoing roads
        for loc in locations:
             if loc not in graph:
                 graph[loc] = []

        all_pairs_distances = {}
        # Compute shortest distances from every location to every other location
        for start_node in locations:
            distances = {loc: float('inf') for loc in locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_dist = distances[current_node]

                # Check if current_node exists in graph (should always if built correctly)
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
            all_pairs_distances[start_node] = distances

        return all_pairs_distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the sum of minimum actions needed for each misplaced package.
        """
        state = node.state

        # Map objects (packages, vehicles) to their current locations if on the ground
        current_ground_locations = {}
        # Map packages to the vehicle they are inside
        package_in_vehicle = {}
        # Map vehicles to their current capacity size
        vehicle_capacity = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_ground_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
            elif parts[0] == "capacity" and len(parts) == 3:
                 vehicle, size = parts[1], parts[2]
                 vehicle_capacity[vehicle] = size # Not used in this simple heuristic, but could be

        total_cost = 0

        # Iterate through all packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location
            # We need to check both (at p l_goal) and ensure it's not inside a vehicle
            # A package at its goal location on the ground satisfies the goal (at p l_goal)
            is_at_goal_on_ground = match(f"(at {package} {goal_location})", "at", package, goal_location) in state
            is_in_vehicle = package in package_in_vehicle

            if is_at_goal_on_ground and not is_in_vehicle:
                 # Package is at goal location and on the ground, goal satisfied for this package
                 continue

            # Package is not at its goal location on the ground. It needs to be moved.

            current_package_location = None
            cost_for_package = 0

            if is_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # The package's effective location is the vehicle's location
                current_package_location = current_ground_locations.get(vehicle)

                if current_package_location is None:
                    # This indicates an inconsistent state (package in vehicle, but vehicle not at a location)
                    # Or perhaps the vehicle itself is in another vehicle (not possible in this domain)
                    # Treat as unreachable for safety in GBFS
                    # print(f"Warning: Vehicle {vehicle} carrying {package} has no location.")
                    return float('inf')

                # Cost: drive from vehicle's current location to goal + drop
                drive_cost = self.distances.get(current_package_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Cannot reach goal location from current vehicle location
                    return float('inf') # State is likely a dead end

                cost_for_package = drive_cost + 1 # drive + drop

            elif package in current_ground_locations:
                # Package is on the ground, not at the goal location
                current_package_location = current_ground_locations[package]

                # Cost: pick + drive from current location to goal + drop
                drive_cost = self.distances.get(current_package_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Cannot reach goal location from current package location
                    return float('inf') # State is likely a dead end

                cost_for_package = 1 + drive_cost + 1 # pick + drive + drop
            else:
                # Package exists in goals but is neither on the ground nor in a vehicle.
                # This indicates an inconsistent state.
                # print(f"Warning: Package {package} is not at any location and not in any vehicle.")
                return float('inf') # State is likely unreachable/invalid

            total_cost += cost_for_package

        return total_cost

# Example of how the Heuristic base class might look (assuming it's provided elsewhere)
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

