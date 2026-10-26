# -*- coding: utf-8 -*-

from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available and has __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts is at least the number of pattern arguments
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# class transportHeuristic(Heuristic): # Use this if Heuristic base class is provided
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move each
    package from its current location to its goal location, independently.
    It sums the estimated costs for each package that is not yet at its goal.
    The cost for a single package is estimated as:
    - If on the ground: 1 (pick-up) + shortest_path_distance (drive) + 1 (drop)
    - If in a vehicle: shortest_path_distance (drive) + 1 (drop)
    Shortest path distances between locations are precomputed using BFS on the
    road network.

    # Assumptions:
    - Any vehicle can pick up and transport any package, ignoring capacity constraints.
    - A vehicle is always available at the package's current location (if on the ground)
      or the vehicle currently carrying it is the one that will transport it.
    - The cost of driving between adjacent locations is 1.
    - The cost of pick-up and drop actions is 1.
    - Roads are bidirectional (inferred from the example domain file).
    - Objects starting with 'p' are packages, objects starting with 'v' are vehicles.

    # Heuristic Initialization
    - Parses the goal facts to identify the target location for each package.
    - Parses static facts to build the road network graph.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize total heuristic cost to 0.
    3. Determine the current status (location or vehicle) for all packages and the location for all vehicles by iterating through the state facts.
    4. Iterate through all packages that have a specified goal location.
    5. For each package:
       a. If the package is already at its goal location (checked by looking for the goal fact in the state), add 0 cost for this package and continue to the next.
       b. If the package is not at its goal:
          i. Determine the package's current "effective" location. If on the ground, it's its location. If in a vehicle, it's the vehicle's location. Handle cases where package/vehicle status is missing (return infinity).
          ii. Find the shortest path distance (minimum drive actions) between the effective current location and the goal location using the precomputed distances. If unreachable, return infinity.
          iii. If the package is currently on the ground, add 1 (for pick-up) to the cost for this package.
          iv. Add the shortest path distance (for driving) to the cost for this package.
          v. Add 1 (for drop) to the cost for this package.
       c. Add the calculated cost for this package to the total heuristic cost.
    6. Return the total heuristic cost.
    """
    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization: Extract goal locations and build road network/distances.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Build road network graph
        self.road_network = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_network.setdefault(l1, set()).add(l2)
                self.road_network.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        # Compute all-pairs shortest paths (APSP) using BFS
        self.shortest_paths = {}
        for start_loc in locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_location):
        """Compute shortest path distances from start_location to all reachable locations."""
        distances = {start_location: 0}
        queue = deque([start_location])
        visited = {start_location}

        while queue:
            current_loc = queue.popleft() # Dequeue using deque

            if current_loc in self.road_network:
                for neighbor in self.road_network[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor) # Enqueue
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if the state is a goal state first. If so, return 0.
        # This ensures h=0 only at goal.
        if self.goals <= state:
             return 0

        # Map locatable (package or vehicle) to its current status/location
        # Format: { obj_name: ('at', location) or ('in', vehicle) }
        current_status = {}
        # Map vehicle to its location
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_status[obj] = ("at", loc)
                # Assume anything with 'at' that is not a package (i.e., not in self.package_goals) is a vehicle
                # This is a simplification based on typical transport domains and the examples.
                if obj not in self.package_goals:
                     vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_status[package] = ("in", vehicle)

        total_cost = 0

        for package, goal_location in self.package_goals.items():
            # If the package is already at its goal location, cost is 0 for this package.
            # We check the overall goal state first, but this check is needed
            # to avoid calculating cost for packages already delivered.
            if f"(at {package} {goal_location})" in state:
                continue

            # Package is not at its goal. Calculate cost for this package.
            cost_for_package = 0

            if package not in current_status:
                 # This package is not mentioned in 'at' or 'in' facts.
                 # This indicates an invalid state or problem definition.
                 # Return infinity to prune.
                 return float('inf')

            status, loc_or_veh = current_status[package]

            if status == "at":
                # Package is on the ground at loc_or_veh
                current_loc = loc_or_veh
                is_in_vehicle = False
            elif status == "in":
                # Package is in vehicle loc_or_veh
                vehicle = loc_or_veh
                if vehicle not in vehicle_locations:
                     # Vehicle location unknown - invalid state?
                     return float('inf')
                current_loc = vehicle_locations[vehicle]
                is_in_vehicle = True
            else:
                 # Unknown status - invalid state?
                 return float('inf')

            # Check if goal location is reachable from current effective location
            if current_loc not in self.shortest_paths or goal_location not in self.shortest_paths[current_loc]:
                 # Goal location is unreachable. Invalid state or unsolvable path.
                 return float('inf')

            drive_distance = self.shortest_paths[current_loc][goal_location]

            if is_in_vehicle:
                # Package is in a vehicle.
                # Cost: Drive vehicle to goal + Drop package
                cost_for_package = drive_distance + 1 # 1 for drop
            else:
                # Package is on the ground.
                # Cost: Pick up package + Drive vehicle to goal + Drop package
                cost_for_package = 1 + drive_distance + 1 # 1 for pick-up, 1 for drop

            total_cost += cost_for_package

        # The total_cost is the sum of costs for packages not at their goal.
        # If the state is not a goal state (checked at the beginning), at least one package
        # is not at its goal, so total_cost will be > 0 (assuming reachable goals).
        return total_cost
