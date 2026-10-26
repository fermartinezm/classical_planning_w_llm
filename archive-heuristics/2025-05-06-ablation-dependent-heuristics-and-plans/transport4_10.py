from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Simple check: number of parts must match number of args for patterns used here.
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package independently.
    The cost for a package is estimated based on its current location (on the ground
    or inside a vehicle) and its goal location, considering the need for pick-up,
    driving, and dropping actions. Shortest path distances on the road network
    are used to estimate drive costs.

    # Assumptions
    - Each package needs to reach a specific goal location on the ground.
    - Vehicles can move between connected locations.
    - Any vehicle can transport any package (ignores capacity constraints for simplicity).
    - A vehicle is available when needed to pick up a package.
    - The road network is static and bidirectional (based on example).
    - The heuristic assumes valid states where packages with goals are either 'at' a location or 'in' a vehicle, and vehicles are 'at' a location.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from static facts
      to enable shortest path calculations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it on the ground at some location `current_l`, or is it inside a vehicle `v`? This is done by examining the state facts `(at package location)` and `(in package vehicle)`.
    2. If the package is inside a vehicle `v`, find the current location `current_l` of that vehicle by examining the state fact `(at vehicle location)`.
    3. Get the package's goal location `goal_l` from the pre-computed goal information.
    4. Calculate the estimated cost for this package:
       - If the package is on the ground at `current_l` (`current_l != goal_l`):
         - It needs to be picked up (1 action).
         - The vehicle needs to drive from `current_l` to `goal_l`. The estimated cost is the shortest path distance `dist(current_l, goal_l)` in the road network, computed using BFS.
         - It needs to be dropped at `goal_l` (1 action).
         - Total estimated cost for this package: 1 (pick-up) + `dist(current_l, goal_l)` (drive) + 1 (drop).
       - If the package is inside a vehicle and the vehicle is at `current_l`:
         - If `current_l == goal_l`:
           - It needs to be dropped at `goal_l` (1 action).
           - Total estimated cost for this package: 1 (drop).
         - If `current_l != goal_l`:
           - The vehicle needs to drive from `current_l` to `goal_l`. The estimated cost is `dist(current_l, goal_l)`.
           - It needs to be dropped at `goal_l` (1 action).
           - Total estimated cost for this package: `dist(current_l, goal_l)` (drive) + 1 (drop).
    5. Sum the estimated costs for all packages that are not yet at their goal location.
    6. If a package is already at its goal location on the ground, its cost is 0 and it contributes nothing to the sum.
    7. If any necessary location is unreachable via the road network, the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph.
        """
        # Assuming task object has 'goals' (frozenset of goal facts)
        # and 'static' (frozenset of static facts like road, capacity-predecessor)
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        # self.goal_locations = {package_name: goal_location_name}
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # Add other goal types if necessary, but 'at' is standard for package delivery

        # Build the road network graph (adjacency list).
        # self.road_graph = {location_name: [neighbor1, neighbor2, ...]}
        self.road_graph = collections.defaultdict(list)
        for fact in static_facts:
            # Road facts are typically (road loc1 loc2)
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional based on example
                self.road_graph[loc2].append(loc1)

        # Capacity-predecessor facts are static but not used in this heuristic
        # for simplicity and speed.

    def get_distance(self, start_loc, end_loc):
        """
        Computes the shortest path distance (number of drive actions) between
        two locations in the road network graph using BFS.
        Returns float('inf') if no path exists.
        """
        if start_loc == end_loc:
            return 0

        # Handle cases where start or end location might not be in the graph
        # (e.g., locations only mentioned in goals but not roads)
        if start_loc not in self.road_graph or end_loc not in self.road_graph:
             # If either location is not part of the road network, distance is infinite
             # unless they are the same location (handled above).
             return float('inf')


        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc has neighbors in the graph
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # frozenset of fact strings

        # Track where packages and vehicles are currently located.
        # package_status: {package_name: ('at', location) or ('in', vehicle_name)}
        # vehicle_locations: {vehicle_name: location}
        package_status = {}
        vehicle_locations = {}

        # Populate package_status and vehicle_locations from the current state
        # We assume objects in goal_locations are packages.
        packages_in_goal = set(self.goal_locations.keys())

        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in packages_in_goal:
                     # This is a package on the ground
                     # Only record if it's not already marked as 'in' a vehicle
                     # (a package cannot be both at a location and in a vehicle simultaneously)
                     if obj not in package_status or package_status[obj][0] != 'in':
                         package_status[obj] = ('at', loc)
                 else:
                     # Assume it's a vehicle (or other locatable)
                     # We only care about vehicles for finding package locations
                     vehicle_locations[obj] = loc
             elif parts[0] == 'in' and len(parts) == 3:
                 p, v = parts[1], parts[2]
                 if p in packages_in_goal:
                     # This is a package inside a vehicle
                     package_status[p] = ('in', v)
                 # We don't know the location of v from this fact, need the 'at' fact

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at the goal location on the ground
            if (f"(at {package} {goal_location})" in state):
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at the goal. Find its current status.
            current_status = package_status.get(package)

            if current_status is None:
                 # This package is in the goal list but not found in the state facts
                 # as either 'at' or 'in'. This indicates an invalid state or problem definition.
                 # Return infinity as this state is likely not on a path to a solvable state.
                 # print(f"Heuristic Error: Package {package} with goal {goal_location} not found in state!")
                 return float('inf')

            status_type, current_loc_or_vehicle = current_status

            if status_type == 'at':
                current_location = current_loc_or_vehicle
                # Package is on the ground at current_location, needs to go to goal_location
                # Cost: pick-up (1) + drive (dist) + drop (1)
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'):
                    # print(f"Heuristic Error: No path from {current_location} to {goal_location} for package {package}")
                    return float('inf') # Cannot reach goal location
                total_cost += 1 + dist + 1

            elif status_type == 'in':
                vehicle = current_loc_or_vehicle
                # Package is inside a vehicle. Find vehicle's location.
                vehicle_current_location = vehicle_locations.get(vehicle)
                if vehicle_current_location is None:
                     # Vehicle location not found? Should not happen in valid states.
                     # print(f"Heuristic Error: Vehicle {vehicle} carrying {package} not found at any location!")
                     return float('inf') # Indicate unsolvable or invalid state path

                # Cost: drive (dist) + drop (1)
                dist = self.get_distance(vehicle_current_location, goal_location)
                if dist == float('inf'):
                    # print(f"Heuristic Error: No path from vehicle location {vehicle_current_location} to {goal_location} for package {package}")
                    return float('inf') # Cannot reach goal location
                total_cost += dist + 1

        return total_cost
