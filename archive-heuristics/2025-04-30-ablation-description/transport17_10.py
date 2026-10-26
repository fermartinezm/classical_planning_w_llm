from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of arguments in the pattern.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current location to its goal location,
    assuming vehicles are always available and have sufficient capacity. It sums
    the estimated costs for each package independently.

    # Assumptions
    - The cost of pick-up is 1.
    - The cost of drop is 1.
    - The cost of drive is 1 per road segment.
    - Vehicles are always available at the required locations when needed (this is a relaxation).
    - Vehicles always have sufficient capacity (this is a relaxation).
    - The shortest path between locations is used for driving cost.
    - The heuristic sums the costs for each package independently, ignoring potential
      synergies (e.g., one vehicle transporting multiple packages) or conflicts
      (e.g., multiple packages needing the same vehicle or path).
    - The goal only involves packages being at specific locations.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:

    1. Determine the package's current status:
       - Is it at a location `l_current`?
       - Is it inside a vehicle `v` which is at location `l_v`?

    2. Calculate the estimated cost for this package:
       - If the package is at `l_current` (and `l_current` is not the goal):
         - It needs to be picked up (1 action).
         - A vehicle needs to drive from `l_current` to the goal location `l_goal` (distance `dist(l_current, l_goal)` actions).
         - It needs to be dropped at `l_goal` (1 action).
         - Estimated cost for this package = 1 (pick-up) + `dist(l_current, l_goal)` (drives) + 1 (drop).
       - If the package is inside a vehicle `v` which is at `l_v` (and `l_v` is not the goal):
         - The vehicle needs to drive from `l_v` to the goal location `l_goal` (distance `dist(l_v, l_goal)` actions).
         - It needs to be dropped at `l_goal` (1 action).
         - Estimated cost for this package = `dist(l_v, l_goal)` (drives) + 1 (drop).
       - If the package is inside a vehicle `v` which is already at the goal location `l_goal`:
         - It needs to be dropped at `l_goal` (1 action).
         - Estimated cost for this package = 1 (drop).

    3. Sum the estimated costs for all packages that are not at their goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network and shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are of the form (at package location)
            if match(goal, "at", "*", "*"):
                _, obj, location = get_parts(goal)
                # Assuming objects starting with 'p' are packages based on examples
                if obj.startswith('p'):
                     self.goal_locations[obj] = location
                # Note: If goals include vehicle locations or other predicates,
                # this heuristic might not be 0 at the true goal state.

        # Build the road graph and compute shortest paths.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional based on example instances
                self.road_graph.setdefault(l2, []).append(l1)

        self.locations = list(locations) # Store list of all locations
        self.dist = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        Returns a dictionary dist[l1][l2] = shortest_distance.
        Unreachable locations have a large finite distance.
        """
        distances = {}
        # Use a large value to represent infinity for unreachable locations
        large_value = len(self.locations) * 3 # Consistent large value

        for start_node in self.locations:
            distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0 # Distance from node to itself is 0

            while q:
                current_loc, current_dist = q.popleft()

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = current_dist + 1
                        q.append((neighbor, current_dist + 1))

            # Fill in large_value for unreachable locations from start_node
            for loc in self.locations:
                 if loc not in distances[start_node]:
                     distances[start_node][loc] = large_value

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        package_location_or_vehicle = {} # {package: location or vehicle_name}
        vehicle_location = {} # {vehicle: location}

        # Extract current locations and containment relationships from the state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                # Assuming 'v' objects are vehicles and 'p' objects are packages
                if obj.startswith('v'):
                    vehicle_location[obj] = location
                elif obj.startswith('p'):
                    package_location_or_vehicle[obj] = location
            elif predicate == "in":
                package, vehicle = args
                # Assuming objects starting with 'p' are packages
                if package.startswith('p'):
                    package_location_or_vehicle[package] = vehicle # Store vehicle name

        total_cost = 0  # Initialize action cost counter.
        large_value = len(self.locations) * 3 # Consistent large value

        # Iterate through packages that have a goal location specified.
        for package, goal_location in self.goal_locations.items():
            current_status = package_location_or_vehicle.get(package)

            if current_status is None:
                 # Package status not found in state - unexpected. Add penalty.
                 total_cost += large_value
                 continue

            is_at_goal = False
            if current_status == goal_location: # Case: (at p goal_location)
                 is_at_goal = True
            elif current_status in vehicle_location: # Case: (in p v)
                 vehicle = current_status
                 if vehicle_location.get(vehicle) == goal_location: # Case: (in p v) and (at v goal_location)
                     is_at_goal = True

            if is_at_goal:
                continue # Package is already at its goal, cost is 0 for this package.

            # Package is not at its goal. Calculate cost.
            if current_status in vehicle_location:
                # Package is in a vehicle (current_status is vehicle name)
                vehicle = current_status
                vehicle_loc = vehicle_location.get(vehicle)

                if vehicle_loc is None:
                     # Vehicle location not found - unexpected. Add penalty.
                     total_cost += large_value
                     continue

                # Cost = distance from vehicle_loc to goal_location + 1 (drop)
                # Use the precomputed distance. It will be large if unreachable.
                # Check if vehicle_loc is a valid start node in our distance map
                if vehicle_loc in self.dist: # Check if BFS started from vehicle_loc
                    cost_drives = self.dist[vehicle_loc][goal_location]
                    total_cost += cost_drives + 1 # +1 for drop action
                else:
                    # Should not happen if all locations are included in BFS, but as a safeguard
                    total_cost += large_value

            else:
                # Package is at a location (current_status is location name)
                package_loc = current_status # This is l_current
                # Cost = 1 (pick-up) + distance from package_loc to goal_location + 1 (drop)
                # Use the precomputed distance. It will be large if unreachable.
                # Check if package_loc is a valid start node in our distance map
                if package_loc in self.dist: # Check if BFS started from package_loc
                    cost_drives = self.dist[package_loc][goal_location]
                    total_cost += 1 + cost_drives + 1 # +1 for pick-up, +1 for drop
                else:
                     # Should not happen if all locations are included in BFS, but as a safeguard
                    total_cost += large_value

        return total_cost
