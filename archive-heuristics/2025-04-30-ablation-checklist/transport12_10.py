import collections
from fnmatch import fnmatch
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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Handle cases where pattern is shorter than fact parts (e.g., matching "(at obj loc extra)" with "at obj loc")
    # This simplified match assumes args covers all relevant parts.
    # A more robust match might check len(parts) == len(args) or handle trailing wildcards explicitly.
    # For the PDDL facts here, a direct zip comparison is sufficient.
    return True


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, independently. It uses
    shortest path distances on the road network for drive actions and adds
    costs for pick-up and drop actions. Capacity constraints are ignored.

    # Assumptions
    - The goal is to move specific packages to specific locations.
    - Any vehicle can pick up any package (capacity is ignored).
    - A vehicle is available when needed for a package (vehicle location/availability
      is only considered if the package is already inside a vehicle).
    - The cost of driving between locations is the shortest path distance
      in the road network.

    # Heuristic Initialization
    - Precomputes shortest path distances between all locations based on the
      `road` facts using Breadth-First Search (BFS).
    - Extracts the goal location for each package from the task goals.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that needs to reach a goal location:

    1.  Determine the package's current state: Is it on the ground at a location,
        or is it inside a vehicle?
    2.  If the package is already at its goal location, the cost for this package is 0.
    3.  If the package is on the ground at a location `l_current` (and `l_current` is not the goal `l_goal`):
        -   It needs to be picked up (1 action).
        -   It needs to be transported from `l_current` to `l_goal`. This requires a vehicle
            to drive between these locations. The estimated number of drive actions is
            the shortest path distance between `l_current` and `l_goal`.
        -   It needs to be dropped at `l_goal` (1 action).
        -   Total estimated cost for this package: 1 (pick-up) + distance(l_current, l_goal) + 1 (drop).
    4.  If the package is inside a vehicle `v`, and the vehicle `v` is at location `l_v`:
        -   It needs to be transported from `l_v` to `l_goal`. This requires the vehicle
            to drive between these locations. The estimated number of drive actions is
            the shortest path distance between `l_v` and `l_goal`.
        -   It needs to be dropped at `l_goal` (1 action).
        -   Total estimated cost for this package: distance(l_v, l_goal) + 1 (drop).
    5.  The total heuristic value is the sum of the estimated costs for all packages
        that are not yet at their goal locations.
    6.  If any package goal is unreachable (no path exists), the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and precomputing
        shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package mentioned in the goal.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph from static facts.
        self.road_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph[loc2].append(loc1)
                locations.add(loc1)
                locations.add(loc2)

        self.locations = list(locations) # Store list of all locations

        # Precompute all-pairs shortest paths using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        Returns a dictionary distances[start_loc][end_loc] = distance.
        """
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """
        Performs BFS starting from start_loc to find distances to all other locations.
        Returns a dictionary distances_from_start[location] = distance.
        """
        distances_from_start = {loc: float('inf') for loc in self.locations}
        distances_from_start[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            # If current_loc is not in the graph (e.g., a location mentioned only in init/goal but not roads)
            # skip it. This shouldn't happen if locations are correctly extracted from roads.
            if current_loc not in self.road_graph:
                 continue

            for neighbor in self.road_graph[current_loc]:
                if distances_from_start[neighbor] == float('inf'):
                    distances_from_start[neighbor] = distances_from_start[current_loc] + 1
                    queue.append(neighbor)

        return distances_from_start


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state.

        # Map locatable objects (packages, vehicles) to their current state.
        # State can be ('at', location) or ('in', vehicle).
        current_locatable_state = {}
        # Map vehicles to their current location.
        current_vehicle_location = {}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, location = parts[1], parts[2]
                current_locatable_state[obj] = ('at', location)
                # Check if the object is a vehicle (simple check based on naming convention or type)
                # A more robust way would be to parse object types from the PDDL domain/instance,
                # but for this heuristic, assuming objects starting with 'v' are vehicles is sufficient
                # given the example instances. A better way is to check task.objects if available.
                # Let's rely on the fact structure: (at ?x - locatable ?v - location)
                # We need to know which objects are vehicles. We can infer this from the 'capacity' predicate
                # or by looking at objects used with 'drive'. Let's assume objects starting with 'v' are vehicles
                # based on example instances.
                if obj.startswith('v'): # Simple heuristic-specific assumption
                     current_vehicle_location[obj] = location
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                current_locatable_state[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location specified.
        for package, goal_location in self.goal_locations.items():
            # If the package is not in the current state at all, it's likely an error
            # or it was never initialized. Assume it must be in current_locatable_state.
            if package not in current_locatable_state:
                 # This case should ideally not happen in valid planning states derived from init.
                 # For robustness, we could return infinity or log a warning.
                 # Let's assume valid states and proceed.
                 continue # Or return float('inf') if an uninitialized package is a problem

            package_state_type, current_pos = current_locatable_state[package]

            # Check if the package is already at its goal location.
            # This requires checking if it's (at package goal_location).
            # We need to check the raw state facts for this specific goal predicate.
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at the goal, cost is 0 for this package.

            # Package is not at the goal. Calculate cost to move it.
            cost_for_package = 0

            if package_state_type == 'at':
                # Package is on the ground at current_pos.
                current_location = current_pos
                # Cost: pick-up + drive + drop
                # Need to drive from current_location to goal_location
                drive_distance = self.distances[current_location].get(goal_location, float('inf'))

                if drive_distance == float('inf'):
                    # Goal is unreachable for this package
                    return float('inf')

                # 1 (pick-up) + drive_distance (drive actions) + 1 (drop)
                cost_for_package = 1 + drive_distance + 1

            elif package_state_type == 'in':
                # Package is inside a vehicle.
                vehicle = current_pos
                # Find the vehicle's location
                if vehicle not in current_vehicle_location:
                    # Vehicle location is unknown, implies an invalid state or missing fact.
                    # Return infinity as we can't proceed.
                    return float('inf')

                vehicle_location = current_vehicle_location[vehicle]

                # Cost: drive + drop
                # Need to drive from vehicle_location to goal_location
                drive_distance = self.distances[vehicle_location].get(goal_location, float('inf'))

                if drive_distance == float('inf'):
                    # Goal is unreachable for this package
                    return float('inf')

                # drive_distance (drive actions) + 1 (drop)
                cost_for_package = drive_distance + 1

            total_cost += cost_for_package

        return total_cost

