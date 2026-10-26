from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the planning environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
# This is typically not needed in the actual planning environment
# where heuristics inherit from a provided base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the minimum actions
    needed for each package independently, considering pick-up, drop, and driving costs.

    # Assumptions
    - The primary goal is to move packages to their target locations.
    - Capacity constraints are ignored for simplicity and efficiency.
    - A suitable vehicle is assumed to be available or reachable for each package's transport needs.
    - The road network is static and provides paths between locations.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build the road network graph from static 'road' facts.
    - Compute all-pairs shortest paths (distances) between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current physical location of every locatable object (packages and vehicles) that is on the ground using `(at ?x ?l)` facts.
    2. Identify which packages are inside which vehicles using `(in ?p ?v)` facts.
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `l_goal`:
       - Check if the package is already at its goal location `(at p l_goal)`. If yes, the cost for this package is 0.
       - If the package is not at its goal:
         - Check if the package is inside a vehicle `v` (`(in p v)` is true).
         - If the package is inside vehicle `v`:
           - The package's current physical location `l_current` is the location of vehicle `v`, found using the `(at v l_v)` fact from step 1.
           - If `l_current` is the goal location `l_goal`:
             - The minimum action needed is: drop (1).
             - Add `1` to the total cost.
           - If `l_current` is not the goal location `l_goal`:
             - The minimum actions needed are: drive from `l_current` to `l_goal` (distance) + drop (1).
             - Add `self.distances[l_current][l_goal] + 1` to the total cost. Handle unreachable locations by returning infinity.
         - If the package is not inside a vehicle (meaning it must be on the ground at some location `l_current`):
           - The package's current physical location `l_current` is found using the `(at p l)` fact from step 1.
           - If `l_current` is not the goal location `l_goal`:
             - The minimum actions needed are: pick-up (1) + drive from `l_current` to `l_goal` (distance) + drop (1).
             - Add `1 + self.distances[l_current][l_goal] + 1` to the total cost. Handle unreachable locations by returning infinity.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network, and computing distances.
        """
        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and compute distances.
        self.distances = self._compute_distances(task.static, self.goal_locations.values())

    def _compute_distances(self, static_facts, goal_locations):
        """
        Build the road network graph from static 'road' facts and compute
        all-pairs shortest paths using BFS.
        Includes all locations mentioned in road facts and goal facts.
        """
        graph = {}
        locations = set()

        # Build the graph and collect all locations from road facts
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                graph[l1].append(l2)

        # Add goal locations to the set of locations, in case they are isolated in road facts
        locations.update(goal_locations)

        # Ensure all locations are in the graph dictionary, even if they have no roads
        for loc in locations:
             if loc not in graph:
                 graph[loc] = []

        all_locations = list(locations)

        # Initialize distance matrix
        distances = {l1: {l2: float('inf') for l2 in all_locations} for l1 in all_locations}

        # Compute shortest paths using BFS from each location
        for start_node in all_locations:
            distances[start_node][start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()

                # Handle locations with no outgoing roads
                if u not in graph:
                    continue

                for v in graph[u]:
                    if distances[start_node][v] == float('inf'):
                        distances[start_node][v] = distances[start_node][u] + 1
                        queue.append(v)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is a goal state. If yes, heuristic is 0.
        if self.goals <= state:
             return 0

        # Track current physical location of all locatable objects (packages and vehicles) on the ground
        current_physical_location_on_ground = {}
        # Track which package is inside which vehicle
        package_in_vehicle = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                # (at ?x ?l) where ?x is locatable (vehicle or package)
                obj, location = args
                current_physical_location_on_ground[obj] = location
            elif predicate == "in":
                # (in ?p ?v) where ?p is package and ?v is vehicle
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            # Package is not at goal (because we didn't return 0 yet). Determine its current status.
            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Get the vehicle's location from the facts about objects on the ground
                current_location = current_physical_location_on_ground.get(vehicle)
                if current_location is None:
                     # Vehicle location is unknown - indicates an invalid state or problem definition issue
                     return float('inf') # State is likely unsolvable

                # Package is in vehicle at current_location, needs to reach goal_location
                if current_location == goal_location:
                    # Vehicle is at the goal location, just need to drop
                    total_cost += 1 # drop action
                else:
                    # Vehicle needs to drive and then drop
                    # Ensure current_location and goal_location are in distances map
                    if current_location in self.distances and goal_location in self.distances.get(current_location, {}):
                         drive_cost = self.distances[current_location][goal_location]
                         if drive_cost != float('inf'):
                             total_cost += drive_cost + 1 # drive + drop
                         else:
                             # Goal location is unreachable from current location
                             return float('inf') # State is likely unsolvable
                    else:
                         # Location not found in precomputed distances (shouldn't happen if _compute_distances is correct)
                         return float('inf')

            else:
                # Package is on the ground. Find its location.
                current_location = current_physical_location_on_ground.get(package)
                if current_location is None:
                     # Package location is unknown - indicates an invalid state or problem definition issue
                     return float('inf') # State is likely unsolvable

                # Package is at current_location, needs to reach goal_location
                # Minimum actions: pick-up + drive + drop
                # Ensure current_location and goal_location are in distances map
                if current_location in self.distances and goal_location in self.distances.get(current_location, {}):
                    drive_cost = self.distances[current_location][goal_location]
                    if drive_cost != float('inf'):
                        total_cost += 1 + drive_cost + 1 # pick-up + drive + drop
                    else:
                        # Goal location is unreachable
                        return float('inf') # State is likely unsolvable
                else:
                    # Location not found in precomputed distances
                    return float('inf')

        return total_cost
