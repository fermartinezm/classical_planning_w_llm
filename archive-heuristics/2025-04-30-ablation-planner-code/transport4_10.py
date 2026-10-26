# Need to import the base class if it's part of the framework
# from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # If the number of parts doesn't match the number of args and no wildcards are used, it's not a match.
    # This check is not strictly necessary with zip and fnmatch, but can be a quick fail.
    # if len(parts) != len(args) and '*' not in args:
    #      return False
    # Use zip to handle cases where parts and args might have different lengths due to wildcards
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is available in the environment
# class transportHeuristic(Heuristic):
class transportHeuristic: # Define without inheritance if base class is not provided
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated costs
    for each package that is not yet at its goal location. The cost for a package
    includes picking it up (if on the ground), driving it to the goal location,
    and dropping it. The driving cost is estimated using the shortest path distance
    in the road network. Vehicle capacity and the ability of a single vehicle to
    transport multiple packages are not explicitly modeled, making the heuristic
    non-admissible but potentially efficient and effective for greedy search.

    # Assumptions
    - The goal is defined by the target locations for specific packages.
    - The road network is static and bidirectional.
    - Shortest path distance in the road network represents the minimum number of drive actions between two locations.
    - Any vehicle can be used to transport any package, provided capacity allows (capacity is not strictly enforced in the cost calculation itself, but the pick/drop actions requiring capacity are counted).
    - All locations relevant to package movement are part of the road network graph or reachable within it.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph (adjacency list) from static facts.
    - Computes shortest path distances between all pairs of locations using Breadth-First Search (BFS).
    - Extracts capacity-predecessor relationships (though not directly used in the current cost calculation, it provides domain context).

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that has a specified goal location `l_goal`:

    1. Determine the package's current status and location:
       - Iterate through the current state facts.
       - If `(at p l_current)` is found, the package is on the ground at `l_current`.
       - If `(in p v)` is found, the package is inside vehicle `v`. Then, find the location of vehicle `v` by looking for `(at v l_v)` in the state. The package's effective location for calculating drive cost is `l_v`.

    2. Calculate the estimated cost for this package based on its status relative to `l_goal`:
       - If the package is on the ground at `l_goal`: The package is already at its destination. Estimated cost for this package is 0.
       - If the package is on the ground at `l_current` where `l_current != l_goal`:
         - It needs to be picked up (1 action).
         - It needs to be transported from `l_current` to `l_goal`. The minimum number of drive actions required is the shortest path distance `dist(l_current, l_goal)`.
         - It needs to be dropped at `l_goal` (1 action).
         - Estimated cost for this package = 1 (pick) + dist(l_current, l_goal) (drive) + 1 (drop).
       - If the package is inside vehicle `v` which is located at `l_v`:
         - If `l_v == l_goal`: The vehicle is already at the destination location. The package only needs to be dropped (1 action). Estimated cost = 1 (drop).
         - If `l_v != l_goal`: The vehicle needs to drive from `l_v` to `l_goal`, and then the package needs to be dropped. Estimated cost = dist(l_v, l_goal) (drive) + 1 (drop).

    3. Sum the estimated costs calculated for all packages that have a goal location specified.

    The total heuristic value is the sum of these individual package costs. This sum is 0
    if and only if all packages with specified goals are already on the ground at their
    respective goal locations. If the overall state is the goal state (which typically
    only requires packages to be at goal locations in this domain), the heuristic will be 0.
    If the state is not the goal state, the heuristic will be greater than 0, provided
    there is at least one package not yet at its goal location.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the road network."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and compute shortest paths.
        self.locations = set()
        self.road_graph = defaultdict(set)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, l1, l2 = parts
                    self.locations.add(l1)
                    self.locations.add(l2)
                    self.road_graph[l1].add(l2)
                    self.road_graph[l2].add(l1) # Roads are bidirectional

        # Compute all-pairs shortest paths using BFS from each location.
        self.shortest_paths = {}
        for start_loc in list(self.locations): # Use list to avoid modifying set during iteration
            self.shortest_paths[start_loc] = self._bfs(start_loc)

        # Store capacity-predecessor relationships (optional for this simple heuristic)
        self.capacity_predecessors = {}
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    _, s1, s2 = parts
                    self.capacity_predecessors[s1] = s2 # s1 is smaller than s2

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node might not be in locations if it's only mentioned in init/goal, not roads
             # This shouldn't happen in a well-formed problem, but handle defensively.
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is still valid and has neighbors in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        # Ensure both locations are known nodes in the graph
        if loc1 not in self.locations or loc2 not in self.locations:
             # This can happen if a location is mentioned in init/goal but not in any road fact.
             # Treat as unreachable for heuristic purposes.
             return 1000 # Large penalty

        # Look up precomputed distance
        # Use .get() with a default of infinity in case loc2 is not reachable from loc1
        distance = self.shortest_paths.get(loc1, {}).get(loc2, float('inf'))

        if distance == float('inf'):
             # Locations are in the graph but not connected
             return 1000 # Large penalty for disconnected locations
        return distance

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is the goal state.
        # In this domain, the goal is typically just package locations.
        # A state is a goal state if all goal facts are present.
        is_goal_state = all(goal in state for goal in self.goals)
        if is_goal_state:
            return 0

        # Track where packages and vehicles are currently located or contained.
        current_package_status = {} # Maps package -> ('at', location) or ('in', vehicle)
        current_vehicle_locations = {} # Maps vehicle -> location

        # Parse the current state to find locations of packages and vehicles
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects starting with 'p' are packages and 'v' are vehicles
                # This relies on naming convention from examples.
                if obj.startswith('p'):
                     current_package_status[obj] = ('at', loc)
                elif obj.startswith('v'):
                     current_vehicle_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                current_package_status[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location specified.
        for package, goal_location in self.goal_locations.items():
            # If the package is not even mentioned in the current state, something is wrong, skip.
            if package not in current_package_status:
                 # This shouldn't happen in valid state transitions from a valid initial state
                 # where all goal packages exist.
                 # print(f"Warning: Package {package} with goal {goal_location} not found in state.")
                 continue # Cannot estimate cost if package status is unknown

            status_type, current_loc_or_vehicle = current_package_status[package]

            # Case 1: Package is on the ground
            if status_type == 'at':
                current_location = current_loc_or_vehicle
                # If already at goal, cost is 0 for this package, continue to next package
                if current_location == goal_location:
                    continue

                # Package is on the ground at a location != goal
                # Needs: pick-up (1) + drive (dist) + drop (1)
                drive_cost = self.get_distance(current_location, goal_location)
                total_cost += 1 + drive_cost + 1

            # Case 2: Package is inside a vehicle
            elif status_type == 'in':
                vehicle = current_loc_or_vehicle
                # Find the vehicle's location
                if vehicle not in current_vehicle_locations:
                    # Vehicle containing package is not at any location? Error state?
                    # print(f"Warning: Vehicle {vehicle} containing {package} not found at any location.")
                    continue # Cannot estimate cost if vehicle location is unknown

                vehicle_location = current_vehicle_locations[vehicle]

                # If vehicle is at the goal location, just need to drop
                if vehicle_location == goal_location:
                    total_cost += 1 # drop
                else:
                    # Vehicle needs to drive to goal location, then drop
                    drive_cost = self.get_distance(vehicle_location, goal_location)
                    total_cost += drive_cost + 1 # drive + drop

        # The heuristic value must be 0 only for goal states.
        # We already handled the `is_goal_state` case at the beginning.
        # If we reach here, it's not a goal state. The calculated total_cost
        # should be > 0 if any package needs moving. If total_cost is 0 here,
        # it means all packages with goals are at their goal locations, which
        # implies it *is* a goal state based on typical transport goals.
        # So, the initial check is sufficient.

        return total_cost
