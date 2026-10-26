import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of args for a valid match
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function for BFS shortest path calculation
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from a start_node to all reachable nodes
    in an unweighted graph using BFS.
    Returns a dictionary {node: distance}.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure the current_node exists as a key in the graph (handles isolated nodes)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently. The cost for a package includes pick-up,
    drop, and the estimated number of drive actions based on the shortest path
    distance in the road network.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - The heuristic ignores vehicle capacity constraints.
    - The heuristic assumes a vehicle is available when needed for pick-up or drop,
      and that vehicle movement cost is independent of vehicle availability or type.
    - The estimated drive cost between two locations is the shortest path distance
      in the road network.
    - All packages that are part of the goal condition exist in the initial state
      and subsequent states.

    # Heuristic Initialization
    - Parses goal conditions to identify the target location for each package.
    - Parses static facts (`road` predicates) to build an undirected graph
      representing the road network.
    - Computes all-pairs shortest paths in the road network using BFS and stores
      the distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is a goal state. If yes, the heuristic value is 0.
    2. Identify the current location or containing vehicle for every package that
       is specified in the goal conditions.
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `goal_l`:
       a. Find the current status of package `p` in the state. This will be either
          `(at p current_l)` or `(in p current_v)`.
       b. If the fact `(at p goal_l)` is present in the current state, the package
          is already at its goal. The cost for this package is 0. Continue to the
          next package.
       c. If the package is currently at location `current_l` (`(at p current_l)` is in state):
          - Estimate the number of drive actions needed for a vehicle to go from
            `current_l` to `goal_l`. This is the shortest path distance `d` between
            `current_l` and `goal_l` in the road network, obtained from precomputed distances.
          - If `goal_l` is unreachable from `current_l`, the state is likely unsolvable
            or requires complex maneuvers not captured by this simple heuristic;
            return a large value.
          - Otherwise, the package needs to be picked up (1 action), transported
            (`d` drive actions), and dropped (1 action).
          - The estimated cost for this package is `1 + d + 1 = 2 + d`.
       d. If the package is currently inside a vehicle `v` (`(in p v)` is in state):
          - Find the location `vehicle_l` of vehicle `v` (`(at v vehicle_l)` must be in state).
          - If the vehicle's location is not found, the state is inconsistent;
            return a large value.
          - Estimate the number of drive actions needed for vehicle `v` to go from
            `vehicle_l` to `goal_l`. This is the shortest path distance `d` between
            `vehicle_l` and `goal_l`, obtained from precomputed distances.
          - If `goal_l` is unreachable from `vehicle_l`, return a large value.
          - Otherwise, the package needs to be transported (`d` drive actions) and
            dropped (1 action).
          - The estimated cost for this package is `d + 1`.
       e. Add the estimated cost for package `p` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location

        # Build the road network graph from static facts.
        self.road_graph = collections.defaultdict(list)
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Roads are bidirectional
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Compute all-pairs shortest paths using BFS.
        # self.distances[(l1, l2)] stores the shortest distance from l1 to l2.
        self.distances = {}
        for start_loc in self.all_locations:
            distances_from_start = bfs_shortest_paths(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

        # Define a large number for unreachable locations
        self.unreachable_distance = 1000000 # Use a large number instead of infinity

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returning a large value if unreachable."""
        if loc1 == loc2:
            return 0
        # If either location is not in our precomputed graph (e.g., isolated node),
        # distance is effectively infinite unless they are the same location.
        if loc1 not in self.all_locations or loc2 not in self.all_locations:
             return self.unreachable_distance

        return self.distances.get((loc1, loc2), self.unreachable_distance)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the state is a goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Track where locatable objects (packages and vehicles) are currently.
        # Maps object name to its location or container.
        current_status = {}
        # Track vehicle locations specifically for when packages are inside them.
        vehicle_locations = {}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                current_status[obj] = location
                # Assuming anything starting with 'v' is a vehicle based on domain example
                if obj.startswith('v'):
                    vehicle_locations[obj] = location
            elif predicate == "in":
                obj, container = args # obj is package, container is vehicle
                current_status[obj] = container

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Find the package's current status (location or vehicle).
            current_loc_or_container = current_status.get(package)

            # If package status isn't found, it's an unexpected state.
            # Treat as unreachable goal for this package.
            if current_loc_or_container is None:
                 return self.unreachable_distance

            # Check if the package is already at its goal location.
            # This requires checking if the fact (at package goal_location) is in the state.
            # We must check the *exact* goal fact.
            if f"(at {package} {goal_location})" in state:
                # Package is already at the goal. Cost for this package is 0.
                continue

            # Package is not at the goal. Calculate its contribution to the heuristic.
            package_cost = 0

            # Determine if the package is at a location or in a vehicle.
            # If current_loc_or_container is a vehicle name (e.g., starts with 'v'),
            # it's in a vehicle. Otherwise, it's at a location.
            # This relies on naming conventions or type information not directly available here.
            # A more robust check: if current_loc_or_container is a key in vehicle_locations, it's a vehicle.
            is_in_vehicle = current_loc_or_container in vehicle_locations

            if not is_in_vehicle:
                # Case 1: Package is on the ground at current_location.
                current_location = current_loc_or_container
                # Package is at current_location, needs to go to goal_location.
                # Needs pick-up (1), drive (d), drop (1).
                drive_distance = self.get_distance(current_location, goal_location)

                if drive_distance == self.unreachable_distance:
                    # Goal is unreachable from current location
                    return self.unreachable_distance # Return large value for unsolvable state

                package_cost = 1 + drive_distance + 1 # pick-up + drive + drop

            else:
                # Case 2: Package is inside a vehicle.
                vehicle_name = current_loc_or_container
                # Find the location of the vehicle.
                vehicle_location = vehicle_locations.get(vehicle_name)

                if vehicle_location is None:
                    # Vehicle carrying the package is not located anywhere? Invalid state.
                    return self.unreachable_distance

                # Package is in vehicle at vehicle_location, needs to go to goal_location.
                # Needs drive (d), drop (1).
                drive_distance = self.get_distance(vehicle_location, goal_location)

                if drive_distance == self.unreachable_distance:
                    # Goal is unreachable from vehicle's current location
                    return self.unreachable_distance # Return large value for unsolvable state

                package_cost = drive_distance + 1 # drive + drop

            total_cost += package_cost

        return total_cost
