# Assuming heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch

# Define utility functions used by the heuristic
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
    # Ensure the number of parts matches the number of args, assuming standard PDDL fact structure
    if len(parts) != len(args):
         return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_paths(graph, start_node):
    """
    Performs BFS from a start_node to find shortest distances to all reachable nodes.
    graph: adjacency list {node: [neighbor1, neighbor2, ...]}
    start_node: the node to start BFS from
    Returns: dictionary {node: distance} for all reachable nodes.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is in graph keys, even if it has no neighbors
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
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, summing the estimates for all packages
    that are not yet at their goal. It calculates the shortest path distance
    for vehicles using the road network.

    # Assumptions:
    - The goal is always to have packages at specific locations (`(at ?p ?l)`).
    - Capacity constraints are relaxed: it's assumed that a suitable vehicle
      can always be made available for pickup/dropoff at a location if needed,
      without considering the state of vehicle capacities beyond their current
      location.
    - The cost of moving a vehicle between locations is the shortest path
      distance in the road network.
    - The cost of pick-up and drop actions is 1.
    - Packages are objects that can be 'in' a vehicle or 'at' a location.
    - Vehicles are objects that can have 'capacity' and be 'at' a location,
      and packages can be 'in' them. Object types are inferred from predicate usage
      in the initial state, goals, and static facts.

    # Heuristic Initialization
    - Infers package and vehicle objects by examining predicate usage in the initial state, goals, and static facts.
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static `road` facts.
    - Collects all relevant locations from static facts, initial state, and goals.
    - Pre-calculates shortest path distances between all pairs of relevant locations
      using Breadth-First Search (BFS) on the road network graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment status of every package
       and the location of every vehicle by iterating through the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `l_goal` (as determined in initialization):
        a. Find the current status of `p` (its location if on the ground, or the vehicle it's in).
        b. If `p` is currently on the ground at its goal location (`(at p l_goal)` is true in the state), the cost for this package is 0. Continue to the next package.
        c. If `p` is not on the ground at its goal location:
           - If `p` is on the ground at `l_current` (`l_current` is a location and `l_current != l_goal`):
             - Estimated cost for this package: 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop). The shortest distance is retrieved from the pre-calculated distances. If the goal is unreachable, a large penalty is added.
           - If `p` is inside vehicle `v` which is at `l_current`:
             - If `l_current == l_goal`:
               - Estimated cost for this package: 1 (drop).
             - If `l_current != l_goal`:
               - Estimated cost for this package: shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop). The shortest distance is retrieved from the pre-calculated distances. If the goal is unreachable, a large penalty is added.
           - If the package status is unknown or invalid, a large penalty is added (indicates an unexpected state).
        d. Add the estimated cost for package `p` to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Infer packages and vehicles by examining predicate usage
        self.packages = set()
        self.vehicles = set()
        all_locations_set = set()

        # Examine initial state, goals, and static facts to infer types and collect locations
        for fact in initial_state | static_facts | self.goals:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts

             predicate = parts[0]
             if predicate == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Objects appearing in 'at' can be packages or vehicles
                 # Locations appear as the second argument
                 all_locations_set.add(loc)
             elif predicate == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 self.packages.add(package)
                 self.vehicles.add(vehicle)
             elif predicate == "capacity" and len(parts) == 3:
                 vehicle, size = parts[1], parts[2]
                 self.vehicles.add(vehicle)
             elif predicate == "road" and len(parts) == 3:
                 l1, l2 = parts[1], parts[2]
                 all_locations_set.add(l1)
                 all_locations_set.add(l2)
             # Ignore capacity-predecessor and other potential predicates

        # Refine packages/vehicles: Any object appearing as the first arg of 'at'
        # that is not already identified as a vehicle must be a package.
        # This handles packages that are initially on the ground and never 'in' in the initial state.
        # Also handles vehicles that might not have capacity mentioned initially.
        potential_locatables = set()
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 potential_locatables.add(obj)

        # Objects in potential_locatables that are not vehicles must be packages
        self.packages.update(potential_locatables - self.vehicles)
        # Objects in potential_locatables that are not packages must be vehicles
        self.vehicles.update(potential_locatables - self.packages)


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are of the form (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                if obj in self.packages: # Only track package goals
                    self.goal_locations[obj] = location
            # Ignore other types of goals if any

        # Build the road network graph using all collected locations
        all_locations = list(all_locations_set) # Convert to list for consistent iteration order if needed, set is fine too
        self.road_graph = {loc: [] for loc in all_locations} # Initialize with all locations
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                # Ensure locations are in our collected set before adding roads
                if l1 in self.road_graph and l2 in self.road_graph: # Check if locations are known
                    self.road_graph[l1].append(l2)
                    self.road_graph[l2].append(l1) # Roads are bidirectional

        # Pre-calculate shortest path distances
        self.distances = {}
        for start_loc in all_locations:
            distances_from_start = bfs_shortest_paths(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

        # Define a large penalty for unreachable goals or invalid states
        self.UNREACHABLE_PENALTY = 1000 # Arbitrary large number


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_current_status = {} # Maps package -> location or vehicle
        vehicle_locations = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_status[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles:
                     package_current_status[package] = vehicle
            # Ignore other predicates like capacity, road, capacity-predecessor in the state

        total_cost = 0

        # Iterate through packages that have a goal location defined
        for package, goal_loc in self.goal_locations.items():
            current_status = package_current_status.get(package)

            # If package is not found in the state, this indicates an issue or an irrelevant object.
            # For a well-formed problem, all goal packages should be in the initial state.
            # If a package is not in the state, it cannot be moved, so the goal is likely unreachable.
            # Add a large penalty.
            if current_status is None:
                 total_cost += self.UNREACHABLE_PENALTY
                 continue

            # Check if the package is already at the goal location on the ground
            # This is the desired final state for the package goal.
            if current_status == goal_loc and f"(at {package} {goal_loc})" in state:
                 continue # Package is at goal on the ground, cost is 0 for this package

            # Package is not at the goal location on the ground. It needs to move.

            # Case 1: Package is on the ground at current_loc (which is not goal_loc)
            # Check if current_status is a location by seeing if it's in our graph keys
            if current_status in self.road_graph:
                current_loc = current_status
                # Cost: pickup + drive + drop
                # Need distance from current_loc to goal_loc
                dist = self.distances.get((current_loc, goal_loc))

                if dist is None:
                    # Goal is unreachable from current location in the road network.
                    total_cost += self.UNREACHABLE_PENALTY
                else:
                    # Cost = pickup (1) + drive (dist) + drop (1)
                    total_cost += 1 + dist + 1

            # Case 2: Package is inside a vehicle v
            elif current_status in self.vehicles: # Check if current_status is a vehicle
                vehicle = current_status
                vehicle_loc = vehicle_locations.get(vehicle) # Get vehicle's location

                # If vehicle location is unknown, state is likely invalid.
                if vehicle_loc is None:
                    total_cost += self.UNREACHABLE_PENALTY
                    continue

                # Package is in vehicle at vehicle_loc. Goal is goal_loc.
                # If vehicle_loc == goal_loc: need to drop. Cost = 1.
                # If vehicle_loc != goal_loc: need to drive + drop. Cost = distance(vehicle_loc, goal_loc) + 1.

                dist = self.distances.get((vehicle_loc, goal_loc))
                if dist is None:
                     # Goal is unreachable from vehicle's current location.
                     total_cost += self.UNREACHABLE_PENALTY
                else:
                    if vehicle_loc == goal_loc:
                        # Cost = drop (1)
                        total_cost += 1
                    else:
                        # Cost = drive (dist) + drop (1)
                        total_cost += dist + 1
            else:
                 # current_status is neither a known location nor a known vehicle? Invalid state?
                 total_cost += self.UNREACHABLE_PENALTY


        # The heuristic should be 0 only for goal states.
        # Our logic adds cost for packages NOT at their goal location on the ground.
        # If all packages are at their goal location on the ground, the loop correctly results in total_cost = 0.
        # If the state is NOT the goal state, at least one package is not at its goal on the ground,
        # so its contribution will be >= 1, making total_cost > 0.
        # Thus, h=0 iff the state is a goal state.

        return total_cost
