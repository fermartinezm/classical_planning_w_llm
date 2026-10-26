from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Split by space, handle potential multiple spaces or leading/trailing spaces
    parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at v1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest path distances
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from a start_node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start_node.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node} # Use a set for faster lookup
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node is a key in the graph before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, based on its current state (on the ground or in a vehicle)
    and its goal location. The cost includes loading/unloading actions and
    the shortest path distance (number of drive actions) required for the
    package or its vehicle to reach the goal location.

    # Assumptions
    - The cost of each action (load, unload, drive) is 1.
    - Vehicle capacity is not explicitly modeled in the heuristic calculation
      beyond the need for a vehicle to transport a package.
    - Road network allows movement between locations.
    - Shortest path distance in the road network is a reasonable estimate
      for the number of drive actions.
    - The heuristic sums costs for packages independently, potentially
      overestimating by double-counting shared vehicle trips.
    - All packages with goal locations are present in the initial state
      and subsequent states, either on the ground or in a vehicle.
    - All vehicles are located at some location in the state.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representation of the road network from the static facts.
    - Precomputes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Determine the current location or containing vehicle for every package
       that has a goal, and the current location for every vehicle.
       - Iterate through the state facts.
       - If a fact is `(at obj loc)`:
         - If `obj` is a package with a goal, record its location.
         - If `obj` is a vehicle, record its location.
       - If a fact is `(in p v)`:
         - If `p` is a package with a goal, record that it is in vehicle `v`.
    3. For each package `p` that has a goal location `goal_loc_p`:
       a. Check if `p` is currently at `goal_loc_p`. If yes, this package
          contributes 0 to the heuristic.
       b. If `p` is not at `goal_loc_p`, determine its current status:
          i. If `p` is on the ground at `current_loc_p`:
             - This package needs to be loaded (1 action), transported
               (shortest_distance(`current_loc_p`, `goal_loc_p`) drive actions),
               and unloaded (1 action).
             - Estimated cost for this package: 1 + shortest_distance(`current_loc_p`, `goal_loc_p`) + 1.
             - Add this cost to the total heuristic.
          ii. If `p` is inside a vehicle `v`:
              - Find the current location of vehicle `v`, `v_loc`.
              - If `v_loc` is the same as `goal_loc_p`:
                 - This package only needs to be unloaded (1 action).
                 - Estimated cost for this package: 1.
                 - Add this cost to the total heuristic.
              - If `v_loc` is different from `goal_loc_p`:
                 - This package needs the vehicle to drive from `v_loc` to
                   `goal_loc_p` (shortest_distance(`v_loc`, `goal_loc_p`) drive actions),
                   and then needs to be unloaded (1 action).
                 - Estimated cost for this package: shortest_distance(`v_loc`, `goal_loc_p`) + 1.
                 - Add this cost to the total heuristic.
       c. If at any point a required location or vehicle status is missing
          (e.g., package is 'in' a vehicle but vehicle location is unknown,
          or goal location is unreachable), return `float('inf')` as the state
          likely leads to an unsolvable path or is malformed.
    4. Return the total computed cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically like (at packageX locationY)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road graph (adjacency list).
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = []
                self.road_graph[loc1].append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph[loc2].append(loc1)

        # Add all locations mentioned in goals to the set of locations,
        # even if they have no roads. This ensures BFS is attempted from/to these locations.
        all_relevant_locations = set(locations)
        for loc in self.goal_locations.values():
             all_relevant_locations.add(loc)

        # Precompute all-pairs shortest paths.
        self.all_pairs_distances = {}
        for start_loc in all_relevant_locations:
             # Run BFS from every relevant location. BFS handles isolated nodes correctly.
             self.all_pairs_distances[start_loc] = bfs(self.road_graph, start_loc)


    def get_distance(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance between two locations.
        Returns float('inf') if locations are not connected or not known.
        """
        # Check if both locations are in our precomputed distances map
        if loc1 in self.all_pairs_distances and loc2 in self.all_pairs_distances[loc1]:
            return self.all_pairs_distances[loc1][loc2]
        elif loc1 == loc2:
             return 0 # Distance to self is 0
        else:
             # One or both locations are not in the set of locations encountered
             # during initialization (from roads or goals). They might be
             # isolated locations or invalid references. Treat as unreachable.
             return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Check if the goal is reached. If so, heuristic is 0.
        # This is a required property for greedy best-first search termination.
        # We check this first for efficiency and correctness.
        if self.goals <= state:
             return 0

        # Track current state of packages and vehicles
        package_status = {} # {package_name: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle_name: location}

        # Populate package_status and vehicle_locations from the current state
        # Iterate through facts to find package and vehicle locations/containment
        # We assume any object with an '(at obj loc)' fact that is not a package
        # with a goal is a vehicle. This is a heuristic simplification.
        # We assume any object in an '(in p v)' fact is a package 'p' and vehicle 'v'.
        possible_vehicles = set()
        packages_in_vehicle_set = set() # Use a set for quick lookup

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # We only care about packages that have a goal
                if package in self.goal_locations:
                    package_status[package] = ('in', vehicle)
                    possible_vehicles.add(vehicle)
                    packages_in_vehicle_set.add(package)

        # Now process 'at' facts to get vehicle locations and package locations (if not in vehicle)
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # If this object is a package with a goal AND it's not marked as 'in' a vehicle
                 if obj in self.goal_locations and obj not in packages_in_vehicle_set:
                      package_status[obj] = ('at', loc)
                 # If this object is one we identified as a possible vehicle (from 'in' fact)
                 # or if it's not a package with a goal (heuristic assumption)
                 # This second part `obj not in self.goal_locations` is a heuristic guess
                 # that anything else with an 'at' fact is a vehicle.
                 elif obj in possible_vehicles or obj not in self.goal_locations:
                      vehicle_locations[obj] = loc


        total_cost = 0

        # Iterate through packages that have goals
        for package, goal_location in self.goal_locations.items():
            # If a package with a goal is not found in the state's 'at' or 'in' facts,
            # it implies an invalid or unreachable state. Return infinity.
            if package not in package_status:
                 return float('inf')

            status_type, current_loc_or_vehicle = package_status[package]

            # Check if the package is already at its goal location
            # This check is slightly redundant because the initial goal check handles
            # the state where *all* goals are met. But checking per package
            # here simplifies the logic flow for calculating individual contributions.
            if status_type == 'at' and current_loc_or_vehicle == goal_location:
                continue # Package is at goal, contributes 0 cost

            # Package is not at its goal location, calculate its contribution
            if status_type == 'at':
                # Package is on the ground, not at goal
                current_loc = current_loc_or_vehicle
                # Needs Load + Drive + Unload
                dist = self.get_distance(current_loc, goal_location)
                if dist == float('inf'):
                    # Goal is unreachable from current location
                    return float('inf') # Return infinity if any goal is unreachable
                total_cost += 1 # Load action
                total_cost += dist # Drive actions
                total_cost += 1 # Unload action

            elif status_type == 'in':
                # Package is inside a vehicle
                vehicle = current_loc_or_vehicle
                # Need to find the vehicle's location
                if vehicle not in vehicle_locations:
                     # Vehicle location is unknown. This state is problematic.
                     return float('inf')

                vehicle_loc = vehicle_locations[vehicle]

                if vehicle_loc == goal_location:
                    # Vehicle is at the goal location, package just needs Unload
                    total_cost += 1 # Unload action
                else:
                    # Vehicle is not at the goal location, needs Drive + Unload
                    dist = self.get_distance(vehicle_loc, goal_location)
                    if dist == float('inf'):
                        # Goal is unreachable from vehicle's current location
                        return float('inf') # Return infinity if any goal is unreachable
                    total_cost += dist # Drive actions
                    total_cost += 1 # Unload action

        return total_cost
