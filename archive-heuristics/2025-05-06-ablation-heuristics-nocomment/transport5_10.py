# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Parses a PDDL fact string into a list of strings."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

# BFS function to find shortest paths
def bfs(start_node, graph):
    """
    Performs BFS from a start node to find shortest distances to all reachable nodes.
    Assumes graph is an adjacency dictionary {node: [neighbor1, neighbor2, ...]}
    Returns a dictionary {node: distance}. Distance is float('inf') if unreachable.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is a valid node in the graph (has an entry in distances)
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

class transportHeuristic: # Inherit from Heuristic if needed
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required
    pick-up actions, drop actions, and the shortest path drive distances
    for all packages that are not yet at their goal location. It relaxes
    capacity constraints and assumes vehicles are available to perform
    the necessary transports along shortest paths. The heuristic is 0
    if and only if the state is a goal state. It returns infinity for
    states where a package required by the goal is unreachable via the
    road network or cannot be located in the state.

    Assumptions:
    - The road network is static and provided in the static facts.
    - All locations mentioned in the problem (in static facts, initial state, goals)
      are considered nodes in the road network graph.
    - Packages always have a defined location (either at a location or in a vehicle)
      in any valid state.
    - Vehicles always have a defined location in any valid state.
    - The goal state only contains (at ?p ?l) predicates for packages.
    - Objects mentioned in (at ?x ?l) facts that are not packages listed in the goals
      are assumed to be vehicles for the purpose of locating vehicles.

    Heuristic Initialization:
    1. Parses the goal state (`task.goals`) to create a mapping from each package
       object (that has a goal) to its target location. This is stored in
       `self.goal_locations`.
    2. Builds the road network graph (adjacency list representation) from the
       static facts (`task.static`) that match the pattern `(road ?l1 ?l2)`.
       All locations mentioned in road facts, initial state `(at ?x ?l)` facts,
       and goal `(at ?p ?l)` facts are collected to ensure all relevant locations
       are included as nodes in the graph.
    3. Computes all-pairs shortest paths between all unique locations identified
       in the previous step using Breadth-First Search (BFS). Since drive actions
       have a uniform cost of 1, BFS finds the shortest path in terms of number
       of drive actions. The results are stored in `self.dist_matrix`, a dictionary
       mapping a start location to another dictionary mapping an end location
       to the shortest distance. Unreachable locations will have a distance of
       `float('inf')`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the heuristic value `h` to 0.
    2. Iterate through the current state facts (`node.state`) to determine the
       current location or containing vehicle for each package that is listed
       in the goals, and the current location of each vehicle. This information
       is stored in `package_state` and `vehicle_location` dictionaries.
       Objects with an `(at ?x ?l)` fact that are not packages from the goals
       are assumed to be vehicles.
    3. Initialize three sets to track the necessary actions and movements:
        - `packages_needing_pickup`: Stores packages that are currently at a
          location (not in a vehicle) and need to be moved to their goal.
        - `packages_needing_drop`: Stores packages that are currently inside
          a vehicle and need to be moved to their goal or dropped at their
          current location if it's the goal.
        - `required_trips`: Stores unique pairs of `(start_loc, end_loc)`
          representing the minimal vehicle movements required to transport
          misplaced packages.
    4. Iterate through each package and its goal location stored in
       `self.goal_locations`.
    5. For the current package, find its status (`at` or `in`) and location/vehicle
       from the parsed state information.
    6. If the package's state cannot be determined (e.g., package from goal is
       not found in state facts), the goal is likely unreachable from this state,
       so return `float('inf')`.
    7. If the package is currently `(at p l_curr)` and `l_curr` is the goal location,
       the package is already in its final state, so it contributes 0 to the heuristic.
       Continue to the next package.
    8. If the package is not yet at its goal location:
        a. Add the package to `packages_needing_drop`, as it will eventually need
           to be dropped at its goal location.
        b. If the package is currently `(at p l_curr)`:
            i. Add the package to `packages_needing_pickup`, as it needs to be
               picked up by a vehicle.
            ii. If `l_curr` is different from the goal location, add the pair
                `(l_curr, goal_location)` to the `required_trips` set.
        c. If the package is currently `(in p v)`:
            i. Find the current location `l_curr` of vehicle `v` from `vehicle_location`.
            ii. If the vehicle's location `l_curr` is different from the goal location,
                add the pair `(l_curr, goal_location)` to the `required_trips` set.
            iii. If the vehicle containing the package cannot be located, the goal
                 is likely unreachable, so return `float('inf')`.
    9. After processing all packages from the goals, calculate the heuristic value:
        a. Add the total number of packages in `packages_needing_pickup` to `h`.
           (Cost for pick-up actions).
        b. Add the total number of packages in `packages_needing_drop` to `h`.
           (Cost for drop actions).
        c. For each unique `(start_loc, end_loc)` pair in the `required_trips` set,
           look up the shortest distance in `self.dist_matrix`.
           i. If the distance is `float('inf')` (meaning the end location is
              unreachable from the start location), the goal is unreachable,
              so return `float('inf')`.
           ii. Otherwise, add the distance to `h`. (Cost for drive actions).
    10. Return the final calculated value of `h`.
    """
    def __init__(self, task):
        # 1. Parse goals
        self.goal_locations = {}
        # Collect all objects mentioned in goals (assuming they are packages)
        goal_packages = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                goal_packages.add(package)

        # 2. Build road network graph and collect all locations
        self.road_graph = {}
        all_locations = set()

        # Add locations from static road facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph:
                    self.road_graph[l2] = []
                self.road_graph[l1].append(l2)
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations from initial state and goals if not already included
        # (This ensures all relevant locations are in the graph nodes for BFS)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # locatable is at location
                 loc = parts[2]
                 if loc not in self.road_graph:
                     self.road_graph[loc] = []
                 all_locations.add(loc)
             # 'in' facts don't directly give locations, vehicle location is needed
             # 'capacity', 'capacity-predecessor' are not locations

        for loc in self.goal_locations.values():
             if loc not in self.road_graph:
                 self.road_graph[loc] = []
             all_locations.add(loc)

        # Ensure all collected locations are keys in the graph, even if they have no roads
        for loc in all_locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # 3. Compute all-pairs shortest paths
        self.dist_matrix = {}
        # Only compute BFS from locations that are actually nodes in the graph
        for start_node in self.road_graph.keys():
            self.dist_matrix[start_node] = bfs(start_node, self.road_graph)

    def __call__(self, node):
        state = node.state
        h = 0

        # 2. Parse current state
        package_state = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_location = {} # {vehicle: location}

        # First pass: find vehicle locations
        # Assume anything with 'at' that is NOT a package in goals is a vehicle
        # This is a heuristic assumption based on domain structure.
        # Collect all objects with 'at' facts first
        at_facts_objects = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                at_facts_objects.add(parts[1])

        # Identify vehicles as objects with 'at' facts that are not packages in goals
        vehicles = at_facts_objects - set(self.goal_locations.keys())

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in vehicles:
                     vehicle_location[obj] = loc

        # Second pass: find package states for packages in goals
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  obj, loc = parts[1], parts[2]
                  if obj in self.goal_locations: # Only care about packages in goals
                       package_state[obj] = ('at', loc)
             elif parts[0] == 'in':
                  p, v = parts[1], parts[2]
                  if p in self.goal_locations: # Only care about packages in goals
                       package_state[p] = ('in', v)


        # 3. Identify tasks for misplaced packages
        packages_needing_pickup = set()
        packages_needing_drop = set()
        required_trips = set() # {(start_loc, end_loc)}

        # 4. Iterate through each package and its goal location
        for package, goal_location in self.goal_locations.items():
            # 5. Find package state
            if package not in package_state:
                 # Package from goal is not found anywhere in the state facts.
                 # This indicates a problem with the state representation or problem.
                 # Goal is likely unreachable.
                 return float('inf')

            current_status, current_loc_or_veh = package_state[package]

            # 7. Check if goal is already satisfied
            if current_status == 'at' and current_loc_or_veh == goal_location:
                 continue # Package is already at goal location

            # 8. Package is not at goal location, needs action
            packages_needing_drop.add(package) # It will eventually need a drop at the goal

            if current_status == 'at':
                l_curr = current_loc_or_veh
                packages_needing_pickup.add(package) # Needs pick-up
                if l_curr != goal_location:
                     required_trips.add((l_curr, goal_location))

            elif current_status == 'in':
                v = current_loc_or_veh
                # Find vehicle location
                if v in vehicle_location:
                    l_curr = vehicle_location[v]
                    if l_curr != goal_location:
                         required_trips.add((l_curr, goal_location))
                    # If l_curr == goal_location, it only needs a drop, already counted in packages_needing_drop
                else:
                    # Vehicle containing package is not located. Problem with state.
                    return float('inf') # Cannot locate vehicle containing package

        # 9. Sum action costs
        h += len(packages_needing_pickup)
        h += len(packages_needing_drop)

        # 10. Sum drive costs for unique required trips
        for start_loc, end_loc in required_trips:
             # Check if locations exist in the distance matrix (should if all_locations was populated correctly)
             if start_loc in self.dist_matrix and end_loc in self.dist_matrix[start_loc]:
                  dist = self.dist_matrix[start_loc][end_loc]
                  if dist == float('inf'):
                       # Goal location is unreachable from current location
                       return float('inf') # Return infinity
                  h += dist
             else:
                  # This indicates an issue with graph construction or location handling
                  # Should not happen if all_locations includes all relevant locations
                  # and BFS was run from all graph nodes.
                  # Treat as unreachable.
                  return float('inf')

        # Heuristic is 0 iff all packages in goals are (at p l_goal).
        # If packages_needing_pickup, packages_needing_drop, and required_trips are all empty, h is 0.
        # This happens only when the loop in step 4 finds all packages already at their goal.
        # If any package is not at goal, it's added to packages_needing_drop (at least), making h >= 1.
        # So h=0 iff goal is reached.

        return h
