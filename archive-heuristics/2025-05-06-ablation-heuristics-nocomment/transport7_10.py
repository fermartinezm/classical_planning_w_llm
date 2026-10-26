# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Removes leading/trailing parentheses and splits a PDDL fact string."""
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a PDDL fact string matches a pattern using fnmatch."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node, all_nodes):
    """Performs BFS to find shortest distances from start_node to all other nodes."""
    distances = {node: float('inf') for node in all_nodes}
    # Ensure start_node is in the graph/all_nodes list
    if start_node not in all_nodes:
        return distances # Cannot start BFS from an unknown node

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is a known node before updating distance
                if neighbor in all_nodes and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic: # Inherit from Heuristic if available
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the estimated costs for each package that is not yet at its goal
        location. The cost for a package depends on whether it is currently
        at a location or inside a vehicle, and the shortest path distances
        between locations in the road network. It considers the cost to move
        a vehicle to the package (if needed), pick up the package, drive it
        to the goal location, and drop it off.

    Assumptions:
        - The heuristic ignores vehicle capacity constraints.
        - The heuristic treats each package's delivery independently, potentially
          overestimating or underestimating costs when multiple packages could
          share vehicle trips or when vehicles are needed for multiple packages.
        - The road network is static and bidirectional.
        - All goal packages exist in the initial state and subsequent states.
        - Vehicle locations are always specified in the state if they exist.
        - The heuristic assumes that if a package is at a location but not in a vehicle,
          any vehicle can eventually reach that location. It calculates the cost
          based on the closest vehicle's current position.
        - If any required travel (vehicle to package, or package/vehicle to goal)
          involves unreachable locations according to the road network, the heuristic
          returns infinity, indicating a likely unsolvable state.
        - Objects in the state are either vehicles or packages. Vehicles are identified
          primarily by appearing as the second argument in an `(in ?p ?v)` fact.
          Any object appearing in an `(at ?x ?l)` fact that is not identified as a
          vehicle is assumed to be a package.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic performs the following steps:
        1. Stores the goal conditions (`self.goals`).
        2. Parses the goal conditions to identify which packages need to be at which
           specific locations (`self.goal_locations`). It looks for `(at package location)` goals.
        3. Parses the static facts (`task.static`) to build the road network graph.
           It identifies all unique locations mentioned in `(road l1 l2)` facts
           and creates an adjacency list representation (`self.road_graph`). Roads
           are assumed to be bidirectional. It also ensures any location mentioned
           in a goal is included in the list of all locations, even if not in a road fact.
        4. Computes the shortest path distance between every pair of locations
           in the road network using Breadth-First Search (BFS). These distances
           are stored in `self.distances`. If two locations are not connected,
           the distance is considered infinite (`float('inf')`).

    Step-By-Step Thinking for Computing Heuristic:
        In the `__call__` method, for a given state (`node.state`), the heuristic
        computes the estimated cost as follows:
        1. Initializes `total_cost` to 0.
        2. Parses the dynamic facts in the current state to determine the current
           location of each package (`package_locations`) and each vehicle
           (`vehicle_locations`), and which package is inside which vehicle
           (`package_in_vehicle`). It first identifies vehicles and packages
           involved in `(in ...)` facts, then uses `(at ...)` facts to find
           locations, distinguishing between vehicles and packages based on the
           first pass. A package is considered `at` a location only if it's not
           `in` a vehicle.
        3. Iterates through each package that is listed in the goal conditions
           (`self.goal_locations`).
        4. For each goal package `p` with goal location `loc_p_goal`:
           a. Checks if `p` is currently at its goal location (`(at p loc_p_goal)`
              and not `(in p v)` for any `v)`). If yes, this package contributes 0
              to the total cost and the heuristic moves to the next goal package.
           b. If `p` is not at its goal location, determines its current status:
              i. If `p` is at a location `loc_p_current` (i.e., `(at p loc_p_current)`
                 where `loc_p_current != loc_p_goal`, and `p` is not in a vehicle):
                 - The package needs to be picked up, transported, and dropped.
                 - It requires a vehicle to reach `loc_p_current`. The heuristic
                   finds the minimum distance from any vehicle's current location
                   to `loc_p_current` (`min_dist_v_to_p`). If no vehicles exist or
                   can reach the package, the state is deemed unsolvable (`float('inf')`).
                 - It needs to be transported from `loc_p_current` to `loc_p_goal`.
                   The distance is `dist_p_to_goal = self.distances.get((loc_p_current, goal_location), float('inf'))`.
                   If the goal location is unreachable from the package's current location,
                   the state is deemed unsolvable (`float('inf')`).
                 - Otherwise, the estimated cost for this package is
                   `min_dist_v_to_p + 1 (pick-up) + dist_p_to_goal + 1 (drop)`.
              ii. If `p` is inside a vehicle `v` (i.e., `(in p v)`):
                  - Finds the current location of vehicle `v`, `loc_v_current`. If the
                    vehicle's location is unknown, the state is deemed unsolvable (`float('inf')`).
                  - If `loc_v_current` is the goal location `loc_p_goal`:
                    - The package only needs to be dropped. The estimated cost is `1 (drop)`.
                  - If `loc_v_current` is not the goal location `loc_p_goal`:
                    - The vehicle needs to drive from `loc_v_current` to `loc_p_goal`
                      and then drop the package.
                    - The distance is `dist_v_to_goal = self.distances.get((loc_v_current, goal_location), float('inf'))`.
                    - If the goal location is unreachable from the vehicle's current location,
                      the state is deemed unsolvable (`float('inf')`).
                    - Otherwise, the estimated cost is `dist_v_to_goal + 1 (drop)`.
              iii. If a goal package is not found in either `package_locations` or
                   `package_in_vehicle` in the current state, it indicates an inconsistent
                   or unsolvable state, and the heuristic returns infinity.
        5. The costs calculated for each goal package are summed up to get the
           `total_cost`.
        6. The `total_cost` is returned as the heuristic value for the state.
    """
    def __init__(self, task):
        # Assume Heuristic base class has a constructor that takes task
        # super().__init__(task)

        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse goals to get package goal locations
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            if match(goal, "at", "?p", "?l"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location

        # 2. Build road network graph and collect all locations
        self.road_graph = {}
        all_locations = set()
        for fact in self.static_facts:
            if match(fact, "road", "?l1", "?l2"):
                l1, l2 = get_parts(fact)[1:]
                all_locations.add(l1)
                all_locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads

        # Ensure all locations mentioned in goals are in the graph nodes list
        # even if they are isolated (not connected by roads in static facts)
        for loc in self.goal_locations.values():
             all_locations.add(loc)

        self.all_locations = list(all_locations) # Store as list for consistent BFS iteration

        # Ensure graph includes nodes for all locations, even isolated ones
        for loc in self.all_locations:
             self.road_graph.setdefault(loc, set())


        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.all_locations:
            distances_from_start = bfs(self.road_graph, start_loc, self.all_locations)
            for end_loc in self.all_locations:
                 self.distances[(start_loc, end_loc)] = distances_from_start[end_loc]


    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Parse current state for locations and containment
        package_locations = {} # {package: location} if at a location AND not in a vehicle
        vehicle_locations = {} # {vehicle: location}
        package_in_vehicle = {} # {package: vehicle} if in a vehicle

        # First pass: Identify packages and vehicles based on 'in' predicate
        known_vehicles = set()
        known_packages_in_vehicle = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'in' and len(parts) == 3:
                 pkg, veh = parts[1:]
                 known_packages_in_vehicle.add(pkg)
                 known_vehicles.add(veh)
                 package_in_vehicle[pkg] = veh

        # Second pass: Populate locations based on 'at' predicate
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1:]
                 if obj in known_vehicles:
                     vehicle_locations[obj] = loc
                 elif obj in known_packages_in_vehicle:
                     # A package that is IN a vehicle should not also be AT a location
                     # This indicates an inconsistent state. Ignore the 'at' fact for this package.
                     pass
                 else:
                     # If it's not a known vehicle and not a package known to be in a vehicle,
                     # assume it's a package at a location.
                     package_locations[obj] = loc

        # Iterate through goal packages
        for package, goal_location in self.goal_locations.items():
            # Check if the package is currently at its goal location and not in a vehicle
            is_at_goal = False
            if package in package_locations and package_locations[package] == goal_location:
                 # package_locations only contains packages not in vehicles, so this check is sufficient
                 is_at_goal = True

            if is_at_goal:
                continue # Package is done

            # Package is not at the goal location (either at wrong location or in a vehicle)
            package_cost = 0

            if package in package_locations: # Package is at a location (not in a vehicle)
                current_location = package_locations[package]
                # current_location must be different from goal_location if we reached here

                # Needs pickup, drive, drop.
                # Cost = (vehicle travel to package) + pickup + (package travel to goal) + drop

                # Find closest vehicle to the package's current location
                min_dist_v_to_p = float('inf')
                
                usable_vehicles_found = False
                for vehicle, v_loc in vehicle_locations.items():
                    # Ensure vehicle location is a known location in the graph
                    if v_loc in self.all_locations:
                        usable_vehicles_found = True
                        dist = self.distances.get((v_loc, current_location), float('inf'))
                        min_dist_v_to_p = min(min_dist_v_to_p, dist)
                    # else: Vehicle is at an unknown location, cannot use it, ignore it

                # If no usable vehicle can reach the package, the state is likely unsolvable
                if not usable_vehicles_found or min_dist_v_to_p == float('inf'):
                     return float('inf')

                # Distance from package's current location to its goal location
                # Ensure current_location is a known location in the graph
                if current_location not in self.all_locations:
                     return float('inf') # Unsolvable

                dist_p_to_goal = self.distances.get((current_location, goal_location), float('inf'))
                # If package cannot reach goal location, unsolvable
                if dist_p_to_goal == float('inf'):
                     return float('inf')

                # Estimated cost for this package: vehicle travel + pickup + package travel + drop
                package_cost = min_dist_v_to_p + 1 + dist_p_to_goal + 1

            elif package in package_in_vehicle: # Package is in a vehicle
                vehicle = package_in_vehicle[package]

                # Need to ensure the vehicle's location is known
                if vehicle not in vehicle_locations:
                     # Vehicle location unknown, cannot estimate cost
                     # This indicates an inconsistent state representation
                     return float('inf') # Unsolvable

                current_location = vehicle_locations[vehicle]

                # Vehicle needs to drive to goal location and drop
                # Cost = (vehicle travel to goal) + drop

                # Ensure current_location is a known location in the graph
                if current_location not in self.all_locations:
                     return float('inf') # Unsolvable

                dist_v_to_goal = self.distances.get((current_location, goal_location), float('inf'))
                # If vehicle cannot reach goal location, unsolvable
                if dist_v_to_goal == float('inf'):
                     return float('inf')

                package_cost = dist_v_to_goal + 1

            else:
                 # Package is a goal package but not found in 'at' or 'in' predicates in the state.
                 # This should not happen in a valid state derived from the initial state and operators.
                 # Treat as unsolvable.
                 return float('inf')


            total_cost += package_cost

        return total_cost
