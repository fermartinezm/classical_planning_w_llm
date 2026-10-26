from heuristics.heuristic_base import Heuristic
from task import Task

from collections import deque

# Helper function for parsing facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Removes surrounding brackets and splits by space
    parts = fact_string[1:-1].split()
    if not parts:
        return None, []
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function for BFS
def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node."""
    # Ensure start_node is a valid key in the graph (even if isolated)
    if start_node not in graph:
         graph[start_node] = [] # Add if missing

    # Initialize distances to infinity for all nodes currently in the graph
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Check if current_node has neighbors in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Ensure neighbor is also in the graph keys (handles cases where
                # a road points to a location not otherwise mentioned)
                if neighbor not in graph:
                    graph[neighbor] = []
                    distances[neighbor] = float('inf') # Initialize distance for new node

                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances

# Helper function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, locations):
    """Computes shortest path distances between all pairs of locations."""
    shortest_paths = {}
    # Collect all unique locations mentioned in the graph edges and the provided locations set
    all_nodes = set(locations)
    for node in graph:
        all_nodes.add(node)
        for neighbor in graph[node]:
            all_nodes.add(neighbor)

    for start_loc in all_nodes:
         # Ensure graph entry exists for BFS, even if isolated
        if start_loc not in graph:
             graph[start_loc] = []
        shortest_paths[start_loc] = bfs(graph, start_loc)
    return shortest_paths


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs for
    each package that is not yet at its goal location. The estimated cost
    for a single package depends on whether it is currently at a location
    or inside a vehicle. If at a location, it estimates the cost to get
    a suitable vehicle to the package, pick it up, drive it to the goal
    location, and drop it off. If inside a vehicle, it estimates the cost
    to drive the vehicle to the goal location and drop it off. Driving costs
    are estimated using shortest path distances in the road network. Pick-up
    and drop-off actions are assumed to cost 1. Vehicle capacity is considered
    only for the pick-up action (a vehicle cannot pick up if its current
    capacity is the minimum possible).

    Assumptions:
    - The road network is static and provided in the initial state/static facts.
    - Vehicle capacities and capacity-predecessor relationships are static.
    - Packages are always either 'at' a location or 'in' a vehicle.
    - The cost of drive, pick-up, and drop actions is 1.
    - A vehicle can pick up a package if its current capacity is not the smallest capacity size defined by capacity-predecessor facts.
    - The capacity-predecessor facts form a chain down to a smallest capacity size (c_min) which is a predecessor but not a successor in any fact.

    Heuristic Initialization:
    1. Parses static facts to build the road network graph and identify locations.
    2. Parses static facts to determine the capacity hierarchy and find the smallest capacity size (c_min).
    3. Parses initial state and goal facts to identify all vehicles, packages, and locations. Combines these with locations from road facts.
    4. Computes all-pairs shortest paths between all identified locations using BFS on the road network graph.
    5. Parses goal facts to identify the target location for each package.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, return 0.
    2. Initialize the total heuristic value (h_value) to 0.
    3. Create dictionaries for quick lookup of dynamic facts in the current state:
       - `at_locations`: Maps locatable (vehicles, packages) names to their location names if the fact `(at obj loc)` is in the state.
       - `in_vehicle`: Maps package names to the vehicle names they are in if the fact `(in p v)` is in the state.
       - `vehicle_capacities`: Maps vehicle names to their current capacity size names if the fact `(capacity v s)` is in the state.
       - Populate these dictionaries by iterating through the facts in the current state.
    4. Initialize an `unreachable` flag to `False`.
    5. Iterate through each package that has a defined goal location in `self.package_goals`.
    6. For the current package:
       - Get its goal location (`goal_loc`).
       - Check if the package is already at its goal location in the current state (i.e., `package` is in `at_locations` and `at_locations[package] == goal_loc`). If yes, skip this package and continue to the next.
       - If the package is not at the goal, determine the package's current status and location:
         - If the package name is a key in `at_locations`, it is `at` `p_loc = at_locations[package]`. Set `is_in_vehicle = False`.
         - If the package name is a key in `in_vehicle`, it is `in` the vehicle `vehicle_carrying_p = in_vehicle[package]`. Find the location of this vehicle (`p_loc = at_locations.get(vehicle_carrying_p)`). Set `is_in_vehicle = True`.
         - If the package is neither in `at_locations` nor `in_vehicle`, or if it's in a vehicle whose location is unknown (vehicle not in `at_locations`), mark the package as unreachable and add infinity to `h_value`. Continue to the next package.
       - Calculate the estimated cost contribution (`cost_p`) for this package:
         - If `is_in_vehicle` is `True`:
           # Package is in vehicle_carrying_p at p_loc, needs to go to goal_loc
           # Cost is drive from p_loc to goal_loc + drop (1)
           # Check if p_loc and goal_loc are valid keys and the path exists.
           if p_loc in self.shortest_paths and goal_loc in self.shortest_paths[p_loc]:
               drive_cost = self.shortest_paths[p_loc][goal_loc]
               if drive_cost != float('inf'):
                    cost_p = drive_cost + 1 # drive + drop
               else:
                    unreachable = True # Goal location unreachable from current location
           else:
               unreachable = True # Locations not in shortest_paths (disconnected graph?)

         - If `is_in_vehicle` is `False` (package is `at` `p_loc`):
           # Package is at p_loc, needs pickup and drive to goal_loc
           # Cost is min_vehicle_dist_to_p + pickup (1) + drive_p_to_goal + drop (1)
           min_vehicle_dist_to_p = float('inf')
           found_reachable_suitable_vehicle = False

           # Find the closest suitable vehicle
           for vehicle in self.vehicles:
               if vehicle in at_locations:
                   loc_v = at_locations[vehicle]
                   s_v = vehicle_capacities.get(vehicle) # Get current capacity

                   # Check if vehicle can pick up (capacity is not c_min)
                   # A vehicle can pick up if c_min was determined AND its current capacity is not c_min.
                   # If c_min is None, it implies pickup is not possible via capacity rules.
                   can_pickup = (self.c_min is not None and s_v is not None and s_v != self.c_min)

                   if can_pickup:
                       # Check if vehicle location is reachable from package location
                       # We need dist from loc_v to p_loc
                       if loc_v in self.shortest_paths and p_loc in self.shortest_paths[loc_v]:
                           dist_v_to_p = self.shortest_paths[loc_v][p_loc]
                           if dist_v_to_p != float('inf'):
                               found_reachable_suitable_vehicle = True
                               min_vehicle_dist_to_p = min(min_vehicle_dist_to_p, dist_v_to_p)
                           # else: vehicle is suitable but unreachable from package location

           if found_reachable_suitable_vehicle:
               # Found a reachable suitable vehicle
               # Now check if package location is reachable from goal location
               if p_loc in self.shortest_paths and goal_loc in self.shortest_paths[p_loc]:
                   drive_cost_p_to_goal = self.shortest_paths[p_loc][goal_loc]
                   if drive_cost_p_to_goal != float('inf'):
                        cost_p = min_vehicle_dist_to_p + 1 + drive_cost_p_to_goal + 1 # drive_v_to_p + pickup + drive_p_to_goal + drop
                   else:
                        unreachable = True # Goal location unreachable from package location
               else:
                    unreachable = True # Package location or goal location not in shortest_paths
           else:
               # No suitable vehicle found or no reachable suitable vehicle
               unreachable = True # Package cannot be picked up

            # Add cost for this package
            h_value += cost_p

        # If any package was deemed unreachable, the total heuristic is infinity
        if unreachable:
             return float('inf')

        return h_value

