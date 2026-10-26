import logging
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are available

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into its predicate and objects.

    Args:
        fact_string: The string representation of a PDDL fact (e.g., '(at p1 l1)').

    Returns:
        A tuple containing the predicate (string) and a list of objects (strings).
        Returns (None, []) for invalid or empty fact strings.
    """
    # Removes parentheses and splits by space
    parts = fact_string.strip('()').split()
    if not parts:
        return None, []
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required
    actions for each package that is not yet at its goal location.
    For a package P at location L_current needing to reach L_goal:
    - If P is at L_current (not in a vehicle): cost is 1 (pickup) + shortest_path_distance(L_current, L_goal) (drive) + 1 (drop).
    - If P is in a vehicle V at L_current: cost is shortest_path_distance(L_current, L_goal) (drive) + 1 (drop).
    This heuristic ignores vehicle capacity constraints and assumes vehicles
    are available where needed. It is non-admissible but aims to be
    informative for greedy search.

    Assumptions:
    - The road network is connected, or at least all locations relevant
      to package movements are reachable from each other. Unreachable goal
      locations result in infinite heuristic.
    - All locations mentioned in facts and goals are collected and used
      for distance calculations.
    - The heuristic is used in a greedy best-first search (doesn't need
      admissibility).

    Heuristic Initialization:
    1. Collect all unique locations mentioned in static facts (roads),
       initial state facts ('at'), and goal facts ('at').
    2. Parse static facts to build the road network graph (adjacency list)
       using the collected locations.
    3. Compute all-pairs shortest paths between all collected locations
       using BFS. Store these distances.
    4. Parse goal facts to identify the target location for each package.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state (set of facts).
    2. Initialize total heuristic value `h = 0`.
    3. Determine the current location of each locatable object (packages, vehicles) and which packages are in which vehicles:
       - Initialize empty dictionaries/sets: `object_locations = {}`, `package_to_vehicle = {}`.
       - Iterate through state facts.
       - If a fact is `(at obj l)`, record `object_locations[obj] = l`.
       - If a fact is `(in p v)`, record `package_to_vehicle[p] = v`.
    4. For each package `p` that has a goal location `l_p_goal`:
       - Determine the package's current location `l_p_current`:
         - If `p` is in `package_to_vehicle`, find the vehicle `v = package_to_vehicle[p]`. The package's location is `object_locations[v]` if `v` is located.
         - Otherwise, the package's location is `object_locations[p]` if `p` is located.
         - If the package's location cannot be determined (e.g., vehicle not located, package not 'at' or 'in'), it's unreachable, return infinity.
       - If `l_p_current` is equal to `l_p_goal`, the package is done, continue to the next package.
       - Calculate the shortest path distance `d` between `l_p_current` and `l_p_goal` using the precomputed distances.
       - If `d` is infinity (goal unreachable from current location), return infinity.
       - If package `p` was found in `package_to_vehicle` (meaning it's in a vehicle):
         - Add `d + 1` (drive + drop) to `h`.
       - Otherwise (package `p` was found directly in `object_locations` and not in a vehicle):
         - Add `1 + d + 1` (pickup + drive + drop) to `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.road_graph = {}
        self.all_pairs_distances = {}
        self.locations = set() # Keep track of all locations

        # 1. Collect all unique locations from static, initial, and goal facts
        for fact_string in task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'road' and len(objects) == 2:
                l1, l2 = objects
                self.locations.add(l1)
                self.locations.add(l2)
            # 'at' facts should not strictly be in static PDDL, but check just in case
            elif predicate == 'at' and len(objects) == 2:
                 obj, loc = objects
                 self.locations.add(loc)

        for fact_string in task.initial_state:
             predicate, objects = parse_fact(fact_string)
             if predicate == 'at' and len(objects) == 2:
                 obj, loc = objects
                 self.locations.add(loc)

        for goal_fact_string in task.goals:
            predicate, objects = parse_fact(goal_fact_string)
            if predicate == 'at' and len(objects) == 2:
                package, location = objects
                self.locations.add(location)
                self.package_goals[package] = location # Also store goals here

        # 2. Build the road network graph using collected locations
        for loc in self.locations:
             self.road_graph[loc] = [] # Initialize adjacency list for all locations

        for fact_string in task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'road' and len(objects) == 2:
                l1, l2 = objects
                # Add directed edge. Assuming bidirectionality requires both (road l1 l2) and (road l2 l1)
                if l1 in self.road_graph: # Ensure location is known
                    self.road_graph[l1].append(l2)


        # 3. Compute all-pairs shortest paths using BFS
        self.all_pairs_distances = self._compute_all_pairs_shortest_paths()


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations."""
        distances = {}
        for start_node in self.locations:
            distances[start_node] = self._bfs(start_node)
        return distances

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        dist = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Should not happen if locations are collected correctly, but safety check
             return dist

        dist[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is a valid key in road_graph, handle isolated nodes
            if u in self.road_graph: # Check if the node has any outgoing roads defined
                for v in self.road_graph[u]:
                    if v in dist and dist[v] == float('inf'): # Ensure neighbor is a known location and not visited
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        h_value = 0

        # 3. Determine the current location of each locatable object (packages, vehicles)
        object_locations = {}
        package_to_vehicle = {}

        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'at' and len(objects) == 2:
                obj, loc = objects
                object_locations[obj] = loc
            elif predicate == 'in' and len(objects) == 2:
                package_obj, vehicle_obj = objects
                package_to_vehicle[package_obj] = vehicle_obj

        # 4. For each package with a goal, determine its current location and calculate cost
        for package, goal_location in self.package_goals.items():
            current_location = None
            is_in_vehicle = False

            if package in package_to_vehicle:
                is_in_vehicle = True
                vehicle = package_to_vehicle[package]
                if vehicle in object_locations:
                    current_location = object_locations[vehicle]
                else:
                    # Vehicle containing package is not located -> unreachable
                    logging.debug(f"Vehicle {vehicle} containing package {package} is not located in state.")
                    return float('inf')
            elif package in object_locations:
                # Package is not in a vehicle, check if it's 'at' a location
                current_location = object_locations[package]
            else:
                # Package is neither 'at' nor 'in' -> unreachable
                logging.debug(f"Package {package} not found at any location or in any vehicle in state.")
                return float('inf')

            # If package is already at goal, cost is 0 for this package
            if current_location == goal_location:
                continue

            # Calculate distance from current location to goal location
            # Ensure both locations are in the precomputed distances
            if current_location not in self.all_pairs_distances or goal_location not in self.all_pairs_distances.get(current_location, {}):
                 # This means the goal location is unreachable from the current location
                 logging.debug(f"Goal location {goal_location} unreachable from current location {current_location} for package {package}.")
                 return float('inf')

            distance = self.all_pairs_distances[current_location][goal_location]

            # If goal is unreachable, return infinity
            if distance == float('inf'):
                logging.debug(f"Goal location {goal_location} unreachable from current location {current_location} for package {package}.")
                return float('inf')

            # Add cost based on package's current state (at location or in vehicle)
            if is_in_vehicle:
                # Package is in a vehicle: needs drive + drop
                h_value += distance + 1
            else:
                # Package is at a location: needs pickup + drive + drop
                h_value += 1 + distance + 1

        return h_value
