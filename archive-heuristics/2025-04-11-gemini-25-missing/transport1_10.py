import itertools
from collections import deque
from fnmatch import fnmatch
# Assuming the heuristic base class is available at this path
# If the environment uses a different structure, adjust the import path accordingly.
from heuristics.heuristic_base import Heuristic

# Helper function to extract predicate and arguments from a PDDL fact string
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string like '(pred arg1 arg2)'."""
    return fact[1:-1].split()

# BFS implementation for shortest paths in unweighted graph
def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    to all other nodes in the graph.

    Args:
        graph (dict): Adjacency list representation of the graph {node: [neighbors]}.
        start_node: The node to start the search from.
        all_nodes (set): A set of all nodes expected in the graph.

    Returns:
        dict: A dictionary mapping each node to its shortest distance from start_node.
              Unreachable nodes will have a distance of float('inf').
    """
    distances = {node: float('inf') for node in all_nodes}
    # Check if start_node is valid and exists within the graph context
    if start_node not in all_nodes:
        # If start node itself isn't part of the defined nodes, return all inf
        # This case should ideally not happen if all_nodes is correctly populated.
        return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Use graph.get to safely access neighbors, defaulting to empty list if node has no edges
        for neighbor in graph.get(current_node, []):
            # Process neighbor only if it's a known node and not yet visited (dist is inf)
            if neighbor in distances and distances[neighbor] == float('inf'):
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances

# Function to compute all-pairs shortest paths
def compute_all_pairs_shortest_paths(graph, all_nodes):
    """
    Computes all-pairs shortest paths (APSP) for the given graph using BFS
    starting from each node.

    Args:
        graph (dict): Adjacency list representation of the graph.
        all_nodes (set): A set of all nodes in the graph.

    Returns:
        dict: A dictionary where keys are (start_node, end_node) tuples and
              values are the shortest distance (number of edges/drive actions)
              between them. Unreachable pairs have a distance of float('inf').
    """
    all_distances = {}
    for start_node in all_nodes:
        # Compute shortest distances from this start_node to all other nodes
        node_distances = bfs(graph, start_node, all_nodes)
        for end_node in all_nodes:
            # Store the computed distance, defaulting to inf if unreachable
            dist = node_distances.get(end_node, float('inf'))
            all_distances[(start_node, end_node)] = dist

    return all_distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It calculates the shortest path distance (number of drive actions)
    for necessary transport and adds the fixed costs for pick-up (1 action) and
    drop (1 action). The heuristic value is the sum of these estimated costs
    over all packages that are not yet in their target location. It is designed
    for greedy best-first search and is likely non-admissible.

    # Assumptions
    - The `(road l1 l2)` predicates define an unweighted graph of locations.
    - Roads are assumed to be bidirectional, meaning if there's a road from l1 to l2,
      there's also one from l2 to l1. The graph construction reflects this based on
      common PDDL domain patterns where both directions are listed if bidirectional.
    - The cost of moving an empty vehicle to a package's location *before* pick-up
      is *not* included in the estimate. This simplifies the calculation.
    - Vehicle capacity constraints (related to `capacity` and `capacity-predecessor`
      predicates) are ignored. The heuristic assumes any vehicle can pick up any
      package if needed, without capacity limits.
    - The graph of locations might be disconnected. If a package's goal location is
      unreachable from its current location (or the location of the vehicle carrying it),
      the heuristic returns infinity, signaling a potentially unsolvable state or path.

    # Heuristic Initialization (`__init__`)
    - Extracts all unique location names and package names from the task definition
      (using objects listed in initial state, static facts, and goals).
    - Parses static `(road ?l1 ?l2)` facts to build an adjacency list representation
      of the location connectivity graph.
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS)
      on the location graph. The result `self.distances[(l1, l2)]` stores the minimum
      number of `drive` actions required to travel between location `l1` and `l2`.
    - Parses goal conditions, specifically `(at ?p ?l)` facts, to identify the
      target location (`goal_loc`) for each package `p`. Stores these in
      `self.goal_locations`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  Initialize the total heuristic estimate `h = 0`.
    2.  Parse the current state (`node.state`) to build dictionaries mapping:
        - `package_locations`: package -> location, for packages currently at a location (`at p loc`).
        - `package_in_vehicle`: package -> vehicle, for packages currently inside a vehicle (`in p v`).
        - `vehicle_locations`: vehicle -> location, for all vehicles currently at a location (`at v loc`).
    3.  Iterate through each package `p` and its corresponding `goal_loc` stored in `self.goal_locations`.
    4.  For each package `p`:
        a. Check if the goal fact `(at p goal_loc)` is present in the current `state`.
           If yes, this package's goal is satisfied, so contribute 0 to `h` and proceed to the next package.
        b. Determine the package's current situation based on the parsed state:
           - **Case 1: Package `p` is at a location `current_loc`.**
             Found via `package_locations`. Calculate the shortest path distance `dist = self.distances[(current_loc, goal_loc)]`. If `dist` is infinity (unreachable), return `float('inf')`. Otherwise, the estimated cost for this package is `cost_p = 1 (pick-up) + dist (drive actions) + 1 (drop)`.
           - **Case 2: Package `p` is inside vehicle `v`.**
             Found via `package_in_vehicle`. Find the vehicle's location `vehicle_loc` using `vehicle_locations`. If `vehicle_loc` is not found (indicating an inconsistent state, e.g., vehicle exists but has no location fact), return `float('inf')`. Calculate `dist = self.distances[(vehicle_loc, goal_loc)]`. If `dist` is infinity, return `float('inf')`. Otherwise, the estimated cost for this package is `cost_p = dist (drive actions) + 1 (drop)`.
           - **Case 3: Package `p`'s state is unknown** (neither `at` a location nor `in` a vehicle).
             This indicates an invalid or unexpected state according to the domain model; return `float('inf')`.
        c. Add the calculated `cost_p` for the current package to the total heuristic value `h`.
    5.  After checking all packages, return the total heuristic value `h`.
        - `h = 0` if and only if all package goals `(at p goal_loc)` are satisfied in the state.
        - `h = float('inf')` if any goal is found to be unreachable from the package's current state.
        - `h > 0` otherwise, providing a non-negative estimate of the remaining actions.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task information:
        - Extracts locations and packages.
        - Builds the road network graph.
        - Computes all-pairs shortest paths (APSP) for driving distances.
        - Stores package goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract all unique location and package names
        self.locations = set()
        self.packages = set()
        # Use initial state, static facts, and goals to find all objects and infer types
        all_facts = task.initial_state | static_facts | task.goals
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             args = parts[1:]

             # Infer types based on predicate structure from domain file
             if pred == 'road' and len(args) == 2: # (road ?l1 ?l2 - location)
                 self.locations.add(args[0])
                 self.locations.add(args[1])
             elif pred == 'at' and len(args) == 2: # (at ?x - locatable ?v - location)
                 # ?v is location
                 self.locations.add(args[1])
                 # We identify packages more reliably from 'in' predicates or goals
             elif pred == 'in' and len(args) == 2: # (in ?x - package ?v - vehicle)
                 # ?x is package
                 self.packages.add(args[0])
             # Add more specific type inference if needed (e.g., from capacity predicates for vehicles)

        # Refine package identification and find goal locations
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Focus on '(at package location)' goals
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Assume the first argument of 'at' in a goal is a package
                self.packages.add(package)
                self.goal_locations[package] = location
                # Ensure goal location is registered as a location
                self.locations.add(location)

        # 2. Build adjacency list for the location graph
        adj = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edges. Assuming bidirectionality: add edge in both directions.
                # If domain guarantees both (road l1 l2) and (road l2 l1) for bidirectional roads,
                # this might add duplicate neighbors, but list append is okay.
                # If only one direction is listed per road, this correctly makes it bidirectional.
                if l1 in adj: adj[l1].append(l2)
                if l2 in adj: adj[l2].append(l1)

        # 3. Compute all-pairs shortest paths
        # Pass the complete set of known locations to ensure BFS covers all nodes
        self.distances = compute_all_pairs_shortest_paths(adj, self.locations)


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the actions needed to reach the goal state.
        """
        state = node.state
        heuristic_value = 0

        # 1. Parse current state to find locations of packages and vehicles
        package_locations = {}  # package -> location (if 'at')
        package_in_vehicle = {} # package -> vehicle (if 'in')
        vehicle_locations = {}  # vehicle -> location (if 'at')

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    package_locations[obj] = loc
                else:
                    # Assume non-package object with 'at' predicate is a vehicle
                    vehicle_locations[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args[0], args[1]
                # Ensure the object 'in' a vehicle is a known package
                if package in self.packages:
                    package_in_vehicle[package] = vehicle
                # else: Log warning? Might be objects other than packages in vehicles if domain changes.

        # 2. Calculate cost for each package goal not yet satisfied
        for package, goal_loc in self.goal_locations.items():
            cost_p = 0

            # Check if goal is already satisfied in the current state
            # Construct the goal fact string for efficient lookup in the state set
            goal_fact_str = f"(at {package} {goal_loc})"
            if goal_fact_str in state:
                continue  # This package is already at its goal location

            # Determine the current situation of the package
            current_loc_direct = package_locations.get(package)
            vehicle_carrying = package_in_vehicle.get(package)

            if current_loc_direct is not None:
                # Case 1: Package is at location 'current_loc_direct'
                current_loc = current_loc_direct
                # Retrieve precomputed shortest path distance
                # Use .get for safety, though keys should exist if APSP was computed correctly
                dist = self.distances.get((current_loc, goal_loc), float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from the package's current location
                    return float('inf')
                # Estimated cost: pick-up(1) + drive(dist) + drop(1)
                cost_p = 1 + dist + 1

            elif vehicle_carrying is not None:
                # Case 2: Package is in vehicle 'vehicle_carrying'
                vehicle = vehicle_carrying
                vehicle_loc = vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                    # The location of the vehicle carrying the package is unknown.
                    # This implies an inconsistent or partially specified state.
                    # Cannot estimate cost reliably. Return infinity.
                    # print(f"Warning: Location of vehicle {vehicle} carrying {package} unknown in state {state}")
                    return float('inf')

                # Retrieve precomputed shortest path distance from vehicle's location
                dist = self.distances.get((vehicle_loc, goal_loc), float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from the vehicle's current location
                    return float('inf')
                # Estimated cost: drive(dist) + drop(1)
                cost_p = dist + 1
            else:
                # Case 3: Package's state is unknown (not 'at' location, not 'in' vehicle)
                # This indicates an invalid state according to the domain model,
                # or the package might not exist in the initial state (unlikely for goal packages).
                # print(f"Warning: State of package {package} is undetermined in heuristic calculation.")
                return float('inf')

            # Add the cost for this package to the total heuristic value
            heuristic_value += cost_p

        # Return the total estimated cost for all unsatisfied package goals
        return heuristic_value
