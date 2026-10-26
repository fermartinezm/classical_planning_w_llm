# Helper functions for parsing PDDL facts
from fnmatch import fnmatch
from collections import deque
import sys # Import sys for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return [] # Or handle error appropriately
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper functions for graph traversal (BFS)
def bfs(graph, start):
    """Computes shortest path distances from start node in a graph."""
    distances = {start: 0}
    queue = deque([start])
    # Ensure start node is in the graph keys if it exists
    if start not in graph:
        # If start is not in the graph (e.g., isolated location), it can only reach itself.
        # The distances dict is already {start: 0}.
        pass # No neighbors to explore
    else:
        while queue:
            current = queue.popleft()
            dist = distances[current]
            # Check if current node has neighbors in the graph
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of nodes."""
    all_distances = {}
    # Collect all unique locations mentioned in the graph (both keys and values)
    locations = set(graph.keys())
    for neighbors in graph.values():
        locations.update(neighbors)

    # Ensure all locations are in the graph keys, even if they have no roads
    for loc in locations:
        graph.setdefault(loc, set())

    # Compute BFS from every location
    for start_node in locations:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances

# Import the base heuristic class
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each package
    to satisfy its goal location predicate `(at package location)`. It sums the estimated
    costs for each package independently, ignoring vehicle capacity constraints and
    vehicle availability for loading.

    # Assumptions
    - The goal for each package is to be on the ground at a specific location `(at package location)`.
    - A package can be on the ground at a location or inside a vehicle.
    - The cost for a package involves loading (if on the ground), driving the vehicle it's in, and unloading.
    - Driving cost is the shortest path distance in the road network.
    - Vehicle capacity and availability are ignored in the cost calculation for simplicity.
    - Package size is assumed to be 'c0' unless specified otherwise (not specified in examples).
      Vehicle capacity hierarchy is parsed but not strictly enforced for individual package costs.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build the road network graph from static facts.
    - Compute all-pairs shortest path distances on the road network.
    - Extract vehicle capacities and the capacity predecessor hierarchy (parsed but not used
      in the core cost calculation per package in this simplified version).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status of every package:
       - Is it on the ground at a location `(at p l)`? If so, what is `l`?
       - Is it inside a vehicle `(in p v)`? If so, what is `v`, and where is `v` `(at v l)`?
    2. For each package `p` with goal location `goal_l`:
       a. Check if the goal `(at p goal_l)` is already satisfied in the current state. If yes, cost for `p` is 0.
       b. If the goal is not satisfied:
          i. If `p` is on the ground at `current_l` (`(at p current_l)` is true, `current_l != goal_l`):
             - It needs to be loaded (1 action).
             - It needs to be transported from `current_l` to `goal_l`. Minimum drive actions = `dist(current_l, goal_l)`.
             - It needs to be unloaded at `goal_l` (1 action).
             - Estimated cost for `p` = 1 (load) + `dist(current_l, goal_l)` + 1 (unload).
          ii. If `p` is inside vehicle `v` which is at `vehicle_l` (`(in p v)` and `(at v vehicle_l)` are true):
             - If `vehicle_l == goal_l`:
                - It needs to be unloaded at `goal_l` (1 action).
                - Estimated cost for `p` = 1 (unload).
             - If `vehicle_l != goal_l`:
                - It needs to be transported from `vehicle_l` to `goal_l`. Minimum drive actions = `dist(vehicle_l, goal_l)`.
                - It needs to be unloaded at `goal_l` (1 action).
                - Estimated cost for `p` = `dist(vehicle_l, goal_l)` + 1 (unload).
    3. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically '(at package location)'
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # 2. Build road network and compute distances
        self.road_network = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.road_network.setdefault(loc1, set()).add(loc2)
                self.road_network.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all locations mentioned in goals are in the network, even if isolated
        for loc in self.goal_locations.values():
             locations.add(loc)
             self.road_network.setdefault(loc, set()) # Add location node if not present

        # Ensure all locations mentioned in road facts are keys in the network dict
        for loc in locations:
             self.road_network.setdefault(loc, set())

        self.distances = compute_all_pairs_shortest_paths(self.road_network)

        # 3. Extract vehicle capacities and capacity hierarchy (parsed but not used in cost calculation)
        self.vehicle_capacities = {}
        self.capacity_predecessors = set() # Store (smaller, larger) pairs
        self.capacity_successors = {} # Store {smaller: {larger, ...}}
        self.sizes = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "capacity" and len(parts) == 3:
                    vehicle, size = parts[1], parts[2]
                    self.vehicle_capacities[vehicle] = size
                    self.sizes.add(size)
                elif parts[0] == "capacity-predecessor" and len(parts) == 3:
                    s1, s2 = parts[1], parts[2]
                    self.capacity_predecessors.add((s1, s2))
                    self.capacity_successors.setdefault(s1, set()).add(s2)
                    self.sizes.add(s1)
                    self.sizes.add(s2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of packages and vehicles
        package_status = {} # Maps package -> {'type': 'at' or 'in', 'obj': location or vehicle}
        vehicle_locations = {} # Maps vehicle -> location

        # First pass: find vehicle locations and identify vehicles
        vehicles = set(self.vehicle_capacities.keys())
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in vehicles: # Check if obj is a known vehicle
                      vehicle_locations[obj] = loc

        # Second pass: find package status and identify packages
        packages = set(self.goal_locations.keys())
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "at" and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in packages: # Check if obj is a known package
                         package_status[obj] = {'type': 'at', 'obj': loc}
                elif parts[0] == "in" and len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    if package in packages and vehicle in vehicles: # Check if package and vehicle are known
                         package_status[package] = {'type': 'in', 'obj': vehicle}

        total_cost = 0

        # Handle packages that need to reach their goal
        for package, goal_location in self.goal_locations.items():
            # Ensure package status is known (it should be if it's in the goal list)
            if package not in package_status:
                 # This indicates an unexpected state where a goal package is not located.
                 # This shouldn't happen in valid states reachable from the initial state.
                 # Returning infinity indicates this path is likely invalid or unsolvable.
                 return float('inf')

            status = package_status[package]

            # Check if the goal (at package goal_location) is already satisfied
            goal_fact = f"(at {package} {goal_location})"
            if goal_fact in state:
                 # Package is already on the ground at the goal location
                 continue # Cost is 0 for this package

            # Goal is not satisfied, calculate cost
            package_cost = 0

            if status['type'] == 'at': # Package is on the ground at status['obj']
                current_location = status['obj']
                # Needs load (1) + drive + unload (1)
                package_cost += 1 # Load

                # Drive cost from current_location to goal_location
                drive_cost = 0
                # Ensure locations exist in the distance map
                if current_location in self.distances and goal_location in self.distances.get(current_location, {}):
                     drive_cost = self.distances[current_location][goal_location]
                else:
                     # Locations are disconnected or not in the graph - problem is likely unsolvable
                     return float('inf')

                package_cost += drive_cost
                package_cost += 1 # Unload

            elif status['type'] == 'in': # Package is inside vehicle status['obj']
                vehicle = status['obj']
                # Find vehicle's location
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                     # Vehicle location unknown - inconsistent state or vehicle not at any location
                     # Problem likely unsolvable from this state.
                     return float('inf')

                # Needs drive + unload (1)
                # Drive cost from vehicle_location to goal_location
                drive_cost = 0
                if vehicle_location != goal_location:
                     # Ensure locations exist in the distance map
                     if vehicle_location in self.distances and goal_location in self.distances.get(vehicle_location, {}):
                          drive_cost = self.distances[vehicle_location][goal_location]
                     else:
                          # Locations are disconnected - problem likely unsolvable
                          return float('inf')

                package_cost += drive_cost
                package_cost += 1 # Unload

            total_cost += package_cost

        # The heuristic is 0 iff all goal facts (at p goal_l) are true, as checked at the start of the loop for each package.
        return total_cost
