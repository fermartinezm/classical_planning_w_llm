# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import defaultdict, deque

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Check if fact is a string and starts/ends with parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return empty list for invalid fact formats
    return []

# Utility function to match PDDL facts (similar to examples)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def build_road_graph(static_facts):
    """
    Builds an adjacency list representation of the road network graph
    from static (road l1 l2) facts.
    """
    graph = defaultdict(set)
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'road':
            l1, l2 = parts[1], parts[2]
            graph[l1].add(l2)
            # Note: Assumes roads are directed as specified in facts.
            # If roads are implicitly bidirectional, the PDDL should contain
            # both (road l1 l2) and (road l2 l1). The example static facts
            # show pairs, supporting this assumption.
    return graph

def compute_shortest_paths(graph):
    """
    Computes all-pairs shortest paths in the road graph using BFS.
    Returns a dictionary where shortest_paths[l1][l2] is the distance
    from l1 to l2. Unreachable locations have distance float('inf').
    """
    # Collect all unique locations from the graph
    locations = set(graph.keys())
    for neighbors in graph.values():
        locations.update(neighbors)
    locations = sorted(list(locations)) # Ensure consistent order

    shortest_paths = {}
    for start_node in locations:
        shortest_paths[start_node] = {}
        # BFS from start_node
        queue = deque([(start_node, 0)])
        visited = {start_node}
        shortest_paths[start_node][start_node] = 0

        while queue:
            current_node, dist = queue.popleft()

            # Check if current_node exists in the graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        shortest_paths[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    # Fill in unreachable paths with infinity
    for l1 in locations:
        if l1 not in shortest_paths:
             shortest_paths[l1] = {} # Should not happen if locations are collected correctly
        for l2 in locations:
             if l2 not in shortest_paths[l1]:
                 shortest_paths[l1][l2] = float('inf') # Indicate unreachable

    return shortest_paths


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity
    and availability constraints. It sums the minimum actions needed for each
    package independently.

    # Assumptions
    - Unit cost for all actions (drive, pick-up, drop).
    - Vehicles are always available when needed to pick up or drop a package.
    - Vehicles have sufficient capacity to carry packages when needed.
    - The road network is static and provides shortest paths for driving.
    - All locations involved in the problem (initial state, goals, roads) are connected or paths are well-defined.
    - Packages are either on the ground at a location or inside a vehicle. Vehicles are always at a location.

    # Heuristic Initialization
    - Parses the goal conditions to create a mapping from each package to its goal location.
    - Identifies all packages that are part of the goal.
    - Parses the static `road` facts to build a graph representing the road network.
    - Computes all-pairs shortest paths between all locations using Breadth-First Search (BFS) on the road graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost `h` to 0.
    2. Create quick lookups for facts in the current state by iterating through the state facts once:
       - `packages_in_vehicles`: Map package names to the vehicle name they are currently inside (only for packages relevant to the goal).
       - `all_objects_at_location`: Map object names (packages or vehicles) to the location name they are currently at on the ground.
    3. For each package `p` that is listed in the problem's goals (obtained during initialization):
    4. Get the goal location `goal_l` for package `p` from the precomputed mapping.
    5. Determine the current status of package `p` using the lookups created in step 2:
       - Check if `p` is currently inside a vehicle by looking it up in `packages_in_vehicles`.
       - Check if `p` is currently on the ground at a location by looking it up in `all_objects_at_location`.
    6. If the package `p` is currently on the ground at its goal location `goal_l` (i.e., `all_objects_at_location.get(p) == goal_l`):
       - This package has reached its goal location and requires 0 further actions. Continue to the next package.
    7. If the package `p` is currently inside a vehicle `v` (i.e., `p` is a key in `packages_in_vehicles`):
       - Get the vehicle name `v` from `packages_in_vehicles[p]`.
       - Find the current location `vehicle_l` of vehicle `v` by looking it up in `all_objects_at_location`.
       - If the vehicle's location `vehicle_l` is not found (which indicates an invalid state), treat this package's goal as unreachable or skip it.
       - Otherwise, the package needs to be transported from `vehicle_l` to `goal_l` inside the vehicle. The minimum number of drive actions required is the shortest path distance between `vehicle_l` and `goal_l`, obtained from the precomputed `self.shortest_paths`.
       - The package then needs to be dropped at `goal_l` (1 action).
       - Add `shortest_path_distance(vehicle_l, goal_l) + 1` to the total heuristic cost `h`.
    8. If the package `p` is currently on the ground at a location `current_l` (i.e., `p` is a key in `all_objects_at_location`, and `current_l` is not `goal_l`, as handled in step 6):
       - The package needs to be picked up (1 action).
       - The package needs to be transported from `current_l` to `goal_l`. The minimum number of drive actions required is the shortest path distance between `current_l` and `goal_l`, obtained from `self.shortest_paths`.
       - The package then needs to be dropped at `goal_l` (1 action).
       - Add `1 + shortest_path_distance(current_l, goal_l) + 1` to the total heuristic cost `h`.
    9. If the package's status is not found (neither on ground nor in vehicle), it indicates an invalid state or a package not relevant to the goal, and it is skipped (implicitly adding 0 cost). We assume packages in goals are always in a valid state representation.
    10. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # We also need a list of all packages involved in goals
        self.packages_in_goals = set(self.goal_locations.keys())

        # Build road graph and compute shortest paths
        road_graph = build_road_graph(static_facts)
        self.shortest_paths = compute_shortest_paths(road_graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Create quick lookups for facts in the current state
        packages_in_vehicles = {} # Map package -> vehicle
        all_objects_at_location = {} # Map object -> location for all 'at' facts

        for fact in state: # state is a frozenset of strings
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "at":
                  obj, location = parts[1], parts[2]
                  all_objects_at_location[obj] = location
             elif predicate == "in":
                  package, vehicle = parts[1], parts[2]
                  # Only track packages that are in the goals
                  if package in self.packages_in_goals:
                       packages_in_vehicles[package] = vehicle

        # Now process packages in goals
        total_cost = 0

        for package in self.packages_in_goals:
            goal_location = self.goal_locations[package]

            # Check current status of the package
            current_location_if_on_ground = all_objects_at_location.get(package)
            is_in_vehicle = package in packages_in_vehicles

            # Case 1: Package is already at the goal location (on the ground)
            if current_location_if_on_ground == goal_location:
                 continue # Cost is 0 for this package

            # Case 2: Package is in a vehicle
            elif is_in_vehicle:
                 vehicle_name = packages_in_vehicles[package]
                 vehicle_location = all_objects_at_location.get(vehicle_name) # Get vehicle's location

                 if vehicle_location is None:
                     # Vehicle location unknown - indicates invalid state or unreachable goal
                     # print(f"Warning: Location of vehicle {vehicle_name} carrying {package} not found.")
                     total_cost += float('inf') # Add infinity cost for unreachable goal
                     continue # Skip this package if its vehicle's location is unknown

                 current_location_for_distance = vehicle_location
                 # Cost: drive from vehicle_location to goal_location + drop
                 dist = self.shortest_paths.get(current_location_for_distance, {}).get(goal_location, float('inf'))
                 cost_for_package = dist + 1

            # Case 3: Package is on the ground (and not at the goal, handled in Case 1)
            elif current_location_if_on_ground is not None:
                 current_location_for_distance = current_location_if_on_ground

                 # Cost: pick-up + drive from current_location to goal_location + drop
                 dist = self.shortest_paths.get(current_location_for_distance, {}).get(goal_location, float('inf'))
                 cost_for_package = 1 + dist + 1 # = 2 + dist

            else:
                 # Case 4: Package exists (in goals) but is neither 'at' a location nor 'in' a vehicle.
                 # This indicates an invalid state or a package not relevant to the goal.
                 # print(f"Warning: Status of package {package} unknown in state.")
                 total_cost += float('inf') # Add infinity cost for unreachable goal
                 continue # Skip this package

            # Add cost for this package
            total_cost += cost_for_package

        return total_cost
