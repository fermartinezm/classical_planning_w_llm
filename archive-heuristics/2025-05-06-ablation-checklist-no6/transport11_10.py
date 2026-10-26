from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict
import math

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function
def bfs_shortest_path(graph, start):
    """
    Computes shortest path distances from a start node to all reachable nodes
    in a graph using BFS.

    Args:
        graph: Adjacency list representation of the graph (dict: node -> list of neighbors).
        start: The starting node.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from start.
    """
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in graph: # Handle nodes that might be in locations list but not graph (e.g., isolated)
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal location.
    It sums the estimated costs for each package independently.
    The cost for a package is estimated based on its current state (on ground or in vehicle)
    and the shortest road distance from its current location (or vehicle's location)
    to its goal location.

    Cost breakdown per package not at goal location on the ground:
    - If on ground at current_l (current_l != goal_l):
      Pick-up (1) + Drive (shortest_path(current_l, goal_l)) + Drop (1) = 2 + shortest_path
    - If in vehicle v at vehicle_l (vehicle_l != goal_l):
      Drive (shortest_path(vehicle_l, goal_l)) + Drop (1) = 1 + shortest_path
    - If in vehicle v at goal_l:
      Drop (1) = 1

    This heuristic ignores vehicle availability and capacity constraints beyond
    the package's current state (in/out of vehicle).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, road network,
        object types, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.goal_locations = {}
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()
        locatables = set() # Temporary set for objects that are 'at' a location

        road_graph = defaultdict(list)

        # 1. Process goals to find packages and goal locations
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)
                self.locations.add(location) # Add goal locations

        # 2. Process static facts to find locations, sizes, and build road graph
        for fact in static_facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "road":
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    road_graph[l1].append(l2)
                    road_graph[l2].append(l1) # Assuming bidirectional roads
                elif parts[0] == "capacity-predecessor":
                    s1, s2 = parts[1], parts[2]
                    self.sizes.add(s1)
                    self.sizes.add(s2)
                # Add other static predicates if needed for object identification?
                # The domain only has road and capacity-predecessor as static.

        # 3. Process initial state to find vehicles, locations, sizes, and locatables
        for fact in initial_state:
             parts = get_parts(fact)
             if parts:
                 if parts[0] == "at":
                     obj, loc = parts[1], parts[2]
                     locatables.add(obj) # Could be package or vehicle
                     self.locations.add(loc)
                 elif parts[0] == "in":
                     package, vehicle = parts[1], parts[2]
                     self.packages.add(package) # Ensure packages in initial state are included
                     self.vehicles.add(vehicle)
                 elif parts[0] == "capacity":
                     vehicle, size = parts[1], parts[2]
                     self.vehicles.add(vehicle)
                     self.sizes.add(size)

        # 4. Identify vehicles from locatables that are not packages
        for obj in locatables:
            if obj not in self.packages:
                self.vehicles.add(obj)

        # Ensure all identified locations are in the graph keys, even if isolated
        for loc in self.locations:
             road_graph.setdefault(loc, [])

        # 5. Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in self.locations:
            distances = bfs_shortest_path(road_graph, start_loc)
            for end_loc, dist in distances.items():
                self.shortest_paths[(start_loc, end_loc)] = dist

        # Note: Unreachable locations will not have an entry in self.shortest_paths.
        # We handle this in __call__ by assigning infinity.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current state of packages and vehicles
        package_current_state = {} # Maps package -> location or vehicle
        vehicle_current_location = {} # Maps vehicle -> location

        # Populate current state information from the node's state facts
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "at":
                    obj, loc = parts[1], parts[2]
                    if obj in self.packages:
                        package_current_state[obj] = loc
                    elif obj in self.vehicles: # Explicitly check if it's a known vehicle
                         vehicle_current_location[obj] = loc
                    # Ignore 'at' facts for other types if any

                elif parts[0] == "in":
                    package, vehicle = parts[1], parts[2]
                    if package in self.packages and vehicle in self.vehicles: # Ensure known objects
                        package_current_state[package] = vehicle
                    # Ignore 'in' facts for other types if any

                # Ignore capacity facts for heuristic calculation in __call__

        total_cost = 0

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state (e.g., not 'at' or 'in'), something is wrong.
            # Assuming valid states where packages are always located or in a vehicle.
            if package not in package_current_state:
                 # This case should ideally not happen in a valid state space exploration
                 # print(f"Warning: Package {package} not found in state {state}")
                 # Assign infinity cost as it's likely an invalid or terminal state not leading to goal.
                 return math.inf # Unsolvable from this state


            current_pos = package_current_state[package]

            # Case 1: Package is on the ground (current_pos is a location)
            if current_pos in self.locations:
                current_location = current_pos
                # If package is at its goal location on the ground, cost is 0 for this package
                if current_location == goal_location:
                    continue # Cost is 0 for this package

                # Package is on the ground at a different location
                # Needs Pick-up, Drive, Drop
                # Estimated actions: 1 (pick) + shortest_path (drive) + 1 (drop)
                drive_cost = self.shortest_paths.get((current_location, goal_location), math.inf)

                if drive_cost == math.inf:
                    # Cannot reach goal location by driving from current location
                    return math.inf # Unsolvable from this state
                else:
                    total_cost += 1 + drive_cost + 1 # Pick-up + Drive + Drop

            # Case 2: Package is in a vehicle (current_pos is a vehicle name)
            elif current_pos in self.vehicles:
                vehicle = current_pos
                if vehicle not in vehicle_current_location:
                    # Vehicle carrying package is not at any location (invalid state?)
                    # print(f"Warning: Vehicle {vehicle} carrying {package} not found at any location in state {state}")
                    return math.inf # Unsolvable from this state

                vehicle_location = vehicle_current_location[vehicle]

                # If vehicle is at the package's goal location
                if vehicle_location == goal_location:
                    # Needs Drop
                    total_cost += 1
                else:
                    # Vehicle needs to drive to goal location, then drop
                    # Estimated actions: shortest_path (drive) + 1 (drop)
                    drive_cost = self.shortest_paths.get((vehicle_location, goal_location), math.inf)

                    if drive_cost == math.inf:
                         # Cannot reach goal location by driving from vehicle's location
                         return math.inf # Unsolvable from this state
                    else:
                         total_cost += drive_cost + 1 # Drive + Drop
            # else: current_pos is neither a location nor a vehicle? Invalid state?
            # print(f"Warning: Package {package} has unknown state {current_pos} in state {state}")
            # return math.inf # Unsolvable from this state


        return total_cost
