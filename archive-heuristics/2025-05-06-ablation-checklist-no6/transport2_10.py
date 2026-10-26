# Required imports
from collections import deque
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS helper function
def bfs(graph, start):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    in a graph represented as an adjacency list.
    """
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # If the start node is not in the graph (e.g., an isolated location),
    # its distance to itself is 0, and to all others remains infinity.
    if start not in distances:
        distances[start] = 0
        return distances # No neighbors to explore

    distances[start] = 0
    queue = deque([start])

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        # Check if current node has neighbors in the graph
        if current in graph:
            for neighbor in graph[current]:
                # Use get with default float('inf') for safety, although all nodes
                # in the graph should be in the distances dict initialized above.
                if distances.get(neighbor, float('inf')) == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


# Define the heuristic class, inheriting from Heuristic base class
# class transportHeuristic(Heuristic): # Uncomment this line in the actual environment
class transportHeuristic: # Use this for standalone code generation
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost by summing the minimum actions required for each package
    to reach its goal location independently, ignoring vehicle capacity and
    shared vehicle trips.

    Cost per package:
    - If at goal location (on ground): 0
    - If in vehicle at goal location: 1 (drop)
    - If on ground not at goal: 1 (pick-up) + distance(current, goal) + 1 (drop)
    - If in vehicle not at goal: distance(vehicle_location, goal) + 1 (drop)
    """

    def __init__(self, task):
        """
        Initializes the heuristic by building the road network graph,
        computing all-pairs shortest paths, and extracting package goal locations.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Build road graph and identify all locations mentioned in road facts
        self.road_graph = {}
        road_locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                road_locations.add(l1)
                road_locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                # Assuming bidirectional roads based on example
                self.road_graph.setdefault(l2, []).append(l1)

        # Ensure all locations from road facts are nodes in the graph dict, even if isolated
        for loc in road_locations:
             self.road_graph.setdefault(loc, []) # Add node with empty list if not already a key

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Iterate through all locations identified from road facts
        for start_loc in road_locations:
            distances_from_start = bfs(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                self.distances[(start_loc, end_loc)] = dist

        # 3. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other goal types if any (transport domain only has 'at' goals for packages)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # Track current status of packages (location string or vehicle name string)
        package_status = {}
        # Track current location of vehicles (location string)
        vehicle_location_map = {}

        # Infer types and populate status/location maps
        # Objects appearing as 2nd arg of 'in' or 1st of 'capacity' are vehicles
        # Objects appearing as 1st arg of 'in' or 1st of 'at' in goals are packages
        vehicles_in_state_or_goals = set()
        packages_in_state_or_goals = set()

        # Infer from state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                packages_in_state_or_goals.add(pkg)
                vehicles_in_state_or_goals.add(veh)
            elif parts[0] == 'capacity':
                 veh, size = parts[1], parts[2]
                 vehicles_in_state_or_goals.add(veh)

        # Infer from goals
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                  packages_in_state_or_goals.add(parts[1]) # Object in 'at' goal is a package

        # Populate status and location maps based on inferred types
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in vehicles_in_state_or_goals:
                    vehicle_location_map[obj] = loc
                elif obj in packages_in_state_or_goals:
                    package_status[obj] = loc
                # Ignore other 'at' facts for objects not identified as packages/vehicles we care about
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                if pkg in packages_in_state_or_goals:
                    package_status[pkg] = veh # Status is the vehicle it's in
                # Ignore 'in' facts for objects not identified as packages we care about


        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_status = package_status.get(package)

            # If package is not in the state facts at all, assume it's already handled (cost 0)
            # This might happen if a package is delivered and its facts are removed,
            # or if the state representation is minimal.
            # However, standard planners usually keep all objects/facts unless deleted.
            # Let's assume goal objects are always present in state facts until goal is met.
            if current_status is None:
                 # Check if the goal fact is directly in the state.
                 if f'(at {package} {goal_location})' in state:
                      continue # Package is at goal, cost 0
                 else:
                      # Package from goal is not in state and goal fact is not in state.
                      # This state is likely invalid or represents something outside the
                      # expected PDDL state representation. Treat as unsolvable.
                      return float('inf')


            # Case 1: Package is on the ground
            # If current_status is not a known vehicle, assume it's a location string.
            if current_status not in vehicles_in_state_or_goals:
                 current_location = current_status
                 if current_location == goal_location:
                     continue # Cost is 0 (already at goal)
                 else:
                     # Get distance from current location to goal location
                     # If current_location or goal_location are not in the road network,
                     # the distance will be inf (handled by get default).
                     dist = self.distances.get((current_location, goal_location), float('inf'))

                     if dist == float('inf'):
                         # Unreachable goal location from current location via road network
                         return float('inf')

                     # The cost is pick-up + drive + drop
                     total_cost += 1 + dist + 1

            # Case 2: Package is in a vehicle
            elif current_status in vehicles_in_state_or_goals: # Check if status is a vehicle name
                 vehicle = current_status
                 vehicle_location = vehicle_location_map.get(vehicle)

                 if vehicle_location is None:
                     # Vehicle exists but its location is unknown? Invalid state.
                     return float('inf')

                 if vehicle_location == goal_location:
                     # Package is in vehicle, vehicle is at goal location
                     total_cost += 1 # Need 1 action: drop
                 else:
                     # Package is in vehicle, vehicle is not at goal location
                     # Get distance from vehicle location to goal location
                     # If vehicle_location or goal_location are not in the road network,
                     # the distance will be inf (handled by get default).
                     dist = self.distances.get((vehicle_location, goal_location), float('inf'))

                     if dist == float('inf'):
                         # Unreachable goal location from vehicle location via road network
                         return float('inf')

                     # The cost is drive + drop
                     total_cost += dist + 1
            else:
                 # current_status is neither a known location nor a known vehicle.
                 # This indicates an unexpected state fact format or object type.
                 # Treat as unsolvable or error.
                 return float('inf')

        return total_cost
