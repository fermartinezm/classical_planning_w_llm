from fnmatch import fnmatch
from collections import deque
# from heuristics.heuristic_base import Heuristic # Assuming base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic: # Inherit from Heuristic if base class is available
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost to reach the goal by summing the minimum actions
    required for each package to reach its goal location, ignoring
    vehicle capacity and coordination.

    Cost per package not at its goal:
    - If at location L (L != Goal): 1 (pick-up) + dist(L, Goal) (drives) + 1 (drop)
    - If in vehicle V at location L_v (L_v != Goal): dist(L_v, Goal) (drives) + 1 (drop)
    - If in vehicle V at location L_v (L_v == Goal): 1 (drop)

    Shortest path distances are precomputed using BFS on the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, identifying
        packages and vehicles, building the road network, and precomputing
        shortest path distances.
        """
        self.goal_locations = {}
        self.packages = set()
        self.vehicles = set()
        locations = set()

        # Extract goals and identify packages/locations
        for goal in task.goals:
            if match(goal, "at", "*", "*"):
                 _, package, location = get_parts(goal)
                 self.goal_locations[package] = location
                 self.packages.add(package) # Packages are objects in goal 'at'
                 locations.add(location)

        # Extract static info: roads, capacity (to identify vehicles)
        self.road_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "road":
                _, l1, l2 = parts
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional
            elif predicate == "capacity":
                 _, vehicle, size = parts
                 self.vehicles.add(vehicle) # Vehicles are objects in 'capacity'
            # capacity-predecessor is not needed

        # Identify objects and locations from initial state
        # A locatable object is anything that appears as the first argument of 'at' or 'in'.
        # A vehicle is anything that appears in a 'capacity' fact.
        # A package is any locatable object that is not a vehicle.
        all_locatable_objects = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "at":
                 _, obj, loc = parts
                 all_locatable_objects.add(obj)
                 locations.add(loc)
             elif predicate == "in":
                 _, package, vehicle = parts
                 all_locatable_objects.add(package)
                 all_locatable_objects.add(vehicle) # Vehicle is also a locatable object
             elif predicate == "capacity":
                 _, vehicle, size = parts
                 self.vehicles.add(vehicle) # Ensure vehicles set is populated from init too

        # Populate packages set: any locatable object not identified as a vehicle
        for obj in all_locatable_objects:
             if obj not in self.vehicles:
                 self.packages.add(obj)

        # Ensure all locations are keys in the graph, even if isolated
        for loc in locations:
             self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in self.road_graph:
            distances = self._bfs(self.road_graph, start_loc)
            for end_loc, dist in distances.items():
                if dist != float('inf'):
                    self.shortest_paths[(start_loc, end_loc)] = dist
                # If dist is inf, it means unreachable, which is handled by get() later.

    def _bfs(self, graph, start_node):
        """Performs BFS to find shortest distances from start_node."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node might not be in graph keys if it's an isolated location
             # mentioned only in 'at' facts, not 'road' facts.
             # In this case, it's unreachable from anywhere else, and nothing is reachable from it (except itself).
             # Distances remain inf, except for start_node itself if it was a valid location.
             if start_node in distances: # Check if it was added to locations set
                 distances[start_node] = 0
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node has neighbors in the graph (it should if it's a key)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        package_status = {} # package -> ('at', loc) or ('in', vehicle)
        vehicle_locations = {} # vehicle -> loc

        # Populate current status and locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                _, obj, loc = parts
                if obj in self.packages:
                    package_status[obj] = ('at', loc)
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
                # else: print(f"Warning: 'at' fact with unknown object type: {fact}")
            elif predicate == "in":
                 _, package, vehicle = parts
                 # Ensure package and vehicle are known types, though they should be from init/goals
                 if package in self.packages and vehicle in self.vehicles:
                     package_status[package] = ('in', vehicle)
                 # else: print(f"Warning: 'in' fact with unknown object types: {fact}")
            # Ignore capacity facts for heuristic calculation

        total_cost = 0

        # Iterate through packages that have a goal
        for package, goal_location in self.goal_locations.items():
            # If a package is a goal but doesn't appear in the state, it's unreachable.
            if package not in package_status:
                 return float('inf')

            current_status = package_status[package]

            if current_status[0] == 'at':
                current_location = current_status[1]
                if current_location != goal_location:
                    # Needs pick-up (1) + drive (dist) + drop (1)
                    dist = self.shortest_paths.get((current_location, goal_location), float('inf'))
                    if dist == float('inf'): return float('inf') # Goal unreachable for this package
                    total_cost += 1 + dist + 1
            elif current_status[0] == 'in':
                vehicle = current_status[1]
                # Vehicle carrying package must be at a location
                if vehicle not in vehicle_locations:
                    # Vehicle carrying package is not at any location? Unreachable.
                    return float('inf')

                vehicle_location = vehicle_locations[vehicle]

                if vehicle_location != goal_location:
                    # Needs drive (dist) + drop (1)
                    dist = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))
                    if dist == float('inf'): return float('inf') # Goal unreachable for this package
                    total_cost += dist + 1
                else: # vehicle_location == goal_location
                    # Needs drop (1)
                    total_cost += 1

        return total_cost
