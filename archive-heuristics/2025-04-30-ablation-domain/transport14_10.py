# Standard library imports
from collections import deque
from fnmatch import fnmatch

# Local imports (assuming heuristic_base is in a parent directory or sibling module)
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False # Number of elements must match pattern length
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the number of actions required to move each package
    to its goal location independently. It considers the current state of the
    package (on the ground or in a vehicle) and the shortest road distance
    between locations. It sums the estimated costs for all packages that are
    not yet at their goal.

    Assumptions:
    - Each package needs to reach a specific goal location.
    - Vehicles are available to transport packages (availability and capacity are simplified).
    - The cost of moving a vehicle between adjacent locations is 1.
    - The shortest path distance between locations is used for vehicle travel cost.
    - Loading and unloading a package each cost 1 action.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability at package locations is ignored.
    - The road network is such that all relevant locations (initial package/vehicle locations, goal locations) are within the same connected component, allowing finite path distances in solvable problems.

    Heuristic Initialization:
    - Extract goal locations for each package from the task goals.
    - Build a graph representing the road network from static facts.
    - Compute all-pairs shortest path distances between locations using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each package, determine its goal location from the pre-calculated goal mapping.
    2. For each package, find its current status in the state:
       - Is it on the ground at some location `l_current`? (Check for `(at package l_current)`)
       - Is it inside a vehicle `v`? (Check for `(in package v)`)
    3. If the package is inside a vehicle `v`, find the current location `l_v` of that vehicle. (Check for `(at v l_v)`)
    4. If the package is already at its goal location (i.e., `l_current == l_goal` and it's on the ground), its contribution to the heuristic is 0.
    5. If the package is on the ground at `l_current` and `l_current != l_goal`:
       - It needs to be loaded (1 action).
       - It needs to be transported by a vehicle from `l_current` to `l_goal`. The cost is estimated as the shortest path distance `distance(l_current, l_goal)`.
       - It needs to be unloaded at `l_goal` (1 action).
       - Estimated cost for this package: 1 + distance(l_current, l_goal) + 1.
    6. If the package is inside a vehicle `v` at `l_v` and `l_v != l_goal`:
       - It needs to be transported by the vehicle from `l_v` to `l_goal`. The cost is estimated as the shortest path distance `distance(l_v, l_goal)`.
       - It needs to be unloaded at `l_goal` (1 action).
       - Estimated cost for this package: distance(l_v, l_goal) + 1.
    7. Sum the estimated costs for all packages that are not yet at their goal location. This sum is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, road network,
        and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at obj loc) fact
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's a (road loc1 loc2) fact
                    _, loc1, loc2 = parts
                    locations.add(loc1)
                    locations.add(loc2)
                    if loc1 not in self.road_graph:
                        self.road_graph[loc1] = []
                    if loc2 not in self.road_graph:
                        self.road_graph[loc2] = []
                    # Add road in both directions as per examples
                    self.road_graph[loc1].append(loc2)
                    self.road_graph[loc2].append(loc1)

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        all_locations_in_graph = set(self.road_graph.keys()) # Use keys to get actual nodes in graph
        for start_loc in all_locations_in_graph:
            self.shortest_paths[start_loc] = self._bfs(start_loc, all_locations_in_graph)

        # Capacity information is ignored in this simple heuristic.
        # Size ordering is ignored.

    def _bfs(self, start_node, all_nodes):
        """
        Perform BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary mapping reachable node to distance.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Only process if current_node is a valid key in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is a valid node we are tracking distances for
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        # Return only reachable nodes with their finite distances
        return {node: dist for node, dist in distances.items() if dist != float('inf')}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached. If so, heuristic is 0.
        # This is important for correctness (h=0 iff goal).
        if self.goals <= state:
             return 0

        # Track current status of packages and vehicles.
        package_status = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: location}
        vehicles = set() # Set of objects identified as vehicles

        # First pass: Identify vehicles and package status (in vehicle)
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'capacity' and len(parts) == 3:
                  vehicles.add(parts[1]) # (capacity vehicle size)
             elif parts[0] == 'in' and len(parts) == 3:
                  pkg, veh = parts[1], parts[2]
                  vehicles.add(veh)
                  if pkg in self.goal_locations: # It's a package we care about
                      package_status[pkg] = ('in', veh)

        # Second pass: Identify package status (at location) and vehicle locations
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we care about
                     # Only update if not already marked as 'in' a vehicle
                     if obj not in package_status:
                         package_status[obj] = ('at', loc)
                elif obj in vehicles: # It's a vehicle
                     vehicle_locations[obj] = loc
                # else: It's some other object 'at' a location, ignore for this heuristic

        total_cost = 0

        # Iterate through packages that need to reach a goal.
        for package, goal_location in self.goal_locations.items():
            # If package is not in our status map, it's not in the state facts.
            # This shouldn't happen in valid states of a solvable problem.
            # For robustness, add a large penalty and continue.
            if package not in package_status:
                 total_cost += 100000
                 continue

            current_status, current_obj_or_loc = package_status[package]

            # Check if the package is already at the goal location on the ground.
            # If it's in a vehicle at the goal, it still needs unloading.
            if current_status == 'at' and current_obj_or_loc == goal_location:
                continue # Package is already at its final destination on the ground.

            # Calculate cost based on current status
            if current_status == 'at': # Package is on the ground at current_obj_or_loc
                l_current = current_obj_or_loc
                # Needs load, drive, unload
                # Cost = 1 (load) + distance(l_current, l_goal) + 1 (unload)

                # Get distance from current location to goal location
                # Check if l_current is a valid location in our graph
                if l_current not in self.shortest_paths:
                     # Current location not in graph? Problem with state or graph building.
                     total_cost += 100000
                     continue

                distances_from_current = self.shortest_paths[l_current]

                # Check if goal_location is reachable from l_current
                if goal_location not in distances_from_current:
                     # Goal location unreachable from current location via roads.
                     # This state is likely not on a path to the goal, or problem is unsolvable.
                     # Add a large penalty.
                     total_cost += 1 + 10000 + 1 # Load + Penalty + Unload
                else:
                     distance = distances_from_current[goal_location]
                     total_cost += 1 + distance + 1 # Load + Drive + Unload

            elif current_status == 'in': # Package is inside vehicle current_obj_or_loc
                vehicle = current_obj_or_loc
                # Find vehicle location
                if vehicle not in vehicle_locations:
                     # Vehicle containing package is not 'at' any location? Invalid state.
                     total_cost += 100000
                     continue

                l_v = vehicle_locations[vehicle]

                # Needs drive (if l_v != l_goal), unload
                # Cost = distance(l_v, l_goal) + 1 (unload)

                # Get distance from vehicle location to goal location
                # Check if l_v is a valid location in our graph
                if l_v not in self.shortest_paths:
                     # Vehicle location not in graph? Problem with state or graph building.
                     total_cost += 100000
                     continue

                distances_from_vehicle = self.shortest_paths[l_v]

                # Check if goal_location is reachable from l_v
                if goal_location not in distances_from_vehicle:
                     # Goal location unreachable from vehicle location via roads.
                     # Add a large penalty.
                     total_cost += 10000 + 1 # Penalty + Unload
                else:
                     distance = distances_from_vehicle[goal_location]
                     total_cost += distance + 1 # Drive + Unload

        return total_cost
