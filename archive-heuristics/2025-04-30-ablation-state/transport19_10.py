from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# BFS function for shortest path in the road network
def bfs(graph, start_node):
    """
    Performs Breadth-First Search to find shortest distances from start_node
    in the given graph.
    """
    distances = {node: float('inf') for node in graph}
    # Ensure start_node is a valid node in the graph keys
    if start_node not in graph:
         # If start_node is not in the graph (e.g., an isolated location not on any road),
         # distance to itself is 0, others remain infinity.
         if start_node in distances: # Ensure start_node is a known location
             distances[start_node] = 0
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Check if current_node has neighbors in the graph (should always be true if in graph keys)
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their respective goal locations. It calculates the cost for
    each package independently and sums these costs. The cost for a single
    package is estimated based on whether it needs to be picked up, driven
    to the goal location, and dropped off.

    # Assumptions
    - Each package needs to be moved individually to its goal location.
    - Vehicle capacity constraints are ignored.
    - The cost of moving a package includes:
        - 1 action for picking up the package (if it's on the ground at the wrong location).
        - The shortest path distance (number of drive actions) for a vehicle to travel from the package's current location (or its carrier vehicle's location) to the package's goal location.
        - 1 action for dropping off the package (if it's in a vehicle).
    - The road network is static and bidirectional.
    - If a package or vehicle is in a location not connected to the road network, or if the goal location is unreachable, the heuristic returns infinity.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Identifies all relevant locations and vehicles from the initial state and static facts.
    - Builds a graph representing the road network based on static `(road l1 l2)` facts.
    - Computes the shortest path distance between all pairs of locations in the road network using Breadth-First Search (BFS). These distances are stored for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the location of each package (either on the ground at a location or inside a vehicle) and the location of each vehicle.
    2. Initialize the total estimated cost to 0.
    3. For each package that needs to reach a specific goal location (as defined in the task goals):
        a. Check if the package is already at its goal location in the current state. If yes, this package contributes 0 to the total cost.
        b. If the package is currently on the ground at a location different from its goal:
            - Add 1 to the cost (for the `pick-up` action).
            - Find the shortest distance (number of `drive` actions) from the package's current location to its goal location using the pre-computed distances. Add this distance to the cost.
            - Add 1 to the cost (for the `drop` action).
            - If the goal location is unreachable from the current location via the road network, the heuristic for this state is infinity.
        c. If the package is currently inside a vehicle:
            - Find the current location of the vehicle carrying the package.
            - Find the shortest distance (number of `drive` actions) from the vehicle's current location to the package's goal location. Add this distance to the cost.
            - Add 1 to the cost (for the `drop` action).
            - If the goal location is unreachable from the vehicle's current location, the heuristic for this state is infinity.
        d. If the package's state (at a location or in a vehicle) is not found, or if a required vehicle's location is not found, the state is considered invalid or unsolvable, and the heuristic is infinity.
    4. Return the accumulated total cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and computing shortest path distances.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Needed to identify initial objects/locations

        self.all_locations = set()
        self.all_vehicles = set()
        self.goal_locations = {} # package -> goal_location mapping

        # Identify vehicles by the 'capacity' predicate in initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "capacity":
                if len(parts) > 1: # Expects (capacity vehicle size)
                    vehicle_name = parts[1]
                    self.all_vehicles.add(vehicle_name)

        # Collect all locations mentioned in initial state (at facts), static facts (road facts), and goals (at facts)
        all_facts_to_parse = set(self.initial_state) | set(self.static) | set(self.goals)
        for fact in all_facts_to_parse:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "at":
                 # (at obj loc)
                 if len(parts) == 3:
                     self.all_locations.add(parts[2])
             elif predicate == "road":
                 # (road loc1 loc2)
                 if len(parts) == 3:
                     self.all_locations.add(parts[1])
                     self.all_locations.add(parts[2])

        # Store goal locations for packages
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at":
                # (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                    # Ensure goal location is in our known locations
                    self.all_locations.add(location)


        # Build road graph
        self.road_graph = {loc: set() for loc in self.all_locations}
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "road":
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    # Ensure locations from road facts are in our collected locations
                    if l1 in self.road_graph and l2 in self.road_graph:
                         self.road_graph[l1].add(l2)
                         self.road_graph[l2].add(l1)


        # Compute shortest path distances between all pairs of locations
        self._flattened_distances = {}
        for start_loc in self.all_locations:
            distances_from_start = bfs(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                 self._flattened_distances[(start_loc, end_loc)] = dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Parse current state to find package and vehicle locations/states
        package_state = {} # package -> ('at', loc) or ('in', vehicle)
        vehicle_location = {} # vehicle -> loc

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3:
                    obj_name = parts[1]
                    loc_name = parts[2]
                    # Check if this object is a package we care about or a vehicle
                    if obj_name in self.goal_locations:
                         package_state[obj_name] = ('at', loc_name)
                    elif obj_name in self.all_vehicles:
                         vehicle_location[obj_name] = loc_name
            elif predicate == "in":
                 if len(parts) == 3:
                    package_name = parts[1]
                    vehicle_name = parts[2]
                    # Check if this package is one we care about
                    if package_name in self.goal_locations:
                         package_state[package_name] = ('in', vehicle_name)


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach a goal
        for package, l_goal in self.goal_locations.items():
            # Check if package is in the state at all (should be either 'at' or 'in')
            if package not in package_state:
                 # Package state is unknown -> invalid state
                 return float('inf')

            state_type, current_loc_or_vehicle = package_state[package]

            # Check if package is already at the goal
            if state_type == 'at' and current_loc_or_vehicle == l_goal:
                continue # Package is already at its goal, cost is 0 for this package

            # Package is not at the goal, calculate cost to move it
            if state_type == 'at':
                # Package is on the ground at l_current
                l_current = current_loc_or_vehicle
                # Cost: pick-up + drive + drop
                drive_cost = self._flattened_distances.get((l_current, l_goal), float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location
                    return float('inf')

                total_cost += 1 # pick-up action
                total_cost += drive_cost # drive actions
                total_cost += 1 # drop action

            elif state_type == 'in':
                # Package is inside a vehicle
                vehicle = current_loc_or_vehicle
                # Need vehicle's location
                if vehicle not in vehicle_location:
                    # Vehicle location unknown -> invalid state
                    return float('inf')

                l_vehicle = vehicle_location[vehicle]
                # Cost: drive + drop
                drive_cost = self._flattened_distances.get((l_vehicle, l_goal), float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from vehicle's current location
                    return float('inf')

                total_cost += drive_cost # drive actions
                total_cost += 1 # drop action

        return total_cost
