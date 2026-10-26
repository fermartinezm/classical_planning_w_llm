import heapq
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator are available in the environment where this heuristic runs
# from task import Operator, Task

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at p1 l1)' -> ('at', 'p1', 'l1')
    Handles potential malformed strings defensively.
    """
    if not fact_string or not fact_string.startswith('(') or not fact_string.endswith(')'):
        return None # Not a valid fact string format
    content = fact_string[1:-1].strip()
    if not content:
        return None # Empty fact
    return tuple(content.split())

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        estimated costs for each package that is not yet at its goal location.
        The estimated cost for a package depends on its current state (at a
        location or inside a vehicle) and its goal location. It includes the
        cost of pickup (if needed), drop (if needed), and the shortest path
        distance the package needs to travel (either carried by a vehicle or
        already inside one). The shortest path distances between locations are
        precomputed using BFS on the road network.

    Assumptions:
        - The goal state primarily consists of (at ?p ?l) facts for packages.
          Any object appearing in an (at ?x ?l) goal fact is considered a package.
        - Vehicle capacity is ignored (relaxed).
        - The cost of getting a vehicle to a package's location for pickup
          is ignored (relaxed).
        - All actions have a cost of 1.
        - Roads are bidirectional.

    Heuristic Initialization:
        1. Parse the static facts to build the road network graph. Identify all locations
           mentioned in static facts, initial state, and goals.
        2. Compute the shortest path distance between all pairs of locations
           using Breadth-First Search (BFS) starting from each location.
           Store these distances in a dictionary of dictionaries: `shortest_distances[l1][l2]` gives the distance from l1 to l2.
        3. Identify the goal location for each package from the task's goal facts.
           Store this in a dictionary: `package_goals[package_name] = goal_location`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state (a frozenset of fact strings).
        2. Initialize the total heuristic value `h_value` to 0.
        3. Iterate through the facts in the current state to determine the current status
           of packages and vehicles:
           - Create dictionaries `package_current_location`, `package_in_vehicle`, and `vehicle_current_location`.
           - For each fact `(at obj loc)`: if `obj` is in `package_goals`, record `package_current_location[obj] = loc`; otherwise, record `vehicle_current_location[obj] = loc`.
           - For each fact `(in pkg veh)`: record `package_in_vehicle[pkg] = veh`.
        4. For each package `package` and its goal location `goal_loc` in `package_goals`:
            a. Check if the package is already at its goal location in the current state:
               If `package` is in `package_current_location` and `package_current_location[package] == goal_loc`, this package contributes 0 to the heuristic. Continue to the next package.
            b. If the package is not at its goal location, calculate its estimated cost contribution:
               - Initialize `package_cost = 0`.
               - If `package` is in `package_current_location`:
                 - Get the current location `current_loc = package_current_location[package]`.
                 - Get the shortest distance `distance = self.shortest_distances.get(current_loc, {}).get(goal_loc, float('inf'))`.
                 - If `distance` is infinity, the goal is unreachable from this location in the road network, return `float('inf')` immediately.
                 - The estimated cost is 1 (pickup) + `distance` (drive) + 1 (drop). Add this to `package_cost`.
               - If `package` is in `package_in_vehicle`:
                 - Get the vehicle `vehicle = package_in_vehicle[package]`.
                 - Find the vehicle's current location `current_loc = vehicle_current_location.get(vehicle)`.
                 - If `current_loc` is None:
                     # Vehicle location is unknown - implies a problem or unreachable goal
                     return float('inf')
                 - Get the shortest distance `distance = self.shortest_distances.get(current_loc, {}).get(goal_loc, float('inf'))`.
                 - If `distance` is infinity:
                     # If goal is unreachable from vehicle's current location
                     return float('inf')

                 - If current_loc == goal_loc:
                    # Vehicle is already at the goal location, just needs drop
                    package_cost = 1 # drop
                 else:
                    # Vehicle needs to drive to goal location and drop
                    package_cost = distance + 1 # drive + drop
               - If the package is neither `at` a location nor `in` a vehicle (should not happen for packages that exist and have goals), return float('inf'). # Defensive check
            c. Add `package_cost` to the total `h_value`.
        5. Return the final `h_value`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.locations = set()
        self.road_graph = defaultdict(set)
        self.shortest_distances = {} # Stores shortest_distances[l1][l2]

        # 1. Parse static facts to build road graph and get all locations
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact and fact[0] == 'road' and len(fact) == 3:
                l1, l2 = fact[1], fact[2]
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assuming roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)

        # Ensure all locations mentioned in init/goals are included, even if not in static roads (unlikely but safe)
        for fact_string in task.initial_state:
             fact = parse_fact(fact_string)
             if fact and fact[0] == 'at' and len(fact) == 3:
                 self.locations.add(fact[2])
        for goal_fact_string in task.goals:
             goal_fact = parse_fact(goal_fact_string)
             if goal_fact and goal_fact[0] == 'at' and len(goal_fact) == 3:
                 self.locations.add(goal_fact[2])


        # 2. Compute shortest path distances between all pairs of locations using BFS
        for start_loc in self.locations:
            self.shortest_distances[start_loc] = self._bfs(start_loc)

        # 3. Identify goal location for each package
        for goal_fact_string in task.goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact and goal_fact[0] == 'at' and len(goal_fact) == 3:
                obj, loc = goal_fact[1], goal_fact[2]
                # Assume any object in an (at ?x ?l) goal is a package
                self.package_goals[obj] = loc

    def _bfs(self, start_location):
        """Performs BFS from a start location to find distances to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_location not in self.locations:
             # Start location might not be in the collected locations if it only appears
             # in facts that weren't processed (e.g., capacity). This shouldn't happen
             # for 'at' or 'road' locations, but being defensive.
             return distances # All distances remain inf

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.road_graph: # Handle locations with no roads
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state
        h_value = 0

        # Determine current status of packages and vehicles
        package_current_location = {}
        package_in_vehicle = {}
        vehicle_current_location = {}

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact and fact[0] == 'at' and len(fact) == 3:
                obj, loc = fact[1], fact[2]
                # Check if the object is one of the packages we have a goal for
                if obj in self.package_goals:
                    package_current_location[obj] = loc
                else:
                    # Assume anything else with 'at' is a vehicle
                    vehicle_current_location[obj] = loc

            elif fact and fact[0] == 'in' and len(fact) == 3:
                package, vehicle = fact[1], fact[2]
                package_in_vehicle[package] = vehicle

        # Calculate heuristic for each package not at its goal
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at goal
            if package in package_current_location and package_current_location[package] == goal_loc:
                continue # Package is already at its goal

            # Package is not at goal, calculate its contribution
            package_cost = 0

            if package in package_current_location:
                # Package is at a location, needs pickup, travel, and drop
                current_loc = package_current_location[package]
                # Cost = 1 (pickup) + distance(current_loc, goal_loc) + 1 (drop)
                # Need to handle cases where current_loc or goal_loc might not be in precomputed distances
                # (e.g., isolated locations). The _bfs handles this by returning inf.
                # Access distances defensively.
                distance = self.shortest_distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if distance == float('inf'):
                    # If goal is unreachable from current location in the road network
                    return float('inf')

                package_cost = 1 + distance + 1

            elif package in package_in_vehicle:
                # Package is inside a vehicle, needs travel and drop
                vehicle = package_in_vehicle[package]
                current_loc = vehicle_current_location.get(vehicle)

                if current_loc is None:
                     # Vehicle location is unknown - implies a problem or unreachable goal
                     return float('inf')

                # Need to handle cases where current_loc or goal_loc might not be in precomputed distances
                distance = self.shortest_distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if distance == float('inf'):
                     # If goal is unreachable from vehicle's current location
                     return float('inf')

                if current_loc == goal_loc:
                    # Vehicle is already at the goal location, just needs drop
                    package_cost = 1 # drop
                else:
                    # Vehicle needs to drive to goal location and drop
                    package_cost = distance + 1 # drive + drop
            else:
                 # Package is neither 'at' a location nor 'in' a vehicle.
                 # This shouldn't happen for a package that exists and has a goal.
                 # It implies the package is "lost". Treat as unreachable goal.
                 return float('inf')

            h_value += package_cost

        return h_value
