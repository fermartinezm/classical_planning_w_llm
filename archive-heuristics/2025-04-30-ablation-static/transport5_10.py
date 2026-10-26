from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    from its current location to its goal location, summing the costs for all
    packages that are not yet at their goal. It relaxes capacity constraints
    and vehicle availability, assuming any package can be picked up and moved
    by some vehicle.

    # Assumptions
    - The goal for each package is always a specific ground location (at ?p ?l).
    - Vehicle capacity and specific vehicle assignments are ignored.
    - Any location can be reached from any other reachable location via the road network.
    - All goal packages are present in the state representation with either an 'at' or 'in' fact.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static 'road' facts and identifies all locations.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. For each package that has a goal location defined:
       a. Check if the package is already at its goal location on the ground. If yes, this package requires 0 further actions for its goal.
       b. If not at the goal, determine the package's current physical location. This is either its ground location (if `(at package location)` is true) or the location of the vehicle it is currently inside (if `(in package vehicle)` and `(at vehicle location)` are true).
       c. Determine if the package is currently inside a vehicle.
       d. Calculate the cost for this package:
          - Add 1 for a 'pick-up' action if the package is currently on the ground.
          - Add the shortest path distance (number of 'drive' actions) from the package's current physical location to its goal location in the road network. If the goal is unreachable, the total heuristic will be infinity.
          - Add 1 for a 'drop' action.
       e. Sum the costs calculated for all packages that are not yet at their goal location.
    3. Return the total summed cost. If any required goal location is unreachable from the package's current location, return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are only (at package location)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goals if they exist, as the heuristic is tailored for (at p l)

        # Build the road graph and collect all locations.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store locations for BFS

        # Compute all-pairs shortest paths using BFS from each location.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS starting from start_node to find shortest paths to all reachable nodes."""
        queue = deque([(start_node, 0)])
        visited = {start_node: 0} # Stores (location: distance)

        while queue:
            current_loc, dist = queue.popleft()
            self.shortest_paths[(start_node, current_loc)] = dist

            # Get neighbors from the road graph, handle locations with no outgoing roads
            neighbors = self.road_graph.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the goal is already reached.
        if self.goals.issubset(state):
             return 0 # Goal state reached

        # Track current locations of locatables (packages and vehicles).
        # Map object name -> its immediate location (ground loc or vehicle name)
        current_immediate_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_immediate_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_immediate_locations[package] = vehicle # Package is inside vehicle

        # Determine the effective physical location for each package relevant to goals.
        # Map package name -> its physical location (ground loc or vehicle's loc)
        effective_package_locations = {}
        package_is_in_vehicle = {} # Map package name -> boolean

        for package in self.goal_locations.keys():
             immediate_loc = current_immediate_locations.get(package)

             if immediate_loc is None:
                 # Package not found in state facts - should not happen for goal packages
                 # in a well-formed state. Treat as unreachable.
                 return float('inf') # Problem likely unsolvable if a goal package is missing

             # Check if the immediate location is a vehicle name by seeing if it's NOT a known location.
             if immediate_loc not in self.locations: # immediate_loc is a vehicle name
                 vehicle_loc = current_immediate_locations.get(immediate_loc)
                 if vehicle_loc is None:
                     # Vehicle location unknown - problem likely unsolvable
                     return float('inf')
                 effective_package_locations[package] = vehicle_loc
                 package_is_in_vehicle[package] = True
             else: # immediate_loc is a location name
                 effective_package_locations[package] = immediate_loc
                 package_is_in_vehicle[package] = False


        total_cost = 0

        # Calculate cost for each package that needs to be moved.
        for package, goal_location in self.goal_locations.items():
            # If the package is already at the goal location on the ground, it's done for this package.
            if f"(at {package} {goal_location})" in state:
                continue

            current_l = effective_package_locations.get(package)

            # This check is redundant due to the earlier check for immediate_loc being None,
            # but kept for clarity or if logic changes.
            if current_l is None:
                 return float('inf')

            package_cost = 0

            # Cost for pickup (if not already in a vehicle)
            if package_is_in_vehicle.get(package, False): # Use .get for safety, though should be in map
                 # Already in vehicle, no pickup needed
                 pass
            else:
                 # On the ground, needs pickup
                 package_cost += 1

            # Cost for driving
            drive_cost = self.shortest_paths.get((current_l, goal_location), float('inf'))
            if drive_cost == float('inf'):
                # Goal location is unreachable from current location
                return float('inf')
            package_cost += drive_cost

            # Cost for drop
            package_cost += 1 # Always needs a drop to be (at p l)

            total_cost += package_cost

        return total_cost
