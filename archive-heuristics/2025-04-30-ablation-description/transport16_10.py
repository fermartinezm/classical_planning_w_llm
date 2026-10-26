# Need to import the base Heuristic class and deque for BFS
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to move all packages that are not at their
    goal location. It calculates the cost for each package independently,
    ignoring vehicle capacity constraints and the possibility of transporting
    multiple packages in one trip. The cost for a package is estimated based
    on whether it's on the ground or in a vehicle, and the shortest road
    distance to its goal location.

    # Assumptions
    - Roads are bidirectional.
    - Any vehicle can transport any package (capacity is ignored).
    - The cost of pick-up, drop, and drive actions is 1.
    - The heuristic sums the estimated costs for each package independently.
    - The heuristic assumes that if a package is in a vehicle, the vehicle's
      location is known, and if a package is on the ground, its location is known.
    - The goal for a package is always to be on the ground at a specific location.

    # Heuristic Initialization
    - Parses the goal conditions to identify the target location for each package
      that needs to be at a specific location.
    - Parses the static facts to build a graph of locations connected by roads.
    - Computes and stores the shortest road distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle, and which
       packages are inside which vehicles, by parsing the 'at' and 'in' facts
       in the current state.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location (extracted during initialization):
       a. Check if the package is already on the ground at its goal location. If yes, the cost
          for this package is 0, move to the next package.
       b. If the package is not at its goal location:
          i. If the package is currently inside a vehicle:
             - Find the current location of that vehicle.
             - The estimated cost for this package is 1 (for the drop action)
               plus the shortest road distance from the vehicle's current location
               to the package's goal location. If the vehicle's location is unknown
               or the goal is unreachable, the cost is infinite.
          ii. If the package is currently on the ground at a location:
              - The estimated cost for this package is 1 (for the pick-up action)
                plus the shortest road distance from the package's current location
                to its goal location, plus 1 (for the drop action). If the package's
                location is unknown or the goal is unreachable, the cost is infinite.
          iii. If the package's location is unknown (neither on ground nor in vehicle,
               should not happen in valid states), the cost is considered infinite.
       c. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and precomputing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse goal conditions to find package goal locations
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # We only care about 'at' goals for objects, assuming they are packages
            # A more robust approach would check object types if available.
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # 2. Build the road graph from static facts
        self.road_graph = {}
        locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)

        # Also add locations from initial state and goals, just in case some locations exist but have no roads
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                  locations.add(parts[2]) # Add location from (at obj loc)
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                  locations.add(parts[2]) # Add location from (at obj loc)

        for loc in locations:
            self.road_graph[loc] = set()

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Ensure locations are in the graph keys (added from the 'locations' set)
                if l1 in self.road_graph and l2 in self.road_graph:
                    self.road_graph[l1].add(l2)
                    self.road_graph[l2].add(l1) # Assuming bidirectional roads

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations = list(locations) # Use a list for consistent iteration order

        for start_loc in all_locations:
            dist_from_start = self._bfs(start_loc, self.road_graph)
            for end_loc in all_locations:
                 # Store distance, default to infinity if not found (unreachable)
                 self.distances[(start_loc, end_loc)] = dist_from_start.get(end_loc, float('inf'))


    def _bfs(self, start_node, graph):
        """
        Performs BFS from a start_node to find distances to all reachable nodes.
        Returns a dictionary mapping reachable node to distance.
        """
        queue = deque([(start_node, 0)])
        visited = {start_node}
        distances = {start_node: 0}

        # Use .get(start_node, []) to handle nodes that might be in 'locations'
        # but have no road connections (isolated nodes).
        # The BFS naturally explores only reachable nodes.
        # The distances dictionary will only contain entries for reachable nodes.
        # The calling code in __init__ handles unreachable pairs by defaulting to infinity.

        while queue:
            current_node, dist = queue.popleft()

            # Use .get(current_node, []) to handle nodes that might be in 'locations'
            # but have no road connections (isolated nodes).
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return distances


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify current location/status of objects
        current_locations = {} # {object_name: location_name} for 'at' facts
        package_in_vehicle = {} # {package_name: vehicle_name} for 'in' facts

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in' and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                package_in_vehicle[pkg] = veh

        # 2. Calculate total cost based on packages not at goal
        total_cost = 0

        for package, goal_location in self.package_goals.items():
            # Check if package is already at goal location on the ground
            # Goal is (at package goal_location)
            is_at_goal = (package in current_locations and current_locations[package] == goal_location)

            if is_at_goal:
                 continue # Goal achieved for this package

            # Package is not at goal. Estimate cost.
            package_cost = float('inf') # Default to infinity if location is unknown or unreachable

            if package in package_in_vehicle:
                # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in current_locations:
                    # Vehicle location is known
                    vehicle_location = current_locations[vehicle]
                    # Cost = 1 (drop) + distance(vehicle_location, goal_location)
                    drive_cost = self.distances.get((vehicle_location, goal_location), float('inf'))
                    if drive_cost != float('inf'):
                         package_cost = 1 + drive_cost
                    # else: package_cost remains inf

            elif package in current_locations:
                # Package is on the ground
                package_location = current_locations[package]
                # Cost = 1 (pick) + distance(package_location, goal_location) + 1 (drop)
                drive_cost = self.distances.get((package_location, goal_location), float('inf'))
                if drive_cost != float('inf'):
                    package_cost = 2 + drive_cost
                # else: package_cost remains inf
            # else: package location is unknown, package_cost remains inf

            total_cost += package_cost

        return total_cost
