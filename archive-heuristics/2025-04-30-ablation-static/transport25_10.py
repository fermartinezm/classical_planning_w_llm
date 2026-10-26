from collections import deque
import math
from heuristics.heuristic_base import Heuristic # Assuming this exists

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, considering the road network
    distances. It sums the estimated costs for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost of pick-up, drop, and driving between adjacent locations is 1.
    - Vehicle capacity and availability are ignored (assumes a vehicle is always
      available when needed and has sufficient capacity).
    - The cost of driving between non-adjacent locations is the shortest path
      distance in the road network.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest paths between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Determine the current location or containing vehicle for every package
       and vehicle in the state by examining 'at' and 'in' facts.
    3. For each package whose goal location is known:
       a. Check if the package is already at its goal location. If yes, the cost
          for this package is 0, continue to the next package.
       b. If the package is currently on the ground at a location (not its goal):
          - Estimate the cost as 1 (pick-up) + shortest_path_distance(current_location, goal_location) (drive) + 1 (drop).
          - If the goal location is unreachable from the current location, the state is likely unsolvable; return infinity.
          - Add this cost to the total heuristic.
       c. If the package is currently inside a vehicle:
          - Find the current location of the vehicle.
          - Estimate the cost as shortest_path_distance(vehicle_location, goal_location) (drive) + 1 (drop).
          - If the goal location is unreachable from the vehicle's location, the state is likely unsolvable; return infinity.
          - Add this cost to the total heuristic.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting package goals, building the road
        network, and computing shortest paths.
        """
        super().__init__(task) # Call the base class constructor

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations

        # 1. Extract package goal locations
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # 2. Find all locations and build road graph
        all_locations = set()
        road_graph = {} # Adjacency list: {location: [neighbor1, neighbor2, ...]}

        # Collect locations from initial state (at facts)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 # (at locatable location)
                 location = parts[2]
                 all_locations.add(location)

        # Collect locations and build graph from static facts (road facts)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                # (road loc1 loc2)
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)
                if l1 not in road_graph:
                    road_graph[l1] = []
                road_graph[l1].append(l2)
            # Also collect locations from static 'at' facts if any (less common)
            elif parts[0] == "at":
                 location = parts[2]
                 all_locations.add(location)


        # Ensure all locations found are keys in the graph, even if they have no outgoing roads
        for loc in all_locations:
            if loc not in road_graph:
                road_graph[loc] = []

        self.all_locations = list(all_locations) # Store as list or keep as set
        self.road_graph = road_graph

        # 3. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.all_locations:
            distances = {loc: math.inf for loc in self.all_locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                # Check if u exists in road_graph keys before iterating its neighbors
                # This handles cases where a location might be in all_locations but not have any road facts involving it as a source
                if u in self.road_graph:
                    for v in self.road_graph[u]:
                        if distances[v] == math.inf:
                            distances[v] = distances[u] + 1
                            queue.append(v)
            self.shortest_paths[start_node] = distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Map locatables (packages, vehicles) to their current status (location or vehicle)
        current_status_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                # (at locatable location)
                locatable, location = parts[1], parts[2]
                current_status_map[locatable] = location
            elif parts[0] == "in":
                # (in package vehicle)
                package, vehicle = parts[1], parts[2]
                current_status_map[package] = vehicle

        # Check if the state is a goal state
        # This check is crucial for the heuristic to be 0 only at the goal
        if self.goals <= state:
             return 0

        # Calculate cost for each package not at its goal
        for package, goal_l in self.package_goals.items():
            # Check if package is already at goal
            # This check is needed for packages that are individually at their goal
            # while other packages are not. The heuristic sums costs for
            # packages *not* at their goal.
            if f"(at {package} {goal_l})" in state:
                continue # Package is at goal, cost is 0 for this package

            current_status = current_status_map.get(package)

            if current_status is None:
                 # Package not found in state facts (should not happen in valid states)
                 # Treat as unsolvable or very high cost
                 return math.inf # Cannot find package, likely unsolvable

            # Case 1: Package is on the ground at current_l
            # Check if current_status is a location by seeing if it's a key in shortest_paths
            # (only locations are keys in shortest_paths)
            if current_status in self.shortest_paths:
                current_l = current_status
                # Needs pick-up, drive, drop
                drive_cost = self.shortest_paths[current_l].get(goal_l, math.inf)

                if drive_cost == math.inf:
                    # Goal location is unreachable from current location
                    return math.inf # Unsolvable state

                cost = 1 + drive_cost + 1 # pick + drive + drop
                total_cost += cost

            # Case 2: Package is inside a vehicle
            else: # current_status is a vehicle name
                vehicle = current_status
                current_v_l = current_status_map.get(vehicle)

                if current_v_l is None or current_v_l not in self.shortest_paths:
                    # Vehicle location not found or invalid
                    return math.inf # Cannot find vehicle location, likely unsolvable

                # Needs drive, drop
                drive_cost = self.shortest_paths[current_v_l].get(goal_l, math.inf)

                if drive_cost == math.inf:
                    # Goal location is unreachable from vehicle's location
                    return math.inf # Unsolvable state

                cost = drive_cost + 1 # drive + drop
                total_cost += cost

        return total_cost
