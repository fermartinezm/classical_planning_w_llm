from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         return [] # Not a valid fact string format
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards `*` are allowed in `args`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    graph: adjacency list {node: {neighbor1, neighbor2, ...}}
    start_node: the node to start BFS from
    Returns: {node: distance}
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not a known location in the graph
        return distances # All distances remain infinity

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is a valid key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location,
    assuming vehicle availability and capacity are not bottlenecks. It sums the
    estimated costs for all packages not yet at their goals.

    # Assumptions
    - The cost of moving a package is the sum of:
        - 1 action to pick it up (if on the ground and not at goal).
        - The shortest path distance (in terms of drive actions) for a vehicle
          carrying the package from its current location (or the vehicle's location)
          to the package's goal location.
        - 1 action to drop it (if not already dropped at goal).
    - Vehicle availability and capacity constraints are ignored. It is assumed
      that a suitable vehicle is always available when needed at the required location.
    - The shortest path between any two locations can be traversed by any vehicle.
    - Goal conditions only involve packages being at specific locations (`(at package location)`).
    - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations and roads from the static facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle by parsing the state facts.
       - `(at obj loc)` facts tell us where locatable objects (packages, vehicles) are on the ground.
       - `(in pkg veh)` facts tell us which package is inside which vehicle.
    2. For each package `p` that has a goal location `l_goal`:
       a. Check if the package is already at its goal location (`(at p l_goal)` is in the state). If yes, cost for this package is 0.
       b. If not at the goal, find the package's current status:
          - If `(at p l_current)` is in the state: The package is on the ground at `l_current`.
          - If `(in p v)` is in the state: The package is inside vehicle `v`. Find vehicle `v`'s location `l_v` from `(at v l_v)` fact.
          - If the package is not found in any `at` or `in` fact, and is not at its goal, the state is likely malformed or unsolvable. Return infinity.
       c. Calculate the estimated cost for the package based on its current status and goal:
          - If on the ground at `l_current` (`l_current != l_goal`): Cost = 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
          - If inside vehicle `v` at `l_v`:
            - If `l_v == l_goal`: Cost = 1 (drop).
            - If `l_v != l_goal`: Cost = shortest_distance(`l_v`, `l_goal`) (drive) + 1 (drop).
          - If any required distance is infinite, the state is likely unsolvable for this package. Return infinity.
       d. Add the estimated cost for this package to the total heuristic value.
    3. Return the total sum of costs for all packages not at their goals.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road graph, and computing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # 2. Build the road graph and identify all locations.
        self.road_graph = {}
        locations = set()

        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "road" and len(parts) == 3:
                 l1, l2 = parts[1], parts[2]
                 locations.add(l1)
                 locations.add(l2)
                 self.road_graph.setdefault(l1, set()).add(l2)
                 self.road_graph.setdefault(l2, set()).add(l1) # Roads are typically bidirectional

        # Ensure all locations found are keys in the graph, even if they have no roads
        for loc in locations:
             self.road_graph.setdefault(loc, set())

        # 3. Compute all-pairs shortest paths.
        self.location_distances = {}
        for start_loc in self.road_graph:
            self.location_distances[start_loc] = bfs(self.road_graph, start_loc)

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        # If either location is not in our precomputed distances (e.g., not a valid location object)
        if loc1 not in self.location_distances or loc2 not in self.location_distances.get(loc1, {}):
             return float('inf') # Should not happen for valid locations in the graph
        return self.location_distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current locations of packages and vehicles.
        package_status = {} # {package_name: location_name or vehicle_name}
        vehicle_locations = {} # {vehicle_name: location_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects starting with 'p' are packages, 'v' are vehicles.
                if obj.startswith('p'):
                    package_status[obj] = loc # Package is at a location
                elif obj.startswith('v'):
                    vehicle_locations[obj] = loc # Vehicle is at a location
            elif pred == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 # Assuming pkg is a package and veh is a vehicle based on domain definition
                 package_status[pkg] = veh # Package is inside a vehicle

        total_cost = 0  # Initialize action cost counter.

        # 2. Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location.
            # This is the most direct way to check if the goal for this package is met.
            if f"(at {package} {goal_location})" in state:
                continue # Goal met for this package, cost is 0.

            # Package is not at its goal. Find its current status.
            current_status = package_status.get(package)

            if current_status is None:
                 # This package from the goal is not found in any 'at' or 'in' fact in the state,
                 # and we already checked it's not at its goal. This state is problematic
                 # or the package doesn't exist in this state (shouldn't happen in valid states).
                 # Assign infinity as it seems unreachable or state is invalid.
                 return float('inf')

            # Case A: Package is inside a vehicle `v`.
            if current_status in vehicle_locations: # Check if the status is a vehicle name that has a location
                vehicle = current_status
                vehicle_location = vehicle_locations[vehicle]

                # Cost to move package from inside vehicle to goal location.
                # Needs 1 drop action + drive actions if vehicle is not at goal.
                cost_for_package = 1 # Drop action
                if vehicle_location != goal_location:
                    dist = self.get_distance(vehicle_location, goal_location)
                    if dist == float('inf'):
                         # Cannot reach goal location from vehicle's current location.
                         return float('inf')
                    cost_for_package += dist # Drive actions

                total_cost += cost_for_package

            # Case B: Package is on the ground at `l_current`.
            elif current_status in self.road_graph: # Check if the status is a known location
                 current_location = current_status

                 # Cost to move package from current location to goal location.
                 # Needs 1 pick-up action + drive actions + 1 drop action.
                 cost_for_package = 1 # Pick-up action
                 dist = self.get_distance(current_location, goal_location)
                 if dist == float('inf'):
                      # Cannot reach goal location from package's current location.
                      return float('inf')
                 cost_for_package += dist # Drive actions
                 cost_for_package += 1 # Drop action

                 total_cost += cost_for_package

            # Case C: Package status is a vehicle name, but that vehicle is not in vehicle_locations.
            # This means the package is 'in' a vehicle, but the vehicle is not 'at' any location. Invalid state.
            # Or package status is something else unexpected.
            else:
                 # This indicates a state representation error.
                 return float('inf')

        # If the loop finishes and total_cost is 0, it means all packages in goal_locations
        # were found to satisfy the `(at package goal_location)` condition in the state.
        # This implies the goal is reached (assuming goals are only package locations).
        # If total_cost > 0, the goal is not reached.
        # The heuristic is 0 iff the goal is reached (for packages).
        # This aligns with the requirement that h=0 only for goal states in this domain.

        return total_cost
