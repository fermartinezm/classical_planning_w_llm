# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for malformed facts
        return []

    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(start_node, graph):
    """Compute shortest paths from start_node to all reachable nodes in graph."""
    distances = {start_node: 0}
    queue = deque([start_node]) # Use deque for efficient pop(0)
    visited = {start_node}

    while queue:
        current_node = queue.popleft() # Dequeue from the left

        # Check if current_node exists in the graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


# Define the heuristic class
# Inherit from Heuristic if using the base class:
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the cost of picking up, dropping,
    and driving, using shortest path distances on the road network. The heuristic
    is the sum of the estimated costs for each package that is not yet at its goal.

    # Assumptions
    - The road network is static and bidirectional (if road A-B exists, B-A exists).
    - Any vehicle can transport any package, ignoring specific capacity levels beyond
      the basic ability to pick up/drop.
    - The cost of a drive action is 1 in the PDDL domain, but the heuristic uses
      the number of road segments (shortest path length) as the estimated cost
      for driving between two locations. This is a common relaxation.
    - A vehicle is assumed to be available to pick up a package if needed at its
      current location. The cost of getting a vehicle to the package's location
      for pick-up is not explicitly modeled beyond the pick-up action itself.

    # Heuristic Initialization
    - Parses static facts to build the road network graph, identifying all locations
      and connections.
    - Computes all-pairs shortest paths between all identified locations using BFS.
      This precomputation makes the per-state heuristic calculation faster.
    - Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. For each package that has a specified goal location:
       a. Determine the package's current status: Is it on the ground at a location,
          or is it inside a vehicle? This is done by checking the state for facts
          like `(at package location)` or `(in package vehicle)`.
       b. If the package is already at its goal location (checked by looking for the
          `(at package goal_location)` fact in the state), it contributes 0 to the total cost.
       c. If the package is on the ground at a location L_current and needs to reach L_goal:
          - The estimated cost for this package is 1 (for the pick-up action)
            + the shortest path distance from L_current to L_goal (for the drive actions)
            + 1 (for the drop action).
       d. If the package is inside a vehicle V, and vehicle V is currently at location L_current,
          and the package needs to reach L_goal:
          - The estimated cost for this package is the shortest path distance from
            L_current to L_goal (for the drive actions)
            + 1 (for the drop action).
       e. If the package's current physical location (or the vehicle's location if the package
          is inside a vehicle) is not a known location in the road network, or if the goal
          location is unreachable from the current physical location via the road network,
          the heuristic returns infinity for this state, indicating it's likely a dead end
          or unsolvable from this state.
    3. The total heuristic value for the state is the sum of the estimated costs
       calculated for all packages that are not yet at their respective goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the road network graph and collect all locations
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(loc2, []).append(loc1)

        # Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in locations:
            self.shortest_paths[start_loc] = bfs(start_loc, self.road_graph)

        # Store goal locations for each package
        self.goal_locations = {}
        # Also identify all package names from goals for easier lookup
        self.package_names = set()
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
                self.package_names.add(package)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached
        if self.goals <= state:
             return 0

        # Track current status of locatable objects (packages and vehicles)
        # Maps object name to its location (if 'at') or vehicle name (if 'in')
        current_status_map = {}
        # Maps vehicle name to its physical location
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_status_map[obj] = loc
                # Identify vehicles: objects in 'at' facts that are NOT packages from the goals
                if obj not in self.package_names:
                     vehicle_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Only track 'in' for objects that are packages in the goals
                if package in self.package_names:
                    current_status_map[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at its goal, cost is 0 for this package

            # Get the package's current status (location or vehicle)
            current_status = current_status_map.get(package)

            # If a package listed in goals is not found in the state's 'at' or 'in' facts,
            # something is wrong with the state representation or the problem definition.
            # Treat this as an unreachable state.
            if current_status is None:
                 return float('inf')

            current_package_physical_location = None
            is_in_vehicle = False

            if current_status in vehicle_locations: # Package is in a vehicle
                is_in_vehicle = True
                vehicle = current_status
                current_package_physical_location = vehicle_locations.get(vehicle)
                # If vehicle location not found, state is likely invalid
                if current_package_physical_location is None:
                     return float('inf')
            else: # Package is on the ground at current_status (which is a location)
                current_package_physical_location = current_status

            # Ensure the current physical location is a known location in the graph
            if current_package_physical_location not in self.shortest_paths:
                 # This location is isolated or not part of the road network graph built
                 # from static facts. Likely an unreachable state.
                 return float('inf')

            # Get the shortest path distance from the package's current physical location to its goal location
            # Use .get() with a default of None to handle cases where goal_location is not reachable
            drive_cost = self.shortest_paths[current_package_physical_location].get(goal_location)

            # If goal location is unreachable from the current physical location
            if drive_cost is None:
                return float('inf')

            # Calculate cost based on whether the package is in a vehicle or on the ground
            if is_in_vehicle:
                # Cost: Drive vehicle from current location to goal location + Drop package
                total_cost += drive_cost + 1 # drive actions + drop action
            else:
                # Cost: Pick up package + Drive from current location to goal location + Drop package
                total_cost += 1 + drive_cost + 1 # pick-up action + drive actions + drop action


        return total_cost
