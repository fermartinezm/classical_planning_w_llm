# Need deque for BFS
from collections import deque

# Assume Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like "(predicate arg1 arg2)"
    if not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts, though unlikely with a proper planner state
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated minimum actions for each package
    independently, ignoring vehicle capacity and sharing. The estimate for a package
    is based on its current location (on the ground or in a vehicle) and the shortest
    path distance in the road network to its goal location.

    # Assumptions
    - Roads are bidirectional (or treated as such for distance calculation).
    - Action costs are uniform (1 per action).
    - Vehicle capacity constraints are ignored in the cost calculation.
    - Vehicle availability is ignored in the cost calculation.
    - The only goal conditions considered are `(at package location)`.
    - Package names start with 'p', vehicle names start with 'v'. (Used for simple type checking).

    # Heuristic Initialization
    - Builds an adjacency list representation of the road network from `road` facts.
    - Stores the goal location for each package from `at` goal facts.
    - Initializes a cache for storing shortest path distances computed by BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Iterate through the state facts to build mappings of:
       - Locatable objects (packages, vehicles) to their current status (ground location or containing vehicle).
       - Vehicles to their current ground location.
    3. Iterate through each package that has a specified goal location (extracted during initialization):
       a. Check if the package is already at its goal location in the current state. If the fact `(at package goal_location)` exists in the state, add 0 to the total cost for this package and proceed to the next package.
       b. If the package is not at its goal:
          i. Determine the effective starting location for transport. If the package is on the ground at `current_l` (i.e., `(at package current_l)` is in the state), the starting location is `current_l`. If the package is inside a vehicle `v` (i.e., `(in package v)` is in the state), find the vehicle's current ground location `vehicle_l` (i.e., `(at v vehicle_l)` is in the state); the starting location is `vehicle_l`.
          ii. Determine the base action cost for the package independent of travel. If the package is on the ground, it needs a pick-up and a drop action (base cost 2). If the package is in a vehicle, it only needs a drop action (base cost 1).
          iii. Calculate the shortest path distance in the road network graph from the effective starting location to the package's goal location using Breadth-First Search (BFS). Cache the results of BFS runs from different starting locations to improve efficiency if the same start location is encountered again.
          iv. If the goal location is unreachable from the starting location (BFS returns infinity distance), the problem is likely unsolvable from this state, and the heuristic returns infinity.
          v. Add the base action cost plus the shortest path distance (representing the minimum number of drive actions required) to the total cost for this package.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting road network and goal locations.
        """
        # Assuming task object has .goals and .static attributes
        self.goals = task.goals
        static_facts = task.static

        self.road_graph = {}
        self.goal_locations = {}
        # Cache for BFS results: {start_loc: {dest_loc: dist, ...}, ...}
        self.distances = {}

        # Build road graph from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(l2, set()).add(l1)

        # Store goal locations for packages from goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def bfs(self, start_node):
        """
        Performs BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary {location: distance}.
        """
        distances = {start_node: 0}
        queue = deque([start_node])

        # Handle cases where start_node might not be in the road graph (e.g., isolated location)
        if start_node not in self.road_graph and start_node in distances:
             # It's an isolated node, distance to itself is 0, no neighbors to explore
             return distances
        elif start_node not in self.road_graph:
             # Start node is not even a known location in the graph
             return {} # Cannot reach anything

        # Ensure start_node is in the graph if it has neighbors
        if start_node in self.road_graph:
            queue = deque([start_node])
        else:
             # If start_node is not in road_graph and has no neighbors, BFS is trivial
             return {start_node: 0} if start_node in distances else {}


        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Use .get() with default empty set for robustness if a node has no outgoing roads defined
            for neighbor in self.road_graph.get(current_node, set()):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        total_cost = 0

        # Map locatable objects (packages, vehicles) to their current status
        # Status can be a location string (if on ground) or a vehicle string (if in vehicle)
        current_status = {}
        # Map vehicles to their ground location
        vehicle_locations = {}

        # Build current status and vehicle locations maps by iterating through state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                # Simple check to identify vehicles based on naming convention
                # This assumes vehicle names start with 'v' as seen in examples.
                # A more robust approach would require object type information from the task definition.
                if obj.startswith('v'):
                    vehicle_locations[obj] = loc
            elif predicate == 'in':
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside this vehicle

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_l in self.goal_locations.items():
            # Check if package is already at goal
            # Checking for the exact fact string is efficient
            if f'(at {package} {goal_l})' in state:
                continue # Package is already at its goal, cost is 0 for this package

            pkg_current_status = current_status.get(package)

            if pkg_current_status is None:
                 # Package not found in the state facts (neither at nor in)
                 # This indicates an invalid state or an unhandled case.
                 # For heuristic purposes, treat as unreachable goal.
                 return float('inf')

            start_l = None
            cost_multiplier = 0 # Base cost (pick/drop)

            # Determine effective start location and base cost based on package status
            if pkg_current_status.startswith('v'): # Package is in a vehicle (assuming vehicle names start with 'v')
                vehicle = pkg_current_status
                start_l = vehicle_locations.get(vehicle)
                if start_l is None:
                    # Vehicle location not found in state - invalid state?
                    return float('inf')
                cost_multiplier = 1 # Needs 1 drop action

            else: # Package is on the ground
                start_l = pkg_current_status
                cost_multiplier = 2 # Needs 1 pick-up and 1 drop action

            # Calculate distance from start_l to goal_l using BFS and cache
            if start_l not in self.distances:
                self.distances[start_l] = self.bfs(start_l)

            # Get distance from cache. Default to infinity if goal is unreachable from start_l.
            # Check if start_l exists in distances cache first, in case BFS returned empty dict
            # for an isolated start_l not in road_graph.
            if start_l not in self.distances:
                 # This case should be covered by the BFS return {} logic, but defensive check
                 dist = float('inf')
            else:
                 dist = self.distances[start_l].get(goal_l, float('inf'))


            if dist == float('inf'):
                # Goal location is unreachable from the package's current/vehicle's location
                return float('inf')

            # Total cost for this package is base actions + drive actions
            # If start_l == goal_l, dist is 0, correctly adding only the base cost.
            package_cost = cost_multiplier + dist
            total_cost += package_cost

        return total_cost
