from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, ignoring vehicle capacity constraints and potential
    conflicts between packages or vehicles. It calculates the shortest path
    distance for vehicles on the road network.

    # Assumptions:
    - Vehicle capacity is ignored. Any vehicle can carry any package.
    - Any vehicle can be used to transport any package.
    - The cost of moving a package is the sum of:
        - 1 (pick-up action, if package is on the ground)
        - Shortest path distance (drive actions) from the package's current
          effective location (where it is, or where its vehicle is) to its goal location.
        - 1 (drop action, if package needs to be dropped at the goal).
    - If a package is already at its goal location on the ground, its cost is 0.
    - If a package is in a vehicle at its goal location, it only needs to be dropped (cost 1).
    - The road network is static and bidirectional.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest paths on the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each package from the task goals.
    2. For each package with a goal:
       a. Determine its current status: Is it on the ground at a location, or inside a vehicle?
       b. If inside a vehicle, find the vehicle's current location. This is the package's
          "effective" current location for transport purposes.
       c. If the package is already on the ground at its goal location, the cost for this package is 0.
       d. If the package is not at its goal location:
          - Calculate the shortest path distance between the package's effective current
            location and its goal location using the precomputed distances.
          - Add 1 for the 'pick-up' action (if the package is currently on the ground).
          - Add the shortest path distance for the 'drive' actions.
          - Add 1 for the 'drop' action (needed if the package is not already on the ground at the goal).
    3. Sum the costs for all packages that are not yet at their goal location on the ground.
    4. If any required location is unreachable via the road network, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        self.static_facts = task.static  # Facts that are not affected by actions.

        # 1. Store goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # We only care about package 'at' goals for this heuristic
            if parts[0] == 'at' and parts[1].startswith('p'): # Assuming packages start with 'p'
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # 2. Build the road network graph
        self.graph = {}
        all_locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Add locations mentioned in goals even if they have no roads (isolated nodes)
        all_locations.update(self.package_goals.values())

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in all_locations:
            self.distances[start_node] = self._bfs(start_node, self.graph, all_locations)

    def _bfs(self, start_node, graph, all_nodes):
        """Performs BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in all_nodes} # Initialize distances for all known locations
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists as a key in the graph (it might be an isolated location)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map packages to their current status (on ground location or vehicle)
        package_status = {} # { package: location_or_vehicle }
        vehicle_locations = {} # { vehicle: location }

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # We need to distinguish vehicles and packages.
                # Assuming objects starting with 'v' are vehicles and 'p' are packages.
                if obj.startswith('v'):
                    vehicle_locations[obj] = loc
                elif obj.startswith('p'):
                    package_status[obj] = loc # Package is on the ground
                # else: other locatables? (none in domain)
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                if package.startswith('p') and vehicle.startswith('v'):
                     package_status[package] = vehicle # Package is inside a vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # If the package doesn't appear in the state at all, it's likely an error or unreachable.
            if package not in package_status:
                 return float('inf') # Package state unknown, assume unreachable

            current_status = package_status[package]

            # Check if package is already at goal on the ground
            if current_status == goal_location and package not in [v for v in vehicle_locations]: # Check if status is location and not a vehicle name
                 # A more robust check: check if current_status is a location string, not a vehicle string
                 # We don't have object types here, so rely on string format or fact structure.
                 # If current_status is a location string, it won't be a key in vehicle_locations.
                 # If current_status is a location string AND the fact (at package goal_location) is in state
                 if current_status in self.distances and (f'(at {package} {goal_location})' in state):
                     continue # Package is on the ground at the goal location

            # Determine the package's effective current location (either its ground location or its vehicle's location)
            current_effective_location = None
            needs_pickup = False # Does the package need a pick-up action?

            if current_status in vehicle_locations: # Package is inside a vehicle
                vehicle = current_status
                current_effective_location = vehicle_locations.get(vehicle)
                needs_pickup = False # Already picked up
            elif current_status in self.distances: # Package is on the ground at a known location
                current_effective_location = current_status
                needs_pickup = True # Needs pick-up

            # If we couldn't determine the effective location, it's an issue.
            if current_effective_location is None:
                 # This could happen if a package is 'in' a vehicle, but the vehicle is not 'at' any location.
                 # Or if the package is 'at' a location not in our graph.
                 return float('inf') # Unreachable or invalid state

            # Calculate the drive cost from the effective current location to the goal location
            drive_cost = self.distances.get(current_effective_location, {}).get(goal_location, float('inf'))

            # If the goal location is unreachable from the current location, return infinity.
            if drive_cost == float('inf'):
                return float('inf')

            # Calculate the cost for this package
            package_cost = 0
            if needs_pickup:
                package_cost += 1 # Cost for pick-up action

            package_cost += drive_cost # Cost for drive actions

            # The package needs to be dropped if it's not already on the ground at the goal.
            # It's on the ground at the goal if its status is the goal location AND it's not in a vehicle.
            # We already handled the case where it's on the ground at the goal with 'continue'.
            # So, if we reach here, it needs a drop action.
            package_cost += 1 # Cost for drop action

            total_cost += package_cost

        return total_cost

