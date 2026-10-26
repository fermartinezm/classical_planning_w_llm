# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Utility functions (can be placed outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class definition
# Inherit from Heuristic if available in the planning framework
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    package independently. The cost for a package is estimated based on its
    current location (or the location of the vehicle carrying it) and its
    goal location, using shortest path distances on the road network.

    # Assumptions
    - The heuristic assumes that a suitable vehicle is always available
      to pick up a package at its current location.
    - It ignores vehicle capacity constraints.
    - It ignores the cost of moving an empty vehicle to a package's location
      for pick-up.
    - It assumes all road connections are bidirectional.
    - Each drive action has a cost of 1. Pick-up and drop actions also have a cost of 1.

    # Heuristic Initialization
    - The road network is built from the static `road` facts.
    - Shortest path distances between all pairs of locations are pre-computed
      using Breadth-First Search (BFS) on the road network graph.
    - The goal location for each package is extracted from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that is not yet at its goal location `goal_l`:
    1. Determine the package's current location. This is either its direct location
       if it's on the ground (`(at p current_l)`) or the location of the vehicle
       carrying it if it's inside a vehicle (`(in p v)` and `(at v vehicle_l)`).
    2. If the package is already at `goal_l`, the cost for this package is 0.
    3. If the package is on the ground at `current_l` (`current_l != goal_l`):
       - The estimated cost for this package is 1 (pick-up) + distance(`current_l`, `goal_l`) + 1 (drop).
       - The distance is the shortest path distance on the road network.
    4. If the package is inside a vehicle `v` which is at `vehicle_l`:
       - If `vehicle_l == goal_l`: The estimated cost is 1 (drop).
       - If `vehicle_l != goal_l`: The estimated cost is distance(`vehicle_l`, `goal_l`) + 1 (drop).
    5. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the road network and pre-computing
        shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the road network graph
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on examples
                self.road_graph.setdefault(l2, set()).add(l1)

        self.locations = list(locations) # Store locations for BFS

        # Pre-compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Identify all packages involved in the goal
        self.goal_packages = set(self.goal_locations.keys())


    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest distances
        to all other nodes in the road graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """
        Returns the pre-computed shortest distance between two locations.
        Returns float('inf') if no path exists.
        """
        if loc1 == loc2:
            return 0
        # Check if loc1 is a known location and if loc2 is reachable from loc1
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             return self.distances[loc1][loc2]
        # This case happens if loc1 or loc2 is not in the set of locations
        # found from 'road' facts, or if there's no path.
        # In a valid problem, all relevant locations should be connected or listed.
        # Returning inf indicates unreachable.
        return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current location/status of packages and vehicles
        package_status = {} # {package: {'type': 'at'/'in', 'location': loc/vehicle}}
        vehicle_locations = {} # {vehicle: location}

        # Populate package_status and vehicle_locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                # Determine if obj is a vehicle or package.
                # Rely on goal_packages and common naming conventions ('p' for package, 'v' for vehicle).
                # This is a heuristic assumption based on typical PDDL naming.
                if obj in self.goal_packages or obj.startswith('p'):
                     package_status[obj] = {'type': 'at', 'location': loc}
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc
                # Ignore other 'at' facts if any appear

            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                # Ensure the object is a package relevant to the goal or named like one
                if package in self.goal_packages or package.startswith('p'):
                    package_status[package] = {'type': 'in', 'location': vehicle}
                # Ignore 'in' facts for non-packages if any exist

            # Ignore other predicates like 'capacity', 'capacity-predecessor', 'road' etc.
            # as they are static or not directly location/containment status for this heuristic.


        total_cost = 0  # Initialize heuristic cost

        # Consider only packages that are part of the goal
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state (e.g., wasn't in initial state), skip it.
            # This shouldn't happen in valid PDDL, but defensive coding.
            if package not in package_status:
                 # If a goal package is not found, it's likely unreachable or an invalid state.
                 # Returning infinity is appropriate.
                 return float('inf')

            current_status = package_status[package]

            # Check if the package is already at its goal location
            if current_status['type'] == 'at' and current_status['location'] == goal_location:
                continue # Package is already at goal

            # Estimate cost for this package
            package_cost = 0

            if current_status['type'] == 'at':
                # Package is on the ground at current_l
                current_l = current_status['location']
                # Cost: pick-up (1) + drive (distance) + drop (1)
                distance = self.get_distance(current_l, goal_location)
                if distance == float('inf'):
                    # If goal is unreachable from current location, return infinity
                    return float('inf')
                package_cost = 1 + distance + 1 # pick + drive + drop

            elif current_status['type'] == 'in':
                # Package is inside a vehicle v
                vehicle = current_status['location']
                # Find the vehicle's location
                if vehicle not in vehicle_locations:
                    # This state is likely invalid or represents a package in a vehicle
                    # that isn't 'at' any location (shouldn't happen in valid states).
                    # Treat as unreachable for safety.
                    return float('inf')

                vehicle_l = vehicle_locations[vehicle]

                if vehicle_l == goal_location:
                    # Vehicle is at the goal location, just need to drop
                    package_cost = 1 # drop
                else:
                    # Vehicle needs to drive to goal_location, then drop
                    distance = self.get_distance(vehicle_l, goal_location)
                    if distance == float('inf'):
                        return float('inf')
                    package_cost = distance + 1 # drive + drop

            total_cost += package_cost

        # The heuristic is 0 if and only if all goal conditions are met.
        # The current calculation sums costs for packages *not* at the goal.
        # If all packages are at the goal, the loop is skipped, and total_cost remains 0.
        # This matches the requirement.

        return total_cost
