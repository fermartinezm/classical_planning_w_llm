# Assuming heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected input robustness, could raise error or return empty
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


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It calculates the cost for each package
    independently, summing up the minimum actions (pick-up, drop, and vehicle
    drive steps) needed for that package based on its current location or the
    location of the vehicle carrying it. It uses precomputed shortest path
    distances between locations based on the road network.

    # Assumptions
    - The heuristic assumes that a suitable vehicle is always available when
      needed to pick up or transport a package. It ignores vehicle capacity
      constraints and the need to move a vehicle to a package's initial
      location if the package is on the ground.
    - The cost of a 'drive' action is considered to be 1 per road segment traversed
      in the shortest path between two locations.
    - 'pick-up' and 'drop' actions each cost 1.
    - The road network is static and bidirectional (if road l1 l2 exists, road l2 l1 also exists).
      (The example instance confirms bidirectionality by including both (road l1 l2) and (road l2 l1)).
    - Packages are the objects whose goal locations are specified in the task.
    - Vehicles are objects that appear in 'capacity' or 'in' predicates.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task's goal conditions.
    - Builds a graph representation of the road network from the static 'road' facts.
    - Identifies all potential vehicle names from static 'capacity' facts.
    - Computes the shortest path distance (number of drive actions) between all pairs
      of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify the current location or containing vehicle for every package
        that has a goal location.
    2.  Identify the current location for every object that is a vehicle.
        (Vehicles are identified based on appearing in 'capacity' or 'in' predicates).
    3.  Initialize the total heuristic cost to 0.
    4.  For each package `p` that has a goal location `loc_p_goal`:
        a.  Determine the package's current status: Is it on the ground at some location `loc_p_current`
            (i.e., `(at p loc_p_current)` is true)? Or is it inside a vehicle `v` (i.e., `(in p v)` is true)?
            Find the effective current physical location of the package (either `loc_p_current` if on ground,
            or the location of vehicle `v` if inside a vehicle).
        b.  If the package's effective current location is the same as its goal location `loc_p_goal`,
            the cost for this package is 0. Continue to the next package.
        c.  If the package is not at its goal, calculate the cost for this package:
            -   The cost includes the minimum number of 'drive' actions needed to move from the package's
                effective current location to its goal location. This is the shortest path distance.
            -   If the package is currently on the ground, it needs a 'pick-up' action (cost 1).
            -   The package needs a 'drop' action at the goal location (cost 1).
            -   Total cost for this package = distance(effective_current_location, goal_location) + (1 if on ground) + 1.
        d.  Add the calculated cost for package `p` to the `total_cost`.
    5.  Return the `total_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts,
           building the road network graph, and precomputing distances."""
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and find all locations.
        self.road_graph = {}
        self.locations = set()
        for fact in self.static:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph:
                    self.road_graph[l2] = []
                self.road_graph[l1].append(l2)
                # Assuming roads are bidirectional based on domain examples
                self.road_graph[l2].append(l1)

        # Ensure all locations from static facts are in the graph keys, even if they have no roads
        for loc in self.locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # Identify all potential vehicles from static capacity facts
        self.all_vehicles = set()
        for fact in self.static:
             predicate, *args = get_parts(fact)
             if predicate == "capacity":
                 vehicle = args[0]
                 self.all_vehicles.add(vehicle)

        # Precompute all-pairs shortest paths using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations."""
        distances = {}
        for start_node in self.locations:
            distances[start_node] = self._bfs(start_node)
        return distances

    def _bfs(self, start_node):
        """Performs BFS starting from start_node to find distances to all other nodes."""
        dist = {node: float('inf') for node in self.locations}
        dist[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in the graph keys, even if it has no neighbors
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[current_node] + 1
                        queue.append(neighbor)
        return dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map packages to their current location (if on ground) or vehicle (if in vehicle)
        package_current_status = {} # { package: location_or_vehicle }
        # Map vehicles to their current location
        vehicle_current_locations = {} # { vehicle: location }

        # Update vehicle list from current state (in case new vehicles appear or capacity changes)
        # Vehicles can appear in 'capacity' or 'in' facts in the state.
        current_vehicles = set(self.all_vehicles) # Start with vehicles from static facts
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "capacity":
                 vehicle = args[0]
                 current_vehicles.add(vehicle)
             elif predicate == "in":
                 package, vehicle = args
                 current_vehicles.add(vehicle)

        # Populate status and locations from the state
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                if obj in self.goal_locations: # It's a package on the ground
                    package_current_status[obj] = location
                elif obj in current_vehicles: # It's a vehicle
                    vehicle_current_locations[obj] = location
                # else: it's another locatable we don't care about its location

            elif predicate == "in":
                package, vehicle = args
                if package in self.goal_locations: # It's a package we care about
                    package_current_status[package] = vehicle # Package is inside this vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package and its goal location
        for package, goal_location in self.goal_locations.items():
            current_status_val = package_current_status.get(package)

            if current_status_val is None:
                 # Package status not found in state. Should not happen if state is valid.
                 # If a package we care about isn't located anywhere, the state is likely invalid/unreachable.
                 return float('inf')

            package_effective_location = None
            is_in_vehicle = False

            if current_status_val in self.locations: # Package is on the ground at a location
                package_effective_location = current_status_val
                is_in_vehicle = False
            elif current_status_val in current_vehicles: # Package is inside a vehicle
                vehicle = current_status_val
                package_effective_location = vehicle_current_locations.get(vehicle)
                is_in_vehicle = True
                if package_effective_location is None:
                     # Vehicle location not found. Cannot calculate cost for this package.
                     # If a package is in a vehicle whose location is unknown, this state is likely invalid/unreachable.
                     return float('inf')
            else:
                 # Status is neither a known location nor a known vehicle with location. Malformed state?
                 # Assign infinity for unexpected states.
                 return float('inf')


            # If the package is already at its goal, cost is 0 for this package.
            if package_effective_location == goal_location:
                continue

            # Package is not at its goal. Calculate cost.
            cost_for_package = 0

            # Cost includes:
            # 1. Drive cost from effective current location to goal location.
            # 2. Pick-up cost (if on ground).
            # 3. Drop cost (always needed if not at goal).

            # 1. Drive cost
            # Ensure effective location and goal location are in the precomputed distances
            if package_effective_location not in self.distances or goal_location not in self.distances[package_effective_location]:
                 # This implies the goal location is unreachable from the current location.
                 # This state might be part of an unsolvable branch, or the graph is disconnected.
                 # Returning infinity signals this is a bad state.
                 return float('inf')

            distance_to_goal = self.distances[package_effective_location][goal_location]
            # If distance is infinity, the goal is unreachable from here.
            if distance_to_goal == float('inf'):
                 return float('inf')

            cost_for_package += distance_to_goal # Drive cost (number of road segments)

            # 2. Pick-up cost (if on ground)
            if not is_in_vehicle:
                 cost_for_package += 1 # Pick-up action cost

            # 3. Drop cost (always needed if not already at goal)
            cost_for_package += 1 # Drop action cost

            total_cost += cost_for_package

        return total_cost
