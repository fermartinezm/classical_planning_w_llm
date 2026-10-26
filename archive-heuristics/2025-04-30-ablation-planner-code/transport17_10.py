# Assume Heuristic base class is imported from 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

from collections import deque

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class transportHeuristic: # Inherit from Heuristic in actual framework
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and potential
    synergies (like carrying multiple packages).

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles can carry packages.
    - The road network is undirected (if road A B exists, road B A exists).
    - Any vehicle can pick up any package if they are at the same location
      (capacity constraints are simplified/ignored).
    - The cost of driving between locations is the shortest path distance
      in the road network.
    - The cost of pick-up and drop actions is 1.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Precomputes shortest path distances between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:

    1. Determine the package's current status:
       - Is it on the ground at some location `current_l`? (`(at package current_l)`)
       - Is it inside a vehicle `v`? (`(in package v)`)

    2. If the package is on the ground at `current_l`:
       - If `current_l` is the goal location, the cost for this package is 0 (this case is covered by the initial goal check).
       - If `current_l` is not the goal location:
         - It needs to be picked up (1 action).
         - It needs to be transported by a vehicle from `current_l` to the goal location `goal_l`. The minimum number of drive actions is the shortest path distance between `current_l` and `goal_l`.
         - It needs to be dropped at `goal_l` (1 action).
         - The estimated cost for this package is 1 (pick-up) + shortest_path_distance(`current_l`, `goal_l`) + 1 (drop).

    3. If the package is inside a vehicle `v`:
       - Find the current location of the vehicle `v` (`(at v vehicle_l)`).
       - If `vehicle_l` is the goal location `goal_l`:
         - It needs to be dropped at `goal_l` (1 action).
         - The estimated cost for this package is 1 (drop).
       - If `vehicle_l` is not the goal location `goal_l`:
         - The vehicle needs to be transported from `vehicle_l` to `goal_l` with the package inside. The minimum number of drive actions is the shortest path distance between `vehicle_l` and `goal_l`.
         - It needs to be dropped at `goal_l` (1 action).
         - The estimated cost for this package is shortest_path_distance(`vehicle_l`, `goal_l`) + 1 (drop).

    4. The total heuristic value is the sum of the estimated costs for all packages
       that are not yet at their goal locations.

    A heuristic value of 0 is returned only when all packages are at their goal locations.
    If a package's goal location is unreachable from its current location (or its vehicle's location)
    via the road network, the shortest path distance will be infinite, resulting in an
    infinite heuristic value, correctly indicating an unsolvable state (or a state
    from which the goal is unreachable).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and precomputing shortest paths.
        """
        # Assuming task object has 'goals' (frozenset of goal facts)
        # and 'static' (frozenset of static facts like road, capacity-predecessor)
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            # Goal is expected to be (at ?p ?l)
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Build the road graph and get all locations
        self.locations = set()
        self.road_graph = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(loc2, []).append(loc1)

        # Precompute all-pairs shortest paths using BFS from each location
        self.shortest_paths = {} # (start_loc, end_loc) -> distance

        for start_loc in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                current_loc = queue.popleft()

                # Store the distance from start_loc to current_loc
                self.shortest_paths[(start_loc, current_loc)] = distances[current_loc]

                # Explore neighbors
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)

        # Ensure distances are stored for all pairs, even unreachable ones (will be inf)
        # This loop is technically redundant if BFS explores all reachable nodes,
        # but ensures any pair lookup is handled.
        for l1 in self.locations:
            for l2 in self.locations:
                 if (l1, l2) not in self.shortest_paths:
                      self.shortest_paths[(l1, l2)] = float('inf')


    def get_shortest_path_distance(self, loc1, loc2):
        """Retrieves the precomputed shortest path distance between two locations."""
        # Handle cases where locations might not be in the graph (e.g., goal location is isolated)
        if loc1 not in self.locations or loc2 not in self.locations:
             return float('inf')
        return self.shortest_paths.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Check if the state is a goal state
        if self.goals.issubset(state):
             return 0

        # Track current locations of vehicles that might be carrying packages
        vehicle_current_location = {}
        # Track package status (at location or in vehicle) for packages in goals
        package_current_status = {} # package -> {'type': 'at'/'in', 'location': loc/vehicle}

        # Iterate through the state facts once to build lookup dictionaries
        # First pass: Find packages and their status (at or in)
        vehicles_carrying_packages_names = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # If this object is one of the packages we need to move
                if obj in self.package_goals:
                     package_current_status[obj] = {'type': 'at', 'location': loc}

            elif predicate == 'in' and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 if package in self.package_goals:
                      package_current_status[package] = {'type': 'in', 'location': vehicle} # location is the vehicle name
                      vehicles_carrying_packages_names.add(vehicle) # Mark vehicle as one whose location we need


        # Second pass: Find locations of vehicles that are carrying packages we care about
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  if obj in vehicles_carrying_packages_names:
                       vehicle_current_location[obj] = loc


        total_cost = 0

        # Calculate cost for each package whose goal is not satisfied
        for package, goal_location in self.package_goals.items():
            goal_fact = f'(at {package} {goal_location})'
            if goal_fact in state:
                continue # This package goal is already satisfied

            # Package goal is not satisfied, calculate cost
            cost_for_package = 0
            current_status = package_current_status.get(package)

            if current_status is None:
                 # Package is in goals but not found in state (neither 'at' nor 'in').
                 # This implies an issue with the state representation or problem definition.
                 # Treat as unreachable.
                 return float('inf')

            current_type = current_status['type']
            current_loc_or_vehicle = current_status['location']

            if current_type == 'at': # Package is on the ground
                package_location = current_loc_or_vehicle
                # Needs pick-up, drive, drop
                # Cost = 1 (pick-up) + distance(package_location, goal_location) + 1 (drop)
                dist = self.get_shortest_path_distance(package_location, goal_location)
                if dist == float('inf'):
                    # Goal is unreachable for this package
                    return float('inf') # Return infinity for the whole state
                cost_for_package = 1 + dist + 1

            elif current_type == 'in': # Package is inside a vehicle
                vehicle_name = current_loc_or_vehicle
                vehicle_location = vehicle_current_location.get(vehicle_name)

                if vehicle_location is None:
                    # Vehicle location not found in state. Invalid state?
                    # A package is in a vehicle, but the vehicle is not 'at' any location.
                    # Treat as unreachable.
                    return float('inf')

                # Needs drive (while in vehicle), drop
                # Cost = distance(vehicle_location, goal_location) + 1 (drop)
                dist = self.get_shortest_path_distance(vehicle_location, goal_location)
                if dist == float('inf'):
                     # Goal is unreachable for this package
                     return float('inf') # Return infinity for the whole state
                cost_for_package = dist + 1

            total_cost += cost_for_package

        return total_cost
