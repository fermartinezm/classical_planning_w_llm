# Add necessary imports
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but used in example)
# We can simplify this if we only need exact matches or simple wildcard logic
# Let's keep it as is for now, as it was in the example.
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Any vehicle can carry any package (capacity is ignored).
    - A vehicle is available whenever and wherever a package needs to be picked up.
    - The cost of driving between locations is the shortest path distance on the road network.
    - Actions have a cost of 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment status for every package that has a goal location.
       - A package can be on the ground at a location `l` (`(at package l)`).
       - A package can be inside a vehicle `v` (`(in package v)`).
    2. Identify the current location for every vehicle.
       - A vehicle is on the ground at a location `l` (`(at vehicle l)`).
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that needs to reach a goal location `goal_l`:
       - Check if the package is already at its goal location on the ground (`(at p goal_l)` is in the state). If yes, the cost for this package is 0, continue to the next package.
       - Find the package's current state (either 'at' a location or 'in' a vehicle).
       - If the package is currently on the ground at `current_l` (`(at p current_l)` is in the state, where `current_l != goal_l`):
         - Estimate the cost to move this package:
           - 1 action to pick up the package.
           - The shortest path distance (number of drive actions) from `current_l` to `goal_l`.
           - 1 action to drop the package at `goal_l`.
           - Total cost for this package: 1 + distance(`current_l`, `goal_l`) + 1.
       - If the package is currently inside a vehicle `v` (`(in p v)` is in the state):
         - Find the current location of vehicle `v`, say `v_current_l` (`(at v v_current_l)` is in the state).
         - Estimate the cost to move this package:
           - The shortest path distance (number of drive actions) from `v_current_l` to `goal_l`.
           - 1 action to drop the package at `goal_l`.
           - Total cost for this package: distance(`v_current_l`, `goal_l`) + 1.
       - Add the estimated cost for the package to the total heuristic cost.
    5. Return the total heuristic cost. If any required location is unreachable, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the road network."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Identify all packages (those appearing in goals).
        self.packages = set(self.goal_locations.keys())

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                # Fact is (road loc1 loc2)
                if len(args) == 2:
                    l1, l2 = args
                    self.road_graph.setdefault(l1, set()).add(l2)
                    # Assuming roads are bidirectional based on example instance structure
                    self.road_graph.setdefault(l2, set()).add(l1)
                    locations.add(l1)
                    locations.add(l2)

        self.locations = list(locations) # Store as list if needed, though dict keys are fine.

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_l in self.locations:
            self.distances[start_l] = {}
            queue = deque([(start_l, 0)])
            visited = {start_l}
            self.distances[start_l][start_l] = 0

            while queue:
                current_l, dist = queue.popleft()

                # Check if current_l exists in the graph keys before accessing neighbors
                # This handles cases where a location might be in 'locations' set but have no roads
                if current_l in self.road_graph:
                    for next_l in self.road_graph[current_l]:
                        if next_l not in visited:
                            visited.add(next_l)
                            self.distances[start_l][next_l] = dist + 1
                            queue.append((next_l, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Quickly check if it's a goal state by checking if all goal facts are in the state.
        # This is a necessary condition for h=0.
        if self.goals <= state:
            return 0

        # Track where packages and vehicles are currently located or contained.
        current_package_location_info = {} # package -> {'type': 'at'/'in', 'location': l / 'vehicle': v}
        current_vehicle_locations = {}     # vehicle -> location

        # Populate current locations/containment from the state facts.
        # Iterate through state facts once.
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args
                # Check if the object is one of the packages we care about (i.e., has a goal)
                if obj in self.packages:
                    current_package_location_info[obj] = {'type': 'at', 'location': loc}
                # Assume any other object with 'at' predicate is a vehicle
                # This is a simplification based on domain structure and object types.
                # We assume objects in 'at' facts that are not packages are vehicles.
                elif obj not in self.packages:
                     current_vehicle_locations[obj] = loc

            elif predicate == 'in' and len(args) == 2:
                package, vehicle = args
                 # Check if the object is one of the packages we care about (i.e., has a goal)
                if package in self.packages:
                    current_package_location_info[package] = {'type': 'in', 'vehicle': vehicle}
                # Vehicles are not 'in' anything in this domain, so no need to track vehicles here.

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that is not at its goal.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at the goal location on the ground
            # We need to check the exact goal fact string
            goal_fact_str = f"(at {package} {goal_location})"
            if goal_fact_str in state:
                continue # Package is already at its goal location on the ground

            # Find the package's current state info
            current_info = current_package_location_info.get(package)

            # If package state is not found, it's an unexpected state.
            # This implies the package is neither at a location nor in a vehicle.
            # This shouldn't happen in a valid state reachable from the initial state.
            # Return infinity to prune this path.
            if current_info is None:
                 return float('inf')


            if current_info['type'] == 'at':
                current_l = current_info['location']

                # Cost: pick + drive + drop
                # We need to find the distance from current_l to goal_l
                # Check if current_l is a known location before lookup
                if current_l not in self.distances:
                     return float('inf') # Unknown location

                dist = self.distances[current_l].get(goal_location)

                # If goal location is unreachable from current location, return infinity
                if dist is None:
                    return float('inf')

                # Cost = 1 (pick) + distance (drive) + 1 (drop)
                total_cost += 1 + dist + 1

            elif current_info['type'] == 'in':
                current_v = current_info['vehicle']

                # Find the location of the vehicle
                v_current_l = current_vehicle_locations.get(current_v)

                # If vehicle location is not found, it's an unexpected state.
                # This implies a package is in a vehicle, but the vehicle's location is unknown.
                # This shouldn't happen in a valid state.
                if v_current_l is None:
                    return float('inf')

                # Cost: drive + drop
                # We need to find the distance from the vehicle's location to the goal_l
                 # Check if vehicle's current location is a known location before lookup
                if v_current_l not in self.distances:
                     return float('inf') # Unknown location

                dist = self.distances[v_current_l].get(goal_location)

                 # If goal location is unreachable from vehicle's current location, return infinity
                if dist is None:
                    return float('inf')

                # Cost = distance (drive) + 1 (drop)
                total_cost += dist + 1

        return total_cost
