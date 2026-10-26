from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
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
    This heuristic estimates the cost to reach the goal by summing the estimated
    minimum actions required for each package that is not yet at its goal location.
    The estimated cost for a single package includes:
    - If the package is on the ground: pick-up + drive + drop actions.
    - If the package is in a vehicle: drive + drop actions.
    The number of drive actions is estimated by the shortest path distance
    (number of road segments) between the relevant locations.

    # Assumptions
    - The cost of pick-up, drop, and drive actions is 1.
    - Vehicles are always available when needed at the package's current location
      or the vehicle's current location (ignores vehicle location and capacity constraints
      beyond the package being in a vehicle).
    - Shortest path distances between locations are pre-calculated based on static road facts.
    - All locations mentioned in the problem (initial state, goals, roads) are part of
      a connected road network, or paths between necessary locations exist.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goal conditions.
    - Builds the road network graph from static `road` facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package that is on the ground and every vehicle.
       Also, identify which package is inside which vehicle.
    2. Initialize total heuristic cost to 0.
    3. For each package specified in the goal conditions:
        a. Determine the package's goal location from the pre-computed goal mapping.
        b. Check if the package is already at its goal location in the current state.
           This means either `(at package goal_location)` is true, or `(in package vehicle)`
           is true for some vehicle, and `(at vehicle goal_location)` is true.
           If the package is at its goal, continue to the next package.
        c. If the package is not at its goal:
            i. Find the package's current status: Is it on the ground at some location `l_current`, or is it inside a vehicle `v`?
            ii. If the package is on the ground at `l_current`:
                - Estimate the cost as 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
            iii. If the package is inside a vehicle `v`:
                - Find the current location `l_v` of vehicle `v`.
                - Estimate the cost as shortest_distance(`l_v`, `l_goal`) (drive) + 1 (drop).
            iv. Add this estimated cost for the package to the total heuristic cost.
            Note: If a required distance is not found (locations are disconnected), a large cost is added.
            If a package's status (at location or in vehicle) or a vehicle's location is unknown
            in the current state, a large cost is added, assuming an invalid or problematic state.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.task = task # Store task for access if needed
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and compute shortest paths.
        self.distances = self._compute_shortest_paths(static_facts)

    def _compute_shortest_paths(self, static_facts):
        """
        Builds the road graph from static facts and computes all-pairs shortest paths.
        Returns a dictionary mapping (location1, location2) pairs to their distance.
        """
        graph = {}
        locations = set()

        # Build graph from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                _, loc1, loc2 = parts
                locations.add(loc1)
                locations.add(loc2)
                graph.setdefault(loc1, set()).add(loc2)
                graph.setdefault(loc2, set()).add(loc1) # Roads are bidirectional

        # Compute shortest paths using BFS from each location
        distances = {}
        for start_loc in locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to satisfy all package goal locations.
        """
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        package_locations = {} # Maps package -> location (if on ground)
        package_in_vehicle = {} # Maps package -> vehicle (if in vehicle)
        vehicle_locations = {} # Maps vehicle -> location

        # Populate current locations and containment based on the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == "at":
                # (at ?x ?l) where ?x is locatable (vehicle or package)
                obj, loc = parts[1], parts[2]
                # We can identify packages by checking if they are in the goal_locations map.
                # Other 'at' objects are assumed to be vehicles for the purpose of this heuristic.
                if obj in self.goal_locations:
                     package_locations[obj] = loc
                else:
                     # Assume it's a vehicle or other locatable not explicitly a package goal
                     vehicle_locations[obj] = loc # Store vehicle locations

            elif predicate == "in":
                # (in ?p ?v) where ?p is package, ?v is vehicle
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.
        LARGE_COST = 1000 # Penalty for unreachable or unknown states

        # Iterate through each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location
            is_at_goal = False
            if package in package_locations and package_locations[package] == goal_location:
                 is_at_goal = True
            elif package in package_in_vehicle:
                 vehicle = package_in_vehicle[package]
                 if vehicle in vehicle_locations and vehicle_locations[vehicle] == goal_location:
                      is_at_goal = True

            if is_at_goal:
                continue # Package is already at its goal, no cost added for this package

            # Package is not at its goal location. Estimate cost.
            if package in package_locations:
                # Package is on the ground at package_locations[package]
                current_location = package_locations[package]
                # Cost = pick-up (1) + drive (distance) + drop (1)
                # Need distance from current_location to goal_location
                drive_cost = self.distances.get((current_location, goal_location))

                if drive_cost is None:
                    # Goal location is unreachable from the package's current location on the ground
                    total_cost += LARGE_COST
                else:
                    total_cost += 1 + drive_cost + 1 # pick-up + drive + drop

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_locations:
                    current_vehicle_location = vehicle_locations[vehicle]
                    # Cost = drive (distance) + drop (1)
                    # Need distance from current_vehicle_location to goal_location
                    drive_cost = self.distances.get((current_vehicle_location, goal_location))

                    if drive_cost is None:
                        # Goal location is unreachable from the vehicle's current location
                        total_cost += LARGE_COST
                    else:
                         total_cost += drive_cost + 1 # drive + drop
                else:
                    # Package is in a vehicle, but vehicle location is unknown in the state.
                    total_cost += LARGE_COST

            else:
                # Package location/status is unknown (not at, not in).
                total_cost += LARGE_COST

        return total_cost

