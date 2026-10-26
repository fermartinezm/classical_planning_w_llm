# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque # Use deque for efficient BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise error
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for malformed facts
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the costs for all
    packages not yet at their destination. The cost for a package includes
    pickup, drop, and the shortest path distance the package needs to travel
    while inside a vehicle. It ignores vehicle capacity and availability
    for simplicity and efficiency.

    # Assumptions
    - The road network is static. The heuristic computes shortest paths based on
      the `(road ?l1 ?l2)` facts.
    - Each package needs to reach a specific goal location specified in the task goals.
    - The cost of pickup, drop, and drive actions is 1.
    - Vehicle capacity is ignored.
    - Vehicle initial positions are ignored when a package is on the ground;
      it's assumed a vehicle can reach the package's location. The heuristic
      only counts the drive cost *with* the package.
    - Objects starting with 'v' are vehicles, and objects listed in goal
      '(at ?p ?l)' facts are packages.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build the road network graph from static `(road ?l1 ?l2)` facts.
    - Compute the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Create mappings for the current state by iterating through state facts:
       - `package_status`: Maps package name to its status ('at', 'in') and value (location or vehicle name).
       - `vehicle_locations`: Maps vehicle name to its current location.
    3. For each package `p` that has a goal location `L_goal` (identified during initialization):
        a. Retrieve the current status and position/vehicle for package `p` from `package_status`. If the package's status is not found in the state, add a large penalty.
        b. If the status is 'at' and the current location is `L_goal`, the cost for this package is 0. Continue to the next package.
        c. If the status is 'at' and the current location is `L_curr` (`L_curr != L_goal`):
           - This package needs to be picked up (1 action), driven from `L_curr` to `L_goal` (estimated by the shortest path distance `dist(L_curr, L_goal)` drive actions), and dropped at `L_goal` (1 action).
           - Add `1 + dist(L_curr, goal_location) + 1` to the total cost. Use precomputed distances. Handle unreachable locations with a large penalty.
        d. If the status is 'in' and the package is inside vehicle `V`:
           - Retrieve the current location of vehicle `V` from `vehicle_locations`, say `L_curr_v`. If the vehicle's location is not found, add a large penalty.
           - Cost for this package: drive (dist(L_curr_v, goal_location)) + drop (1).
           - Add `dist(L_curr_v, goal_location) + 1` to the total cost. Use precomputed distances. Handle unreachable locations with a large penalty.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
            # Assuming no other goal types for packages in this domain

        # Build the road network graph and collect all locations.
        self.roads = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store locations as a list

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """
        Perform BFS starting from start_loc to find shortest distances to all
        reachable locations. Stores results in self.distances.
        """
        queue = deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}
        self.distances[(start_loc, start_loc)] = 0

        while queue:
            current_loc, current_dist = queue.popleft()

            # Get neighbors from the road graph
            neighbors = self.roads.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[(start_loc, neighbor)] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))

        # Locations not visited from start_loc are unreachable.
        # Their distance from start_loc will not be in self.distances.
        # We handle this during lookup in __call__ using .get() with a default.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Populate current status and locations from the state facts.
        package_status = {} # Maps package -> ('at', location) or ('in', vehicle)
        vehicle_locations = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj, location = args
                # Check if this object is one of the packages we care about (from goals)
                if obj in self.package_goals:
                    package_status[obj] = ('at', location)
                # Check if this object is a vehicle (simple heuristic based on name prefix)
                # This assumes vehicles are the only other locatables besides packages.
                elif obj.startswith('v'):
                     vehicle_locations[obj] = location
                # else: it's some other locatable we don't need to track for this heuristic

            elif predicate == "in" and len(args) == 2:
                 package, vehicle = args
                 # Assume the first argument of 'in' is always a package
                 if package in self.package_goals: # Ensure it's a package we care about
                     package_status[package] = ('in', vehicle)
                 # else: it's a package not in our goals, ignore for heuristic

        total_cost = 0  # Initialize action cost counter.
        unreachable_penalty = 1000000 # Large cost for unreachable goals

        # Iterate through each package that needs to reach a goal location.
        for package, goal_location in self.package_goals.items():
            # Get the current status of the package.
            # If a package is not mentioned in any 'at' or 'in' fact in the state,
            # it's an unexpected state. Add a penalty.
            current_status_info = package_status.get(package)

            if current_status_info is None:
                 # Package status not found. Treat as unreachable or problematic.
                 # print(f"Warning: Status of package {package} not found in state.")
                 total_cost += unreachable_penalty
                 continue

            status, current_pos_or_veh = current_status_info

            # Case 1: Package is at a location.
            if status == 'at':
                L_curr = current_pos_or_veh
                if L_curr == goal_location:
                    # Package is already at its goal. Cost is 0 for this package.
                    continue
                else:
                    # Package needs pickup, drive, drop.
                    # Cost: 1 (pickup) + dist(L_curr, goal_location) + 1 (drop)
                    distance = self.distances.get((L_curr, goal_location), float('inf'))
                    if distance == float('inf'):
                        total_cost += unreachable_penalty
                    else:
                        total_cost += 2 + distance

            # Case 2: Package is inside a vehicle.
            elif status == 'in':
                vehicle = current_pos_or_veh
                # Find the location of the vehicle.
                L_curr_v = vehicle_locations.get(vehicle)

                if L_curr_v is None:
                    # Vehicle location not found. Treat as unreachable or problematic.
                    # print(f"Warning: Location of vehicle {vehicle} containing package {package} not found.")
                    total_cost += unreachable_penalty
                    continue

                # Package needs drive (if vehicle not at goal), drop.
                # Cost: dist(L_curr_v, goal_location) + 1 (drop)
                distance = self.distances.get((L_curr_v, goal_location), float('inf'))
                if distance == float('inf'):
                     total_cost += unreachable_penalty
                else:
                    total_cost += 1 + distance

            # else: Unexpected status (should not happen based on domain)

        # The heuristic is 0 if and only if total_cost is 0, which happens
        # iff all packages in package_goals are currently at their goal_location.
        # This aligns with the goal condition for this domain.

        return total_cost
