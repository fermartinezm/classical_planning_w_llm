from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base.py is available in a 'heuristics' directory
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package from its current location to its goal location, assuming
    unlimited vehicle capacity and availability. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Any vehicle can carry any package (ignores capacity constraints).
    - A suitable vehicle is always available at the package's current location
      if needed for pick-up.
    - The road network is static and provides bidirectional travel if a road
      exists in both directions. The heuristic uses shortest path distance
      on this network.
    - The cost of pick-up, drop, and drive actions is 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts.
    - Collects all relevant locations mentioned in the problem.
    - Computes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it on the ground at a location `L`, or is it inside a vehicle `V`?
    2. If the package is on the ground at `L`:
       - If `L` is the goal location, the package is done (cost 0 for this package).
       - If `L` is not the goal location, it needs to be picked up (1 action), transported by a vehicle from `L` to the goal location (shortest path distance actions), and then dropped at the goal location (1 action). The total estimated cost for this package is `distance(L, goal_location) + 2`.
    3. If the package is inside a vehicle `V`:
       - Determine the current location of vehicle `V`, say `L_V`.
       - If `L_V` is the goal location, the package only needs to be dropped (1 action). The total estimated cost for this package is 1.
       - If `L_V` is not the goal location, the vehicle needs to transport the package from `L_V` to the goal location (shortest path distance actions), and then the package needs to be dropped (1 action). The total estimated cost for this package is `distance(L_V, goal_location) + 1`.
    4. The total heuristic value for the state is the sum of the estimated costs for all packages that are not yet at their goal location on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting package goals, building the
        road network, and precomputing shortest path distances.
        """
        # Extract goal locations for each package
        self.package_goals = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Build the road network graph and collect all relevant locations
        self.road_graph = {}
        locations = set()

        # Collect locations from static road facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Add locations from initial state 'at' facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                loc = parts[2]
                locations.add(loc)

        # Add locations from goal 'at' facts
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == "at":
                  loc = parts[2]
                  locations.add(loc)

        # Ensure all collected locations are in the graph dictionary keys
        # This is important so BFS is attempted from/to all relevant locations.
        for loc in locations:
             if loc not in self.road_graph:
                  self.road_graph[loc] = [] # Add node with no outgoing edges


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations_list = list(locations) # Convert set to list for consistent iteration order (optional)

        for start_node in all_locations_list:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc has neighbors in the graph (it should, due to the loop above)
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        current_package_at = {}  # package -> loc (if on ground)
        current_package_in = {}  # package -> vehicle (if in vehicle)
        current_vehicle_locations = {} # vehicle -> loc

        # Parse the current state to find locations of packages and vehicles
        # We assume any object in self.package_goals is a package.
        # Any object appearing as the second argument of '(in ... obj)' is a vehicle.
        # Any object appearing as the first argument of '(at obj loc)' that is *not* a package must be a vehicle.
        all_packages_in_goals = set(self.package_goals.keys())

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # Check if the object is a package based on whether it's in our goal list
                if obj in all_packages_in_goals:
                    current_package_at[obj] = loc
                # Assume anything else with an 'at' predicate is a vehicle
                else:
                    current_vehicle_locations[obj] = loc
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                # Only track packages that are relevant to the goal
                if pkg in all_packages_in_goals:
                    current_package_in[pkg] = veh
                # We don't need to track vehicles here, their location is in 'at' facts


        total_cost = 0  # Initialize action cost counter.

        # Iterate through all packages that have a goal location
        for package, goal_loc in self.package_goals.items():
            cost_for_package = 0

            # Case 1: Package is on the ground
            if package in current_package_at:
                current_loc = current_package_at[package]
                if current_loc != goal_loc:
                    # Needs pick-up, drive, and drop
                    if (current_loc, goal_loc) not in self.distances:
                        # Goal is unreachable from current location
                        return float('inf') # Problem is unsolvable from this state
                    dist = self.distances[(current_loc, goal_loc)]
                    cost_for_package = dist + 2 # 1 for pick-up, dist for drive, 1 for drop
                # If current_loc == goal_loc, cost_for_package is 0, which is correct.

            # Case 2: Package is inside a vehicle
            elif package in current_package_in:
                vehicle = current_package_in[package]
                # Find the vehicle's current location
                if vehicle not in current_vehicle_locations:
                    # This state is inconsistent: package is in vehicle, but vehicle location is unknown.
                    # Return infinity as it's likely an unsolvable or malformed state.
                    return float('inf')

                current_loc_v = current_vehicle_locations[vehicle]

                if current_loc_v != goal_loc:
                    # Needs drive and drop
                    if (current_loc_v, goal_loc) not in self.distances:
                         # Goal is unreachable from vehicle's current location
                         return float('inf') # Problem is unsolvable from this state
                    dist = self.distances[(current_loc_v, goal_loc)]
                    cost_for_package = dist + 1 # dist for drive, 1 for drop
                else: # current_loc_v == goal_loc
                    # Needs only drop
                    cost_for_package = 1

            # Case 3: Package from goal list is not found in 'at' or 'in' facts.
            # This implies a malformed state representation or the package doesn't exist.
            # In a valid state, every package should be locatable.
            # Return infinity as it's likely an unsolvable or malformed state.
            else:
                 return float('inf')

            total_cost += cost_for_package

        # The heuristic is 0 if and only if total_cost is 0.
        # This occurs only when all goal packages are found in current_package_at
        # and their current_loc matches their goal_loc. This correctly identifies goal states.
        return total_cost
