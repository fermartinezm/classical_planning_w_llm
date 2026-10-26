from fnmatch import fnmatch
from collections import deque
# from heuristics.heuristic_base import Heuristic # Assuming this is available

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
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

# class transportHeuristic(Heuristic): # Inherit from Heuristic if needed
class transportHeuristic: # Provide just the class as requested
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the estimated
    costs for each package that is not yet at its goal location. The estimated
    cost for a package is based on its current state (on the ground or in a vehicle)
    and the shortest road distance to its goal location.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Capacity constraints are ignored (optimistic assumption).
    - A vehicle is always available to pick up a package if needed (optimistic assumption).
    - The shortest path distance between locations is the minimum number of drive actions.
    - Objects appearing as the first argument in 'at' goal facts are considered packages.
    - Objects appearing as the first argument in 'at' facts in the state, which are not identified as packages, are considered vehicles.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a road network graph from the static `road` facts to compute shortest distances.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that needs to be at location `l_goal` according to the goal state:
    1. Check if `p` is already at `l_goal` on the ground (i.e., `(at p l_goal)` is in the current state). If yes, the cost for this package is 0.
    2. If `p` is not at `l_goal`, determine its current status:
       a. If `p` is on the ground at `l_current` (i.e., `(at p l_current)` is in the state, where `l_current != l_goal`):
          - It needs to be picked up (1 action).
          - It needs to be transported by a vehicle from `l_current` to `l_goal`. The minimum number of drive actions is the shortest distance `dist(l_current, l_goal)` in the road network.
          - It needs to be dropped at `l_goal` (1 action).
          - Estimated cost for this package: 1 (pick) + `dist(l_current, l_goal)` + 1 (drop).
       b. If `p` is inside a vehicle `v` (i.e., `(in p v)` is in the state):
          - Find the current location of vehicle `v`, say `l_v` (i.e., `(at v l_v)` is in the state).
          - The vehicle needs to be transported from `l_v` to `l_goal`. The minimum number of drive actions is `dist(l_v, l_goal)`.
          - It needs to be dropped at `l_goal` (1 action).
          - Estimated cost for this package: `dist(l_v, l_goal)` + 1 (drop).
    3. The total heuristic value is the sum of the estimated costs for all packages not at their goal location.
    4. If any required location is unreachable via the road network, the distance is considered infinite, and the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the road graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        self.packages = set() # Keep track of all packages mentioned in goals
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                obj, location = args
                # Assume objects in 'at' goals are packages
                self.goal_locations[obj] = location
                self.packages.add(obj)


        # Build the road network graph from static facts.
        self.road_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                self.road_graph[loc1].append(loc2)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

    def bfs(self, start_loc, end_loc):
        """
        Computes the shortest path distance between two locations using BFS on the road graph.
        Returns float('inf') if the end location is unreachable from the start location.
        """
        if start_loc == end_loc:
            return 0

        # Ensure start and end locations are known
        if start_loc not in self.all_locations or end_loc not in self.all_locations:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            # Check neighbors
            if current_loc in self.road_graph:
                for next_loc in self.road_graph[current_loc]:
                    if next_loc == end_loc:
                        return dist + 1
                    if next_loc not in visited:
                        visited.add(next_loc)
                        queue.append((next_loc, dist + 1))

        # If BFS completes without finding the end_loc
        return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current status of packages and locations of vehicles.
        # Status can be {'type': 'at', 'location': loc} or {'type': 'in', 'vehicle': v}
        current_package_status = {}
        current_vehicle_locations = {} # {vehicle: location}

        # First pass: Identify packages in vehicles
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                if package in self.packages: # Only track packages we care about (in goals)
                    current_package_status[package] = {'type': 'in', 'vehicle': vehicle}

        # Second pass: Identify locations of vehicles and packages on the ground
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    # This object is a package. If it's not already marked as 'in' a vehicle, it's on the ground.
                    if obj not in current_package_status:
                         current_package_status[obj] = {'type': 'at', 'location': loc}
                else:
                    # This object is not a package we care about. Assume it's a vehicle.
                    current_vehicle_locations[obj] = loc

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # Need to check the exact fact string in the state set
            goal_fact_string = f"(at {package} {goal_location})"
            if goal_fact_string in state:
                 continue # Goal achieved for this package

            # Package is not at the goal location on the ground. Find its current status.
            status = current_package_status.get(package)

            if status is None:
                 # This package is in the goals but not found in 'at' or 'in' facts in the state.
                 # This indicates an invalid state or the package was never initialized.
                 # For heuristic purposes, treat as infinitely costly.
                 return float('inf')

            if status['type'] == 'at':
                # Package is on the ground at current_loc
                current_loc = status['location']
                # Cost: pick + drive + drop
                d = self.bfs(current_loc, goal_location)
                if d == float('inf'):
                    return float('inf')
                total_cost += 1 + d + 1 # pick (1) + drive (d) + drop (1)

            elif status['type'] == 'in':
                # Package is in a vehicle
                vehicle = status['vehicle']
                # Find vehicle location
                vehicle_loc = current_vehicle_locations.get(vehicle)
                if vehicle_loc is None:
                    # Vehicle location not found. This shouldn't happen in a valid state
                    # where a package is in a vehicle, the vehicle must have a location.
                    return float('inf')

                # Cost: drive vehicle + drop package
                d = self.bfs(vehicle_loc, goal_location)
                if d == float('inf'):
                    return float('inf')
                total_cost += d + 1 # drive (d) + drop (1)

        return total_cost
