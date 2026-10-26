# from fnmatch import fnmatch # Not used in this implementation
from heuristics.heuristic_base import Heuristic
import collections

# Helper function to parse PDDL facts
def get_parts(fact_string):
    """Extract predicate and arguments from a PDDL fact string."""
    return fact_string[1:-1].split()

# Helper function for BFS
def bfs_shortest_paths(locations, roads):
    """
    Computes shortest path distances between all pairs of locations
    in the road network using BFS.

    Args:
        locations: A list or set of all location objects.
        roads: A list of tuples (l1, l2) representing road connections.

    Returns:
        A dictionary where distances[(l1, l2)] is the shortest distance
        from l1 to l2. Returns float('inf') if no path exists.
    """
    adj = collections.defaultdict(set)
    for l1, l2 in roads:
        adj[l1].add(l2)
        adj[l2].add(l1) # Assuming roads are bidirectional

    distances = {}
    all_locations_list = list(locations) # Use a list for consistent iteration order

    for start_node in all_locations_list:
        q = collections.deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0

        while q:
            current_loc, dist = q.popleft()

            for neighbor in adj[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    # Ensure all pairs have a distance entry (inf if unreachable)
    for l1 in all_locations_list:
        for l2 in all_locations_list:
            if (l1, l2) not in distances:
                 distances[(l1, l2)] = float('inf')

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, ignoring vehicle capacity and assuming
    vehicles are always available for pick-up. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Vehicle capacity constraints are ignored.
    - A vehicle is assumed to be available at any location where a package needs
      to be picked up.
    - Roads are bidirectional.
    - The cost of each action (drive, pick-up, drop) is 1.
    - The goal is solely defined by the 'at' locations of packages.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Determine the current location of every locatable object (packages and vehicles)
       and which packages are inside which vehicles by iterating through the state facts.
    3. For each package 'p' that has a goal location 'l_goal' defined in the problem:
       a. Check if the package is currently at its goal location on the ground.
          If '(at p l_goal)' is in the state, the cost for this package is 0. Continue to the next package.
       b. If the package is not at its goal location:
          i. If the package is currently at a location 'l_curr' (i.e., '(at p l_curr)' is in the state, and l_curr != l_goal):
             - Estimated cost for this package: 1 (pick-up) + distance('l_curr', 'l_goal') (drive) + 1 (drop).
             - If distance('l_curr', l_goal) is infinity, the goal is unreachable, return infinity.
             - Add this cost to the total heuristic cost.
          ii. If the package is currently inside a vehicle 'v' (i.e., '(in p v)' is in the state):
             - Find the current location 'l_v' of vehicle 'v' (i.e., find '(at v l_v)' in the state).
             - Estimated cost for this package: distance('l_v', 'l_goal') (drive) + 1 (drop).
             - If distance('l_v', l_goal) is infinity, the goal is unreachable, return infinity.
             - Add this cost to the total heuristic cost.
          iii. If the package's location is not found (neither 'at' nor 'in'), this indicates an invalid state or a package
               without a goal specified in the initial state (which shouldn't happen if it's listed in the goal). Return infinity.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Assuming goals are only (at ?p ?l) for packages
            if parts[0] == "at":
                # The first argument of a goal 'at' predicate is assumed to be a package.
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Extract road network and locations
        roads = []
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                roads.append((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # Compute shortest path distances between all locations
        self.distances = bfs_shortest_paths(list(locations), roads)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Map current locations of locatables (packages and vehicles)
        current_locations = {}
        # Map which package is in which vehicle
        package_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                locatable, location = parts[1], parts[2]
                current_locations[locatable] = location
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground
            if (package in current_locations and current_locations[package] == goal_location):
                 # Package is at goal location on the ground, cost is 0 for this package
                 continue

            # If not at goal, calculate cost based on current state
            if package in current_locations: # Package is on the ground at current_locations[package]
                current_loc = current_locations[package]
                # Cost = pick-up + drive + drop
                # Assuming a vehicle is available at current_loc
                drive_cost = self.distances.get((current_loc, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # If goal is unreachable for this package, the whole state is likely unsolvable
                    return float('inf')
                total_cost += 1 + drive_cost + 1 # pick + drive + drop
            elif package in package_in_vehicle: # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in current_locations: # Vehicle location is known
                    vehicle_loc = current_locations[vehicle]
                    # Cost = drive + drop
                    drive_cost = self.distances.get((vehicle_loc, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         # If goal is unreachable for this package, the whole state is likely unsolvable
                         return float('inf')
                    total_cost += drive_cost + 1 # drive + drop
                else:
                    # This state shouldn't happen in a valid problem (vehicle must be somewhere)
                    # But as a fallback, assume infinite cost if vehicle location is unknown
                    return float('inf')
            else:
                 # Package location is unknown - this indicates an invalid state or a package
                 # without a goal specified in the initial state (which shouldn't happen
                 # if it's listed in the goal). Return infinity.
                 return float('inf')

        # The heuristic is 0 if and only if total_cost is 0.
        # total_cost is 0 iff all packages in self.package_goals were found
        # to be at their goal locations in the state.
        # This assumes the goal is exactly the conjunction of (at ?p ?l_goal)
        # for all p in self.package_goals. This matches the example problems.

        return total_cost
