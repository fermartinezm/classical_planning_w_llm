from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available as specified
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current location to its goal location,
    independently for each package. It uses shortest path distances on the road
    network to estimate drive costs.

    # Assumptions
    - The heuristic assumes that a suitable vehicle is always available
      to transport any package whenever needed, ignoring vehicle locations,
      capacities, and assignments.
    - The cost of picking up a package is 1.
    - The cost of dropping a package is 1.
    - The cost of driving a vehicle between two locations is the shortest
      path distance (number of road segments) between them.
    - The heuristic sums the estimated costs for each package independently.

    # Heuristic Initialization
    - Extracts all unique locations mentioned in the problem (initial state,
      goals, and static road facts).
    - Builds a graph representing the road network based on static `road` facts.
    - Computes all-pairs shortest path distances between all relevant locations
      using Breadth-First Search (BFS).
    - Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package. A package can be
       either at a physical location `l` (predicate `(at p l)`) or inside
        a vehicle `v` (predicate `(in p v)`).
    2. Identify the current location of every vehicle (predicate `(at v l)`).
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `l_goal`:
       a. Get the current location of `p`, let's call it `current_loc_p`.
       b. If `current_loc_p` is the same as `l_goal`, the package is already
          at its goal, so add 0 to the total cost for this package.
       c. If `current_loc_p` is a physical location (i.e., `(at p current_loc_p)`
          is true in the state):
          - The package needs to be picked up (cost +1).
          - A vehicle needs to drive from `current_loc_p` to `l_goal`.
            Estimate this drive cost as the shortest path distance between
            `current_loc_p` and `l_goal` (cost + `dist(current_loc_p, l_goal)`).
          - The package needs to be dropped at `l_goal` (cost +1).
          - Total cost for this package: `1 + dist(current_loc_p, l_goal) + 1`.
       d. If `current_loc_p` is a vehicle `v` (i.e., `(in p v)` is true in the state):
          - Find the current physical location of vehicle `v`, let's call it `l_v`.
          - The vehicle needs to drive from `l_v` to `l_goal`.
            Estimate this drive cost as the shortest path distance between
            `l_v` and `l_goal` (cost + `dist(l_v, l_goal)`).
          - The package needs to be dropped at `l_goal` (cost +1).
          - Total cost for this package: `dist(l_v, l_goal) + 1`.
       e. If the required distance cannot be found (e.g., goal location is
          unreachable from the current location in the road network), return
          a large value to indicate this path is likely not leading to the goal.
    5. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static  # Facts that are not affected by actions.

        # 1. Extract all unique locations mentioned in the problem
        all_locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                all_locations.add(l1)
                all_locations.add(l2)

        # Include locations from initial state and goals
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 all_locations.add(loc)

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 obj, loc = get_parts(goal)[1:]
                 all_locations.add(loc)

        # 2. Build the road graph
        self.road_graph = {loc: set() for loc in all_locations}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Roads are bidirectional

        # 3. Compute all-pairs shortest path distances using BFS
        self.distances = {}
        for start_loc in all_locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while queue:
                current_loc, dist = queue.popleft()

                # Ensure current_loc is a valid key in the graph
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

        # 4. Store goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.package_goals[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_locations = {} # Maps package name to its location (string) or vehicle name (string)
        vehicle_locations = {} # Maps vehicle name to its location (string)

        # Populate package_locations and vehicle_locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 3: # Facts like (at obj loc) or (in obj vehicle)
                 predicate = parts[0]
                 obj1 = parts[1]
                 obj2 = parts[2]

                 if predicate == "at":
                     # obj1 is locatable (vehicle or package), obj2 is location
                     # Check if obj1 is a package (by looking up in package_goals)
                     if obj1 in self.package_goals:
                         package_locations[obj1] = obj2 # Package is at a location
                     else:
                         vehicle_locations[obj1] = obj2 # Assume obj1 is a vehicle at a location
                 elif predicate == "in":
                     # obj1 is package, obj2 is vehicle
                     package_locations[obj1] = obj2 # Package is inside a vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package and its goal location
        for package, goal_location in self.package_goals.items():
            current_loc_p = package_locations.get(package)

            # If package location is not found in the state, something is wrong.
            # This shouldn't happen in a valid state representation.
            if current_loc_p is None:
                 # print(f"Warning: Package {package} location not found in state.")
                 return 1000000 # Large value indicating invalid state or unsolvable path

            # If package is already at goal, cost is 0 for this package.
            if current_loc_p == goal_location:
                continue

            # If package is not at goal, it needs transport.
            # Estimate minimum actions: pick-up, drive, drop.

            # Case 1: Package is on the ground at current_loc_p
            # Check if current_loc_p is a location string (i.e., a key in self.distances)
            if current_loc_p in self.distances:
                # Package is at a location, not in a vehicle.
                # Needs pick-up (1), drive, drop (1).
                dist = self.distances.get(current_loc_p, {}).get(goal_location)

                if dist is None:
                     # Goal location is unreachable from current location by driving.
                     # This state is likely not on a path to the goal. Return a large value.
                     # print(f"Warning: Distance from {current_loc_p} to {goal_location} not found (package on ground).")
                     return 1000000 # Large value indicating likely unsolvable path from here.

                total_cost += 1 # Pick-up action
                total_cost += dist # Drive actions
                total_cost += 1 # Drop action

            # Case 2: Package is inside a vehicle
            # current_loc_p is the vehicle name.
            # Check if current_loc_p is a known vehicle name (i.e., has a location in vehicle_locations)
            elif current_loc_p in vehicle_locations:
                vehicle_name = current_loc_p
                l_v = vehicle_locations.get(vehicle_name) # Get vehicle's location

                if l_v is None:
                     # Vehicle location not found? Should not happen in a valid state.
                     # print(f"Warning: Location for vehicle {vehicle_name} not found (package in vehicle).")
                     return 1000000 # Large value

                # Package is in a vehicle at l_v.
                # Needs drive from l_v to goal_location, drop (1).
                dist = self.distances.get(l_v, {}).get(goal_location)

                if dist is None:
                     # Goal location is unreachable from vehicle's location by driving.
                     # This state is likely not on a path to the goal. Return a large value.
                     # print(f"Warning: Distance from vehicle location {l_v} to package goal {goal_location} not found.")
                     return 1000000 # Large value

                total_cost += dist # Drive actions
                total_cost += 1 # Drop action
            else:
                # This case should not happen in a valid state if package_locations is populated correctly.
                # It means the package's location is neither a known location nor a known vehicle.
                # print(f"Warning: Package {package} has unknown location type: {current_loc_p}")
                return 1000000 # Large value

        return total_cost
