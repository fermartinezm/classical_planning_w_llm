from fnmatch import fnmatch
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts (optional, but good practice)
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

# Define the heuristic class
# Assuming Heuristic base class is available in the environment, e.g., from heuristics.heuristic_base import Heuristic
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package that
    is not yet at its goal. The cost for a package depends on whether it is
    on the ground or inside a vehicle, and involves pick-up/drop actions plus
    the shortest path distance (in terms of drive actions) required to get
    the package or its containing vehicle to the package's goal location.

    # Assumptions
    - The road network is static and provides the movement options for vehicles.
    - Vehicle capacity is ignored for simplicity and efficiency. Any vehicle
      is assumed to be able to pick up any package if at the same location.
    - Vehicle availability at a package's location is ignored. It is assumed
      a vehicle can be moved to the package's location if needed.
    - The cost of each action (drive, pick-up, drop) is 1.
    - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles.
      (This assumption is made for parsing state facts to distinguish object types).

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static `(road l1 l2)` facts.
    - Computes the shortest path distance (number of drive actions) between
      all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and every vehicle. A package
       can be on the ground `(at p l)` or inside a vehicle `(in p v)`. A vehicle
       is always on the ground `(at v l)`. This is done by parsing the state facts,
       assuming object names starting with 'p' are packages and 'v' are vehicles.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `loc_goal` (extracted during initialization):
        a. Check if the goal fact `(at p loc_goal)` is already present in the current state.
           If yes, the cost for this package is 0, and we move to the next package.
        b. If the goal fact is not in the state:
            i. Find the package's current status from the parsed state: Is it `(at p loc_curr)`
               or `(in p v)`?
            ii. If `(at p loc_curr)`:
                - The estimated cost for this package is 1 (for the pick-up action) +
                  the shortest_distance from `loc_curr` to `loc_goal` (for drive actions) +
                  1 (for the drop action).
            iii. If `(in p v)`:
                - Find the current location of vehicle `v`, say `loc_v`, using the parsed `(at v loc_v)` fact.
                - The estimated cost for this package is the shortest_distance from `loc_v` to `loc_goal`
                  (for drive actions) + 1 (for the drop action).
            iv. If the required shortest_distance is infinite (no path exists), the state is considered
                to lead to an unreachable goal for this package, and the total heuristic returns infinity.
        c. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total sum as the heuristic value. If any package's goal was unreachable,
       infinity is returned.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args[0], args[1]
                self.goal_locations[package] = location

        # Build the road network graph and compute all-pairs shortest paths.
        self.distances = {}
        self.locations = set()
        road_graph = {}

        # Extract locations and build graph from road facts
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                road_graph.setdefault(l1, []).append(l2)

        # Also add locations mentioned in initial state and goals to ensure they are included
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc)
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 self.locations.add(loc)

        # Ensure all locations are in the graph keys, even if they have no outgoing roads
        for loc in self.locations:
             road_graph.setdefault(loc, [])

        # Compute shortest paths from every location to every other location using BFS
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0 # Distance to self is 0

            while q:
                current_loc, dist = q.popleft()

                # Store the distance found
                self.distances[(start_node, current_loc)] = dist

                # Explore neighbors
                if current_loc in road_graph: # Check if current_loc has outgoing roads
                    for neighbor in road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

            # After BFS from start_node, ensure distances to all other locations are stored
            # (reachable ones were stored in the loop, unreachable ones remain inf)
            for loc in self.locations:
                 if (start_node, loc) not in self.distances:
                      self.distances[(start_node, loc)] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        current_status = {} # Maps object (package or vehicle) to its location or vehicle it's in
        vehicle_locations = {} # Maps vehicle to its location

        # Identify packages and vehicles based on naming convention (p* for package, v* for vehicle)
        # This is a simplifying assumption based on provided examples.
        packages_in_state = set()
        vehicles_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                if obj.startswith('p'):
                    packages_in_state.add(obj)
                elif obj.startswith('v'):
                    vehicles_in_state.add(obj)
                    vehicle_locations[obj] = loc # Store vehicle location
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside vehicle
                packages_in_state.add(package)
                vehicles_in_state.add(vehicle) # Ensure vehicle is known

        # Consider all packages that have a goal or are currently in the state
        all_packages_to_consider = set(self.goal_locations.keys()) | packages_in_state


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package in all_packages_to_consider:
            goal_location = self.goal_locations.get(package)

            # If package has no goal, skip it
            if goal_location is None:
                 continue

            # Check if the package is already at its goal location predicate-wise
            # The goal is always (at package goal_location)
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal, calculate cost
            current_status_of_package = current_status.get(package)

            if current_status_of_package is None:
                 # Package with a goal is not mentioned in 'at' or 'in' facts. Invalid state?
                 # Assume unreachable goal for this package from this state.
                 return float('inf')

            # Check if the package is on the ground or in a vehicle
            if current_status_of_package in self.locations:
                # Package is on the ground at current_status_of_package
                loc_curr = current_status_of_package
                # Cost: pick-up + drive + drop
                # Need distance from loc_curr to goal_location
                drive_cost = self.distances.get((loc_curr, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # Goal is unreachable from package's current location on the ground
                    return float('inf')
                total_cost += 1 + drive_cost + 1 # pick-up + drive + drop

            elif current_status_of_package in vehicles_in_state: # Check if it's a known vehicle
                # Package is inside a vehicle (current_status_of_package is the vehicle name)
                vehicle = current_status_of_package
                loc_v = vehicle_locations.get(vehicle) # Get the vehicle's location

                if loc_v is None:
                    # Vehicle exists but its location is not known? Invalid state.
                    return float('inf')

                # Cost: drive + drop
                # Need distance from vehicle's location to package's goal location
                drive_cost = self.distances.get((loc_v, goal_location), float('inf'))
                if drive_cost == float('inf'):
                     # Goal is unreachable for the vehicle carrying the package
                     return float('inf')

                total_cost += drive_cost + 1 # drive + drop
            else:
                 # current_status_of_package is neither a location nor a known vehicle. Invalid state?
                 return float('inf')

        # If we reached here without returning inf, the state is reachable and cost is finite.
        # If total_cost is 0, it means all packages with goals are (at p goal_l), which is the goal state.
        # So h=0 iff state is goal is satisfied.
        return total_cost
