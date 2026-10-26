from fnmatch import fnmatch
from collections import deque
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
    This heuristic estimates the total number of actions (pick-up, drop, drive)
    required to move all packages to their goal locations. It calculates a cost
    for each package not at its goal, summing up the actions needed for that
    package independently, including the shortest path distance for driving.

    # Assumptions
    - Roads are bidirectional.
    - Any vehicle can transport any package (ignoring specific capacity constraints beyond the existence of *some* capacity).
    - Vehicles are available when needed at the package's current location if the package is on the ground (ignores vehicle location and availability for pick-up).
    - The cost of driving between two locations is the number of road segments (shortest path distance).

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on `road` facts from the static information.
    - Computes all-pairs shortest path distances between locations using BFS on the road graph.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that is not currently at its goal location `goal_l_p` (as defined in the task goals):
    1. Determine the package's current state by examining the state facts: Is it on the ground at `current_l_p` (fact `(at p current_l_p)`) or inside a vehicle `current_v_p` (fact `(in p current_v_p)`)? If it's in a vehicle, find the vehicle's current location `current_l_v` (fact `(at current_v_p current_l_v)`).
    2. If the package is on the ground at `current_l_p` and `current_l_p` is different from `goal_l_p`:
       - It needs to be picked up (1 action).
       - It needs to be transported from `current_l_p` to `goal_l_p`. The estimated cost for this driving is the shortest path distance between `current_l_p` and `goal_l_p` (number of drive actions).
       - It needs to be dropped at `goal_l_p` (1 action).
       - The total estimated cost for this package is 1 (pick) + distance(`current_l_p`, `goal_l_p`) + 1 (drop).
    3. If the package is inside a vehicle `current_v_p` which is at `current_l_v`, and `current_l_v` is different from `goal_l_p`:
       - It needs to be transported from `current_l_v` to `goal_l_p` inside the vehicle. The estimated cost is the shortest path distance between `current_l_v` and `goal_l_p`.
       - It needs to be dropped at `goal_l_p` (1 action).
       - The total estimated cost for this package is distance(`current_l_v`, `goal_l_p`) + 1 (drop).
    4. If the package is already at its goal location (either `(at p goal_l_p)` is true, or `(in p ?v)` is true and `(at ?v goal_l_p)` is true, although the heuristic simplifies this by just checking `current_pos == goal_l_p`), the cost for this package is 0.
    5. The total heuristic value is the sum of the estimated costs for all packages that are not yet at their goal location.

    This heuristic is non-admissible because it sums costs for packages independently, potentially overcounting drive actions if a single vehicle trip moves multiple packages towards their goals. It also ignores vehicle capacity and specific vehicle availability constraints.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between all locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road graph and collect all unique locations.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads

        # Add any locations mentioned in goals or initial state that might not be in road facts
        # (e.g., isolated locations, though less common in transport domains)
        for loc in self.goal_locations.values():
             locations.add(loc)
        # Note: Initial state locations are not directly available in task.initial_state as just locations,
        # they are part of 'at' facts. We'll handle this during BFS initialization or rely on
        # locations discovered from roads and goals covering relevant places.
        # A more robust approach would parse all 'at' facts in the initial state to get all locations.
        # For now, assume locations in roads and goals cover the necessary graph.

        # Compute all-pairs shortest path distances.
        self.distances = {}
        all_locations_list = list(locations) # Use a list for consistent ordering if needed, or just iterate set
        for start_loc in all_locations_list:
             # Ensure start_loc is in the graph keys if it has roads, otherwise BFS from it is trivial/impossible
             if start_loc in self.road_graph or start_loc in locations: # Check if it's a known location
                self.distances[start_loc] = self._bfs(start_loc, all_locations_list)


    def _bfs(self, start_loc, all_locations):
        """Perform BFS from start_loc to find distances to all other locations."""
        distances_from_start = {loc: float('inf') for loc in all_locations}
        distances_from_start[start_loc] = 0
        queue = deque([start_loc])

        # Handle isolated locations not in the road graph keys but present in all_locations
        if start_loc not in self.road_graph and start_loc in all_locations:
             # It's an isolated location. Distances to itself is 0, others are inf.
             return distances_from_start


        while queue:
            current_loc = queue.popleft()
            current_dist = distances_from_start[current_loc]

            # Check if current_loc has any roads connected
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances_from_start[neighbor] == float('inf'):
                        distances_from_start[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances_from_start

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_locations = {} # Maps package -> location string or vehicle string
        vehicle_locations = {} # Maps vehicle -> location string

        # Populate current locations from state facts
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                # Need to distinguish packages from vehicles.
                # Assume objects in goal_locations are packages.
                if obj in self.goal_locations:
                     package_locations[obj] = location
                else:
                     # Assume anything else with 'at' is a vehicle for this heuristic
                     vehicle_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args
                package_locations[package] = vehicle # Package is inside vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # If a package in the goal is not mentioned in the current state's
            # 'at' or 'in' facts, it's an unexpected state.
            if package not in package_locations:
                 # This shouldn't happen in valid states reachable from a valid initial state
                 # where all goal packages exist. Return infinity to indicate a potential issue
                 # or an unsolvable path if this state was reached.
                 return float('inf')

            current_pos = package_locations[package] # This is either a location string or a vehicle string

            # Check if the package is already at its goal location
            # If the package is 'in' a vehicle, it's not 'at' the goal location yet.
            if current_pos == goal_location:
                continue # Package is already where it needs to be

            # Case 1: Package is on the ground at current_pos
            # Check if current_pos is a location by seeing if it's in our distance map keys.
            if current_pos in self.distances:
                current_l = current_pos
                # Cost: pick (1) + drive (distance) + drop (1)
                # Need distance from current_l to goal_l
                dist = self.distances.get(current_l, {}).get(goal_location, float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from current location
                    return float('inf') # Problem likely unsolvable or heuristic fails

                total_cost += 2 + dist

            # Case 2: Package is inside a vehicle current_pos
            else: # current_pos is a vehicle name
                current_v = current_pos
                # Need the location of the vehicle
                if current_v not in vehicle_locations:
                     # Vehicle location unknown - indicates state representation issue or error
                     # Or perhaps vehicle is not 'at' any location (e.g., initial state error)
                     # Return infinity as heuristic might be misleading
                     return float('inf')

                current_l_v = vehicle_locations[current_v]

                # Cost: drive (distance) + drop (1)
                # Need distance from vehicle's current location to goal_l
                dist = self.distances.get(current_l_v, {}).get(goal_location, float('inf'))

                if dist == float('inf'):
                     # Goal location is unreachable from vehicle's current location
                     return float('inf') # Problem likely unsolvable or heuristic fails

                total_cost += 1 + dist

        # The loop sums costs only for packages NOT at their goal.
        # If total_cost is 0 after the loop, it means all goal packages were already at their goal.
        # This correctly results in h=0 for goal states.

        return total_cost
