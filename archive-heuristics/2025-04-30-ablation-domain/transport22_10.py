from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return an empty list for malformed facts, which match() will handle
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must exactly match the number of arguments in the pattern.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each package not at its goal location,
    the estimated cost to move that package to its goal. The cost for a package is estimated as the number
    of load/unload actions plus the shortest path distance (number of drive actions) required.
    It ignores vehicle capacity and the possibility of transporting multiple packages simultaneously.

    # Assumptions
    - The cost of load, unload, and drive actions is 1.
    - Any package can be transported by any vehicle (ignoring size constraints).
    - A suitable vehicle is always available when needed to pick up a package.
    - The road network is static and provides bidirectional connections.
    - All locations mentioned in initial state, goals, or road facts are relevant nodes in the graph.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on the static `road` facts and locations mentioned in initial state/goals.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, return 0.
    2. Identify the current location of every object (packages and vehicles) by parsing the state facts (`at` predicate).
    3. Identify which packages are currently inside which vehicles by parsing the state facts (`in` predicate).
    4. Initialize the total heuristic cost to 0.
    5. For each package that has a specified goal location:
       a. Check if the package is already at its goal location in the current state using the `at` predicate. If yes, skip this package.
       b. Determine the package's current status: is it on the ground or inside a vehicle?
       c. If the package is currently inside a vehicle V:
          - Find the current location L_v of vehicle V. If the vehicle's location is unknown, the state is inconsistent, return infinity.
          - The estimated cost for this package is 1 (for unload) + the shortest path distance from L_v to the package's goal location (for drive actions). If the goal is unreachable from L_v, return infinity.
       d. If the package is currently on the ground at location L:
          - Find the current location L of the package. If the package's location is unknown, the state is inconsistent, return infinity.
          - The estimated cost for this package is 1 (for load) + the shortest path distance from L to the package's goal location (for drive actions) + 1 (for unload). If the goal is unreachable from L, return infinity.
       e. Add the estimated cost for this package to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the road network graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1:]
                self.goal_locations[package] = location

        # Collect all locations mentioned in the problem (roads, initial state, goals)
        locations = set()
        self.road_graph = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1:]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        # Add locations from initial state and goals to ensure they are nodes in the graph
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 loc = parts[2]
                 locations.add(loc)
                 self.road_graph.setdefault(loc, set()) # Add location node if not already present

        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at" and len(parts) == 3:
                 loc = parts[2]
                 locations.add(loc)
                 self.road_graph.setdefault(loc, set()) # Add location node if not already present

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_loc, all_locations):
        """
        Performs BFS from a start location to find shortest paths to all other locations.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in all_locations}
        # Handle cases where start_loc might not be in the collected locations (e.g., malformed input)
        if start_loc not in distances:
             # Cannot compute paths from an unknown location
             return distances # All distances remain inf

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc is in the road_graph keys before iterating
            # get(current_loc, set()) handles locations added from init/goals that have no roads
            for neighbor in self.road_graph.get(current_loc, set()):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Track where objects are currently located.
        current_locations = {} # object -> location
        package_in_vehicle = {} # package -> vehicle

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1:]
                current_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1:]
                package_in_vehicle[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if (f"(at {package} {goal_location})" in state):
                 continue # Package is already at its goal

            # Determine package's current status and location
            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Need vehicle's current location
                if vehicle not in current_locations:
                     # This indicates an inconsistent state (package in vehicle, but vehicle location unknown)
                     # For heuristic, treat as very high cost or unreachable
                     return float('inf')

                current_loc_v = current_locations[vehicle]

                # Cost: Unload + Drive
                # Need distance from vehicle's current location to package's goal location
                # Check if locations exist in shortest_paths and if path exists
                if current_loc_v not in self.shortest_paths or goal_location not in self.shortest_paths[current_loc_v]:
                     # This could happen if goal_location or current_loc_v was not in the set of all_locations during BFS
                     # or if goal is unreachable from current location.
                     return float('inf') # Indicate this path is likely bad

                dist = self.shortest_paths[current_loc_v][goal_location]

                if dist == float('inf'):
                    # Goal is unreachable from current vehicle location
                    return float('inf')

                total_cost += 1 # Unload action (cost 1)
                total_cost += dist # Drive actions (cost 1 each)

            else:
                # Package is on the ground
                if package not in current_locations:
                     # This indicates an inconsistent state (package not in vehicle, but location unknown)
                     return float('inf')

                current_loc_p = current_locations[package]

                # Cost: Load + Drive + Unload
                # Need distance from package's current location to its goal location
                if current_loc_p not in self.shortest_paths or goal_location not in self.shortest_paths[current_loc_p]:
                     return float('inf')

                dist = self.shortest_paths[current_loc_p][goal_location]

                if dist == float('inf'):
                    # Goal is unreachable from current package location
                    return float('inf')

                total_cost += 1 # Load action (cost 1)
                total_cost += dist # Drive actions (cost 1 each)
                total_cost += 1 # Unload action (cost 1)

        return total_cost
