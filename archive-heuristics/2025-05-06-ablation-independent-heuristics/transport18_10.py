# Need to import necessary classes from the planning framework
# Assuming Heuristic and Task are available from these imports
from heuristics.heuristic_base import Heuristic
from task import Task

# Need deque and defaultdict for graph processing
from collections import deque, defaultdict

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required
    actions for each package that is not yet at its goal location.
    The cost for a package depends on its current status (at a location or
    in a vehicle) and the shortest path distance from its current location
    to its goal location in the road network.

    Assumptions:
    - The state representation is consistent (a package is either at one
      location or in one vehicle, and a vehicle is at one location).
    - The road network is static and provided in the initial state facts.
    - Capacity constraints are simplified: The heuristic assumes a vehicle
      can eventually be made available with sufficient capacity to pick up
      a package, and focuses primarily on the movement costs (pick-up, drive, drop).
      It does not model complex capacity management or vehicle availability.
    - The heuristic is non-admissible, designed for greedy best-first search.

    Heuristic Initialization:
    1. Parses the goal facts to identify which packages need to be at which
       locations. Stores this mapping in `self.package_goals`.
    2. Parses the static 'road' facts to build a graph representation of the
       road network. Identifies all unique locations mentioned in road facts
       and goal facts.
    3. Computes the shortest path distance between all pairs of these locations
       using Breadth-First Search (BFS) starting from each location. Stores
       these distances in `self.distances`.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (represented as a frozenset of fact strings):
    1. Initialize the total heuristic value `h` to 0.
    2. Create dictionaries to quickly look up the current location of vehicles
       and the current status of packages (`at` a location or `in` a vehicle).
    3. Iterate through the facts in the current state:
       - If a fact is `(at obj loc)`:
         - If `obj` is a package whose goal is tracked, record its status as `('at', loc)`.
         - If `obj` is assumed to be a vehicle (e.g., starts with 'v'), record its location.
       - If a fact is `(in package vehicle)`:
         - If `package` is a package whose goal is tracked, record the vehicle it is in.
    4. After iterating through state facts, process the 'in' facts: For each package
       recorded as being 'in' a vehicle, find the location of that vehicle from the
       recorded vehicle locations. Record the package's status as `('in', vehicle, vehicle_location)`.
    5. For each package `p` that is listed in `self.package_goals` with a target
       location `goal_l`:
       a. Check if the goal fact `(at p goal_l)` is already true in the current state. If yes,
          this package contributes 0 to the heuristic, so continue to the next
          package.
       b. Get the current status of package `p` from the dictionary built in steps 3 and 4.
       c. If the package's status is unknown (e.g., not found in `at` or `in` facts, or vehicle location unknown),
          return `float('inf')` as the state might be inconsistent or unsolvable.
       d. Determine the package's current location `current_l` based on its status
          (either directly from `('at', current_l)` or from the vehicle's location
          in `('in', vehicle, current_l)`).
       e. If `current_l` or `goal_l` are not found in the precomputed distances
          (meaning they are not part of the known network or unreachable), return `float('inf')`.
       f. Get the shortest path distance `d` from `current_l` to `goal_l` from
          `self.distances`.
       g. Calculate the cost contribution for package `p` based on its status type:
          - If status is `('at', current_l)`: Cost = `d + 2` (pick-up, drive, drop).
          - If status is `('in', vehicle, current_l)`:
            - If `current_l == goal_l`: Cost = `1` (drop).
            - If `current_l != goal_l`: Cost = `d + 1` (drive, drop).
       h. Add this cost to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = self._parse_package_goals(task.goals)
        self.distances = self._precompute_distances(task.static)

    def _parse_package_goals(self, goals):
        """Parses goal facts to map packages to their target locations."""
        package_goals = {}
        for goal_fact in goals:
            # Goal facts are typically (at package location)
            predicate, args = self._parse_fact(goal_fact)
            if predicate == 'at' and len(args) == 2:
                package = args[0]
                location = args[1]
                package_goals[package] = location
        return package_goals

    def _parse_fact(self, fact_str):
        """Parses a fact string into predicate and arguments."""
        # Remove parentheses and split by space
        # Handle potential empty string or malformed fact
        if not fact_str or not isinstance(fact_str, str) or fact_str[0] != '(' or fact_str[-1] != ')':
             return None, []
        content = fact_str[1:-1].strip()
        if not content:
            return None, []
        parts = content.split()
        if not parts:
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def _precompute_distances(self, static_facts):
        """Builds road graph and computes all-pairs shortest paths."""
        graph = defaultdict(list)
        locations = set()

        for fact_str in static_facts:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'road' and len(args) == 2:
                l1, l2 = args
                graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Add locations from package goals to ensure they are included
        for loc in self.package_goals.values():
             locations.add(loc)

        distances = {}
        # Ensure all locations found are keys in the distances dict, even if isolated
        for loc in locations:
             distances[loc] = {} # Initialize inner dict

        for start_node in locations:
            # Run BFS from each location
            distances[start_node].update(self._bfs(graph, start_node, locations))

        return distances

    def _bfs(self, graph, start_node, all_nodes):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        dist = {node: float('inf') for node in all_nodes}
        if start_node in all_nodes: # Ensure start_node is one of the known locations
            dist[start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                current_dist = dist[u]

                # Check if u is in graph (it might be a location with no outgoing roads)
                # If u is not in graph, it has no neighbors, loop is skipped.
                if u in graph:
                    for v in graph[u]:
                        # Ensure neighbor v is one of the known locations
                        if v in all_nodes and dist[v] == float('inf'):
                            dist[v] = current_dist + 1
                            queue.append(v)
        return dist

    def __call__(self, node):
        """Computes the heuristic value for the given state."""
        state = node.state
        h = 0

        # Map packages and vehicles to their current locations/status
        package_current_status = {} # {package: ('at', location) or ('in', vehicle, location)}
        vehicle_locations = {}      # {vehicle: location}
        package_in_vehicle_map = {} # {package: vehicle} # Temp storage for 'in' facts

        # Pass 1: Find vehicle locations and package 'at' locations, and package 'in' vehicle facts
        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'at' and len(args) == 2:
                obj, loc = args
                # Check if obj is a package in our goals
                if obj in self.package_goals:
                     package_current_status[obj] = ('at', loc)
                # Assume objects starting with 'v' are vehicles based on domain naming convention
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                 package, vehicle = args
                 if package in self.package_goals:
                     package_in_vehicle_map[package] = vehicle

        # Pass 2: Process packages that are 'in' vehicles, using the vehicle locations found
        for package, vehicle in package_in_vehicle_map.items():
             if vehicle in vehicle_locations:
                 vehicle_loc = vehicle_locations[vehicle]
                 package_current_status[package] = ('in', vehicle, vehicle_loc)
             # else: vehicle location unknown, package status remains unknown, handled later


        # Calculate heuristic contribution for each package in the goal
        for package, goal_l in self.package_goals.items():
            # Check if the goal fact is already true
            if f'(at {package} {goal_l})' in state:
                continue # Goal for this package is met

            # Get package's current status
            status = package_current_status.get(package)

            if status is None:
                 # Package is not 'at' any location and not 'in' any vehicle whose location is known.
                 # This indicates an inconsistent state or the package was
                 # never at the initial location specified.
                 # Return infinity to signal an issue or unsolvable state from here.
                 # print(f"Warning: Status of package {package} unknown in state.")
                 return float('inf')

            status_type = status[0]
            current_l = None

            if status_type == 'at':
                current_l = status[1] # ('at', location)
            elif status_type == 'in':
                current_l = status[2] # ('in', vehicle, location)

            # Ensure current_l is a valid location and goal_l is a valid location
            # Check if distance is computable (i.e., locations are in our distance map)
            # and if the goal location is reachable from the current location.
            if current_l not in self.distances or goal_l not in self.distances.get(current_l, {}) or self.distances[current_l][goal_l] == float('inf'):
                 # Goal location is unreachable from current location within the known network
                 # print(f"Warning: Goal location {goal_l} unreachable from {current_l} for package {package}.")
                 return float('inf')

            distance = self.distances[current_l][goal_l]

            if status_type == 'at':
                # Package is at current_l, needs pick-up, drive, drop
                h += distance + 2
            elif status_type == 'in':
                # Package is in vehicle at current_l
                if current_l == goal_l:
                    # Needs only drop
                    h += 1
                else:
                    # Needs drive and drop
                    h += distance + 1

        return h
