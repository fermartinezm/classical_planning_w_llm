from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections # Used for BFS queue

# Helper functions
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
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, independently of other packages and
    vehicle capacity constraints. It sums the estimated costs for all packages
    that are not yet at their goal.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles can move between connected locations (roads).
    - Any vehicle can pick up any package at the same location (ignoring capacity).
    - Any vehicle can drop any package it contains at its current location (ignoring capacity).
    - The cost of a 'drive', 'pick-up', or 'drop' action is 1.
    - The cost of moving a vehicle between two locations is the shortest path distance
      in the road network.

    # Heuristic Initialization
    - Parses goal facts to determine the target location for each package.
    - Identifies all locations, vehicles, and packages present in the problem
      definition (from initial state, goals, and static facts) by inferring types
      from predicate usage.
    - Builds a graph of the road network from static facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package,
       and the current location for every vehicle, by examining the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location:
       a. Check if the package is currently at its goal location. If yes, add 0 to cost for this package.
       b. If the package is currently at a location `L` (and `L` is not the goal):
          - Estimate the cost to move this package to its goal location `G`.
          - This requires a 'pick-up' (cost 1), driving a vehicle from `L` to `G`
            (cost = shortest path distance `dist(L, G)`), and a 'drop' (cost 1).
          - Add `1 + dist(L, G) + 1` to the total cost.
          - If `G` is unreachable from `L` via roads, the state is likely unsolvable or very far;
            return infinity.
       c. If the package is currently inside a vehicle `V`:
          - Find the current location `VL` of vehicle `V`.
          - Estimate the cost to move this package to its goal location `G`.
          - This requires driving vehicle `V` from `VL` to `G`
            (cost = shortest path distance `dist(VL, G)`) and a 'drop' (cost 1).
          - Add `dist(VL, G) + 1` to the total cost.
          - If `G` is unreachable from `VL` via roads, return infinity.
       d. If the package's state is not known (not 'at' a location or 'in' a vehicle),
          return infinity as the state is likely invalid.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and building the road network graph.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Collect all potential objects and locations by inferring types from predicate usage
        self.locations = set()
        self.vehicles = set()
        self.packages = set()
        self.sizes = set()
        all_symbols = set()

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            all_symbols.update(parts)
            if parts[0] == "road":
                self.locations.add(parts[1])
                self.locations.add(parts[2])
            elif parts[0] == "capacity-predecessor":
                self.sizes.add(parts[1])
                self.sizes.add(parts[2])

        # Process goal facts
        for goal in self.goals:
             parts = get_parts(goal)
             all_symbols.update(parts)
             if parts[0] == "at":
                 # Assume first arg of 'at' in goal is a package
                 self.packages.add(parts[1])
                 self.locations.add(parts[2])

        # Process initial state facts
        initial_at_objects = set()
        for fact in initial_state:
            parts = get_parts(fact)
            all_symbols.update(parts)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                self.locations.add(loc)
                initial_at_objects.add(obj)
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                self.packages.add(package) # Anything 'in' is a package
                self.vehicles.add(vehicle) # Anything a package is 'in' is a vehicle
            elif parts[0] == "capacity":
                vehicle, size = parts[1], parts[2]
                self.vehicles.add(vehicle) # Anything with 'capacity' is a vehicle
                self.sizes.add(size)

        # Refine typing: Any symbol appearing in an 'at' fact in the initial state
        # that hasn't been classified as a package (from goals/in) or vehicle (from capacity/in)
        # is assumed to be a vehicle. This is an inference based on domain structure.
        for symbol in initial_at_objects:
             if symbol not in self.packages and symbol not in self.vehicles:
                  self.vehicles.add(symbol)

        # Build the road network graph
        road_graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                if l1 in road_graph and l2 in road_graph:
                     road_graph[l1].append(l2)
                     road_graph[l2].append(l1) # Assuming bidirectional roads

        # Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc, road_graph)

    def _bfs(self, start_node, graph):
        """Perform BFS to find shortest distances from start_node."""
        distances = {node: float('inf') for node in graph}
        if start_node in distances:
            distances[start_node] = 0
            queue = collections.deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()

                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if neighbor in distances and neighbor not in visited:
                            visited.add(neighbor)
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        package_current_state = {} # Map package -> location or vehicle
        vehicle_current_location = {} # Map vehicle -> location

        # Populate current locations/states from the current state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_location[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                package_current_state[package] = vehicle

        total_cost = 0

        # Iterate through each package that has a goal
        for package, goal_location in self.package_goals.items():
            current_state = package_current_state.get(package)

            # If package is not found in the current state facts, it's an invalid state
            if current_state is None:
                return float('inf')

            # If package is already at goal, cost is 0 for this package
            if current_state == goal_location:
                continue

            # If package is in a vehicle
            if current_state in self.vehicles:
                vehicle = current_state
                vehicle_loc = vehicle_current_location.get(vehicle)

                # If vehicle location is unknown (invalid state)
                if vehicle_loc is None:
                     return float('inf')

                # Cost to move vehicle to goal location + drop action
                drive_cost = self.shortest_paths.get(vehicle_loc, {}).get(goal_location, float('inf'))

                # If goal is unreachable from vehicle's current location
                if drive_cost == float('inf'):
                     return float('inf')

                total_cost += drive_cost + 1 # drive actions + 1 drop action

            # If package is at a location
            elif current_state in self.locations:
                package_loc = current_state

                # Cost to pick up + drive to goal + drop
                drive_cost = self.shortest_paths.get(package_loc, {}).get(goal_location, float('inf'))

                # If goal is unreachable from package's current location
                if drive_cost == float('inf'):
                     return float('inf')

                total_cost += 1 + drive_cost + 1 # 1 pick-up + drive actions + 1 drop action

            else:
                # Package state is unknown or invalid (e.g., 'at p unknown_object')
                return float('inf')

        return total_cost
