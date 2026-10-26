from heuristics.heuristic_base import Heuristic
from task import Task
import collections

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum estimated
        actions required for each package that is not yet at its goal location.
        For a package at its current location, the cost is estimated as
        1 (pick-up) + shortest_distance(current_location, goal_location) + 1 (drop).
        For a package inside a vehicle, the cost is estimated as
        shortest_distance(vehicle_location, goal_location) + 1 (drop).
        Shortest distances between locations are precomputed using BFS on the
        road network. Capacity and vehicle availability are simplified/ignored
        in the action cost estimation for individual packages.

    Assumptions:
        - The state representation is a frozenset of strings like '(predicate arg1 arg2)'.
        - The goal is a frozenset of facts, potentially including '(at package location)'.
        - The road network is static and defined by '(road l1 l2)' facts in task.static.
        - Packages relevant to the heuristic are those mentioned in '(at package location)'
          goal facts.
        - Vehicle capacity and specific vehicle assignment are not explicitly
          modeled in the heuristic cost calculation for simplicity and efficiency.
          A vehicle is assumed to be available and have sufficient capacity
          when needed for a package.
        - State always represents packages being either 'at' a location or 'in' a vehicle.

    Heuristic Initialization:
        1. Parses goal facts to map each package mentioned in an '(at package location)'
           goal fact to its goal location. Identifies these packages as 'relevant'.
        2. Extracts all location objects (names) from the task facts.
        3. Builds the road network graph (adjacency list) from '(road l1 l2)' static facts.
        4. Computes all-pairs shortest paths between all identified locations using BFS
           starting from each location. Stores distances in a dictionary.

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state (frozenset of facts).
        2. If the state is the goal state (checked using `task.goal_reached`), return 0.
        3. Initialize total heuristic value `h = 0`.
        4. Create temporary mappings for the current state:
           - `current_location_map`: maps object (package or vehicle) to its location if `(at obj l)` is true.
           - `current_package_vehicle`: maps package to vehicle if `(in p v)` is true.
        5. Iterate through each package `p` in the set of 'relevant packages' (those with a goal location).
        6. Get the goal location `l_goal` for package `p`.
        7. Check if `(at p l_goal)` is true in the current state. If yes, the package is at its goal; continue to the next package (cost is 0 for this package).
        8. If the package is not at goal:
           - Check if `p` is in `current_location_map`. If yes, get `l_current = current_location_map[p]`.
             - Estimate cost for this package: 1 (pick-up) + `shortest_distance(l_current, l_goal)` + 1 (drop).
             - If `shortest_distance(l_current, l_goal)` is infinity, the cost for this package is infinity.
           - Else, check if `p` is in `current_package_vehicle`. If yes, get `v = current_package_vehicle[p]`.
             - Find the vehicle's current location `l_v = current_location_map.get(v)`.
             - If `l_v` is found:
               - Estimate cost for this package: `shortest_distance(l_v, l_goal)` + 1 (drop).
               - If `shortest_distance(l_v, l_goal)` is infinity, the cost for this package is infinity.
             - If `l_v` is not found (invalid state), the cost for this package is infinity.
           - Else (package is neither at a location nor in a vehicle - invalid state), the cost for this package is infinity.
        9. If the cost for the current package is infinity, the total heuristic `h` becomes infinity. Break the loop.
        10. Otherwise, add the estimated cost for the current package to `h`.
        11. After iterating through all relevant packages, return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = self._parse_goals(task.goals)
        self.locations = self._get_all_locations(task)
        self.distances = self._compute_shortest_paths(task.static, self.locations)
        self.relevant_packages = set(self.package_goals.keys()) # Packages that need to reach a goal

    def _parse_fact(self, fact_str: str) -> tuple:
        """Parses a fact string like '(predicate arg1 arg2)' into a tuple."""
        # Remove surrounding brackets and split by space
        content = fact_str[1:-1]
        parts = content.split()
        return tuple(parts)

    def _parse_goals(self, goals: frozenset[str]) -> dict[str, str]:
        """Extracts package goal locations from the goal facts."""
        package_goals = {}
        for goal_fact_str in goals:
            fact = self._parse_fact(goal_fact_str)
            # Assuming goals relevant to this heuristic are always (at package location)
            if fact[0] == 'at' and len(fact) == 3:
                package, location = fact[1], fact[2]
                package_goals[package] = location
        return package_goals

    def _get_all_locations(self, task: Task) -> list[str]:
        """Extracts all location objects (names) from the task facts."""
        locations = set()
        # Identify predicates where an argument is a location type
        location_arg_indices = {
            'road': [1, 2], # (road l1 l2)
            'at': [2],     # (at obj l)
        }

        # Iterate through all possible ground facts in the task
        # task.facts contains all ground facts from initial state, static, and goals
        for fact_str in task.facts:
             fact = self._parse_fact(fact_str)
             predicate = fact[0]
             if predicate in location_arg_indices:
                 arg_indices = location_arg_indices[predicate]
                 for idx in arg_indices:
                     if idx < len(fact): # Ensure index is valid
                         locations.add(fact[idx])

        return list(locations)

    def _compute_shortest_paths(self, static_facts: frozenset[str], locations: list[str]) -> dict[tuple[str, str], float]:
        """Computes all-pairs shortest paths using BFS."""
        graph = collections.defaultdict(list)
        for fact_str in static_facts:
            fact = self._parse_fact(fact_str)
            if fact[0] == 'road' and len(fact) == 3:
                l1, l2 = fact[1], fact[2]
                graph[l1].append(l2)

        distances = {}
        for start_loc in locations:
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

            # For locations not reachable from start_loc, distance remains infinity
            for other_loc in locations:
                 if (start_loc, other_loc) not in distances:
                     distances[(start_loc, other_loc)] = float('inf')

        return distances

    def __call__(self, node) -> float:
        state = node.state

        # Return 0 if the goal is reached
        if self.task.goal_reached(state):
             return 0

        h_value = 0
        current_location_map = {} # Maps obj -> loc for any obj that is 'at' a location
        current_package_vehicle = {} # Maps package -> vehicle for any package that is 'in' a vehicle

        # Build current state mappings
        for fact_str in state:
            fact = self._parse_fact(fact_str)
            if fact[0] == 'at' and len(fact) == 3:
                obj, loc = fact[1], fact[2]
                current_location_map[obj] = loc
            elif fact[0] == 'in' and len(fact) == 3:
                package, vehicle = fact[1], fact[2]
                current_package_vehicle[package] = vehicle

        # Calculate heuristic for each relevant package
        for package in self.relevant_packages:
            l_goal = self.package_goals[package]

            # Check if package is already at its goal location
            # We check the fact string directly for efficiency
            if f'(at {package} {l_goal})' in state:
                 continue # Cost is 0 for this package

            # Package is not at goal, calculate cost
            package_cost = float('inf')

            # Case 1: Package is at a location
            if package in current_location_map:
                l_current = current_location_map[package]
                dist = self.distances.get((l_current, l_goal), float('inf'))
                if dist != float('inf'):
                    # Cost = pick-up (1) + drive (dist) + drop (1)
                    package_cost = 1 + dist + 1
                else:
                    package_cost = float('inf') # Cannot reach goal location

            # Case 2: Package is in a vehicle
            elif package in current_package_vehicle:
                vehicle = current_package_vehicle[package]
                l_v = current_location_map.get(vehicle) # Get vehicle's location
                if l_v is not None:
                    dist = self.distances.get((l_v, l_goal), float('inf'))
                    if dist != float('inf'):
                        # Cost = drive (dist) + drop (1)
                        package_cost = dist + 1
                    else:
                        package_cost = float('inf') # Cannot reach goal location
                else:
                    # Vehicle location unknown - invalid state?
                    package_cost = float('inf')

            # Case 3: Package is neither at a location nor in a vehicle (invalid state)
            else:
                 package_cost = float('inf')

            # Add package cost to total heuristic
            if package_cost == float('inf'):
                h_value = float('inf') # If any package is unreachable, goal is unreachable
                break # No need to check other packages
            else:
                h_value += package_cost

        return h_value
