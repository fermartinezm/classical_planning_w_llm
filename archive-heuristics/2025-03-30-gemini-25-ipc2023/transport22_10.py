import math
from collections import deque
# Assuming the heuristic base class is available via this path
# Adjust the import path if necessary based on the project structure.
# e.g., from planning_framework.heuristics.heuristic_base import Heuristic
from heuristics.heuristic_base import Heuristic

def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space. Handles leading/trailing whitespace.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    """
    return fact.strip()[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'transport'.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their designated goal locations. It calculates the cost for each package
    individually based on its current state (at a location or inside a vehicle)
    and sums these costs. The cost includes the estimated number of drive actions
    (based on shortest paths between locations) plus the necessary pick-up and drop actions (1 action each).
    It aims for accuracy to guide greedy best-first search effectively, but is not necessarily admissible.

    # Assumptions
    - The primary goal is to satisfy all `(at package location)` predicates specified in the goal.
      Other goal types (e.g., vehicle locations) might exist but are only checked for final goal state recognition.
    - The cost of moving a package is estimated independently of other packages. Interactions like
      multiple packages in one vehicle are not explicitly modeled for cost calculation beyond using shared drives.
    - The heuristic ignores vehicle capacity constraints (`capacity`, `capacity-predecessor`, `size`).
      It assumes any vehicle can pick up any package if they are at the same location, regardless of capacity.
    - The heuristic assumes any available vehicle can be used for transport, ignoring
      which specific vehicle might be optimal or currently closest for the initial pickup drive. The cost
      calculation focuses on the package's journey.
    - The cost estimate for a package `p` currently at location `loc_p` needing to go to `loc_g`
      is `1 (for pick-up) + shortest_path_dist(loc_p, loc_g) (for drive actions) + 1 (for drop)`.
    - The cost estimate for a package `p` currently inside vehicle `v` which is at `loc_v`, with the package
      needing to go to `loc_g`, is `shortest_path_dist(loc_v, loc_g) (for drive actions) + 1 (for drop)`.
    - Roads (`road l1 l2`) define connectivity. The heuristic uses the shortest path distance in terms
      of the number of `drive` actions. Roads are treated as potentially directed based on the static
      facts provided (e.g., if only `(road l1 l2)` exists, it's a one-way path from l1 to l2).

    # Heuristic Initialization
    - Stores the task object to check for goal completion later using `task.goal_reached()`.
    - Parses the task's goal conditions (`task.goals`) to identify the target location (`loc_g`) for each package (`p`).
      Stores these mappings in the `self.goal_locations` dictionary.
    - Identifies all unique location objects mentioned in `road` predicates (`task.static`), initial `at` predicates
      (`task.initial_state`), and goal `at` predicates (`task.goals`). Stores them in the `self.locations` set.
    - Identifies all unique package objects mentioned as the first argument in goal `at` predicates. Stores them in
      the `self.packages` set.
    - Builds a directed graph representation of the `road` network using the identified locations and
      static `road` facts. The graph is stored as an adjacency list `adj`.
    - Precomputes all-pairs shortest path distances between all known locations using Breadth-First Search (BFS)
      starting from each location. Stores distances in `self.distances[start_loc][end_loc]`.
      If `loc_g` is unreachable from `loc_s`, the distance is `float('inf')`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal State:** If the input `node.state` satisfies all goal conditions defined in the task
        (checked using `self.task.goal_reached(state)`), the heuristic value is 0, indicating the goal is reached.
    2.  **Parse Current State:** Iterate through the facts in `node.state` to determine:
        - The current location of each object (vehicle or package) via `(at obj loc)` facts. Store in `current_locations` dict: `{obj: loc}`.
        - Which package is inside which vehicle via `(in pkg veh)` facts. Store in `package_containers` dict: `{pkg: veh}`.
    3.  **Initialize Cost:** Set `total_cost = 0.0` (using float to handle potential infinity).
    4.  **Iterate Packages:** For each package `p` that has a defined goal location `loc_g` in `self.goal_locations`:
        a.  **Determine Package Status:** Check if `p` is currently in a vehicle (`p in package_containers`) or at a location (`p in current_locations`).
        b.  **Handle Missing Package:** If `p` (which is required for the goal) is neither `in` a vehicle nor `at` a location in the current state, this implies an inconsistency or that the goal is unreachable from this state. Return `float('inf')`.
        c.  **Package at Location:** If `p` is `at` location `loc_p`:
            i.  If `loc_p == loc_g`, the package is already at its destination. Add 0 to `total_cost`.
            ii. If `loc_p != loc_g`, find the shortest path distance `dist = self.distances.get(loc_p, {}).get(loc_g, float('inf'))`. Using `.get()` provides robustness against missing keys.
            iii.If `dist` is `infinity`, the goal location is unreachable for this package. Return `float('inf')`.
            iv. Otherwise, add the estimated cost `1 (pickup) + dist (drive) + 1 (drop)` to `total_cost`.
        d.  **Package in Vehicle:** If `p` is `in` vehicle `v`:
            i.  Find the current location `loc_v` of vehicle `v` using `current_locations.get(v)`.
            ii. If `loc_v` cannot be found (vehicle has no `at` fact), the state is inconsistent. Raise a `ValueError`.
            iii.Find the shortest path distance `dist = self.distances.get(loc_v, {}).get(loc_g, float('inf'))`.
            iv. If `dist` is `infinity`, the goal location is unreachable for this package. Return `float('inf')`.
            v.  Otherwise, add the estimated cost `dist (drive) + 1 (drop)` to `total_cost`.
        e.  **Check for Infinity:** If `total_cost` becomes `infinity` at any point, immediately return `float('inf')`.
    5.  **Handle Zero Cost for Non-Goal States:** After iterating through all packages, if `total_cost` is 0 but the state was determined *not* to be a goal state in Step 1, return 1. This ensures the heuristic is strictly positive for non-goal states, preventing premature termination in search algorithms like Greedy Best-First Search if only the package subgoals are met but other potential goal conditions are not.
    6.  **Return Total Cost:** Return the calculated `total_cost`. If infinity was returned earlier, that value propagates. Otherwise, return the final sum, cast to an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic: stores task, parses goals, identifies objects,
        builds road graph, and precomputes all-pairs shortest path distances.
        """
        self.task = task # Store task for goal checking later
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify all unique location objects from various sources
        self.locations = set()
        potential_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Add locations from road facts
            if parts[0] == "road" and len(parts) == 3:
                potential_locations.add(parts[1])
                potential_locations.add(parts[2])
        for fact in task.initial_state:
            parts = get_parts(fact)
            # Add locations from initial 'at' facts
            if parts[0] == "at" and len(parts) == 3:
                potential_locations.add(parts[2]) # object is at location
        for goal in self.goals:
             parts = get_parts(goal)
             # Add locations from goal 'at' facts
             if parts[0] == "at" and len(parts) == 3:
                 potential_locations.add(parts[2]) # goal is at location

        # Assume all collected names are valid locations in this domain context
        self.locations = potential_locations

        # 2. Identify packages and their goal locations from goal 'at' predicates
        self.packages = set()
        self.goal_locations = {} # package -> goal_location
        for goal in self.goals:
            parts = get_parts(goal)
            # Assume goals of the form (at package location) define package goals
            # A more robust system might use type information if available from the task object
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Heuristic: if the object is the first arg in 'at' goal, it's a package
                self.packages.add(package)
                self.goal_locations[package] = location
                # Ensure goal location is in our known locations set for graph consistency
                if location not in self.locations:
                    # This might happen if a location only appears in the goal
                    # Add it defensively, but it might indicate an issue in the PDDL instance
                    # print(f"Warning: Goal location {location} for package {package} "
                    #       f"was not identified initially. Adding to locations.")
                    self.locations.add(location)

        # 3. Build road graph (directed) and compute shortest paths
        # Adjacency list: map location -> list of reachable locations in one step
        adj = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Add edge only if both ends are known locations
                if l1 in self.locations and l2 in self.locations:
                    adj[l1].append(l2)
                # else:
                    # This might indicate an issue if road refers to non-location objects
                    # print(f"Warning: Road fact {fact} involves unknown/non-location objects.")

        self.distances = self._compute_shortest_paths(self.locations, adj)

    def _compute_shortest_paths(self, locations, adj):
        """
        Computes all-pairs shortest paths using BFS for each location as a source.
        Handles directed graphs based on the adjacency list `adj`.
        Returns a dict of dicts: distances[start_loc][end_loc] = shortest_dist.
        Distance is float('inf') if unreachable.
        """
        distances = {loc: {other_loc: float('inf') for other_loc in locations} for loc in locations}
        for start_node in locations:
            # Basic check if start_node is in the keys (it should be)
            if start_node not in distances: continue

            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # visited_dist stores the shortest distance found *so far* from start_node in this BFS run
            visited_dist = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_dist[current_node]

                # Explore neighbors using the adjacency list
                for neighbor in adj.get(current_node, []):
                    # If neighbor hasn't been reached yet in this BFS run
                    if neighbor not in visited_dist:
                        visited_dist[neighbor] = current_dist + 1
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
                    # For standard BFS on unweighted graphs, the first time we reach a node,
                    # it's via a shortest path from the source for that run.
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Estimates the minimum number of actions (drive, pickup, drop)
        to move all packages specified in the goals to their goal locations.
        Returns 0 for goal states, infinity for unreachable states,
        and a positive integer estimate otherwise.
        """
        state = node.state

        # Check if the goal is already reached using the task's method
        if self.task.goal_reached(state):
            return 0

        total_cost = 0.0 # Use float for potential infinity

        # 1. Parse the current state to find locations and containers
        current_locations = {} # obj -> location, for objects currently 'at' a location
        package_containers = {} # pkg -> vehicle, for packages currently 'in' a vehicle
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif pred == "in" and len(parts) == 3:
                pkg, vehicle = parts[1], parts[2]
                package_containers[pkg] = vehicle

        # 2. Calculate cost for each package based on its goal
        for package, goal_loc in self.goal_locations.items():
            cost_for_package = float('inf') # Default to infinity if status unknown or unreachable

            if package in package_containers:
                # Package is inside a vehicle
                vehicle = package_containers[package]
                vehicle_loc = current_locations.get(vehicle) # Find vehicle's location

                if vehicle_loc is None:
                    # This signifies an inconsistent state where the container vehicle has no location
                    raise ValueError(f"State inconsistent: Vehicle {vehicle} containing "
                                     f"package {package} has no 'at' predicate in state {state}")

                # Get shortest distance from vehicle's location to package's goal location
                # Use .get() for robustness against potentially missing keys in distance map
                dist = self.distances.get(vehicle_loc, {}).get(goal_loc, float('inf'))

                if dist != float('inf'):
                    # Cost = drive distance + 1 drop action
                    cost_for_package = dist + 1
                # else: cost_for_package remains infinity

            elif package in current_locations:
                # Package is at a location on the ground/map
                current_loc = current_locations[package]

                if current_loc == goal_loc:
                    # Package is already at its goal
                    cost_for_package = 0
                else:
                    # Package needs pickup, drive, drop
                    # Get shortest distance from current location to goal location
                    dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                    if dist != float('inf'):
                        # Cost = 1 pickup + drive distance + 1 drop
                        cost_for_package = 1 + dist + 1
                    # else: cost_for_package remains infinity

            else:
                # The package (which is required for the goal) is not 'in' a vehicle
                # and not 'at' a location in the current state.
                # This implies the state is invalid or the goal is unreachable.
                # print(f"Warning: Goal package {package} not found in state {state}")
                return float('inf') # Return infinity immediately

            # Add the cost for this package to the total
            total_cost += cost_for_package

            # If at any point total_cost becomes infinity, the goal is unreachable.
            if total_cost == float('inf'):
                 return float('inf')


        # If total_cost is 0, but we know it's not a goal state (from the check at the start),
        # return 1. This ensures the heuristic is strictly positive for non-goal states,
        # preventing search algorithms from terminating prematurely if only package subgoals are met.
        if total_cost == 0:
             # We already established it's not a goal state overall
             return 1

        # Return the final cost, cast to an integer.
        # Since all components (1, dist) are integers, total_cost should be integer if finite.
        return int(total_cost)
