from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available from heuristics.heuristic_base

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed for individual elements).
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
    required to move each package to its goal location, ignoring vehicle capacity
    and assuming any vehicle can be used. The total heuristic is the sum of
    costs for each package not yet at its goal.

    # Assumptions
    - Roads are bidirectional (as seen in the example instance files).
    - Any vehicle can carry any package (ignoring size/capacity constraints).
    - The cost of pick-up, drop, and drive actions is 1.
    - The heuristic sums the minimum actions needed for each package independently.
    - Goal conditions only involve packages being at specific locations (`(at package location)`).
    - The state representation includes `(at object location)` for all relevant objects
      and `(in package vehicle)` for packages inside vehicles.

    # Heuristic Initialization
    - Builds a graph of locations based on `road` facts found in static information.
    - Computes shortest path distances between all pairs of locations using BFS on the location graph.
    - Extracts the goal location for each package from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every locatable object (packages and vehicles)
       using `(at object location)` facts present in the state.
    2. Identify which packages are currently inside which vehicles using `(in package vehicle)`
       facts present in the state.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a specified goal location (extracted during initialization):
       a. Check if the package is already at its goal location on the ground. If the fact
          `(at package goal_location)` is present in the state, this package is done;
          continue to the next package (cost is 0 for this package).
       b. If the package is not at its goal location, determine its current status:
          - If the package is currently inside a vehicle `v` (found via `(in package v)`
            in the state):
             - Find the vehicle's current location `current_v_l` using `(at v current_v_l)`
               in the state. If the vehicle's location is not found, assume the path
               is unreachable (return infinity).
             - The package needs to be transported from `current_v_l` to `goal_location`
               and then dropped.
             - The estimated cost for this package is the shortest path distance
               between `current_v_l` and `goal_location` (number of drive actions)
               plus 1 (drop action).
             - If the goal location is unreachable from the vehicle's current location
               (based on precomputed distances), return infinity.
          - If the package is currently on the ground at `current_l` (found via
            `(at package current_l)` in the state):
             - If the package's location is not found, assume the path is unreachable
               (return infinity).
             - The package needs to be picked up, transported from `current_l` to
               `goal_location`, and then dropped.
             - The estimated cost for this package is 1 (pick-up action)
               plus the shortest path distance between `current_l` and `goal_location`
               (number of drive actions) plus 1 (drop action).
             - If the goal location is unreachable from the package's current location
               (based on precomputed distances), return infinity.
       c. Add the estimated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost. If any package's goal is unreachable, the
       total cost will be infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        distances, and extracting package goal locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from road facts
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations = list(locations) # Get a list of all unique locations found
        for start_node in all_locations:
            self.distances[start_node] = self._bfs(start_node, all_locations)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            # Use match to safely parse the goal fact
            if match(goal, "at", "*", "*"):
                 _, package, location = get_parts(goal)
                 self.goal_locations[package] = location
            # Assuming goals are always (at package location).
            # If other goal types exist, they are ignored by this heuristic.


    def _bfs(self, start_node, all_nodes):
        """
        Perform BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in all_nodes}
        # start_node is guaranteed to be in all_nodes if all_nodes comes from locations set
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node exists in the graph dictionary before accessing neighbors
            # A location might exist but have no roads connected to it.
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current locations of locatable objects (packages and vehicles)
        current_locations = {}
        # Track which package is in which vehicle
        package_in_vehicle = {}

        for fact in state:
            # Use match for robustness, although direct string parsing is also possible
            # given the expected fact formats. Let's stick to match for consistency.
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            goal_fact_str = f"(at {package} {goal_location})"
            if goal_fact_str in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at its goal location. Calculate cost to move it.

            # Check if the package is in a vehicle
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find the vehicle's current location
                if vehicle not in current_locations:
                     # Vehicle location not found in state? This indicates an invalid state representation
                     # or a problem structure not handled (e.g., vehicle doesn't exist or is not 'at' a location).
                     # For a heuristic, returning inf is a safe assumption for unsolvable paths.
                     return float('inf')

                current_v_l = current_locations[vehicle]

                # If vehicle is at the goal location, only drop is needed
                if current_v_l == goal_location:
                    total_cost += 1 # drop action
                else:
                    # Need to drive vehicle to goal location and then drop
                    # Check if path exists in precomputed distances
                    # Ensure both current_v_l and goal_location are valid nodes in our distance map
                    if current_v_l not in self.distances or goal_location not in self.distances[goal_location]:
                         # This means either the current location or the goal location was not
                         # part of the locations found during graph building (e.g., isolated).
                         return float('inf') # Assume unreachable

                    dist = self.distances[current_v_l][goal_location]

                    if dist == float('inf'):
                       # Goal is unreachable from vehicle's current location
                       return float('inf') # Problem likely unsolvable from here
                    else:
                       total_cost += dist + 1 # drive actions + drop action

            else:
                # Package is on the ground, not at goal
                if package not in current_locations:
                    # Package location not found in state? Invalid state.
                    return float('inf') # Assume unreachable

                current_l = current_locations[package]

                # Need to pick up, drive, and drop
                # Check if path exists in precomputed distances
                if current_l not in self.distances or goal_location not in self.distances[goal_location]:
                     # This means either the current location or the goal location was not
                     # part of the locations found during graph building (e.g., isolated).
                     return float('inf') # Assume unreachable

                dist = self.distances[current_l][goal_location]

                if dist == float('inf'):
                    # Goal is unreachable from package's current location
                    return float('inf') # Problem likely unsolvable from here
                else:
                    total_cost += 1 + dist + 1 # pick action + drive actions + drop action

        # The heuristic is 0 only if all packages are at their goal locations.
        # The loop above only adds cost if a package is NOT at its goal.
        # So if total_cost is 0, it means all packages were skipped (because they were at their goal).
        # This satisfies the requirement that h=0 iff goal state (assuming goal only involves package locations).

        return total_cost
