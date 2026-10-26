import collections
from fnmatch import fnmatch
# Assuming the Heuristic base class is available in the specified path
# If not, you might need to adjust the import based on your project structure
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Removes parentheses and splits the string by spaces.

    Args:
        fact (str): A PDDL fact string, e.g., "(at p1 l1)".

    Returns:
        list: A list containing the predicate and arguments,
              e.g., ["at", "p1", "l1"].
              Returns an empty list if the fact is malformed or empty.
    """
    # Remove leading/trailing whitespace and parentheses
    cleaned_fact = fact.strip()
    if len(cleaned_fact) < 2 or cleaned_fact[0] != '(' or cleaned_fact[-1] != ')':
        return [] # Malformed fact
    # Split the content inside parentheses
    return cleaned_fact[1:-1].split()

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their specified goal locations. It decomposes the problem by calculating
    an estimated cost for each package individually based on its current state
    and goal location, then summing these costs. The cost for a single package
    includes the actions needed to pick it up (if not already in a vehicle),
    drive it to the destination using the shortest path on the road network,
    and drop it there.

    # Assumptions
    - The primary goal conditions are of the form `(at package location)`. Other
      types of goal predicates are not explicitly considered in the heuristic value,
      although the check `task.goal_reached(state)` correctly identifies goal states.
    - Vehicle capacity constraints (predicates `capacity`, `capacity-predecessor`)
      are ignored to simplify the estimation. The heuristic assumes that a suitable
      vehicle will eventually be available and have capacity for any required pick-up.
    - The cost associated with moving a vehicle *to* a package's location before
      picking it up is *not* included in the estimate. It only considers the actions
      directly involved in the package's transit from its current location/vehicle
      to its goal.
    - The shortest path driving distance between any two locations is calculated
      based solely on the static `(road l1 l2)` predicates. Each `drive` action
      is assumed to have a cost of 1.
    - Potential positive interactions (e.g., a single vehicle trip delivering
      multiple packages or picking up a package en route) are ignored. The costs
      for each package are summed independently.
    - Objects are identified partly by naming conventions (e.g., assuming objects
      starting with 'p' are packages, 'v' are vehicles). A more robust implementation
      would ideally use type information if readily available from the `task` object.

    # Heuristic Initialization
    - The constructor (`__init__`) performs necessary pre-computation based on the
      planning task definition (`task`).
    - It parses the `task.goals` to identify the target location for each package
      that has an `(at ...)` goal condition, storing this mapping in `self.package_goals`.
    - It parses the `task.static` facts (facts true in all states) to:
        - Build an adjacency list representation (`self.adj`) of the road network
          defined by the `(road l1 l2)` facts.
        - Collect the set of all unique location names found in roads, goals,
          and initial state `at` predicates (`self.locations`).
    - It computes all-pairs shortest path distances between all identified locations
      using Breadth-First Search (BFS) starting from each location. These distances
      (number of `drive` actions) are stored in a nested dictionary `self.distances[loc1][loc2]`.
      If no path exists between two locations according to the `road` network,
      the distance is stored as `float('inf')`.

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state node (`node.state`):
    1.  Retrieve the current state (`state`) from the input `node`.
    2.  **Goal Check:** Immediately check if `state` satisfies all goal conditions using
        `self.task.goal_reached(state)`. If true, the heuristic value is 0, and we return.
    3.  **State Parsing:** Initialize data structures to store current object locations:
        - `locatable_locations`: Maps objects (packages, vehicles) to the location they are `at`.
        - `package_in_vehicle`: Maps packages to the vehicle they are `in`.
        - `vehicle_locations`: Specifically maps vehicles to their `at` location.
        Iterate through the facts in `state` and populate these dictionaries based on
        `(at ...)` and `(in ...)` predicates.
    4.  **Cost Calculation:** Initialize `heuristic_value = 0.0`. Iterate through each
        package `p` and its corresponding `goal_loc` stored in `self.package_goals`.
        For each package:
        a.  **Check if Goal Met:** See if the package `p` is already at `goal_loc` by
            checking `locatable_locations.get(p) == goal_loc`. If yes, this package's
            goal is satisfied in the current state; continue to the next package.
        b.  **Package on Ground:** If `p` is found in `locatable_locations` (meaning it's
            `at` some `current_loc`), calculate the cost to move it to `goal_loc`:
            - Retrieve the shortest path distance `dist = self.distances[current_loc].get(goal_loc, float('inf'))`.
            - If `dist` is `inf`, the goal is unreachable from this state; return `float('inf')`.
            - Otherwise, add the estimated cost `1 (pick-up) + dist (drive actions) + 1 (drop)`
              to `heuristic_value`.
        c.  **Package in Vehicle:** If `p` is found in `package_in_vehicle` (meaning it's
            `in` some `vehicle`), calculate the cost from the vehicle's current location:
            - Find the vehicle's location `current_vehicle_loc` from `vehicle_locations`.
              If the vehicle has no location (error state), return `float('inf')`.
            - Retrieve the shortest path distance `dist = self.distances[current_vehicle_loc].get(goal_loc, float('inf'))`.
            - If `dist` is `inf`, the goal is unreachable; return `float('inf')`.
            - Otherwise, add the estimated cost `dist (drive actions) + 1 (drop)` to
              `heuristic_value`.
        d.  **Package Missing:** If a package `p` from the goals is neither `at` a location
            nor `in` a vehicle in the current state, this indicates an unexpected or
            invalid state; return `float('inf')`.
    5.  **Final Value:** After iterating through all packages:
        - If `heuristic_value` is `float('inf')`, return it.
        - If `heuristic_value` is 0 but the state is *not* a goal state (checked initially),
          return 1. This prevents the heuristic from being 0 for non-goal states, which
          can be problematic for some search algorithms like Greedy Best-First Search.
        - Otherwise, return the calculated `heuristic_value` converted to an integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task's goals and static facts
        to precompute package destinations and shortest path distances between locations.

        Args:
            task: The planning task object, containing initial state, goals,
                  operators, and static facts.
        """
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals to find package destinations
        self.package_goals = {}
        self.packages = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Focus on goals specifying the final location of a package
            if parts and parts[0] == 'at' and len(parts) == 3:
                 package, location = parts[1], parts[2]
                 # Basic check assuming packages often start with 'p' or similar.
                 # Adapt this check if naming conventions differ.
                 if package.startswith('p'):
                     self.package_goals[package] = location
                     self.packages.add(package)

        # 2. Build road graph and identify all unique locations
        self.locations = set()
        self.adj = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Add edges for the road network graph
                self.adj[loc1].append(loc2)
                # Add locations to the set of known locations
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Ensure locations mentioned in goals or initial state are included,
        # even if they are isolated (not part of any 'road' fact).
        for loc in self.package_goals.values():
            self.locations.add(loc)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 # Add location from initial state 'at' predicates
                 self.locations.add(parts[2])

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        for start_node in self.locations:
            # Initialize distance from start_node to itself as 0
            self.distances[start_node][start_node] = 0
            # Queue stores tuples of (location, distance_from_start_node)
            queue = collections.deque([(start_node, 0)])
            # visited keeps track of nodes for which we've found the shortest path in this BFS run
            visited = {start_node}

            while queue:
                current_loc, current_dist = queue.popleft()

                # Explore neighbors reachable via roads
                if current_loc in self.adj:
                    for neighbor in self.adj[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_dist = current_dist + 1
                            # Update the main distance matrix for the start_node
                            self.distances[start_node][neighbor] = new_dist
                            queue.append((neighbor, new_dist))
            # After BFS from start_node completes, self.distances[start_node]
            # contains shortest path distances to all reachable locations.


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.

        Args:
            node: A search node containing the state (`node.state`) for which
                  to compute the heuristic value. The state is expected to be
                  a frozenset or set of PDDL fact strings.

        Returns:
            float: An estimate of the remaining actions to reach a goal state.
                   Returns 0 for goal states, float('inf') for states detected
                   as unsolvable (regarding package delivery), and a positive
                   integer estimate otherwise.
        """
        state = node.state

        # Optimization: If the state is already a goal state, heuristic is 0.
        if self.task.goal_reached(state):
            return 0

        heuristic_value = 0.0 # Use float to handle potential infinity

        # Parse the current state to find locations of packages and vehicles
        locatable_locations = {} # Maps object -> location it is 'at'
        package_in_vehicle = {} # Maps package -> vehicle it is 'in'
        vehicle_locations = {} # Maps vehicle -> location it is 'at'

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            try:
                if predicate == 'at' and len(args) == 2:
                    obj, loc = args[0], args[1]
                    locatable_locations[obj] = loc
                    # Basic check for vehicle naming convention
                    if obj.startswith('v'):
                        vehicle_locations[obj] = loc
                elif predicate == 'in' and len(args) == 2:
                    package, vehicle = args[0], args[1]
                    # Basic check for package naming convention
                    if package.startswith('p'):
                        package_in_vehicle[package] = vehicle
            except IndexError:
                # Should not happen if get_parts works correctly and facts are well-formed
                # print(f"Warning: Could not parse fact correctly: {fact}")
                continue

        # Calculate estimated cost for each package goal that is not yet satisfied
        for package, goal_loc in self.package_goals.items():
            # Check if this package's goal is already met in the current state
            if locatable_locations.get(package) == goal_loc:
                continue # Goal satisfied for this package.

            # Determine current state of the package (at location or in vehicle)
            current_loc = locatable_locations.get(package)
            if current_loc is not None:
                # Case 1: Package is 'at' a location on the ground.
                # Estimate cost = pick-up + drive + drop
                dist = self.distances[current_loc].get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from package's current location.
                    return float('inf') # This state cannot lead to the goal.

                heuristic_value += (1 + dist + 1)

            elif package in package_in_vehicle:
                # Case 2: Package is 'in' a vehicle.
                # Estimate cost = drive + drop
                vehicle = package_in_vehicle[package]
                current_vehicle_loc = vehicle_locations.get(vehicle)

                if current_vehicle_loc is None:
                     # This indicates an invalid state (vehicle carrying package has no location).
                     # print(f"Error: Vehicle {vehicle} carrying {package} has no 'at' predicate.")
                     return float('inf')

                dist = self.distances[current_vehicle_loc].get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # Goal location is unreachable from the vehicle's current location.
                    return float('inf') # This state cannot lead to the goal.

                heuristic_value += (dist + 1)
            else:
                # Error Case: A package required for the goal is neither 'at' a location
                # nor 'in' a vehicle in the current state. This implies the package
                # doesn't exist or the state is somehow invalid or inconsistent.
                # print(f"Error: Goal package {package} not found 'at' or 'in' state.")
                return float('inf')

        # Final adjustments:
        # If heuristic is calculated as 0 but it's not a goal state (e.g., if there
        # are non-package goals not considered), return 1 to avoid search stagnation.
        if heuristic_value == 0 and not self.task.goal_reached(state):
             # print("Warning: Heuristic calculated 0 for a non-goal state.")
             return 1

        # Return the final computed heuristic value.
        # Return float('inf') if calculated, otherwise convert the sum to int.
        return int(heuristic_value) if heuristic_value != float('inf') else float('inf')

