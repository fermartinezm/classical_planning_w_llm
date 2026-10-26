from collections import deque
from fnmatch import fnmatch
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
    This heuristic estimates the minimum number of actions required to move each
    misplaced package to its goal location, ignoring vehicle capacity and
    availability constraints. It sums the estimated costs for each package
    independently.

    # Assumptions
    - Vehicles can carry only one package at a time (inferred from action effects on capacity).
    - Any vehicle can be used to transport any package (implicitly, as vehicle identity is not strictly tracked per package cost).
    - Vehicles are always available when needed at the package's location or the vehicle's current location.
    - The cost of driving between locations is the shortest path distance in the road network.
    - Pickup and drop actions each cost 1.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Build the road network graph from the static `road` facts, initial state `at` facts, and goal `at` facts to include all relevant locations.
    - Compute all-pairs shortest paths between all relevant locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or container (vehicle) for every package and the location for every vehicle present in the state. This is done by iterating through the state facts `(at ?x ?l)` and `(in ?p ?v)`.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each goal condition `(at ?p ?l_goal)`.
    4. If the goal condition `(at ?p ?l_goal)` is already satisfied in the current state, continue to the next goal.
    5. If the goal is not satisfied, find the current status of package `?p`. This will be either a location `l_curr` (if `(at ?p l_curr)` is in the state) or a vehicle `v` (if `(in ?p v)` is in the state).
    6. If the package `?p` is on the ground at location `l_curr`:
       - The estimated cost for this package is 1 (pick-up) + the shortest path distance from `l_curr` to `l_goal` (drive) + 1 (drop).
       - Add `2 + dist(l_curr, l_goal)` to the total cost. If `l_goal` is unreachable from `l_curr` in the road network, return a large penalty value as the heuristic.
    7. If the package `?p` is inside a vehicle `v`:
       - Find the current location `l_v` of vehicle `v` from the state fact `(at v l_v)`.
       - If `l_v` is not the goal location `l_goal`:
          - The estimated cost for this package is the shortest path distance from `l_v` to `l_goal` (drive) + 1 (drop).
          - Add `dist(l_v, l_goal) + 1` to the total cost. If `l_goal` is unreachable from `l_v`, return a large penalty value.
       - If `l_v` is the goal location `l_goal`:
          - The estimated cost for this package is 1 (drop).
          - Add `1` to the total cost.
    8. If a goal package is not found in the state's `at` or `in` facts, this indicates an invalid state; return a large penalty.
    9. After iterating through all goal conditions, the accumulated `total_cost` is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between locations.
        """
        self.goals = task.goals
        self.static = task.static
        # Assuming task object provides initial state facts as task.initial_state
        # This is needed to include all relevant locations in the graph for BFS.
        self.initial_state_facts = task.initial_state


        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and compute shortest paths.
        # Include locations from static road facts, initial state 'at' facts, and goal 'at' facts.
        self.dist = self._compute_shortest_paths_from_relevant_facts()


    def _compute_shortest_paths_from_relevant_facts(self):
        """
        Build the graph from road facts and compute all-pairs shortest paths
        using BFS from each location found in static facts, initial state, or goals.
        """
        graph = {}
        locations = set()

        # Extract locations from road facts
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                if l1 not in graph:
                    graph[l1] = []
                if l2 not in graph:
                    graph[l2] = []
                # Assuming roads are bidirectional
                graph[l1].append(l2)
                graph[l2].append(l1)

        # Extract locations from initial state 'at' facts
        for fact in self.initial_state_facts:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
                 if loc not in graph:
                     graph[loc] = [] # Add isolated locations

        # Extract locations from goal 'at' facts
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 locations.add(loc)
                 if loc not in graph:
                     graph[loc] = [] # Add isolated locations


        # Compute shortest paths from each location using BFS
        dist = {}
        for start_loc in locations:
            dist[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, d = queue.popleft()
                dist[start_loc][current_loc] = d

                # Get neighbors safely
                for neighbor in graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        return dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track current status of packages (location or vehicle) and locations of vehicles
        pkg_status = {} # Maps package -> location (if on ground) or vehicle (if in vehicle)
        veh_locations = {} # Maps vehicle -> location

        # Populate pkg_status and veh_locations from the current state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Assuming 'v' prefix for vehicles and 'p' for packages based on examples
                if obj.startswith('v'):
                     veh_locations[obj] = loc
                elif obj.startswith('p'):
                     pkg_status[obj] = loc # Package is on the ground
            elif parts[0] == "in":
                pkg, veh = parts[1], parts[2]
                pkg_status[pkg] = veh # Package is in a vehicle

        total_cost = 0
        UNREACHABLE_PENALTY = 1000000 # Large penalty for unreachable locations

        # Iterate through goal conditions to find unsatisfied package goals
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "at":
                 package, goal_location = args
                 # Check if this goal is satisfied in the current state
                 if goal in state:
                     continue # Goal already satisfied for this package

                 # Goal is not satisfied. Estimate cost for this package.
                 # Find package's current status (location or vehicle)
                 current_status = pkg_status.get(package)

                 if current_status is None:
                     # This package is a goal package but not found in the state's at/in facts.
                     # This indicates an invalid state representation or a package that disappeared.
                     # In a real planner, this might indicate an unsolvable state or a bug.
                     # For a heuristic, return a large penalty.
                     # print(f"Warning: Goal package {package} not found in state.")
                     return UNREACHABLE_PENALTY

                 # Determine if the package is on the ground or in a vehicle based on naming convention
                 # Assuming locations do not start with 'v' and vehicles start with 'v'
                 if not current_status.startswith('v'): # It's a location name
                     current_location = current_status
                     # Needs pick-up (1) + drive (dist) + drop (1)
                     # Ensure goal_location is reachable from current_location
                     if current_location in self.dist and goal_location in self.dist.get(current_location, {}):
                          total_cost += 1 # pick-up
                          total_cost += self.dist[current_location][goal_location] # drive
                          total_cost += 1 # drop
                     else:
                          # Goal location is unreachable from current location.
                          return UNREACHABLE_PENALTY

                 # Case 2: Package is inside a vehicle
                 else: # current_status is a vehicle name (starts with 'v')
                     vehicle = current_status
                     vehicle_location = veh_locations.get(vehicle)

                     if vehicle_location is None:
                         # Vehicle carrying package is not located anywhere? Should not happen in valid states.
                         # print(f"Warning: Vehicle {vehicle} carrying {package} not found in state.")
                         return UNREACHABLE_PENALTY

                     if vehicle_location != goal_location:
                         # Needs drive (dist) + drop (1)
                         # Ensure goal_location is reachable from vehicle_location
                         if vehicle_location in self.dist and goal_location in self.dist.get(vehicle_location, {}):
                             total_cost += self.dist[vehicle_location][goal_location] # drive
                             total_cost += 1 # drop
                         else:
                             # Goal location is unreachable from vehicle location.
                             return UNREACHABLE_PENALTY

                     else: # vehicle_location == goal_location
                         # Needs drop (1)
                         total_cost += 1

        return total_cost
