import collections
import math
# The problem description implies the existence of a base class Heuristic.
# We assume it's available in the execution environment, e.g., from heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact_string):
    """Removes parentheses and splits a PDDL fact string into parts.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    """
    return fact_string[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    Estimates the cost to reach the goal state by summing the necessary pick-up and drop actions
    for misplaced packages, and adding an estimate of the driving required. The driving estimate
    is based on the maximum distance any single package needs to travel along the shortest path
    between locations connected by roads. This heuristic is designed for Greedy Best-First Search
    and does not need to be admissible.

    # Assumptions
    - The goal is primarily defined by `(at package location)` predicates. Other goal types are ignored by this heuristic.
    - The road network defined by `(road l1 l2)` predicates is static. The heuristic correctly handles
      both symmetric and asymmetric roads as defined in the static facts.
    - Vehicle capacities (`capacity`, `capacity-predecessor`) are ignored for simplicity to ensure
      efficient computation. This might affect accuracy but keeps the heuristic fast.
    - The heuristic assumes at least one vehicle exists if packages need moving. If a package's goal
      is unreachable via the road network from its current location, the heuristic returns infinity.
    - Vehicles are identified by assuming their names start with 'v' based on provided examples
      (e.g., 'v1', 'v2'). This is a simplifying assumption; a fully robust implementation might
      need access to object type information from the task definition if vehicle names vary.

    # Heuristic Initialization
    - Parses goal conditions (`task.goals`) to identify target locations for each package (`self.goal_locations`).
    - Identifies all unique locations involved in roads or goals (`self.locations`).
    - Parses static `(road l1 l2)` facts (`task.static`) to build an adjacency list representation
      of the location graph (`self.adj`).
    - Computes All-Pairs Shortest Paths (APSP) using Breadth-First Search (BFS) starting from each
      location. Stores these distances in `self.distances`. Unreachable locations have infinite
      distance (`math.inf`).
    - Identifies all packages that appear in goal conditions (`self.packages`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Retrieve the current state's facts (`node.state`).
    2.  Check if all goal facts (`self.goals`) are present in the current state using set comparison
        (`self.goals <= state`). If yes, the state is a goal state, return 0.
    3.  Parse the current state to determine the location of each relevant package and vehicle:
        - Store package ground locations `(at package loc)` in `current_p_locs`.
        - Store which vehicle holds which package `(in package vehicle)` in `current_p_in_v`.
        - Store vehicle locations `(at vehicle loc)` in `current_v_locs`, assuming vehicles are named 'v...'.
    4.  Initialize `num_misplaced_packages = 0` and `max_required_drive = 0`.
    5.  Iterate through each package `p` and its `goal_loc` defined in `self.goal_locations`:
        a. Determine if the package `p` is currently effectively at its `goal_loc`. A package is
           considered *not* at its goal if:
           - It's on the ground `(at p current_loc)` where `current_loc != goal_loc`.
           - It's inside a vehicle `(in p vehicle)` and that vehicle is at `vehicle_loc` where `vehicle_loc != goal_loc`.
           - It's inside a vehicle `(in p vehicle)` and that vehicle *is* at `goal_loc` (because a `drop` action is still needed).
        b. If `p` is not effectively at its `goal_loc`:
           - Increment `num_misplaced_packages`.
           - Determine the starting location for calculating driving distance (`current_loc_for_dist`):
             - If `p` is on the ground, this is its current location `current_loc`.
             - If `p` is in a vehicle, this is the vehicle's current location `vehicle_loc`.
             - Handle potential inconsistencies (e.g., package in vehicle with unknown location, package location unknown) by returning `math.inf`.
           - Calculate the shortest path distance `dist` from `current_loc_for_dist` to `goal_loc` using the precomputed `self.distances`.
           - If `dist` is infinite (meaning `goal_loc` is unreachable from `current_loc_for_dist`), return `math.inf` as the state is likely unsolvable.
           - Update `max_required_drive = max(max_required_drive, dist)`.
    6.  If `num_misplaced_packages` is 0 after checking all packages (and the initial goal check failed), return 0 as a safeguard (this might indicate goals involve predicates other than 'at', which this heuristic doesn't handle).
    7.  Otherwise, the heuristic value is `h = (2 * num_misplaced_packages) + max_required_drive`.
        - The `2 * num_misplaced_packages` term estimates the minimum pick-up (if on ground) and drop actions required for each misplaced package.
        - The `max_required_drive` term estimates the travel cost, using the longest necessary trip for any single package as a proxy for the overall driving effort, avoiding summing potentially parallel trips.
    8.  Return the calculated heuristic value `h` as an integer, ensuring it's non-negative.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing task goals and static facts.

        Args:
            task: The planning task object containing goals, initial state,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static
        self.packages = set()
        self.locations = set()
        self.goal_locations = {} # Stores package -> goal_location mapping
        self.adj = collections.defaultdict(list) # Adjacency list for road graph: location -> [neighbor_location]
        # Stores all-pairs shortest paths: location1 -> location2 -> distance
        # Initialize distances to infinity, self-distance will be set to 0
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: math.inf))

        # 1. Parse goals: identify packages, goal locations, and involved locations
        for goal in self.goals:
            parts = get_parts(goal)
            # Focus on 'at' predicates for package goals
            if parts[0] == 'at' and len(parts) == 3:
                package, loc = parts[1], parts[2]
                # Simple check if the first argument looks like a package (e.g., starts with 'p')
                # This is an assumption based on examples. A robust parser would use type info.
                if package.startswith('p'):
                    self.goal_locations[package] = loc
                    self.packages.add(package)
                    self.locations.add(loc) # Ensure goal locations are in the set

        # 2. Parse static facts: build road graph and collect all locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Add directed edge loc1 -> loc2 based on the PDDL fact
                self.adj[loc1].append(loc2)
                # Add locations to the set of all known locations
                self.locations.add(loc1)
                self.locations.add(loc2)
            # Could parse other static info like capacity-predecessor if needed by the heuristic

        # Ensure all locations mentioned in adj keys/values are in self.locations
        # This handles locations that might only appear as destinations in 'road' facts
        all_adj_locs = set(self.adj.keys())
        for neighbours in self.adj.values():
            all_adj_locs.update(neighbours)
        self.locations.update(all_adj_locs)

        # 3. Compute All-Pairs Shortest Paths (APSP) using BFS from each location
        for start_node in self.locations:
            self.distances[start_node][start_node] = 0
            queue = collections.deque([start_node])
            # visited_bfs stores nodes visited *in this specific BFS run* and their distances from start_node
            visited_bfs = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_bfs[current_node]

                # Use .get() for neighbors in case a location has no outgoing roads
                for neighbor in self.adj.get(current_node, []):
                    # Process neighbor only if it hasn't been visited in this BFS run
                    if neighbor not in visited_bfs:
                        visited_bfs[neighbor] = current_dist + 1
                        # Update the main distance matrix for the start_node
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
                    # In standard BFS on unweighted graphs, the first time a node is reached gives the shortest path.


    def __call__(self, node):
        """
        Computes the heuristic value for a given state node.

        Args:
            node: The node in the search space containing the state (node.state).

        Returns:
            An estimated cost (number of actions) to reach the goal from the node's state.
            Returns 0 for goal states, math.inf for states deemed unsolvable by the heuristic.
        """
        state = node.state

        # Check if the current state satisfies all goal conditions
        if self.goals <= state:
             return 0

        # Find current locations of packages and vehicles from the state facts
        current_p_locs = {} # package -> location (if '(at package location)')
        current_p_in_v = {} # package -> vehicle (if '(in package vehicle)')
        current_v_locs = {} # vehicle -> location (if '(at vehicle location)')

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if len(parts) < 2: continue # Skip malformed facts like "()"

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # Is it a package relevant to the goal?
                     current_p_locs[obj] = loc
                # Heuristic assumption: vehicles are named starting with 'v'
                elif obj.startswith('v'):
                     current_v_locs[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # Is it a package relevant to the goal?
                    current_p_in_v[package] = vehicle

        num_misplaced_packages = 0
        max_required_drive = 0

        # Iterate through all packages that have a defined goal location
        for package, goal_loc in self.goal_locations.items():
            is_at_goal = False # Flag: Is the package effectively at its goal location?
            current_loc_for_dist = None # The location from which driving distance needs calculation

            if package in current_p_locs:
                # Case 1: Package is on the ground
                current_loc = current_p_locs[package]
                if current_loc == goal_loc:
                    is_at_goal = True # Package is at the goal location
                else:
                    # Package is on the ground, but at the wrong location
                    current_loc_for_dist = current_loc
            elif package in current_p_in_v:
                # Case 2: Package is inside a vehicle
                vehicle = current_p_in_v[package]
                if vehicle in current_v_locs:
                    vehicle_loc = current_v_locs[vehicle]
                    # Package is in a known vehicle. It's considered misplaced even if the
                    # vehicle is at the goal, because a 'drop' action is still needed.
                    current_loc_for_dist = vehicle_loc
                    # is_at_goal remains False, as drop is needed
                else:
                    # Inconsistency: package is in a vehicle, but vehicle's location is unknown in the current state.
                    # This implies an invalid or intermediate state, or faulty vehicle detection.
                    # Consider the goal unreachable from this state.
                    # print(f"Heuristic Warning: Vehicle {vehicle} carrying {package} has no known location in state {state}.")
                    return math.inf
            else:
                # Case 3: Package state is unknown (neither 'at' nor 'in').
                # If a package required for the goal is not present in the state, the goal is unreachable.
                # print(f"Heuristic Warning: Package {package} from goal has no state ('at' or 'in') in {state}.")
                return math.inf

            # If the package is not effectively at its goal location, update counts
            if not is_at_goal:
                num_misplaced_packages += 1
                if current_loc_for_dist is not None:
                    # Check if the locations involved exist in our precomputed distance map keys.
                    # This handles cases where a location might appear in the state but not in static facts/goals.
                    if current_loc_for_dist not in self.distances or goal_loc not in self.distances[current_loc_for_dist]:
                         # A location needed for distance calculation wasn't processed during init (e.g., isolated location in state).
                         # print(f"Heuristic Warning: Cannot find precomputed distance from {current_loc_for_dist} to {goal_loc}.")
                         return math.inf # Indicate reachability issue

                    # Retrieve the precomputed shortest distance
                    dist = self.distances[current_loc_for_dist][goal_loc]

                    if dist == math.inf:
                        # The goal location is unreachable from the package's current effective location.
                        return math.inf
                    # Update the maximum driving distance required among all misplaced packages
                    max_required_drive = max(max_required_drive, dist)
                else:
                     # This block should logically not be reached if is_at_goal is False.
                     # Included as a safeguard against logic errors.
                     # print(f"Heuristic Error: Package {package} misplaced but current_loc_for_dist is None.")
                     return math.inf


        # Final heuristic value calculation
        if num_misplaced_packages == 0:
             # If we get here, it means all packages are at their goal locations.
             # This should ideally be caught by the initial `self.goals <= state` check.
             # If that check failed (e.g., goals include non-'at' predicates), returning 0 here is correct.
             return 0
        else:
             # Estimate: 2 actions (pickup/drop) per misplaced package + longest drive needed for any single package
             # The '2' accounts for one pickup (if needed) and one drop per package journey.
             heuristic_value = (2 * num_misplaced_packages) + max_required_drive
             # Return the heuristic value as an integer, ensuring it's non-negative.
             return max(0, int(round(heuristic_value)))
