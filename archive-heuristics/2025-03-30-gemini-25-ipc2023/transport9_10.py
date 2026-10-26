import collections
from fnmatch import fnmatch
# The base class Heuristic is assumed to be available in this path.
from heuristics.heuristic_base import Heuristic
import math

# Helper function to parse PDDL fact strings like '(predicate arg1 arg2)'
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string by removing
    parentheses and splitting the string.

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list[str]: A list containing the predicate name and its arguments.
                   Returns an empty list if the fact is malformed.
    """
    if fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach the
    goal state. It focuses on the packages that need to be moved to their target
    locations as specified in the goal conditions. The heuristic calculates the
    cost for each misplaced package individually and sums these costs. The cost
    for a package includes the actions to pick it up (if not already in a vehicle),
    drive it to the goal location via the shortest path, and drop it there.

    # Assumptions
    - The primary task is delivering packages to specified locations.
    - Goal conditions are primarily of the form `(at package location)`. Other goal types are ignored.
    - All actions (`drive`, `pick-up`, `drop`) have a uniform cost of 1.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored for simplicity.
    - The cost of moving a vehicle *to* a package's location for pickup is ignored. The heuristic
      only estimates the cost starting from when the package is picked up (or if already in a vehicle).
    - Shortest path distances between locations are based on the static `(road l1 l2)` predicates.
    - Interactions between packages (e.g., multiple packages sharing a vehicle trip, detours)
      are ignored. The total heuristic value is the sum of costs estimated for each package independently.

    # Heuristic Initialization
    - Stores the task object for later goal checking (`task.goal_reached`).
    - Extracts goal locations for all packages specified in `task.goals`. Only `(at package location)` goals are processed.
    - Identifies all package, vehicle, and location objects by parsing `task.initial_state` and `task.static`.
      Object types (package vs. vehicle) are inferred based on predicate usage (e.g., `(capacity v s)` implies `v` is a vehicle, `(in p v)` implies `p` is package, `v` is vehicle). Objects involved in `(at o l)` in the initial state are classified based on whether they were already identified as vehicles; otherwise, they are assumed to be packages.
    - Builds an adjacency list representation of the road network from `(road l1 l2)` static facts.
    - Computes all-pairs shortest path distances between all known locations using Breadth-First Search (BFS)
      starting from each location. Distances are stored in `self.distances`. Unreachable locations
      maintain an infinite distance.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal:** First, check if the current state `node.state` already satisfies the goal conditions using `self.task.goal_reached(state)`. If yes, return `0.0`.
    2.  **Initialize Cost:** Set the total heuristic cost `h = 0.0`.
    3.  **Parse State:** Iterate through the facts in the current `state` to determine:
        - The current location of each package `p`: This can be a location `l` (from `(at p l)`) or a vehicle `v` (from `(in p v)`). Store this in `current_package_location`.
        - The current location of each vehicle `v` (from `(at v l)`). Store this in `current_vehicle_location`.
    4.  **Iterate Packages:** For each package `p` that has a goal location `goal_loc` defined in `self.goal_locations`:
        a.  **Get Current Status:** Find the package's current location or container (`current_loc_or_vehicle`) from the parsed state information. If the package's status is not found in the state (e.g., it's mentioned only in the goal), return `math.inf` as the state is considered inconsistent or the goal potentially unreachable from this state representation.
        b.  **Calculate Cost:**
            i.  **Package at Goal:** If `current_loc_or_vehicle == goal_loc` (meaning the package is at the correct location and not inside a vehicle), the cost contribution for this package is `0.0`.
            ii. **Package at Location (Wrong):** If the package is at a location `current_loc` such that `current_loc != goal_loc`:
                - Find the shortest path distance: `dist = self.distances[current_loc].get(goal_loc, math.inf)`.
                - If `dist` is `math.inf`, the goal location is unreachable from the current location; return `math.inf`.
                - Estimate cost: `1.0` (pick-up) + `dist` (drive actions) + `1.0` (drop). Add `2.0 + dist` to `h`.
            iii.**Package in Vehicle:** If the package is inside vehicle `vhc`:
                - Find the vehicle's current location: `vehicle_loc = current_vehicle_location.get(vhc)`. If the vehicle's location is unknown, return `math.inf` (inconsistent state).
                - Find the shortest path distance: `dist = self.distances[vehicle_loc].get(goal_loc, math.inf)`.
                - If `dist` is `math.inf`, the goal location is unreachable from the vehicle's current location; return `math.inf`.
                - Estimate cost: `dist` (drive actions) + `1.0` (drop). Add `1.0 + dist` to `h`.
            iv. **Invalid Status:** If `current_loc_or_vehicle` is neither a known location nor a known vehicle, return `math.inf` (error).
    5.  **Return Value:** After calculating the sum `h` for all packages:
        - If `h` is `0.0` but the state is *not* a goal state (checked in step 1), return `1.0`. This ensures the heuristic value is strictly positive for non-goal states, preventing the search from terminating prematurely if the heuristic calculation yields zero unexpectedly.
        - Otherwise, return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts, initial state,
        and goal conditions from the planning task.
        """
        self.task = task  # Store task for goal checking in __call__
        self.goals = task.goals
        static_facts = task.static
        init_state = task.initial_state

        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.adj = collections.defaultdict(list)
        self.goal_locations = {}  # package name -> goal location name
        # Use float distances to handle math.inf
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: math.inf))

        # 1. Identify locations and build road graph from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj[loc1].append(loc2)
            # Other static facts like capacity-predecessor are ignored

        # 2. Identify packages and vehicles from initial state predicates
        # This inference relies on common patterns in the transport domain.
        temp_vehicles_from_capacity = set()
        temp_vehicles_from_in = set()
        temp_packages_from_in = set()
        objects_at_location = set() # Objects initially at some location

        for fact in init_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'capacity' and len(parts) >= 2:
                temp_vehicles_from_capacity.add(parts[1])
            elif predicate == 'in' and len(parts) == 3:
                temp_packages_from_in.add(parts[1])
                temp_vehicles_from_in.add(parts[2])
            elif predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                objects_at_location.add(obj)
                self.locations.add(loc) # Ensure location is known

        # Consolidate vehicle and package identification
        self.vehicles.update(temp_vehicles_from_capacity)
        self.vehicles.update(temp_vehicles_from_in)
        self.packages.update(temp_packages_from_in)

        # Classify objects initially 'at' a location
        for obj in objects_at_location:
            if obj not in self.vehicles and obj not in self.packages:
                # Assume it's a package if not identified as a vehicle
                self.packages.add(obj)

        # 3. Parse goals to find package destinations and refine object types
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                package, loc = parts[1], parts[2]
                # Ensure this object is treated as a package
                self.packages.add(package)
                if package in self.vehicles:
                    self.vehicles.remove(package) # Correct potential misclassification
                self.goal_locations[package] = loc
                self.locations.add(loc) # Ensure goal location is known
            # Ignore other potential goal types

        # 4. Compute all-pairs shortest paths using BFS
        all_locs = list(self.locations) # Create a stable list of locations
        for start_node in all_locs:
            # Check if start_node has outgoing edges or exists
            if start_node not in self.locations: continue

            self.distances[start_node][start_node] = 0.0
            queue = collections.deque([(start_node, 0.0)])
            # visited_dist tracks distances found so far in this specific BFS run
            visited_dist = {start_node: 0.0}

            while queue:
                current_node, dist = queue.popleft()

                # Use .get() for safe access to adjacency list
                for neighbor in self.adj.get(current_node, []):
                    # Ensure neighbor is a valid location before processing
                    if neighbor in self.locations:
                        new_dist = dist + 1.0
                        # Update distance if this path is shorter than any previously found path
                        # (or if the neighbor hasn't been reached yet)
                        if new_dist < self.distances[start_node][neighbor]:
                            self.distances[start_node][neighbor] = new_dist
                            queue.append((neighbor, new_dist))


    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the given state node.

        Args:
            node: The node in the search space containing the state.

        Returns:
            float: The estimated cost to reach the goal. Returns 0.0 for goal states,
                   math.inf for potentially unreachable goals or inconsistent states,
                   and a positive value otherwise.
        """
        state = node.state

        # Check if the current state is a goal state
        if self.task.goal_reached(state):
            return 0.0

        heuristic_value = 0.0

        # Find current locations of packages and vehicles in the current state
        current_package_location = {}  # package name -> location name or vehicle name
        current_vehicle_location = {}  # vehicle name -> location name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    current_package_location[obj] = loc
                elif obj in self.vehicles:
                    current_vehicle_location[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                pkg, vhc = parts[1], parts[2]
                if pkg in self.packages:
                    current_package_location[pkg] = vhc # Store vehicle name as location

        # Calculate cost for each package based on its goal location
        for package in self.goal_locations:
            goal_loc = self.goal_locations[package]
            current_loc_or_vehicle = current_package_location.get(package)

            if current_loc_or_vehicle is None:
                # Package's current state is unknown (not 'at' or 'in').
                # This indicates an issue with the state or problem definition.
                # print(f"Warning: Location of package {package} not found in state.")
                return math.inf # Signal inconsistency or unreachability

            cost = 0.0
            if current_loc_or_vehicle == goal_loc:
                # Package is at the goal location (and not in a vehicle). Cost is 0.
                cost = 0.0
            elif current_loc_or_vehicle in self.locations:
                # Package is at a location `current_loc` which is not the goal.
                current_loc = current_loc_or_vehicle
                dist = self.distances[current_loc].get(goal_loc, math.inf)
                if dist == math.inf:
                    # Goal location is unreachable from the package's current location.
                    # print(f"Unreachable: Package {package} from {current_loc} to {goal_loc}")
                    return math.inf
                # Cost = pick-up (1) + drive (dist) + drop (1)
                cost = 1.0 + dist + 1.0
            elif current_loc_or_vehicle in self.vehicles:
                # Package is inside vehicle `vhc`.
                vhc = current_loc_or_vehicle
                vehicle_loc = current_vehicle_location.get(vhc)
                if vehicle_loc is None:
                    # The location of the vehicle carrying the package is unknown.
                    # print(f"Warning: Location of vehicle {vhc} carrying {package} not found.")
                    return math.inf # Inconsistent state
                dist = self.distances[vehicle_loc].get(goal_loc, math.inf)
                if dist == math.inf:
                    # Goal location is unreachable from the vehicle's current location.
                    # print(f"Unreachable: Package {package} from vehicle {vhc} at {vehicle_loc} to {goal_loc}")
                    return math.inf
                # Cost = drive (dist) + drop (1)
                cost = dist + 1.0
            else:
                # The package's location is neither a known location nor a known vehicle.
                # print(f"Warning: Unknown location type for package {package}: {current_loc_or_vehicle}")
                return math.inf # Error state

            heuristic_value += cost

        # Final check: Ensure heuristic is non-zero for non-goal states.
        # If heuristic calculated to 0 but it's not a goal state, return 1.0.
        if heuristic_value == 0.0 and not self.task.goal_reached(state):
             # This might happen if self.goal_locations is empty or if all packages
             # happen to be at their destinations but other goal conditions exist (unlikely assumed).
             # Return 1.0 ensures the search progresses.
            return 1.0

        return heuristic_value
