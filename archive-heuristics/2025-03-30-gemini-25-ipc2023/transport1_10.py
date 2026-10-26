from collections import deque
import math # For infinity

# Attempt to import the base class, provide a dummy if not found
try:
    # This assumes the planner framework provides the base class
    # in a module named 'heuristics.heuristic_base'.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    # This allows the code to be syntactically correct and testable standalone.
    class Heuristic:
        """Dummy base class if the actual Heuristic class is not available."""
        def __init__(self, task):
            self.task = task # Store task for potential use by subclasses
        def __call__(self, node):
            raise NotImplementedError("Heuristic calculation not implemented.")

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    Removes parentheses and splits by space.
    Returns an empty list if the fact is malformed or empty.
    """
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith("(") and fact.endswith(")"):
        # Basic split, assumes space separation
        return fact[1:-1].split()
    return []

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move each package
    to its specified goal location. It calculates an estimated cost for each package
    that is not yet at its goal and sums these costs. The cost for a package considers
    the pick-up action (if needed), the drop action, and the minimum number of drive
    actions based on shortest path distances in the road network. It is designed for
    use with greedy best-first search and does not need to be admissible.

    # Assumptions
    - The primary driver of cost is the sequence of actions needed for each package:
      (potentially drive vehicle to package), pickup, drive vehicle with package, drop.
      This heuristic simplifies by estimating pickup(1) + drive_with_package(dist) + drop(1)
      if the package is on the ground, or drive_with_package(dist) + drop(1) if already
      in a vehicle. It ignores the cost of moving the vehicle *to* the package initially,
      aiming for an informative but potentially non-admissible estimate.
    - Vehicle capacity constraints (`capacity`, `capacity-predecessor`) are ignored.
      Any vehicle is assumed capable of picking up a package if co-located.
    - The road network forms an unweighted graph where each 'drive' action has a cost of 1.
      Shortest paths are calculated using BFS.
    - Goals are exclusively of the form `(at package location)`.
    - All necessary objects (packages, locations, vehicles) exist and are consistent
      with the static facts and state representation.

    # Heuristic Initialization
    - Extracts all unique locations from static `road` facts and goal specifications.
    - Parses static `road` facts to build an adjacency list (`self.adj`) representing
      the road network graph. Only `(road l1 l2)` facts are used; bidirectionality
      must be explicit in the PDDL file if required.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS)
      and stores them in `self.distances`. Unreachable pairs have infinite distance.
    - Parses the task's goal conditions (`task.goals`) to identify the target location
      for each package, stored in `self.goal_locations`.
    - Stores the set of packages involved in goals in `self.packages`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize `total_estimated_cost = 0`.
    2. Parse the current state (`node.state`) to build dictionaries mapping:
       - `package_locations`: package -> current location (if `at`).
       - `package_in_vehicle`: package -> vehicle (if `in`).
       - `vehicle_locations`: vehicle -> current location (if `at`).
    3. For each package `p` listed in `self.goal_locations`:
       a. Let `goal_loc` be the target location for `p`.
       b. Construct the goal fact string, e.g., `"(at p1 l2)"`.
       c. Check if this `goal_fact` is present in the current state `state`.
          If yes, this package's goal is met, its cost contribution is 0. Continue to the next package.
       d. Determine the current status of package `p`:
          i. **If `p` is on the ground:** Found in `package_locations` at `current_loc`.
             - Retrieve the shortest path distance `dist = self.distances.get((current_loc, goal_loc), math.inf)`.
             - If `dist` is infinity, the goal is unreachable for this package. Return `math.inf` immediately (the state is unsolvable regarding this goal).
             - Estimated cost contribution = `1 (for pickup) + dist (for driving) + 1 (for drop)`.
          ii. **If `p` is in a vehicle:** Found in `package_in_vehicle` as `vehicle`.
              - Find the vehicle's location `vehicle_loc` from `vehicle_locations`.
              - If `vehicle_loc` is found:
                  - Retrieve `dist = self.distances.get((vehicle_loc, goal_loc), math.inf)`.
                  - If `dist` is infinity, the goal is unreachable. Return `math.inf`.
                  - Estimated cost contribution = `dist (for driving) + 1 (for drop)`.
              - If `vehicle_loc` is not found (vehicle exists but its location is unknown in the state),
                this indicates an inconsistent or unexpected state. Return `math.inf`.
          iii. **If `p`'s status is unknown** (not at goal, not `at`, not `in`): This implies an issue with the state or problem definition. Return `math.inf`.
       e. Add the calculated cost contribution for package `p` to `total_estimated_cost`.
    4. After checking all packages:
       - Verify if the current state `state` satisfies *all* goal conditions defined in `self.task.goals` using set subset comparison (`self.goals <= state`).
       - If `is_goal_state` is true, return 0 (heuristic must be 0 for goal states).
       - Otherwise (state is not a goal state):
         - If `total_estimated_cost` calculated is 0 (this might happen if all package goals are met but other non-package goals exist, or in edge cases), return 1. This ensures the heuristic is always positive for non-goal states, preventing the search from terminating prematurely on non-goal states with h=0.
         - Otherwise, return the calculated `total_estimated_cost`.
    """

    def __init__(self, task):
        """Initializes the heuristic by processing static information and goals."""
        super().__init__(task) # Pass task to base class
        self.goals = task.goals
        static_facts = task.static

        # --- Extract locations, packages, and build road graph ---
        locations = set()
        adj = {}
        self.packages = set()
        self.goal_locations = {} # Map: package_name -> goal_location_name

        # Identify packages and goal locations from goals
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure goal is (at package location) format
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Assume first arg of 'at' goal is package
                self.goal_locations[package] = location
                self.packages.add(package)
                locations.add(location) # Add goal location to known locations

        # Process static facts for roads and add involved locations
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                # Add directed edge for BFS based on PDDL definition
                adj.setdefault(loc1, []).append(loc2)
            # Could potentially identify vehicles or sizes here if needed for a more complex heuristic

        self.locations = locations
        self.adj = adj # Adjacency list for graph traversal

        # --- Compute all-pairs shortest paths ---
        self.distances = self._compute_all_pairs_shortest_paths()


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of known locations using BFS.
        Handles disconnected graphs (unreachable locations will have infinite distance).
        Returns a dictionary mapping (from_loc, to_loc) -> distance.
        """
        distances = {}
        if not self.locations:
            return distances # Return empty dict if no locations defined

        for start_node in self.locations:
            # Initialize distances from start_node: infinity to all, 0 to self
            for loc in self.locations:
                distances[(start_node, loc)] = math.inf
            # Check if start_node is a valid location before setting self-distance
            if start_node in self.locations:
                 distances[(start_node, start_node)] = 0

            queue = deque([(start_node, 0)])
            # visited_bfs stores nodes visited *in this specific BFS run* to avoid cycles
            visited_bfs = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors using the adjacency list
                for neighbor in self.adj.get(current_node, []):
                    # Ensure neighbor is a known location (important for robustness)
                    if neighbor in self.locations and neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        new_dist = dist + 1
                        distances[(start_node, neighbor)] = new_dist
                        queue.append((neighbor, new_dist))
                    # In standard BFS on unweighted graph, the first time visiting gives the shortest path.
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state (node.state).
        Estimates the minimum number of actions required to reach the goal.
        Returns math.inf if the goal is determined to be unreachable.
        """
        state = node.state
        total_estimated_cost = 0

        # --- Parse current state for relevant predicates ---
        package_locations = {} # Map: package -> location
        package_in_vehicle = {} # Map: package -> vehicle
        vehicle_locations = {} # Map: vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    package_locations[obj] = loc
                else:
                    # Assume non-package object with 'at' is a vehicle
                    vehicle_locations[obj] = loc
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args[0], args[1]
                if package in self.packages:
                    package_in_vehicle[package] = vehicle

        # --- Calculate cost for each package goal ---
        for package, goal_loc in self.goal_locations.items():

            # Construct the goal fact string to check against the state
            goal_fact = f"(at {package} {goal_loc})"
            if goal_fact in state:
                continue # This package's goal is met, cost contribution is 0

            cost_p = math.inf # Initialize cost for this package; remains inf if status unknown

            if package in package_locations:
                # Case 1: Package is on the ground at current_loc
                current_loc = package_locations[package]
                # Retrieve pre-computed shortest path distance
                dist = self.distances.get((current_loc, goal_loc), math.inf)

                if dist == math.inf:
                    # Goal location is unreachable from the package's current location
                    return math.inf # State is unsolvable regarding this goal

                # Estimated cost: pickup(1) + drive(dist) + drop(1)
                cost_p = 1 + dist + 1

            elif package in package_in_vehicle:
                # Case 2: Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_locations:
                    vehicle_loc = vehicle_locations[vehicle]
                    # Retrieve pre-computed shortest path distance from vehicle's location
                    dist = self.distances.get((vehicle_loc, goal_loc), math.inf)

                    if dist == math.inf:
                        # Goal location is unreachable from the vehicle's current location
                        return math.inf # State is unsolvable regarding this goal

                    # Estimated cost: drive(dist) + drop(1)
                    cost_p = dist + 1
                else:
                    # Vehicle's location is unknown, cannot estimate cost.
                    # This suggests an inconsistent state representation.
                    return math.inf

            else:
                # Package is not at its goal, not found 'at' a location, and not 'in' a vehicle.
                # This indicates an invalid or unexpected state.
                return math.inf # Cannot determine status or estimate cost.

            # Add the calculated cost for this package to the total
            total_estimated_cost += cost_p


        # --- Final checks: Ensure h=0 iff goal state, h>0 otherwise ---
        # Check if ALL goal conditions specified in the task are met
        is_goal_state = self.goals <= state

        if is_goal_state:
            # If this is truly a goal state, the heuristic value must be 0.
            return 0
        elif total_estimated_cost == 0:
            # If heuristic calculated to 0, but it's not a goal state
            # (e.g., all package goals met but other non-package goals exist, or an edge case).
            # Return 1 to ensure the heuristic value is positive for non-goal states.
            # This helps greedy search make progress.
            return 1
        else:
            # Return the calculated sum of costs for all packages not yet at their goal.
            # Since cost_p is calculated as >= 1 for non-goal packages, this sum will be > 0.
            return total_estimated_cost
