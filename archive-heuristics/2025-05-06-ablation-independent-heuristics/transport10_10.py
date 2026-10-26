from heuristics.heuristic_base import Heuristic
from task import Operator, Task
from collections import deque
import math


def parse_fact(fact_string):
    """Helper function to parse a PDDL fact string."""
    # Remove leading/trailing brackets and split by space
    parts = fact_string.strip("()").split()
    return parts


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the number of actions required to reach the goal state by summing
    up the estimated costs for each package that is not yet at its goal location.
    The cost for a package depends on whether it is currently at a location or
    inside a vehicle, and involves estimated drive actions (based on shortest
    paths) plus pick-up and drop actions.

    Assumptions:
    - Unit cost for all actions (drive, pick-up, drop).
    - The road network is static and bidirectional.
    - Shortest path distances between locations are precomputed using BFS.
    - Capacity constraints are ignored when calculating the minimum vehicle
      approach cost for a package at a location. Any vehicle is assumed
      potentially usable to approach a package.
    - The heuristic value is infinity if a package's goal location is
      unreachable from its current location (or vehicle's location) via the
      road network, or if no vehicles exist to pick up a package at a location.

    Heuristic Initialization:
    1. Parses initial state, goal state, and static facts to identify all
       relevant objects (packages, vehicles, locations, sizes) and their types
       based on the predicates they appear with.
    2. Builds the road network graph from 'road' static facts using the identified
       locations.
    3. Computes all-pairs shortest path distances between all identified
       locations using BFS on the road graph.
    4. Identifies the smallest capacity size ('c0') from 'capacity-predecessor'
       static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize total heuristic value `h` to 0.
    2. Parse the current state to determine the location of each vehicle and
       the state (location or vehicle) of each package. Also record current
       vehicle capacities (though capacity is ignored in distance calculation).
    3. Parse the goal state to determine the target location for each package
       that needs to be at a specific location.
    4. Iterate through each package that has a goal location:
        a. If the package is already at its goal location, its cost is 0. Continue.
        b. Get the package's current state (location or vehicle). If the state
           is unknown or the goal location is not a known location, the package
           cost is infinity. Add to `h` and continue.
        c. If the package is currently at a location `loc_p_current`:
           - It needs a vehicle to pick it up, drive it, and drop it.
           - This requires 1 pick-up action and 1 drop action (cost 2).
           - It requires driving from `loc_p_current` to `loc_p_goal`. The minimum
             drive cost is the shortest path distance `dist(loc_p_current, loc_p_goal)`.
           - It requires a vehicle to first reach `loc_p_current`. Find the minimum
             shortest path distance `dist(loc_v_current, loc_p_current)` over all
             vehicles `v` from their current locations `loc_v_current`. If no
             vehicles exist or are located, this minimum is infinity.
           - The estimated cost for this package is `min_v(dist(loc_v_current, loc_p_current)) + dist(loc_p_current, loc_p_goal) + 2`.
           - If any required distance is infinity (unreachable), the package cost is infinity.
        d. If the package is currently inside a vehicle `v`:
           - Get the vehicle's current location `loc_v_current`. If the vehicle's
             location is unknown or not a known location, the package cost is
             infinity. Add to `h` and continue.
           - It needs the vehicle to drive it to `loc_p_goal` and then drop it.
           - This requires 1 drop action (cost 1).
           - It requires driving from `loc_v_current` to `loc_p_goal`. The minimum
             drive cost is the shortest path distance `dist(loc_v_current, loc_p_goal)`.
           - The estimated cost for this package is `dist(loc_v_current, loc_p_goal) + 1`.
           - If the required distance is infinity (unreachable), the package cost is infinity.
        e. Add the calculated cost for this package to the total heuristic `h`.
    5. Return `h`. If `h` accumulated any infinity cost, return infinity.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        # --- Preprocessing: Identify object types, build graph, compute distances ---

        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()
        self.road_graph = {} # Adjacency list: location -> set of connected locations

        # Collect all relevant facts
        all_relevant_facts = task.initial_state | task.goals | task.static

        # Classify objects and build road graph/sizes
        for fact_string in all_relevant_facts:
            parts = parse_fact(fact_string)
            pred = parts[0]
            if pred == 'road':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional roads
            elif pred == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                self.sizes.add(s1)
                self.sizes.add(s2)
            elif pred == 'in':
                pkg, veh = parts[1], parts[2]
                self.packages.add(pkg)
                self.vehicles.add(veh)
            elif pred == 'capacity':
                veh, size = parts[1], parts[2]
                self.vehicles.add(veh)
                self.sizes.add(size)
            elif pred == 'at':
                obj, loc = parts[1], parts[2]
                self.locations.add(loc)


        # Compute all-pairs shortest paths using BFS
        self.distances = {} # (l1, l2) -> distance
        for start_loc in self.locations:
            queue = deque([(start_loc, 0)])
            visited = {start_loc: 0}
            self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            while queue:
                current_loc, dist = queue.popleft()

                for neighbor in self.road_graph.get(current_loc, set()):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Identify smallest capacity 'c0' (optional for this heuristic, but good practice)
        predecessors = {parse_fact(f)[1] for f in task.static if parse_fact(f)[0] == 'capacity-predecessor'}
        successors = {parse_fact(f)[2] for f in task.static if parse_fact(f)[0] == 'capacity-predecessor'}
        all_sizes_in_hierarchy = predecessors | successors
        # Smallest capacity is the one that is not a successor in the hierarchy
        self.smallest_capacity = next(iter(all_sizes_in_hierarchy - successors), None)


        # --- End Preprocessing ---


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state
        goals = self.goals

        # --- Parse current state ---
        current_package_state = {} # package -> location or vehicle
        current_vehicle_locations = {} # vehicle -> location
        # current_vehicle_capacities = {} # vehicle -> size (not used in current heuristic logic)

        for fact_string in state:
            parts = parse_fact(fact_string)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    current_package_state[obj] = loc
                elif obj in self.vehicles:
                    current_vehicle_locations[obj] = loc
            elif pred == 'in':
                pkg, veh = parts[1], parts[2]
                if pkg in self.packages and veh in self.vehicles:
                     current_package_state[pkg] = veh
            # elif pred == 'capacity':
            #     veh, size = parts[1], parts[2]
            #     if veh in self.vehicles and size in self.sizes:
            #         current_vehicle_capacities[veh] = size
        # --- End Parse current state ---


        # --- Parse goal state to find target locations for packages ---
        goal_package_locations = {} # package -> location
        for fact_string in goals:
            parts = parse_fact(fact_string)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages: # Only care about package goals
                    goal_package_locations[obj] = loc
        # --- End Parse goal state ---


        # --- Calculate total heuristic ---
        h = 0
        for p, loc_p_goal in goal_package_locations.items():
            current_loc_or_veh = current_package_state.get(p)

            # If package is not mentioned in state, or goal location is not a known location
            if current_loc_or_veh is None or loc_p_goal not in self.locations:
                 h += math.inf
                 continue # Cannot calculate finite cost

            # If package is already at goal location
            if current_loc_or_veh == loc_p_goal:
                continue # Cost is 0 for this package

            # If package is at a location
            if current_loc_or_veh in self.locations:
                loc_p_current = current_loc_or_veh

                # Cost to drive package from current location to goal location
                drive_cost_pkg = self.distances.get((loc_p_current, loc_p_goal), math.inf)

                # Cost for a vehicle to approach the package's current location
                min_vehicle_approach_cost = math.inf
                if not self.vehicles: # No vehicles exist
                     min_vehicle_approach_cost = math.inf
                else:
                    for v in self.vehicles:
                        loc_v_current = current_vehicle_locations.get(v)
                        if loc_v_current is not None and loc_v_current in self.locations:
                            approach_dist = self.distances.get((loc_v_current, loc_p_current), math.inf)
                            min_vehicle_approach_cost = min(min_vehicle_approach_cost, approach_dist)

                # Total cost for this package: min_approach + pick-up (1) + drive_with_pkg + drop (1)
                # If vehicle approach is impossible or drive is impossible, cost is infinity
                if min_vehicle_approach_cost == math.inf or drive_cost_pkg == math.inf:
                     package_cost = math.inf
                else:
                     package_cost = min_vehicle_approach_cost + 1 + drive_cost_pkg + 1

                h += package_cost

            # If package is in a vehicle
            elif current_loc_or_veh in self.vehicles:
                v = current_loc_or_veh
                loc_v_current = current_vehicle_locations.get(v)

                # If vehicle location is unknown or not a known location
                if loc_v_current is None or loc_v_current not in self.locations:
                    h += math.inf
                    continue # Cannot calculate finite cost

                # Cost to drive vehicle (with package) from current location to goal location
                drive_cost_pkg = self.distances.get((loc_v_current, loc_p_goal), math.inf)

                # Total cost for this package: drive_with_pkg + drop (1)
                if drive_cost_pkg == math.inf:
                    package_cost = math.inf
                else:
                    package_cost = drive_cost_pkg + 1

                h += package_cost

            # If package state is something else (shouldn't happen if parsing is correct)
            else:
                 h += math.inf


        # Return infinity if any package cost was infinity, otherwise return the sum
        return h if h != math.inf else math.inf
