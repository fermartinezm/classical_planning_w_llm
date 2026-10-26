# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available
from collections import deque # For BFS
import math # For float('inf')

# Helper function to parse PDDL facts represented as strings
def parse_fact(fact_str):
    # Example: '(at p1 l1)' -> ('at', 'p1', 'l1')
    # Remove leading/trailing parentheses and split by space
    parts = fact_str.strip('()').split()
    if not parts:
        return None, []
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum estimated
    steps required for each package that is not yet at its goal location.
    The estimation for a package considers its current location (or vehicle)
    and its goal location, using precomputed shortest path distances
    on the road network. It simplifies by ignoring vehicle capacity constraints
    and assuming any vehicle can be used to pick up a package.

    Assumptions:
    - The road network is static and defined by (road l1 l2) facts in the
      initial state (or static facts).
    - Vehicle capacity is ignored for heuristic calculation.
    - Any vehicle can potentially be used to pick up any package (ignoring size).
    - The cost of any action (drive, pick-up, drop) is 1.
    - Shortest path distances between locations are precomputed using BFS.
    - The goal only consists of (at package location) facts.

    Heuristic Initialization:
    1. Identify all unique locations mentioned in road facts, initial state, and goal state.
    2. Build an adjacency list representation of the road network graph from (road l1 l2) facts.
    3. Compute all-pairs shortest paths between all identified locations using BFS.
       Store these distances in a dictionary `self.location_distances`.
       Handle unreachable locations by storing infinity (`math.inf`).
    4. Extract the goal location for each package from the task's goal facts. Store in `self.package_goal_locations`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state by checking if all goal facts are in the state. If yes, return 0.
    2. Initialize the total heuristic value `h_value` to 0.
    3. Extract the current location of each locatable object (packages and vehicles)
       and which packages are inside which vehicles from the current state facts.
       Store in `package_locations`, `vehicle_locations`, and `package_in_vehicle`.
       Distinguish packages from vehicles based on whether they appear in `self.package_goal_locations`.
    4. Iterate through each package `p` and its goal location `goal_l` in `self.package_goal_locations`.
    5. If the fact `(at p goal_l)` is present in the current state, this package is already at its goal. Continue to the next package.
    6. If the package `p` is not at its goal:
       a. Determine the package's current effective location `current_loc_p`.
          - If `p` is in `package_locations`, `current_loc_p` is its location.
          - If `p` is in `package_in_vehicle`, find the vehicle `v` carrying it, and `current_loc_p` is the location of vehicle `v` from `vehicle_locations`.
          - If the package's location cannot be determined (e.g., not at a location and not in a vehicle), treat as unreachable (infinity cost).
       b. Estimate the cost to move the package from `current_loc_p` to `goal_l`.
          - If the package is currently at `current_loc_p` (not in a vehicle):
            - It needs to be picked up (cost 1).
            - A vehicle needs to reach `current_loc_p`. Find the minimum shortest path distance from any vehicle's current location (`v_loc` from `vehicle_locations`) to `current_loc_p`. Let this be `min_dist_v_to_p_loc`. If no vehicles exist or can reach `current_loc_p`, this is infinity.
            - The vehicle needs to drive from `current_loc_p` to `goal_l`. Cost is `self.location_distances[current_loc_p][goal_l]`. If unreachable, this is infinity.
            - The package needs to be dropped (cost 1).
            - Estimated cost for this package: `min_dist_v_to_p_loc + 1 + self.location_distances[current_loc_p][goal_l] + 1`.
          - If the package is currently in vehicle `v` at `current_loc_v`:
            - The vehicle needs to drive from `current_loc_v` to `goal_l`. Cost is `self.location_distances[current_loc_v][goal_l]`. If unreachable, this is infinity.
            - The package needs to be dropped (cost 1).
            - Estimated cost for this package: `self.location_distances[current_loc_v][goal_l] + 1`.
       c. Add the estimated cost for this package to `h_value`. If any required distance is infinity, the cost for this package is infinity. If the total `h_value` becomes infinity, we can stop early and return infinity.
    7. Return the total `h_value`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        self.road_graph = {}
        self.locations = set()
        self.package_goal_locations = {}

        # 1. & 2. Build road graph and collect locations
        # Collect locations from static facts (roads)
        for fact_str in self.static_facts:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'road':
                if len(objects) == 2:
                    l1, l2 = objects
                    self.locations.add(l1)
                    self.locations.add(l2)
                    if l1 not in self.road_graph:
                        self.road_graph[l1] = []
                    self.road_graph[l1].append(l2)

        # Collect locations from initial state and goals
        # This ensures we include locations where objects start or need to end up,
        # even if they are isolated in the road network definition.
        all_relevant_facts = set(task.initial_state) | set(task.goals)
        for fact_str in all_relevant_facts:
             predicate, objects = parse_fact(fact_str)
             if predicate == 'at':
                 # (at obj loc)
                 if len(objects) == 2:
                     self.locations.add(objects[1])
             # capacity and capacity-predecessor don't involve locations

        # Ensure all collected locations are in the graph dictionary even if they have no outgoing roads
        for loc in self.locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # 3. Compute all-pairs shortest paths using BFS
        self.location_distances = {}
        for start_node in self.locations:
            self.location_distances[start_node] = {}
            # Initialize distances to infinity
            for end_node in self.locations:
                self.location_distances[start_node][end_node] = math.inf
            self.location_distances[start_node][start_node] = 0

            queue = deque([start_node])
            visited = {start_node}

            while queue:
                current_node = queue.popleft()
                current_dist = self.location_distances[start_node][current_node]

                # Check if current_node has outgoing roads in the graph
                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        # Check if neighbor is a known location
                        if neighbor in self.locations and neighbor not in visited:
                            visited.add(neighbor)
                            # Distance is 1 more than current node's distance (cost of drive)
                            self.location_distances[start_node][neighbor] = current_dist + 1
                            queue.append(neighbor)

        # 4. Extract package goal locations
        for goal_fact_str in self.goals:
            predicate, objects = parse_fact(goal_fact_str)
            if predicate == 'at' and len(objects) == 2:
                package, location = objects
                # Assuming goal is always (at package location)
                self.package_goal_locations[package] = location
            # Ignore other potential goal types if any, or handle them

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Initialize heuristic value
        h_value = 0

        # 3. Extract current state information
        package_locations = {} # package -> location (if at location)
        package_in_vehicle = {} # package -> vehicle (if in vehicle)
        vehicle_locations = {} # vehicle -> location

        # Determine which objects are packages vs vehicles based on goal facts
        is_package = set(self.package_goal_locations.keys())

        for fact_str in state:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'at' and len(objects) == 2:
                obj, loc = objects
                if obj in is_package:
                    package_locations[obj] = loc
                else: # Assume it's a vehicle or other locatable not in goals
                    vehicle_locations[obj] = loc
            elif predicate == 'in' and len(objects) == 2:
                pkg, veh = objects
                # Ensure pkg is actually a package we care about
                if pkg in is_package:
                    package_in_vehicle[pkg] = veh
            # Ignore capacity facts for this heuristic

        # 4. Iterate through misplaced packages
        for package, goal_l in self.package_goal_locations.items():
            # 5. Check if package is already at goal
            if (f'(at {package} {goal_l})') in state:
                continue # Package is already at its goal

            # 6. Package is misplaced, calculate its contribution
            estimated_package_cost = math.inf

            # 6a. Determine current location/status
            current_loc_p = None
            vehicle_carrying_p = None

            if package in package_locations:
                current_loc_p = package_locations[package]
            elif package in package_in_vehicle:
                vehicle_carrying_p = package_in_vehicle[package]
                if vehicle_carrying_p in vehicle_locations:
                    current_loc_p = vehicle_locations[vehicle_carrying_p]
                # else: vehicle location unknown? Treat as unreachable.

            if current_loc_p is None:
                 # Package location is unknown (not at a location and vehicle location unknown)
                 estimated_package_cost = math.inf
            else:
                # Ensure current_loc_p and goal_l are known locations in our distance map
                # Check if start and end locations are in the precomputed distances map
                if current_loc_p not in self.location_distances or goal_l not in self.location_distances.get(current_loc_p, {}):
                     estimated_package_cost = math.inf # Cannot compute distance
                else:
                    # 6b. Estimate cost based on status
                    if vehicle_carrying_p is None: # Package is at a location, needs pickup
                        # Needs vehicle to reach current_loc_p
                        min_dist_v_to_p_loc = math.inf
                        # Find the closest vehicle
                        for veh, v_loc in vehicle_locations.items():
                            # Check if vehicle location is in the precomputed distances map
                            if v_loc in self.location_distances and current_loc_p in self.location_distances.get(v_loc, {}):
                                 dist = self.location_distances[v_loc][current_loc_p]
                                 min_dist_v_to_p_loc = min(min_dist_v_to_p_loc, dist)

                        # Needs to drive from current_loc_p to goal_l
                        dist_p_loc_to_goal = self.location_distances[current_loc_p][goal_l]

                        # Estimated cost: drive_to_pickup + pickup + drive_to_dropoff + dropoff
                        if min_dist_v_to_p_loc != math.inf and dist_p_loc_to_goal != math.inf:
                             estimated_package_cost = min_dist_v_to_p_loc + 1 + dist_p_loc_to_goal + 1
                        else:
                             estimated_package_cost = math.inf # Cannot reach pickup or goal

                    else: # Package is in a vehicle, needs dropoff
                        # Needs to drive from current_loc_p (vehicle's loc) to goal_l
                        dist_v_loc_to_goal = self.location_distances[current_loc_p][goal_l]

                        # Estimated cost: drive_to_dropoff + dropoff
                        if dist_v_loc_to_goal != math.inf:
                            estimated_package_cost = dist_v_loc_to_goal + 1
                        else:
                            estimated_package_cost = math.inf # Cannot reach goal

            # 6c. Add to total heuristic value
            if estimated_package_cost == math.inf:
                 h_value = math.inf # If any package is unreachable, the state is likely a dead end or very far
                 break # No need to calculate further
            else:
                 h_value += estimated_package_cost

        # 7. Return total heuristic value
        return h_value
