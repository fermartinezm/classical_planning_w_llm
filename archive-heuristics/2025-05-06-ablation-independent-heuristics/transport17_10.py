import heapq
from collections import deque, defaultdict
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at obj loc)' -> ('at', 'obj', 'loc')
    """
    # Remove parentheses and split by space
    parts = fact_string.strip('()').split()
    return tuple(parts)


class transportHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the transport domain.
    It estimates the cost to reach the goal by summing the minimum estimated
    actions required for each package that is not yet at its goal location.
    For a package not at its goal, the heuristic estimates the cost based on
    its current state:
    - If the package is at a location different from its goal:
      Estimated cost = (cost to get a vehicle with sufficient capacity to package location)
                       + 1 (pick-up)
                       + shortest_path_distance(package_location, goal_location)
                       + 1 (drop).
      The cost to get a vehicle is the minimum shortest path distance from any
      vehicle with sufficient capacity to the package's location. If a suitable
      vehicle is already at the package's location, this cost is 0.
    - If the package is inside a vehicle:
      Estimated cost = shortest_path_distance(vehicle_location, goal_location) + 1 (drop).
      This assumes the vehicle is free to move the package directly to the goal.
    The total heuristic is the sum of these estimated costs for all packages
    not at their goal location.

    Assumptions:
    - The goal state consists only of (at package location) facts for packages.
    - The road network is undirected (if road l1 l2 exists, road l2 l1 also exists).
    - The capacity hierarchy is defined by capacity-predecessor facts, and
      a vehicle can pick up a package if its current capacity is not the minimum
      capacity size (c0).
    - The heuristic ignores potential conflicts or shared vehicle usage between
      packages, treating each package's movement independently.
    - Objects in (at ...) facts that are not packages mentioned in the goal
      are assumed to be vehicles.

    Heuristic Initialization:
    1. Parse the goal facts to create a mapping from each package to its goal location.
    2. Parse the static facts to:
       - Build the road network graph.
       - Identify all unique locations.
       - Determine the minimum capacity size (c0) from capacity-predecessor facts.
    3. Compute all-pairs shortest paths between all locations using BFS. Store
       these distances.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state.
    2. Initialize the total heuristic value to 0.
    3. Create temporary dictionaries/mappings for the current state:
       - package_location: maps package name to its current location (if at a location).
       - package_in_vehicle: maps package name to the vehicle it is inside (if inside a vehicle).
       - vehicle_location: maps vehicle name to its current location.
       - vehicles_with_capacity: maps vehicle name to its current capacity size.
    4. Populate these mappings by iterating through the facts in the current state.
    5. Identify vehicles that have sufficient capacity (capacity size is not c0)
       and their current locations.
    6. Iterate through each package and its goal location stored during initialization.
    7. For the current package:
       - Check if the package is already at its goal location in the current state. If yes, continue to the next package.
       - If not, determine the package's current status:
         - If the package is at a location (found in package_location mapping):
           - Get the current location `l_p` and goal location `l_goal`.
           - Look up shortest path distance `dist_p_goal` from `l_p` to `l_goal`.
           - Calculate the penalty for getting a suitable vehicle to `l_p`:
             - Find the minimum shortest path distance `min_dist_v_p` from any vehicle
               with sufficient capacity to `l_p`.
             - If a suitable vehicle is already at `l_p`, `min_dist_v_p` is 0.
             - If no suitable vehicle exists or none are reachable, `min_dist_v_p` is infinity.
           - Estimated cost for package = `min_dist_v_p` + 1 (pick-up) + `dist_p_goal` + 1 (drop).
           - Handle infinity propagation.
         - If the package is inside a vehicle (found in package_in_vehicle mapping):
           - Get the vehicle `v` and its current location `l_v`.
           - Get the goal location `l_goal`.
           - Look up shortest path distance `dist_v_goal` from `l_v` to `l_goal`.
           - Estimated cost for package = `dist_v_goal` + 1 (drop).
           - Handle infinity propagation.
         - If the package's status is not found (should not happen in valid states),
           estimated cost is infinity.
    8. Add the estimated cost for the package to the total heuristic value.
    9. Return the total heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.road_graph = defaultdict(list)
        self.locations = set()
        self.shortest_paths = {}
        self.min_capacity_size = None # To store the identifier for the minimum capacity (c0)

        # 1. Parse goal facts
        for goal_fact_str in task.goals:
            # Assuming goal facts are always (at package location)
            parts = parse_fact(goal_fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
            # Add other goal types parsing if necessary, but based on examples, only 'at' facts for packages are expected.

        # 2. Parse static facts
        predecessor_sizes = set()
        successor_sizes = set()

        for static_fact_str in task.static:
            parts = parse_fact(static_fact_str)
            if parts[0] == 'road' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.road_graph[loc1].append(loc2)
                self.road_graph[loc2].append(loc1) # Assuming roads are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)
            elif parts[0] == 'capacity-predecessor' and len(parts) == 3:
                s1, s2 = parts[1], parts[2]
                predecessor_sizes.add(s1)
                successor_sizes.add(s2)

        # Determine the minimum capacity size (c0)
        # c0 is the size that is a predecessor but never a successor in capacity-predecessor facts
        potential_c0 = predecessor_sizes - successor_sizes
        if len(potential_c0) == 1:
            self.min_capacity_size = list(potential_c0)[0]
        elif len(potential_c0) > 1:
            # This case shouldn't happen in a well-formed PDDL with a single capacity chain
            logging.warning(f"Found multiple potential minimum capacity sizes: {potential_c0}. Using the first one.")
            self.min_capacity_size = list(potential_c0)[0]
        # else: self.min_capacity_size remains None, indicating no capacity restriction based on c0 was found

        # 3. Compute all-pairs shortest paths
        self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        for start_loc in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                current_loc = queue.popleft()

                # Check if current_loc is in road_graph keys before accessing
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)

            # Store distances from start_loc to all other reachable locations
            for end_loc in self.locations:
                 # Only store if reachable (distance is not inf)
                 if distances[end_loc] != float('inf'):
                    self.shortest_paths[(start_loc, end_loc)] = distances[end_loc]


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Keyword arguments:
        node -- the current state

        Returns:
        The estimated number of actions to reach the goal.
        """
        state = node.state
        h_value = 0

        # Create quick lookup dictionaries for the current state
        package_location = {}
        package_in_vehicle = {}
        vehicle_location = {}
        vehicles_with_capacity = {} # vehicle -> capacity_size

        for fact_str in state:
            parts = parse_fact(fact_str)
            if parts[0] == 'at':
                obj_name, loc_name = parts[1], parts[2]
                # Check if the object is a package we care about (in goals)
                if obj_name in self.package_goals:
                     package_location[obj_name] = loc_name
                # Assume anything else with 'at' is a vehicle for simplicity in this domain
                # A more robust way would involve parsing types from the PDDL domain/instance
                elif obj_name.startswith('v'): # Simple check based on example names
                     vehicle_location[obj_name] = loc_name

            elif parts[0] == 'in':
                 package_name, vehicle_name = parts[1], parts[2]
                 if package_name in self.package_goals: # Only track packages relevant to goals
                     package_in_vehicle[package_name] = vehicle_name

            elif parts[0] == 'capacity':
                 v_name, s_name = parts[1], parts[2]
                 if v_name.startswith('v'): # Assume objects with capacity are vehicles
                     vehicles_with_capacity[v_name] = s_name


        # Identify vehicles that have sufficient capacity (not c0) and their locations
        vehicles_with_sufficient_capacity_loc = {} # vehicle -> location
        for v_name, s_name in vehicles_with_capacity.items():
             # Check if capacity is sufficient (not the minimum size, if c0 was found)
             if self.min_capacity_size is None or s_name != self.min_capacity_size:
                  # Find its location
                  if v_name in vehicle_location:
                      vehicles_with_sufficient_capacity_loc[v_name] = vehicle_location[v_name]
                  # else: vehicle capacity known, but location unknown (should not happen in valid states)


        # Calculate heuristic contribution for each package not at its goal
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at goal
            if package in package_location and package_location[package] == goal_loc:
                continue # Package is at goal, contributes 0

            # Package is not at goal, calculate its cost
            cost_for_package = float('inf') # Initialize with infinity

            if package in package_location:
                # Package is at a location, but not the goal
                current_loc = package_location[package]
                base_cost = 1 + 1 # pick-up + drop

                # Get drive cost from package's current location to goal
                drive_cost_package_to_goal = self.shortest_paths.get((current_loc, goal_loc), float('inf'))

                # Calculate penalty for getting a vehicle to current_loc if needed
                penalty = float('inf') # Assume infinite penalty initially
                vehicle_needed_at_current_loc = True # Assume vehicle is needed unless one is already there

                # Check if any vehicle with sufficient capacity is already at current_loc
                for v_name, v_loc in vehicles_with_sufficient_capacity_loc.items():
                     if v_loc == current_loc:
                         penalty = 0 # Found a suitable vehicle at the location
                         vehicle_needed_at_current_loc = False
                         break # No need to check other vehicles

                # If no suitable vehicle is at current_loc, calculate min drive cost to bring one
                if vehicle_needed_at_current_loc:
                     min_drive_to_package = float('inf')
                     found_reachable_vehicle = False
                     for v_name, v_loc in vehicles_with_sufficient_capacity_loc.items():
                         if (v_loc, current_loc) in self.shortest_paths:
                             dist = self.shortest_paths[(v_loc, current_loc)]
                             min_drive_to_package = min(min_drive_to_package, dist)
                             found_reachable_vehicle = True

                     if found_reachable_vehicle:
                         penalty = min_drive_to_package
                     # else: penalty remains inf if no suitable vehicle exists or none are reachable

                # Total cost for package at location = penalty + base_cost + drive_cost_package_to_goal
                # Ensure inf + finite = inf
                if penalty == float('inf') or drive_cost_package_to_goal == float('inf'):
                     cost_for_package = float('inf')
                else:
                     cost_for_package = penalty + base_cost + drive_cost_package_to_goal


            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_location:
                    # Vehicle location is known
                    vehicle_loc = vehicle_location[vehicle]
                    # Cost = drive (shortest_path) + drop (1)
                    drive_cost_vehicle_to_goal = self.shortest_paths.get((vehicle_loc, goal_loc), float('inf'))

                    if drive_cost_vehicle_to_goal == float('inf'):
                         cost_for_package = float('inf')
                    else:
                         cost_for_package = drive_cost_vehicle_to_goal + 1 # drive + drop
                # else: vehicle location is unknown (should not happen), cost remains inf
            # else: package status is unknown (should not happen in valid states), cost remains inf

            # Add the cost for this package to the total heuristic
            # If cost_for_package is inf, the total heuristic becomes inf
            # This handles cases where a package goal is unreachable from its current state
            if cost_for_package == float('inf'):
                 return float('inf') # If any package goal is unreachable, the state is likely a dead end
            h_value += cost_for_package

        # If the loop finishes, all packages had a finite cost, sum them up.
        # If there are no packages in goals, h_value is 0.
        # If all packages are at their goal, h_value is 0.
        return h_value

