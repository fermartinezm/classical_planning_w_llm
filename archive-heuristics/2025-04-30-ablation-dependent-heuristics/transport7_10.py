from collections import deque
import sys

# Define a large number for unreachable states
UNREACHABLE_COST = sys.maxsize

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each package that is not yet at its goal location.
    For a package on the ground, the estimated cost is 2 (pick-up + drop)
    plus the shortest road distance from its current location to its goal location.
    For a package inside a vehicle, the estimated cost is 1 (drop) plus the
    shortest road distance from the vehicle's current location to the package's
    goal location. The heuristic ignores vehicle capacity constraints and the
    cost for a vehicle to initially reach a package's location if the package
    is on the ground.

    Assumptions:
    - Roads are bidirectional (inferred from example PDDL).
    - Vehicle capacity constraints are ignored.
    - The cost for a vehicle to move to a package's initial location (if the
      package is on the ground) is not explicitly included in the package's cost;
      it's implicitly handled by the search exploring states where vehicles move.
      The heuristic only counts the cost of moving the package itself (pick-up,
      drop, and vehicle travel *with* the package).
    - All locations mentioned in the problem are part of the road network or
      are isolated nodes. Unreachable goal locations result in a very large
      heuristic value.
    - The state representation is consistent (a package is either 'at' a location
      or 'in' a vehicle, and vehicles are always 'at' a location).

    Heuristic Initialization:
    The constructor analyzes the static facts from the task description,
    specifically 'road' and 'capacity-predecessor' predicates, and also
    identifies all packages, vehicles, and locations from the initial state
    and goals. It builds a graph of the road network and precomputes the
    shortest path distances between all pairs of locations using BFS. It also
    stores the goal location for each package.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic cost to 0.
    2. Parse the current state to determine the current location of every
       package (either at a location or in a vehicle) and every vehicle.
       Store these in dictionaries (package_locations, package_in_vehicle,
       vehicle_locations).
    3. Iterate through each package that has a goal location specified in the task.
    4. For a package 'p' with goal location 'l_goal':
       a. Check if '(at p l_goal)' is already true in the current state. If yes,
          this part of the goal is satisfied, and the cost for this package is 0.
          Continue to the next package.
       b. If the goal is not satisfied, find the current status of package 'p'
          by looking it up in the parsed state information (package_locations
          or package_in_vehicle).
       c. If 'p' is currently at a location 'p_loc' (i.e., '(at p p_loc)' is true):
          - Calculate the shortest distance 'dist' from 'p_loc' to 'l_goal'
            using the precomputed distance matrix.
          - If 'l_goal' is unreachable from 'p_loc' (distance is infinity),
            return a large number as the heuristic value for the entire state,
            indicating this state likely cannot reach the goal.
          - Add 2 + 'dist' to the total heuristic cost. This accounts for the
            pick-up action (1), the drive action(s) carrying the package (dist),
            and the drop action (1).
       d. If 'p' is currently in a vehicle 'v' (i.e., '(in p v)' is true):
          - Find the current location 'v_loc' of vehicle 'v' by looking it up
            in the parsed state information (vehicle_locations).
          - If the vehicle's location is not found (should not happen in valid
            states), return a large number.
          - Calculate the shortest distance 'dist' from 'v_loc' to 'l_goal'
            using the precomputed distance matrix.
          - If 'l_goal' is unreachable from 'v_goal' (distance is infinity),
            return a large number.
          - Add 1 + 'dist' to the total heuristic cost. This accounts for the
            drive action(s) carrying the package (dist) and the drop action (1).
       e. If 'p' is not found to be either at a location or in a vehicle (indicates
          an invalid state representation or parsing issue), return a large number.
    5. Return the total calculated heuristic cost.
    """
    def __init__(self, task):
        self.task = task
        self.package_names = set()
        self.vehicle_names = set()
        self.location_names = set()
        self.size_names = set()
        self.road_graph = {}
        self.capacity_predecessors = {} # Not used in this heuristic, but parsed

        # Parse initial state, static facts, and goals to identify objects and static info
        all_relevant_facts = task.initial_state | task.static | task.goals

        for fact in all_relevant_facts:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'capacity':
                self.vehicle_names.add(parts[1])
                self.size_names.add(parts[2])
            elif predicate == 'capacity-predecessor':
                self.size_names.add(parts[1])
                self.size_names.add(parts[2])
                self.capacity_predecessors[parts[1]] = parts[2]
            elif predicate == 'road':
                loc1 = parts[1]
                loc2 = parts[2]
                self.location_names.add(loc1)
                self.location_names.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional
            elif predicate == 'in':
                self.package_names.add(parts[1])
                self.vehicle_names.add(parts[2])
            elif predicate == 'at':
                # Add location from 'at' facts
                self.location_names.add(parts[2])
                # Object type (package/vehicle) will be determined by checking against
                # self.package_names and self.vehicle_names sets populated by 'in'/'capacity'.

        # Ensure all locations from road graph are in location_names
        for loc in self.road_graph:
            self.location_names.add(loc)
            for neighbor in self.road_graph[loc]:
                self.location_names.add(neighbor)

        # Ensure all locations from goals are in location_names
        for goal_fact in task.goals:
            parts = goal_fact.strip('()').split()
            if parts[0] == 'at':
                self.location_names.add(parts[2])

        # Ensure all locations from initial state 'at' facts are in location_names
        for fact in task.initial_state:
            parts = fact.strip('()').split()
            if parts[0] == 'at':
                self.location_names.add(parts[2])


        # Compute all-pairs shortest paths
        self.location_list = sorted(list(self.location_names))
        self.location_to_index = {loc: i for i, loc in enumerate(self.location_list)}
        self.num_locations = len(self.location_list)
        self.distance_matrix = self._compute_all_pairs_shortest_paths()

        # Store goal locations for easy access
        self.package_goal_locations = {} # {package_name: goal_location_name}
        for goal_fact in task.goals:
            parts = goal_fact.strip('()').split()
            if parts[0] == 'at':
                self.package_goal_locations[parts[1]] = parts[2]
            # Assuming goals are only (at package location)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {loc: float('inf') for loc in self.location_names}
        if start_node not in self.location_names:
             # Start node is not a known location, cannot reach anything
             return distances

        distances[start_node] = 0
        queue = deque([start_node])
        # Use a set for visited for faster lookups
        visited = {start_node}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest paths between all pairs of locations using BFS."""
        dist_matrix = [[float('inf')] * self.num_locations for _ in range(self.num_locations)]
        for i, start_loc in enumerate(self.location_list):
            distances_from_start = self._bfs(start_loc)
            for j, end_loc in enumerate(self.location_list):
                dist_matrix[i][j] = distances_from_start[end_loc]
        return dist_matrix

    def get_distance(self, loc1, loc2):
        """Looks up the precomputed shortest distance between two locations."""
        if loc1 not in self.location_to_index or loc2 not in self.location_to_index:
            # This case should ideally not happen if all relevant locations are collected
            # but handles potential inconsistencies defensively.
            return float('inf')
        idx1 = self.location_to_index[loc1]
        idx2 = self.location_to_index[loc2]
        return self.distance_matrix[idx1][idx2]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach a goal state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        package_locations = {} # {package_name: location_name} for packages *at* a location
        package_in_vehicle = {} # {package_name: vehicle_name} for packages *in* a vehicle
        vehicle_locations = {} # {vehicle_name: location_name}

        # Parse the current state to find locations of packages and vehicles
        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'at':
                obj_name = parts[1]
                loc_name = parts[2]
                if obj_name in self.package_names:
                    package_locations[obj_name] = loc_name
                elif obj_name in self.vehicle_names:
                    vehicle_locations[obj_name] = loc_name
                # Ignore other 'at' facts if any
            elif predicate == 'in':
                pkg_name = parts[1]
                veh_name = parts[2]
                if pkg_name in self.package_names and veh_name in self.vehicle_names:
                     package_in_vehicle[pkg_name] = veh_name
                # Ignore other 'in' facts if any

        total_cost = 0

        # Iterate through each package goal
        for p_name, l_goal in self.package_goal_locations.items():
            # Check if this specific package goal is already satisfied
            if f'(at {p_name} {l_goal})' in state:
                continue # This package is already at its goal

            # Find current status of package p_name
            if p_name in package_locations:
                # Package is at a location p_loc
                p_loc = package_locations[p_name]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                dist = self.get_distance(p_loc, l_goal)
                if dist == float('inf'):
                     # Goal is unreachable for this package
                     return UNREACHABLE_COST
                total_cost += 2 + dist
            elif p_name in package_in_vehicle:
                # Package is in a vehicle v_name
                v_name = package_in_vehicle[p_name]
                # Find vehicle's location
                if v_name in vehicle_locations:
                    v_loc = vehicle_locations[v_name]
                    # Cost: drive (distance) + drop (1)
                    dist = self.get_distance(v_loc, l_goal)
                    if dist == float('inf'):
                         # Goal is unreachable for this package
                         return UNREACHABLE_COST
                    total_cost += 1 + dist
                else:
                     # Vehicle containing package is not at any location? Invalid state?
                     # Treat as unreachable.
                     return UNREACHABLE_COST
            else:
                # Package is not at a location and not in a vehicle? Invalid state?
                # Treat as unreachable.
                return UNREACHABLE_COST

        # If we reached here, all individual package goals were either satisfied
        # or contributed a finite cost. The total_cost is the sum.
        # If total_cost is 0, it means all package goals were satisfied,
        # which should have been caught by the initial goal_reached check.
        # If the goal set is empty, goal_reached returns True, and we return 0.
        # So, this return is correct.
        return total_cost
