# Required imports
from collections import deque

# Heuristic class definition
class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum required actions for each package that is not yet at its
        goal location. For a package on the ground, it estimates 2 actions
        (pickup + drop) plus the shortest road distance from its current
        location to its goal location. For a package already in a vehicle,
        it estimates 1 action (drop) plus the shortest road distance from
        the vehicle's current location to the package's goal location.
        The shortest road distances are precomputed using BFS.

    Assumptions:
        - The heuristic assumes that any package can be picked up by any
          vehicle and dropped off, ignoring capacity constraints and
          vehicle availability for initial pickups (except implicitly through
          the vehicle's current location if the package is already in it).
        - It assumes the road network is undirected (if road l1 l2 exists,
          road l2 l1 also exists, which is true in the provided domain file).
        - It assumes all packages that need to be moved have a goal location
          specified in the task goals as an `(at package location)` fact.
        - It assumes that in any valid state, a package is either at a
          location or in a vehicle, and every vehicle is at a location.
        - Vehicle names can be identified from initial state facts that are
          `(at item location)` where `item` is not a package with a goal.

    Heuristic Initialization:
        1. Parse static facts to build the road network graph. Collect all
           unique locations mentioned in road facts.
        2. Parse goal facts to determine the target location for each package
           and collect all goal locations.
        3. Parse initial state facts to identify vehicle names (assuming
           vehicles are the only other locatables initially at locations)
           and collect their initial locations.
        4. Combine all collected locations (from roads, goals, initial state).
        5. Compute shortest path distances between all pairs of these locations
           using Breadth-First Search (BFS). Store these distances.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Create temporary dictionaries to store the current location of
           packages and vehicles, and which package is in which vehicle,
           by iterating through the current state facts.
        3. For each package `p` that has a goal location defined in the task:
            a. Get the package's goal location `goal_loc_p`.
            b. Check if the package `p` is currently on the ground
               (`(at p current_loc_p)` fact exists in the state).
               - If yes, get its `current_loc_p`.
               - If `current_loc_p` is not `goal_loc_p`:
                   - Add 2 to `h` (for the necessary pick-up and drop actions).
                   - Find the shortest distance `dist` between `current_loc_p`
                     and `goal_loc_p` from the precomputed distances.
                   - If `dist` is infinity (unreachable), return infinity
                     immediately as the state is likely unsolvable.
                   - Add `dist` to `h` (for the necessary drive actions).
            c. Else (package `p` is not on the ground, assume it's in a vehicle
               `(in v p)` based on the state parsing).
               - Get the vehicle `v` it is in.
               - Find the vehicle's current location `current_loc_v` from the
                 state parsing.
               - If `current_loc_v` is not `goal_loc_p`:
                   - Add 1 to `h` (for the necessary drop action).
                   - Find the shortest distance `dist` between `current_loc_v`
                     and `goal_loc_p` from the precomputed distances.
                   - If `dist` is infinity (unreachable), return infinity.
                   - Add `dist` to `h` (for the necessary drive actions).
               - Else (`current_loc_v` is `goal_loc_p`):
                   - Add 1 to `h` (for the necessary drop action).
        4. Return the total heuristic value `h`.
        5. If the goal state is reached (checked by the planner before calling
           the heuristic), the heuristic should return 0. The calculation
           above naturally results in 0 if all packages are at their goal
           locations (they won't enter the cost calculation loops).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.road_graph = {}
        self.locations = set()
        self.goal_locations = {}
        self.distances = {}
        self.vehicles = set()

        # 1. Parse static facts and build road graph
        for fact_str in task.static:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            if predicate == 'road':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                self.road_graph.setdefault(loc2, []).append(loc1) # Assuming roads are bidirectional

        # 2. Parse goal facts to get package goal locations and collect goal locations
        for goal_fact_str in task.goals:
             parts = goal_fact_str.strip('()').split()
             if parts[0] == 'at':
                 # Goal is (at package location)
                 package, location = parts[1], parts[2]
                 self.goal_locations[package] = location
                 self.locations.add(location)
                 # Ensure goal locations are in graph keys, even if isolated
                 self.road_graph.setdefault(location, [])

        # 3. Parse initial state facts to identify vehicles and collect initial locations
        # Assuming anything 'at' a location in the initial state that isn't a package goal is a vehicle
        for fact_str in task.initial_state:
             parts = fact_str.strip('()').split()
             if parts[0] == 'at':
                 item = parts[1]
                 # If item is not a package (i.e., not in goal_locations keys), assume it's a vehicle
                 if item not in self.goal_locations:
                     self.vehicles.add(item)
                     self.locations.add(parts[2]) # Add initial vehicle locations to locations set
                     # Ensure initial vehicle locations are in graph keys, even if isolated
                     self.road_graph.setdefault(parts[2], [])

        # 4. Combine all collected locations
        # Ensure all locations mentioned in the graph (keys and values) are in the locations set
        self.locations.update(self.road_graph.keys())
        for neighbors in self.road_graph.values():
             self.locations.update(neighbors)

        # 5. Compute shortest distances using BFS
        # Ensure BFS is run for all collected locations
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        """
        # Use the full set of known locations
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Should not happen if locations are collected correctly, but defensive
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node exists as a key in the graph
            # (handles isolated locations correctly)
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is a known location before checking distance
                    if neighbor in self.locations and distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal, or float('inf')
            if the state is likely unsolvable.
        """
        # The planner typically checks for goal state before calling the heuristic.
        # However, including the check here ensures h=0 for goal states even if
        # called directly.
        if self.task.goal_reached(state):
             return 0

        h = 0
        package_current_loc = {}
        package_current_vehicle = {}
        vehicle_current_loc = {}

        # Parse current state facts
        for fact_str in state:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            if predicate == 'at':
                item, loc = parts[1], parts[2]
                if item in self.goal_locations: # It's a package we care about
                    package_current_loc[item] = loc
                elif item in self.vehicles: # It's a vehicle we identified
                     vehicle_current_loc[item] = loc
                # Ignore other 'at' facts if any

            elif predicate == 'in':
                package, vehicle = parts[1], parts[2]
                # Only track packages that have a goal location
                if package in self.goal_locations:
                    package_current_vehicle[package] = vehicle

        # Iterate through packages that need to reach a goal
        for package, goal_loc in self.goal_locations.items():
            # If package is already at goal, cost is 0 for this package
            # This check is implicitly handled by the logic below, but explicit
            # check is clearer and slightly more efficient if many packages are at goal.
            # However, the loop structure already skips packages at goal.
            # Let's rely on the structure: if package is in package_current_loc
            # and location == goal_loc, it won't enter the cost calculation.

            # Package is not at goal. Calculate cost for this package.
            if package in package_current_loc: # Package is on the ground
                current_loc = package_current_loc[package]
                # This package needs to be picked up, transported, and dropped.
                # Minimum actions: 1 (pickup) + distance(current_loc, goal_loc) + 1 (drop)
                # If current_loc == goal_loc, this case is skipped by the outer loop check.
                # So current_loc != goal_loc here.
                h += 2 # pickup + drop
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    # If goal location is unreachable from package's current location
                    return float('inf')
                h += dist
            elif package in package_current_vehicle: # Package is in a vehicle
                vehicle = package_current_vehicle[package]
                current_loc_v = vehicle_current_loc.get(vehicle)

                if current_loc_v is None:
                     # Vehicle location not found in state - indicates invalid state representation
                     return float('inf')

                # This package needs to be transported (if vehicle not at goal) and dropped.
                # Minimum actions: distance(current_loc_v, goal_loc) + 1 (drop)
                # If current_loc_v == goal_loc, dist is 0, h += 1. Correct.
                h += 1 # drop
                dist = self.distances.get(current_loc_v, {}).get(goal_loc, float('inf'))
                if dist == float('inf'):
                    # If goal location is unreachable from vehicle's current location
                    return float('inf')
                h += dist
            # Else: Package is not 'at' a location and not 'in' a vehicle. This case
            # should ideally not happen in a valid state representation for this domain.
            # We assume all packages are accounted for either 'at' or 'in'.
            # If a package is not in goal_locations, we ignore it.

        return h
