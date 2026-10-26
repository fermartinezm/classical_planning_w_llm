# Assuming heuristics.heuristic_base is available
# from heuristics.heuristic_base import Heuristic
# Assuming collections.deque is available
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes surrounding brackets and splits a fact string into parts."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

class transportHeuristic: # Inherit from Heuristic if available
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal state by summing the minimum actions
    required for each package to reach its goal location, considering pick-up,
    drop, and travel distance. It also adds a penalty for locations where the
    demand for package pick-ups exceeds the available vehicle capacity at that location.

    Assumptions:
    - Roads are bidirectional unless specified otherwise in static facts (code assumes bidirectionality).
    - Packages effectively consume one unit of vehicle capacity. The capacity sizes (c0, c1, etc.) form a linear chain defined by capacity-predecessor facts, where c0 represents 0 available slots, c1 represents 1 available slot, and so on.
    - Goal facts are always of the form (at package location).
    - Objects starting with 'p' are packages and objects starting with 'v' are vehicles (in parsing state facts). This is a common convention but relies on domain naming. A more robust approach would parse object types from the PDDL problem file, but this information is not directly provided by the Task object.
    - All locations mentioned in road facts or initial/goal states are relevant.
    - The road network is connected such that goal locations are reachable from initial locations in solvable problems.

    Heuristic Initialization:
    The constructor (`__init__`) performs the following steps:
    1. Parses static facts (`task.static`) and initial state facts (`task.initial_state`) to identify all locations, vehicles, capacity sizes, and road connections. It builds an adjacency list representation of the road network.
    2. Parses `capacity-predecessor` facts to build a mapping (`capacity_to_int`) from capacity size strings (like 'c0', 'c1') to integers representing the number of available slots (0, 1, etc.). It finds the base capacity ('c0') as the size that is a predecessor but not a successor in any `capacity-predecessor` fact.
    3. Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS) on the road graph. This is stored in a dictionary `self.dist`.
    4. Stores the goal location for each package from `task.goals` in `self.goal_locations`.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic function (`__call__`) computes the estimated cost for a given state (`node.state`) as follows:
    1. It parses the current state to determine the location of each package (either at a location or inside a vehicle) and the location and current capacity of each vehicle. Vehicle capacities are converted from strings to integer slots using the precomputed `capacity_to_int` map.
    2. It identifies the set of packages that are not currently at their specified goal location.
    3. It calculates a base cost by iterating through each package that needs to be moved:
        - If the package is currently at a location `L_start`, it needs a pick-up action (cost 1), a drive action from `L_start` to its goal location `L_goal` (cost equals the shortest distance `dist[L_start][L_goal]`), and a drop action (cost 1). The total base cost added for this package is `1 + dist[L_start][L_goal] + 1`.
        - If the package is currently inside a vehicle `v`, it needs a drive action from the vehicle's current location `L_v` to the package's goal location `L_goal` (cost equals the shortest distance `dist[L_v][L_goal]`), and a drop action (cost 1). The total base cost added for this package is `dist[L_v][L_goal] + 1`.
        - If any required distance is infinite (locations are disconnected), the heuristic returns infinity.
    4. It calculates a capacity penalty. It groups packages that are currently at a location and need to be moved, by their current location. For each location `L` where packages need picking up, it calculates the total number of slots needed (`Needed_slots`) and the total available slots (`Available_slots_at_L`) across all vehicles currently at `L`. If `Needed_slots > `Available_slots_at_L`, there is a capacity deficit. The penalty added to the heuristic is the size of this deficit (`Needed_slots - Available_slots_at_L`). This penalizes states where packages are waiting but there isn't enough vehicle capacity locally to pick them up. This step is skipped if capacity information was not available in the PDDL.
    5. The total heuristic value is the sum of the base costs for all packages needing movement and the capacity penalty.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.initial_state = task.initial_state

        # --- Heuristic Initialization ---
        self.road_graph = {}
        self.locations = set()
        self.capacity_predecessors = {}
        self.capacity_sizes = set()
        self.vehicles = set()

        # Extract info from initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty lines or malformed facts
            if parts[0] == 'capacity':
                if len(parts) > 2: # Ensure fact has enough parts
                    self.vehicles.add(parts[1])
                    self.capacity_sizes.add(parts[2])
            elif parts[0] == 'at':
                 if len(parts) > 2:
                     # Assume objects starting with 'v' are vehicles, 'l' are locations
                     if parts[1].startswith('v'):
                          self.vehicles.add(parts[1])
                     if parts[2].startswith('l'):
                          self.locations.add(parts[2])

        # Extract info from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty lines or malformed facts
            if parts[0] == 'road':
                if len(parts) > 2:
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    if l1 not in self.road_graph: self.road_graph[l1] = []
                    if l2 not in self.road_graph: self.road_graph[l2] = []
                    # Assuming roads are bidirectional
                    self.road_graph[l1].append(l2)
                    self.road_graph[l2].append(l1)
            elif parts[0] == 'capacity-predecessor':
                if len(parts) > 2:
                    s1, s2 = parts[1], parts[2]
                    self.capacity_predecessors[s2] = s1
                    self.capacity_sizes.add(s1)
                    self.capacity_sizes.add(s2)

        # Compute capacity size to integer mapping
        self.capacity_to_int = self._build_capacity_map()

        # Compute all-pairs shortest paths
        self.dist = self._compute_all_pairs_shortest_paths()

        # Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) > 2:
                pkg, loc = parts[1], parts[2]
                self.goal_locations[pkg] = loc


    def _build_capacity_map(self):
        """Builds a mapping from capacity size string to integer (available slots)."""
        capacity_to_int = {}
        if not self.capacity_sizes:
             return capacity_to_int # No capacity info, return empty map

        # Find the root capacity (smallest number of slots, usually c0)
        successors = set(self.capacity_predecessors.keys())
        predecessors = set(self.capacity_predecessors.values())

        # The root is a predecessor but not a successor
        root_candidates = predecessors - successors

        if not root_candidates:
             # This might happen if there's only one capacity size or a cycle (invalid PDDL?)
             # Or if the smallest capacity is not explicitly a predecessor of anything.
             # If capacity_sizes is not empty but no root found, it's an issue with the PDDL definition.
             # For robustness, if there's exactly one size and no predecessors, assume it's c0?
             if len(self.capacity_sizes) == 1 and not self.capacity_predecessors:
                 root_capacity = list(self.capacity_sizes)[0]
             else:
                 # Cannot determine root, capacity heuristic won't work correctly.
                 # Return empty map, heuristic will ignore capacity penalty.
                 # print("Warning: Could not determine root capacity size. Capacity heuristic disabled.")
                 return {}
        else:
             # Assuming exactly one root capacity
             root_capacity = root_candidates.pop()


        capacity_to_int[root_capacity] = 0
        current_capacity = root_capacity
        current_value = 0

        # Build map by following successors (reverse of predecessor map)
        successor_map = {s1: s2 for s2, s1 in self.capacity_predecessors.items()}

        while current_capacity in successor_map:
            next_capacity = successor_map[current_capacity]
            current_value += 1
            capacity_to_int[next_capacity] = current_value
            current_capacity = next_capacity

        return capacity_to_int

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        dist = {}
        for start_node in self.locations:
            dist[start_node] = self._bfs(start_node)
        return dist

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node is not in graph (isolated location with no roads), skip
            if current_node not in self.road_graph:
                 continue

            for neighbor in self.road_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the current state.

        Returns:
            The estimated number of actions to reach a goal state.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic:

        # 1. Parse the current state to get object locations and vehicle capacities.
        pkg_loc = {} # package -> location or vehicle
        veh_loc = {} # vehicle -> location
        veh_cap_str = {} # vehicle -> capacity_string

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty lines or malformed facts
            predicate = parts[0]
            if predicate == 'at' and len(parts) > 2:
                obj, loc = parts[1], parts[2]
                # Determine if obj is package or vehicle based on its presence in self.vehicles
                if obj in self.vehicles:
                    veh_loc[obj] = loc
                else: # Assume it's a package if not a known vehicle
                    pkg_loc[obj] = loc
            elif predicate == 'in' and len(parts) > 2:
                pkg, veh = parts[1], parts[2]
                pkg_loc[pkg] = veh # Package is inside vehicle
            elif predicate == 'capacity' and len(parts) > 2:
                veh, cap_str = parts[1], parts[2]
                veh_cap_str[veh] = cap_str

        # Convert vehicle capacity strings to integer slots
        # If capacity_to_int is empty, capacity is not relevant or defined, treat as infinite slots
        if not self.capacity_to_int:
             veh_slots = {v: float('inf') for v in self.vehicles}
        else:
             # Default to 0 slots if a vehicle's capacity is not listed in the state (shouldn't happen in valid states)
             veh_slots = {veh: self.capacity_to_int.get(veh_cap_str.get(veh), 0) for veh in self.vehicles}


        # 2. Identify packages that are not at their goal location.
        packages_to_move = set()
        for goal in self.goals:
            # Assuming goals are always (at package location)
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) > 2:
                pkg, goal_loc = parts[1], parts[2]
                # Check if the goal fact is NOT true in the current state
                if '(at {} {})'.format(pkg, goal_loc) not in state:
                     packages_to_move.add(pkg)

        # If all goal packages are at their goal, the heuristic is 0.
        if not packages_to_move:
            return 0

        # 3. Calculate the base cost for moving each package.
        # This is a sum of minimum actions (pick-up, drop, travel) assuming
        # vehicles are available and have capacity when needed.
        h = 0
        packages_needing_pickup_at_loc = {} # location -> set of packages needing pickup there

        for pkg in packages_to_move:
            current_loc_or_veh = pkg_loc.get(pkg) # Get package's current location or vehicle
            goal_loc = self.goal_locations.get(pkg) # Get package's goal location

            if goal_loc is None:
                 # Package is in packages_to_move but has no goal location? Should not happen.
                 continue # Skip this package

            if current_loc_or_veh is None:
                 # This package is a goal but not in the state. Should not happen in solvable problems.
                 # Treat as unreachable or very high cost.
                 return float('inf')

            if current_loc_or_veh in self.vehicles: # Package is in a vehicle
                veh = current_loc_or_veh
                current_veh_loc = veh_loc.get(veh) # Get vehicle's location

                if current_veh_loc is None:
                     # Vehicle carrying package is not located anywhere? Should not happen.
                     return float('inf')

                # Cost for package in vehicle: drive from vehicle location to goal + drop
                # Need distance from current_veh_loc to goal_loc
                # Check if locations exist in distance map (handle isolated locations)
                if current_veh_loc not in self.dist or goal_loc not in self.dist.get(current_veh_loc, {}):
                     return float('inf') # Goal location unreachable from vehicle location

                travel_cost = self.dist[current_veh_loc][goal_loc]
                if travel_cost == float('inf'): return float('inf') # Goal location unreachable

                h += 1 # drop action
                h += travel_cost # drive action

            else: # Package is at a location (not in a vehicle)
                current_pkg_loc = current_loc_or_veh # This is L_start

                # Cost for package at location: pick-up + drive from L_start to goal + drop
                # Need distance from current_pkg_loc to goal_loc
                 # Check if locations exist in distance map
                if current_pkg_loc not in self.dist or goal_loc not in self.dist.get(current_pkg_loc, {}):
                     return float('inf') # Goal location unreachable from package location

                travel_cost = self.dist[current_pkg_loc][goal_loc]
                if travel_cost == float('inf'): return float('inf') # Goal location unreachable

                h += 1 # pick-up action
                h += 1 # drop action
                h += travel_cost # drive action

                # This package needs a pick-up at current_pkg_loc
                if current_pkg_loc not in packages_needing_pickup_at_loc:
                    packages_needing_pickup_at_loc[current_pkg_loc] = set()
                packages_needing_pickup_at_loc[current_pkg_loc].add(pkg)


        # 4. Calculate a penalty based on local capacity deficits for pick-ups.
        # If capacity_to_int is empty, capacity is not relevant or defined, skip penalty.
        if self.capacity_to_int:
            for loc, pkgs_at_loc in packages_needing_pickup_at_loc.items():
                needed_slots = len(pkgs_at_loc)
                # Sum available slots of vehicles currently at this location
                # Ensure vehicle location is known before checking equality
                available_slots_at_loc = sum(veh_slots.get(v, 0) for v in self.vehicles if veh_loc.get(v) == loc)

                deficit = needed_slots - available_slots_at_loc
                h += max(0, deficit) # Add the number of extra slots needed as a penalty

        # 5. Return the total estimated cost.
        return h

# Note: The Heuristic base class is assumed to be provided by the environment.
# If not, a minimal base class would be needed:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass
