from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function outside the class
def get_parts(fact):
    """Removes surrounding brackets and splits the fact string into parts."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        minimum actions required for each package that is not currently at
        its goal location. It considers the cost of picking up the package
        (if needed), driving the minimum distance to the goal location, and
        dropping the package. It ignores vehicle capacity and the possibility
        of transporting multiple packages together.

    Assumptions:
        - Vehicle capacity constraints are ignored. Any vehicle is assumed
          to have sufficient capacity to pick up any package.
        - Multiple packages can be transported by the same vehicle, but the
          heuristic calculates the drive cost independently for each package,
          potentially overestimating the benefit of shared rides.
        - Vehicles are assumed to be available to pick up packages when needed.
        - The state representation is complete and accurate regarding the
          location of all packages and vehicles mentioned in the problem.
        - Goal conditions only involve packages being at specific locations.
        - All locations mentioned in facts are part of the road network graph.

    Heuristic Initialization:
        In the constructor, the heuristic precomputes static information:
        1. Identifies all locations, packages, and vehicles present in the
           initial state, goals, operators, and static facts by inspecting
           predicate arguments based on the domain definition.
        2. Builds a graph of locations based on the 'road' facts found in
           the static information. Roads are assumed to be bidirectional.
        3. Computes the shortest path distance (minimum number of drive actions)
           between all pairs of identified locations using Breadth-First Search (BFS).
        4. Stores the goal location for each package that has a goal specified
           in the task's goal conditions.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state using the task's
           `goal_reached` method. If yes, the heuristic value is 0.
        2. If not a goal state, initialize the total heuristic cost to 0.
        3. Iterate through the facts in the current state to determine the
           current location or containing vehicle for every package and the
           current location for every vehicle. Store this information in
           dictionaries for quick lookup.
        4. For each package that has a goal location specified in the task:
            a. Find the package's current status (at a location or in a vehicle)
               from the information gathered in step 3.
            b. If the package's current status is unknown (which should not
               happen in a valid state), return infinity as the state might
               be unreachable or invalid.
            c. If the package is currently at a location `l`:
                - If `l` is not the goal location `goal_l`:
                    - This package needs to be picked up, transported, and dropped.
                    - Add 1 (for the pick-up action) to the total cost.
                    - Find the shortest distance (number of drive actions)
                      between `l` and `goal_l` using the precomputed distances.
                      If the goal location is unreachable from the current
                      location, return infinity. Otherwise, add this distance
                      to the total cost.
                    - Add 1 (for the drop action) to the total cost.
                - If `l` is the goal location, the cost for this package is 0.
            d. If the package is currently in a vehicle `v`:
                - Find the current location `l_v` of vehicle `v` from the
                  information gathered in step 3. If the vehicle's location
                  is unknown (which should not happen in a valid state),
                  return infinity.
                - If `l_v` is not the goal location `goal_l`:
                    - This package needs to be transported (while in the vehicle)
                      and dropped.
                    - Find the shortest distance (number of drive actions)
                      between `l_v` and `goal_l`. If unreachable, return infinity.
                      Otherwise, add this distance to the total cost.
                    - Add 1 (for the drop action) to the total cost.
                - If `l_v` is the goal location:
                    - This package only needs to be dropped.
                    - Add 1 (for the drop action) to the total cost.
        5. Return the accumulated total cost.
    """
    def __init__(self, task):
        self.task = task
        self.goals = task.goals

        # Identify objects by type based on predicate positions in domain definition
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()
        self.capacity_predecessors = {} # s1 -> s2 (s1 is smaller than s2)

        # Collect all literals from initial state, goals, operators, and static facts
        all_literals = set(task.initial_state) | set(task.goals)
        for op in task.operators:
            all_literals |= op.preconditions | op.add_effects | op.del_effects
        all_literals |= task.static

        # First pass: Identify objects based on specific predicates ('in', 'capacity', 'capacity-predecessor', 'road')
        for lit in all_literals:
            if lit.startswith('(not '):
                fact = lit[5:-1]
            else:
                fact = lit[1:-1]

            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == 'in' and len(parts) == 3:
                self.packages.add(parts[1])
                self.vehicles.add(parts[2])
            elif pred == 'capacity' and len(parts) == 3:
                self.vehicles.add(parts[1])
                self.sizes.add(parts[2])
            elif pred == 'capacity-predecessor' and len(parts) == 3:
                self.sizes.add(parts[1])
                self.sizes.add(parts[2])
                self.capacity_predecessors[parts[1]] = parts[2]
            elif pred == 'road' and len(parts) == 3:
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Second pass: Identify objects from 'at' facts. If an object is not
        # already identified as a vehicle, assume it is a package. Add locations.
        for lit in all_literals:
             if lit.startswith('(not '):
                fact = lit[5:-1]
             else:
                fact = lit[1:-1]
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc) # Ensure location is added
                 if obj not in self.vehicles: # If it's not a vehicle we identified
                     self.packages.add(obj) # Assume it's a package

        # Build location graph from road facts
        self.location_graph = {loc: set() for loc in self.locations}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Ensure locations are in our identified set before adding to graph
                if l1 in self.location_graph and l2 in self.location_graph:
                     self.location_graph[l1].add(l2)
                     self.location_graph[l2].add(l1) # Roads are bidirectional

        # Compute all-pairs shortest paths
        self.distances = self.compute_all_pairs_shortest_paths(self.location_graph)

        # Store goal locations for packages
        self.package_goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                pkg, loc = parts[1], parts[2]
                if pkg in self.packages: # Only track goals for identified packages
                    self.package_goal_locations[pkg] = loc

    def compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        # Handle empty graph case
        if not graph:
            return distances

        for start_node in graph:
            distances[start_node] = self.bfs(graph, start_node)
        return distances

    def bfs(self, graph, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in graph}
        # Ensure start_node is in the graph keys
        if start_node not in distances:
             # This indicates a location was identified but not part of the road network graph.
             # Return distances with inf for all nodes as they are unreachable from start_node.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node exists in graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state

        # If goal is reached, heuristic is 0
        if self.task.goal_reached(state):
            return 0

        current_package_state = {} # Maps package -> location (if at) or vehicle (if in)
        current_vehicle_location = {} # Maps vehicle -> location

        # Populate current locations/vehicles from state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    current_package_state[obj] = loc
                elif obj in self.vehicles:
                    current_vehicle_location[obj] = loc
            elif pred == 'in' and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                if pkg in self.packages and veh in self.vehicles:
                     current_package_state[pkg] = veh # Store the vehicle name

        total_cost = 0

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.package_goal_locations.items():
            current_state_val = current_package_state.get(package)

            # If package is not in current_package_state, it means it's not
            # at any location and not in any vehicle. This indicates an issue
            # with the state representation or object identification.
            # Assuming valid states, this case implies the package might not
            # exist or is in an unknown state. For a heuristic, returning inf
            # is a safe bet for potentially unsolvable paths.
            if current_state_val is None:
                 return float('inf')

            if current_state_val in self.locations: # Package is at a location
                current_location = current_state_val
                if current_location != goal_location:
                    # Needs pick-up, drive, drop
                    dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))
                    if dist == float('inf'):
                         return float('inf') # Unreachable goal location
                    total_cost += 1 # pick-up action
                    total_cost += dist # drive actions
                    total_cost += 1 # drop action
                # else: package is at goal location, cost is 0 for this package

            elif current_state_val in self.vehicles: # Package is in a vehicle
                vehicle = current_state_val
                current_vehicle_loc = current_vehicle_location.get(vehicle)

                # If vehicle location not found, indicates issue with state representation
                if current_vehicle_loc is None:
                     return float('inf')

                if current_vehicle_loc != goal_location:
                    # Needs drive, drop
                    dist = self.distances.get(current_vehicle_loc, {}).get(goal_location, float('inf'))
                    if dist == float('inf'):
                         return float('inf') # Unreachable goal location
                    total_cost += dist # drive actions
                    total_cost += 1 # drop action
                else: # Vehicle is at goal location
                    # Needs drop
                    total_cost += 1 # drop action

            # else: current_state_val is neither a location nor a vehicle? Invalid state?
            # Assuming valid states based on PDDL structure.

        return total_cost
