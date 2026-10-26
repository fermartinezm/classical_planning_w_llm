from heuristics.heuristic_base import Heuristic
from task import Task  # Used for type hinting and accessing task structure
from collections import deque
import logging # Optional: for debugging unreachable states

# Configure logging if needed for debugging
# logging.basicConfig(level=logging.INFO)

def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...).
    e.g., '(at p1 l8)' -> ('at', 'p1', 'l8')
    """
    # Remove outer parentheses and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search on a graph to find shortest distances
    from a start node to all other nodes. Assumes unweighted edges (cost 1).
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Ensure current_node is still valid in graph keys (should be if from queue)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing up
    the minimum number of actions required for each package that is not yet
    at its goal location. The cost for a package is estimated based on its
    current status (at a location or in a vehicle) and the shortest driving
    distance to its goal location in the road network.

    Assumptions:
    - This heuristic makes several relaxations:
      - It assumes packages can be moved independently, ignoring potential
        conflicts over vehicles or capacity constraints.
      - It assumes a suitable vehicle is always available at the package's
        current location if needed for pick-up.
      - It only considers the shortest driving distance for the 'drive' part,
        ignoring the need for a vehicle to potentially travel to the package's
        location first if it's not already there.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for any state where all goal packages are
      at reachable locations from their current positions via the road network.

    Heuristic Initialization:
    The heuristic is initialized once per planning task. It parses the task
    description (initial state, goals, static facts) to identify all relevant
    objects (packages, vehicles, locations, sizes). It then builds the road
    network graph based on the '(road l1 l2)' static facts. Finally, it
    precomputes the all-pairs shortest paths between all identified locations
    using Breadth-First Search (BFS), as drive actions have a cost of 1.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Identify objects and locations from initial state, goals, and static facts
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set() # Not strictly needed for this heuristic, but good practice

        def parse_and_collect_objects(fact_string):
            predicate, args = parse_fact(fact_string)
            if predicate == 'in' and len(args) == 2:
                self.packages.add(args[0])
                self.vehicles.add(args[1])
            elif predicate == 'capacity' and len(args) == 2:
                self.vehicles.add(args[0])
                self.sizes.add(args[1])
            elif predicate == 'at' and len(args) == 2:
                # args[0] is locatable (pkg or veh), args[1] is location
                self.locations.add(args[1])
            elif predicate == 'road' and len(args) == 2:
                self.locations.add(args[0])
                self.locations.add(args[1])
            elif predicate == 'capacity-predecessor' and len(args) == 2:
                self.sizes.add(args[0])
                self.sizes.add(args[1])

        # Process initial state, goals, and static facts
        for fact_string in task.initial_state:
            parse_and_collect_objects(fact_string)
        for fact_string in task.goals:
             parse_and_collect_objects(fact_string)
        for fact_string in task.static:
             parse_and_collect_objects(fact_string)

        # Add any locatable object appearing in 'at' facts that wasn't
        # identified as a package from 'in' facts, assuming it's a vehicle.
        all_locatables_in_at = set()
        for fact_string in task.initial_state | task.goals:
             predicate, args = parse_fact(fact_string)
             if predicate == 'at' and len(args) == 2:
                 all_locatables_in_at.add(args[0])
        self.vehicles.update(all_locatables_in_at - self.packages)


        # Build road network graph (adjacency list)
        self.road_graph = {loc: [] for loc in self.locations}
        for fact_string in self.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'road' and len(args) == 2:
                l1, l2 = args
                # Ensure locations are in our collected set before adding edge
                if l1 in self.road_graph and l2 in self.road_graph:
                     self.road_graph[l1].append(l2)

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = bfs(self.road_graph, start_loc)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by verifying if all goal
           facts are present in the state. If yes, the heuristic is 0.
        2. Initialize the total heuristic value `h` to 0.
        3. Create dictionaries to quickly look up the current location of packages
           and vehicles, and which package is in which vehicle, based on the
           facts in the current state.
        4. Identify the target location for each package that is part of the goal
           state (i.e., appears in an '(at ?p ?l_goal)' goal fact).
        5. For each package that has a goal:
           a. Check if the package is already at its goal location in the current
              state. If the fact '(at package goal_location)' is in the state,
              this package contributes 0 to the heuristic, so continue to the
              next package.
           b. If the package is not at the goal, determine its current status
              and location (L_curr):
              - Look for '(at package L_curr)' in the state.
              - If not found, look for '(in package vehicle)' and then
                '(at vehicle L_curr)' in the state.
           c. If the package's current location (L_curr) cannot be determined
              from the state facts, or if L_curr is not a known location from
              the domain, the state is likely problematic or unreachable; return
              float('inf').
           d. Calculate the shortest driving distance (`dist`) from L_curr to the
              package's goal location (L_goal) using the precomputed shortest
              paths stored in `self.shortest_paths`.
           e. If the goal location is unreachable from the current location
              (i.e., `dist` is float('inf')), return float('inf') as the state
              is likely a dead end for this package's goal.
           f. Calculate the estimated actions needed for this specific package
              based on its current status:
              - If the package is currently at L_curr (not in a vehicle):
                It needs a pick-up action (1), followed by driving the vehicle
                to L_goal (dist actions), and finally a drop action (1).
                Estimated cost: 1 + dist + 1 = 2 + dist.
              - If the package is currently in a vehicle at L_curr:
                It needs driving the vehicle to L_goal (dist actions), and
                finally a drop action (1).
                Estimated cost: dist + 1.
           g. Add this estimated cost for the current package to the total
              heuristic value `h`.
        6. After processing all goal packages, return the total heuristic value `h`.

        Args:
            node: The current search node, containing the state (a frozenset of facts).

        Returns:
            An estimated number of actions to reach the goal state, or float('inf')
            if the state is estimated to be a dead end (e.g., a goal package
            is at an unreachable location).
        """
        state = node.state

        # 1. Check if the current state is a goal state.
        if self.goals <= state:
            return 0

        h = 0

        # 3. Map current locations/status for quick lookup
        package_current_location = {}
        package_in_vehicle = {}
        vehicle_location = {}

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'at' and len(args) == 2:
                obj, loc = args
                if obj in self.packages:
                    package_current_location[obj] = loc
                elif obj in self.vehicles:
                    vehicle_location[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                pkg, veh = args
                if pkg in self.packages and veh in self.vehicles:
                     package_in_vehicle[pkg] = veh

        # 4. Identify goal packages and their targets
        goal_packages = {}
        for goal_str in self.goals:
            predicate, args = parse_fact(goal_str)
            if predicate == 'at' and len(args) == 2:
                package, goal_loc = args
                # Only consider goals for known packages at known locations
                if package in self.packages and goal_loc in self.locations:
                    goal_packages[package] = goal_loc
                # else: ignore goal involving unknown object/location

        # 5. For each package that has a goal:
        for package, goal_location in goal_packages.items():
            # 5a. Check if already at goal
            if '(at {} {})'.format(package, goal_location) in state:
                continue # Package is already at its goal

            # 5b. Determine current status and location
            L_curr = None
            is_in_vehicle = False

            if package in package_current_location:
                L_curr = package_current_location[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                if vehicle in vehicle_location:
                    L_curr = vehicle_location[vehicle]
                    is_in_vehicle = True
                else:
                    # 5c. Vehicle location unknown
                    # logging.debug(f"Vehicle {vehicle} carrying package {package} has unknown location in state.")
                    return float('inf') # Cannot determine package location

            # 5c. Package location unknown
            if L_curr is None or L_curr not in self.locations:
                 # This could happen if a goal package is not 'at' any location
                 # and not 'in' any vehicle whose location is known.
                 # logging.debug(f"Package {package} is a goal package but its location is unknown in the state.")
                 return float('inf') # Cannot find package location in state

            # 5d. Calculate shortest distance
            # Ensure L_curr is a valid start node in our precomputed paths
            if L_curr not in self.shortest_paths:
                 # This indicates an issue with location identification or graph building
                 # logging.error(f"Start location {L_curr} not found in precomputed shortest paths.")
                 return float('inf')

            dist_map = self.shortest_paths[L_curr]

            # Ensure goal_location is a valid target in the distance map
            if goal_location not in dist_map:
                 # This indicates an issue with location identification or BFS
                 # logging.error(f"Goal location {goal_location} not found in distance map from {L_curr}.")
                 return float('inf')

            dist = dist_map[goal_location]

            # 5e. Check reachability
            if dist == float('inf'):
                # logging.debug(f"Goal location {goal_location} unreachable from {L_curr} for package {package}.")
                return float('inf') # Goal location is unreachable

            # 5f. Calculate estimated actions for this package
            if is_in_vehicle:
                # Package is in vehicle at L_curr, needs drive and drop
                cost = dist + 1
            else:
                # Package is at L_curr, needs pick-up, drive, drop
                cost = 1 + dist + 1 # pick-up + drive + drop

            # 5g. Add to total
            h += cost

        # 6. Return total heuristic value
        return h
