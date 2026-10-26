from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string by removing parentheses and splitting."""
    # Example: "(at package1 location1)" -> ["at", "package1", "location1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern using fnmatch.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.
    """
    # Summary
    # This heuristic estimates the number of actions required to move all packages
    # to their respective goal locations. It sums the estimated minimum actions
    # needed for each package independently.

    # Assumptions:
    # - Roads are bidirectional (inferred from how the road graph is built).
    # - Package sizes and vehicle capacities are not considered as constraints for loading/carrying.
    # - Any vehicle can transport any package.
    # - Vehicles are always available where needed to pick up or drop off packages (this is a key relaxation).
    # - The graph of locations connected by roads is static.
    # - The goal is solely defined by (at package location) facts for specific packages.
    # - The heuristic value is finite for solvable states (i.e., all goal locations are reachable from relevant initial locations).

    # Heuristic Initialization
    # - Extracts the goal location for each package from the task's goal conditions.
    # - Builds a graph representing the road network based on static 'road' facts.
    # - Collects all unique locations mentioned in road facts and goal 'at' facts.
    # - Computes the shortest path distance between all pairs of these collected locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    # 1. Check if the current state is the goal state using the task's goal conditions. If yes, return 0.
    # 2. Initialize the total heuristic cost to 0.
    # 3. Find the current location of all objects (packages and vehicles) and which packages are inside which vehicles by iterating through the current state's facts.
    # 4. For each package that has a specified goal location in the task:
    #    a. Check if the package is already at its goal location in the current state (i.e., the fact '(at package goal_location)' is present). If yes, skip this package as it requires no further actions.
    #    b. If the package is currently inside a vehicle (i.e., the fact '(in package vehicle)' is present for some vehicle):
    #       i. Find the current location of that vehicle (i.e., the fact '(at vehicle vehicle_location)' is present).
    #       ii. Estimate the cost for this package as the shortest distance from the vehicle's current location to the package's goal location (representing vehicle drive actions), plus 1 action for unloading the package.
    #       iii. Add this estimated cost to the total heuristic cost.
    #    c. If the package is currently on the ground (i.e., the fact '(at package package_location)' is present for some location, and it's not inside a vehicle):
    #       i. Find the package's current location.
    #       ii. Estimate the cost for this package as 1 action for loading it into a vehicle, plus the shortest distance from the package's current location to its goal location (representing vehicle drive actions), plus 1 action for unloading it.
    #       iii. Add this estimated cost to the total heuristic cost.
    #    d. If any required location is unreachable (distance is infinity as determined by BFS), return infinity for the heuristic value, indicating an likely unsolvable state or path.
    # 5. Return the total accumulated cost, which represents the sum of independent minimum action estimates for all packages not yet at their goals.

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the road
        network graph, and computing all-pairs shortest paths.
        """
        self.goals = task.goals  # The set of facts that must hold in goal states.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract goal locations for packages. Assuming goals are only (at package location).
        self.goal_locations = {}
        all_locations = set() # Collect all locations mentioned

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at obj loc) fact
                obj, location = args
                # Assuming any 'at' goal refers to an object (likely a package) that needs moving.
                self.goal_locations[obj] = location
                all_locations.add(location)
            # Ignoring other potential goal types for this heuristic

        # Build the road graph and collect locations from road facts
        self.road_graph = defaultdict(list)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                self.road_graph[loc2].append(loc1) # Assuming roads are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Note: Locations from initial state 'at' facts are not explicitly added here,
        # but BFS from all collected locations should cover the connected components.
        # For robustness, one might parse initial state facts as well.

        # Compute all-pairs shortest paths using BFS
        self.distance = {}
        all_locations_list = list(all_locations) # Get a list of all unique locations
        for start_loc in all_locations_list:
            self.distance[start_loc] = self._bfs(start_loc, all_locations)

    def _bfs(self, start_node, all_nodes):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in all_nodes}
        if start_node not in distances:
             # Start node is not in the set of known locations (e.g., isolated)
             # Cannot reach any other node.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has outgoing roads in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is one of the known locations to avoid errors
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.
        """
        state = node.state

        # If the state is a goal state, heuristic is 0.
        if self.goals <= state:
             return 0

        total_cost = 0

        # Find current locations of all objects (packages and vehicles)
        current_locations = {} # Map object -> location (if 'at')
        packages_in_vehicles = {} # Map package -> vehicle (if 'in')

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                current_locations[obj] = loc
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                packages_in_vehicles[package] = vehicle
            # Ignoring other predicates like 'capacity' for the heuristic calculation

        # Iterate through packages that have a goal location defined in the task
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at its goal location in the current state
            if f'(at {package} {goal_location})' in state:
                continue # This package is already at its goal, no cost added for it

            # Package is not at its goal location. Estimate cost to get it there.
            if package in packages_in_vehicles:
                # Package is inside a vehicle
                vehicle = packages_in_vehicles[package]
                if vehicle in current_locations:
                    vehicle_location = current_locations[vehicle]
                    # Cost: Drive vehicle from its current location to package's goal + Unload
                    # Get distance from vehicle's current location to package's goal location
                    # Use .get() with default {} to handle cases where vehicle_location might not be in self.distance
                    drive_cost = self.distance.get(vehicle_location, {}).get(goal_location, float('inf'))

                    if drive_cost == float('inf'):
                         # Goal location is unreachable from vehicle's current location
                         # This state is likely not on a path to the goal
                         return float('inf') # Indicate unsolvable or very high cost

                    total_cost += drive_cost + 1 # drive actions + unload action

                else:
                     # Vehicle location is unknown - state inconsistency?
                     # Treat as unreachable
                     return float('inf') # Indicate problem

            elif package in current_locations:
                # Package is on the ground
                package_location = current_locations[package]
                # Cost: Load + Drive from package's location to goal + Unload
                # Get distance from package's current location to package's goal location
                # Use .get() with default {} to handle cases where package_location might not be in self.distance
                drive_cost = self.distance.get(package_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal location is unreachable from package's current location
                     # This state is likely not on a path to the goal
                     return float('inf') # Indicate unsolvable or very high cost

                total_cost += 1 + drive_cost + 1 # load action + drive actions + unload action
            else:
                # Package location is unknown (not 'at' a location and not 'in' a vehicle)
                # State inconsistency?
                return float('inf') # Indicate problem

        # The total_cost is the sum of estimated minimum actions for each package
        # that is not yet at its goal location. If we reached here, the state is
        # not the goal state (checked at the beginning), so total_cost should be > 0
        # unless there are no packages to move according to goal_locations, which
        # would mean the heuristic is 0 for non-goal states, violating requirement 2.
        # However, given the domain, the goal is typically moving packages.
        # If total_cost is 0 here, it implies all packages in self.goal_locations are at their goal.
        # If self.goals <= state was false, it implies there are other goal conditions.
        # But based on examples, only (at package location) goals exist.
        # So, if total_cost is 0, it implies self.goals <= state should have been true.
        # The initial check `if self.goals <= state: return 0` is the definitive one.
        # If we pass that, total_cost will be > 0 for any state where packages
        # in goal_locations are not yet at their destinations.
        return total_cost
