import collections
from heuristics.heuristic_base import Heuristic
from task import Task

class transportHeuristic(Heuristic):
    """
    Summary:
        This heuristic estimates the cost to reach the goal state in the transport domain
        by summing the estimated costs for each package to reach its individual goal location.
        It is a domain-dependent, non-admissible heuristic designed for greedy best-first search.
        The estimated cost for a package is based on its current location (at a location or in a vehicle)
        and the shortest path distance via roads to its goal location, plus fixed costs for pickup and dropoff actions.

    Assumptions:
        - The goal state consists of packages being at specific locations, represented by facts like `(at package_name location_name)`.
        - Objects appearing as the first argument of `(at ...)` facts in the goal are packages.
        - Roads are bidirectional if defined in the static facts (e.g., if `(road l1 l2)` is static, assume `(road l2 l1)` is also traversable, even if not explicitly listed). The precomputation handles this by adding edges in both directions.
        - The heuristic ignores vehicle capacity constraints and the need for a vehicle to travel to a package's current location if the package is currently `at` a location. It assumes a suitable vehicle is available for pickup/dropoff and transport.
        - States where a goal package's location cannot be determined (not `at` a location, not `in` a vehicle, or in a vehicle whose location is unknown) are considered infinitely far from the goal.
        - States where a goal package's target location is unreachable from its current location via the road network are considered infinitely far from the goal.

    Heuristic Initialization:
        1. Stores the task object for easy access to goal state checking.
        2. Parses static facts to build the road network graph (adjacency list) and identify all locations mentioned in road facts. It assumes roads are bidirectional and adds edges in both directions.
        3. Parses goal facts to identify packages and their target locations, storing them in a dictionary `package_goals`. It assumes objects in `(at obj loc)` goal facts are packages unless they are identified as locations.
        4. Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS) starting from each location on the road network graph. Unreachable locations are assigned an infinite distance (`float('inf')`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state using `self.task.goal_reached(state)`. If yes, the heuristic value is 0, and the function returns immediately.
        2. Initialize the total heuristic value `h` to 0.
        3. Create temporary dictionaries to quickly look up the current location of locatables (`at_locations`: maps object name to location name) and which package is in which vehicle (`in_vehicles`: maps package name to vehicle name) by iterating through the facts in the current state.
        4. Iterate through each package and its goal location stored in `self.package_goals`.
        5. For the current package:
            a. Construct the string representation of the goal fact for this package (e.g., `'(at p1 l2)'`).
            b. Check if this goal fact is present in the current state. If yes, this part of the goal is satisfied for this package, so add 0 to `h` for this package and move to the next package.
            c. If the goal fact is not in the state, determine the package's current location and whether it is inside a vehicle.
                i. Check if the package is `at` a location by looking up `package_name` in `at_locations`. If found, `current_l` is the location, and `is_in_vehicle` is False.
                ii. If not found in `at_locations`, check if the package is `in` a vehicle by looking up `package_name` in `in_vehicles`. If found, get the `vehicle_name`. Then, find the location of that vehicle by looking up `vehicle_name` in `at_locations`. If the vehicle's location is found, `current_l` is the package's location, and `is_in_vehicle` is True.
                iii. If the package's location cannot be determined by the above steps (e.g., not `at` or `in`, or in a vehicle whose location is unknown), this indicates an inconsistent state representation. Add `float('inf')` to `h` and move to the next package.
            d. If the current location `current_l` was found:
                i. Look up the shortest path distance `dist` from `current_l` to the `goal_location` using the precomputed `self.distances` dictionary. Handle cases where either `current_l` or `goal_location` were not part of the locations found during initialization (e.g., not connected by roads or not in goals), treating the distance as `float('inf')`.
                ii. If the distance `dist` is `float('inf')` (meaning the goal location is unreachable from the current location via the road network), add `float('inf')` to `h`.
                iii. If the distance is finite:
                    - If the package is currently `in` a vehicle (`is_in_vehicle` is True), the estimated cost for this package is `1 + dist` (1 for the drop action + `dist` for the drive action). Add this value to `h`.
                    - If the package is currently `at` a location (`is_in_vehicle` is False), the estimated cost for this package is `2 + dist` (1 for the pick-up action + `dist` for the drive action + 1 for the drop action). Add this value to `h`.
        6. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task # Store task for goal_reached check
        self._parse_static_info(task.static)
        self._parse_goals(task.goals)
        self._compute_shortest_paths()

    def _parse_fact_string(self, fact_str):
        """Helper to parse a PDDL fact string into (predicate, [objects])."""
        # Remove leading/trailing parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts:
            return None
        predicate = parts[0]
        objects = parts[1:]
        return (predicate, objects)

    def _parse_static_info(self, static_facts):
        """Builds road graph and collects locations from static facts."""
        self.road_graph = collections.defaultdict(list)
        self.location_objects = set()
        # capacity_predecessors is not used in this simple heuristic

        for fact_str in static_facts:
            parsed = self._parse_fact_string(fact_str)
            if parsed is None:
                continue

            predicate, objects = parsed

            if predicate == 'road' and len(objects) == 2:
                l1, l2 = objects
                self.location_objects.add(l1)
                self.location_objects.add(l2)
                self.road_graph[l1].append(l2)
                # Assuming roads are bidirectional based on example
                self.road_graph[l2].append(l1)
            # elif predicate == 'capacity-predecessor' and len(objects) == 2:
            #     s1, s2 = objects
            #     self.capacity_predecessors[s1] = s2

    def _parse_goals(self, goal_facts):
        """Extracts package goal locations from goal facts."""
        self.package_goals = {}
        # Assume objects in (at obj loc) goal facts are packages unless they are locations
        for fact_str in goal_facts:
            parsed = self._parse_fact_string(fact_str)
            if parsed is None:
                continue
            predicate, objects = parsed
            if predicate == 'at' and len(objects) == 2:
                obj, loc = objects
                # Basic check: don't treat locations as packages if they appear in goals
                if obj not in self.location_objects:
                     self.package_goals[obj] = loc
                # Note: This might incorrectly identify vehicles as packages if vehicles
                # also have goal locations, but the example problems don't show this.

    def _compute_shortest_paths(self):
        """Computes all-pairs shortest paths between locations using BFS."""
        self.distances = {}
        # Use a list of locations to ensure consistent iteration order
        all_nodes = list(self.location_objects)

        for start_node in all_nodes:
            self.distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            self.distances[start_node][start_node] = 0

            while queue:
                current_node, current_dist = queue.popleft()

                # Ensure current_node is in the graph keys (a location might exist
                # without any roads connected, but still be in location_objects)
                if current_node not in self.road_graph:
                     continue

                for neighbor in self.road_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append((neighbor, current_dist + 1))

        # Ensure all pairs of locations have a distance entry (finite or infinite)
        for l1 in all_nodes:
            if l1 not in self.distances:
                self.distances[l1] = {}
            for l2 in all_nodes:
                 if l2 not in self.distances[l1]:
                     self.distances[l1][l2] = float('inf') # Mark unreachable
                 # Also ensure symmetric infinite distance if l2 -> l1 wasn't found
                 # (though BFS from l2 would find it if a path exists)
                 if l2 in self.distances and l1 not in self.distances[l2]:
                      self.distances[l2][l1] = float('inf')


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates the remaining cost by summing estimated costs for each misplaced package.
        """
        state = node.state

        # Return 0 if the goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0

        # Build temporary state lookups for efficient access
        at_locations = {} # obj -> location (for locatables: packages and vehicles)
        in_vehicles = {}  # package -> vehicle

        for fact_str in state:
            parsed = self._parse_fact_string(fact_str)
            if parsed is None:
                continue
            predicate, objects = parsed
            if predicate == 'at' and len(objects) == 2:
                obj, loc = objects
                at_locations[obj] = loc
            elif predicate == 'in' and len(objects) == 2:
                pkg, veh = objects
                in_vehicles[pkg] = veh

        # Calculate heuristic for each package that has a goal location
        for package_name, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location
            goal_fact_str = f'(at {package_name} {goal_location})'
            if goal_fact_str in state:
                 continue # This package goal is satisfied

            current_l = None
            is_in_vehicle = False

            # Find current location of the package
            if package_name in at_locations:
                current_l = at_locations[package_name]
                is_in_vehicle = False
            elif package_name in in_vehicles:
                vehicle_name = in_vehicles[package_name]
                if vehicle_name in at_locations:
                    current_l = at_locations[vehicle_name]
                    is_in_vehicle = True
                else:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # Inconsistent state.
                    h += float('inf')
                    continue
            else:
                # Package location is unknown (not at a location, not in a vehicle).
                # Inconsistent state.
                h += float('inf')
                continue

            # Calculate distance from current location to goal location
            # Ensure both current_l and goal_location are known locations in our graph
            if current_l not in self.distances or goal_location not in self.distances.get(current_l, {}):
                 # One of the locations wasn't in the original road network/goals
                 dist = float('inf')
            else:
                 dist = self.distances[current_l][goal_location]

            if dist == float('inf'):
                # Goal location is unreachable from current location
                h += float('inf')
            else:
                # Add estimated cost based on current status and distance
                if is_in_vehicle:
                    # Already in vehicle: 1 (drop) + dist (drive)
                    h += 1 + dist
                else:
                    # At location: 1 (pickup) + dist (drive) + 1 (drop)
                    h += 2 + dist

        return h
