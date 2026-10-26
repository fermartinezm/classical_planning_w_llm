import collections
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the estimated costs for each package that is not yet at its goal location.
        For a package not at its goal, the estimated cost is calculated based on
        whether the package is currently at a location or inside a vehicle,
        plus the shortest path distance the package (or its vehicle) needs to travel.
        Capacity constraints are not explicitly modeled.

    Assumptions:
        - The goal state is defined by a set of (at ?p ?l) facts for packages.
        - Road network is static and defined by (road ?l1 ?l2) facts.
        - If (road l1 l2) exists, (road l2 l1) also exists (undirected graph).
        - All locations mentioned in the problem are connected or relevant to package movement.
        - The state representation is a frozenset of strings.
        - The static information is a frozenset of strings.
        - The Task object provides goal facts and static facts.

    Heuristic Initialization:
        - Parses the goal facts to determine the target location for each package.
        - Parses the static facts to build the road network graph.
        - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS).
        - Stores the goal locations and shortest path distances for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Build quick lookup dictionaries for the current state: `at_locations` (mapping object to location) and `in_vehicle` (mapping package to vehicle). This is done by iterating through the state facts once.
        3. Iterate through each package `p` that has a specified goal location `l_goal(p)` in the task's goal state (stored in `self.goal_locations`).
        4. Check if the package `p` is currently at its goal location `l_goal(p)` using the `at_locations` dictionary.
        5. If `p` is at `l_goal(p)`, this package contributes 0 to the heuristic. Continue to the next package.
        6. If `p` is NOT at `l_goal(p)`, determine its current status:
            a. If `p` is in `at_locations`, it is at location `l_current = at_locations[p]`. The estimated cost for this package is:
               - 2 actions (one pick-up, one drop)
               - Plus the shortest path distance from `l_current` to `l_goal(p)` for the vehicle carrying it.
               - Add `2 + shortest_path(l_current, l_goal(p))` to `h`. If the goal location is unreachable from `l_current`, return `math.inf`.
            b. If `p` is not in `at_locations` but is in `in_vehicle`, it is inside vehicle `v = in_vehicle[p]`. Find the location of vehicle `v` using the `at_locations` dictionary: `l_v = at_locations[v]`. The estimated cost for this package is:
               - 1 action (one drop)
               - Plus the shortest path distance from `l_v` to `l_goal(p)` for the vehicle.
               - Add `1 + shortest_path(l_v, l_goal(p))` to `h`. If the vehicle is not at a known location or the goal location is unreachable from `l_v`, return `math.inf`.
            c. If `p` is neither in `at_locations` nor `in_vehicle`, this indicates an unexpected state; return `math.inf`.
        7. After iterating through all packages with goal locations, the total value `h` is the heuristic estimate for the current state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.goal_locations = self._parse_goal_locations(task.goals)
        self.road_graph, self.locations = self._build_road_graph(task.static)
        self.shortest_paths = self._compute_shortest_paths(self.road_graph, self.locations)

    def _parse_goal_locations(self, goals):
        """
        Parses the goal facts to find the target location for each package.

        Args:
            goals: A frozenset of goal facts (strings).

        Returns:
            A dictionary mapping package names to their goal location names.
        """
        goal_locs = {}
        for fact in goals:
            # Example fact: '(at p1 l2)'
            parts = fact.strip('()').split()
            if len(parts) == 3 and parts[0] == 'at':
                obj_name = parts[1]
                loc_name = parts[2]
                # Assuming 'at' facts in goal only refer to packages
                goal_locs[obj_name] = loc_name
        return goal_locs

    def _build_road_graph(self, static_facts):
        """
        Builds an adjacency list representation of the road network.

        Args:
            static_facts: A frozenset of static facts (strings).

        Returns:
            A tuple: (road_graph_dict, set_of_locations).
            road_graph_dict: A dictionary mapping location names to a list of connected location names.
            set_of_locations: A set of all unique location names.
        """
        road_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            # Example fact: '(road l1 l2)'
            parts = fact.strip('()').split()
            if len(parts) == 3 and parts[0] == 'road':
                loc1 = parts[1]
                loc2 = parts[2]
                road_graph[loc1].append(loc2)
                road_graph[loc2].append(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)
        return road_graph, locations

    def _compute_shortest_paths(self, graph, locations):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.

        Args:
            graph: The road network graph (adjacency list).
            locations: A set of all unique location names.

        Returns:
            A dictionary of dictionaries, where shortest_paths[start][end]
            is the distance from start to end. Returns math.inf if unreachable.
        """
        shortest_paths = {}
        for start_node in locations:
            distances = {loc: math.inf for loc in locations}
            distances[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                current_node = queue.popleft()

                # Check if current_node exists as a key in the graph
                # This handles cases where a location might exist but have no roads connected
                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if distances[neighbor] == math.inf:
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
            shortest_paths[start_node] = distances
        return shortest_paths


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            The estimated number of actions to reach a goal state.
        """
        h = 0

        # Build quick lookup dictionaries for current state facts
        at_locations = {} # {object_name: location_name}
        in_vehicle = {} # {package_name: vehicle_name}
        # Iterate through the state facts ONCE
        for fact_str in state:
             parts = fact_str.strip('()').split()
             if len(parts) == 3:
                 predicate = parts[0]
                 obj1 = parts[1]
                 obj2 = parts[2]
                 if predicate == 'at':
                     at_locations[obj1] = obj2
                 elif predicate == 'in':
                     in_vehicle[obj1] = obj2

        # Iterate through packages that have a goal location
        for package, goal_loc in self.goal_locations.items():
            # Check if package is already at its goal location
            if package in at_locations and at_locations[package] == goal_loc:
                 continue # Package is at goal, contributes 0

            # Package is not at goal, calculate its contribution
            package_h = 0
            current_loc = None

            if package in at_locations:
                # Package is at a location l_current
                current_loc = at_locations[package]
                # Cost: pick-up (1) + drop (1) + drive
                package_h += 2
                # Add shortest path distance
                if current_loc in self.shortest_paths and goal_loc in self.shortest_paths[current_loc]:
                     dist = self.shortest_paths[current_loc][goal_loc]
                     if dist == math.inf:
                         # Goal location unreachable from current location
                         return math.inf
                     package_h += dist
                else:
                     # Either current_loc or goal_loc is not in the precomputed paths (shouldn't happen if locations are parsed correctly)
                     # Or goal is unreachable
                     return math.inf

            elif package in in_vehicle:
                # Package is inside a vehicle v
                vehicle_carrying = in_vehicle[package]
                if vehicle_carrying in at_locations:
                    # Vehicle is at location l_v
                    current_loc = at_locations[vehicle_carrying]
                    # Cost: drop (1) + drive
                    package_h += 1
                    # Add shortest path distance
                    if current_loc in self.shortest_paths and goal_loc in self.shortest_paths[current_loc]:
                         dist = self.shortest_paths[current_loc][goal_loc]
                         if dist == math.inf:
                             # Goal location unreachable from vehicle's location
                             return math.inf
                         package_h += dist
                    else:
                         # Either current_loc or goal_loc is not in the precomputed paths
                         # Or goal is unreachable
                         return math.inf
                else:
                    # Vehicle carrying package is not at any location (invalid state?)
                    # Treat as unreachable goal.
                    return math.inf

            else:
                 # Package is neither 'at' nor 'in' (invalid state?)
                 # Treat as unreachable goal.
                 return math.inf

            h += package_h

        return h
