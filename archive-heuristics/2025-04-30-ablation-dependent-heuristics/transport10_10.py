import collections

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding parentheses and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)


class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing the
        estimated costs for each package that is not yet at its goal location.
        For a package not at its goal, the estimated cost includes:
        1. The cost to pick up the package (1 action, if not already in a vehicle).
        2. The cost for a vehicle to travel from the package's current location
           to its goal location (estimated by the shortest path distance in the
           road network).
        3. The cost to drop the package at the goal location (1 action).
        The heuristic ignores vehicle capacity constraints and assumes any package
        can be transported by any vehicle.

    Assumptions:
        - The road network defined by `(road l1 l2)` facts is static.
        - Vehicle capacity constraints are ignored.
        - Any vehicle can carry any package.
        - The goal is always a conjunction of `(at package location)` facts.
        - Objects starting with 'p' are packages, and objects starting with 'v'
          are vehicles. This is inferred from common PDDL naming conventions
          and example instances, as the Task object doesn't provide type information.

    Heuristic Initialization:
        The constructor `__init__` precomputes the following from the static facts:
        - The road network graph: An adjacency list representation where nodes are
          locations and edges are roads.
        - All-pairs shortest paths between locations using BFS from each location.
          This is stored in a dictionary mapping (location1, location2) pairs to
          their shortest distance (number of drive actions).
        - The goal locations for each package by parsing the `task.goals`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h_value` to 0.
        2. Create efficient lookups for relevant facts in the current `state`:
           `at_facts` (object -> location), `in_facts` (package -> vehicle),
           `vehicle_locations` (vehicle -> location). Iterate through the state
           facts to populate these lookups.
        3. Iterate through each package `p` that has a goal location `L_goal`
           identified during initialization (from `self.package_goals`).
        4. For the current package `p`, determine its current status and location:
           - If `p` is a key in `at_facts`, its current location `L_current` is
             `at_facts[p]`, and it is not in a vehicle (`is_in_vehicle = False`).
           - If `p` is a key in `in_facts`, it is in vehicle `v = in_facts[p]`.
             Find the location of vehicle `v` from `vehicle_locations[v]`. This
             is the package's current location `L_current`, and it is in a vehicle
             (`is_in_vehicle = True`).
           - If the package's status cannot be determined (e.g., not `at` any location
             and not `in` any vehicle, or in a vehicle whose location is unknown),
             return `float('inf')` as the state is likely invalid or unreachable.
        5. If the package's current location `L_current` is the same as its goal
           location `L_goal`, the package contributes 0 to the heuristic.
        6. If `L_current` is different from `L_goal`:
           - Get the shortest path distance `travel_cost` between `L_current` and
             `L_goal` from the precomputed `self.shortest_paths`. If there is no
             path, return `float('inf')`.
           - If the package is `is_in_vehicle`, it needs to travel and be dropped.
             Add `travel_cost + 1` (for the drop action) to `h_value`.
           - If the package is not `is_in_vehicle` (it's `at` a location), it needs
             to be picked up, travel, and be dropped. Add `1 + travel_cost + 1`
             (for pickup, travel, and drop actions) to `h_value`.
        7. After processing all packages with goals, return the final `h_value`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.road_graph = collections.defaultdict(list)
        self.package_goals = {}
        self.shortest_paths = {} # Stores (loc1, loc2) -> distance
        self.locations = set() # Keep track of all locations

        # 1. Build the road graph from static facts
        for fact_string in task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'road':
                l1, l2 = fact[1], fact[2]
                self.road_graph[l1].append(l2)
                self.locations.add(l1)
                self.locations.add(l2)

        # 2. Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self._bfs(start_loc)

        # 3. Extract package goal locations
        for goal_fact_string in task.goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact[0] == 'at':
                package, location = goal_fact[1], goal_fact[2]
                self.package_goals[package] = location

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest paths to all other nodes.
        Stores results in self.shortest_paths.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in the graph keys before iterating
            # This handles cases where a location exists but has no outgoing roads defined
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        # Store the computed distances
        for end_node, dist in distances.items():
            if dist != float('inf'): # Only store reachable pairs
                 self.shortest_paths[(start_node, end_node)] = dist

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach the goal.
        """
        h_value = 0

        # Create efficient lookups for current state facts
        at_facts = {} # object -> location
        in_facts = {} # package -> vehicle
        vehicle_locations = {} # vehicle -> location

        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at':
                obj, loc = fact[1], fact[2]
                at_facts[obj] = loc
                # Assume objects starting with 'v' are vehicles
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif fact[0] == 'in':
                pkg, veh = fact[1], fact[2]
                in_facts[pkg] = veh

        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.package_goals.items():
            current_location = None
            is_in_vehicle = False

            # Find package's current status and location
            if package in at_facts:
                current_location = at_facts[package]
                is_in_vehicle = False
            elif package in in_facts:
                vehicle = in_facts[package]
                if vehicle in vehicle_locations:
                    current_location = vehicle_locations[vehicle]
                    is_in_vehicle = True
                else:
                    # Package is in a vehicle, but vehicle location is unknown in state
                    # This implies an invalid state representation or an unreachable state
                    # Return infinity
                    return float('inf')
            else:
                 # Package is not at any location and not in any vehicle
                 # This implies an invalid state representation or an unreachable state
                 # Return infinity
                 return float('inf')


            # Calculate cost for this package if it's not at the goal
            if current_location != goal_location:
                # Package needs to move
                travel_cost = self.shortest_paths.get((current_location, goal_location), float('inf'))

                if travel_cost == float('inf'):
                    # Goal location is unreachable from current location
                    return float('inf') # Problem is unsolvable from this state

                if is_in_vehicle:
                    # Package is in a vehicle, needs travel and drop
                    # Cost = drive actions + drop action
                    h_value += travel_cost + 1
                else:
                    # Package is at a location, needs pickup, travel, and drop
                    # Cost = pickup action + drive actions + drop action
                    h_value += 1 + travel_cost + 1 # pickup + travel + drop

        return h_value
