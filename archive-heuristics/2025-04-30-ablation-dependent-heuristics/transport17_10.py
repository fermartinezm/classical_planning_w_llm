from collections import deque

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the remaining cost to reach the goal by summing
    the estimated minimum number of actions required for each package that
    is not yet at its final destination. The estimation for a single package
    involves the cost of picking it up (if needed), driving it to its goal
    location, and dropping it off. Driving cost is estimated by the shortest
    path distance in the location graph.

    Assumptions:
    - Roads are bidirectional.
    - Vehicle capacity constraints are ignored for the heuristic estimate.
      It is assumed that a suitable vehicle is available or can be moved
      to the package's location and has sufficient capacity for the pick-up
      and transport.
    - The cost of moving a package is estimated as:
        - If package is at location L (and L is not the goal): 1 (pick-up) + distance(L, GoalL) (drive) + 1 (drop)
        - If package is in vehicle V at location L: distance(L, GoalL) (drive) + 1 (drop)
    - Distance is the shortest path distance in the location graph, computed
      using BFS. Unreachable locations result in infinite distance.

    Heuristic Initialization:
    The constructor performs precomputation based on the static information
    from the planning task:
    1. It builds an adjacency list representation of the location graph
       based on the `(road ?l1 ?l2)` facts. It assumes roads are bidirectional.
       It also includes any locations mentioned in the initial state or goals
       even if they have no roads, treating them as potentially isolated nodes.
    2. It computes the shortest path distance between all pairs of locations
       in the graph using Breadth-First Search (BFS) starting from each location.
       These distances are stored in a dictionary.
    3. It extracts the goal location for each package from the task's goal
       facts `(at ?p ?l)` and stores them in a dictionary mapping package
       names to goal location names.

    Step-By-Step Thinking for Computing Heuristic:
    The `__call__` method computes the heuristic value for a given state:
    1. Check if the state is a goal state using `self.task.goal_reached(state)`.
       If it is, the heuristic value is 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Create temporary dictionaries (`at_loc`, `in_veh`) to store the current
       location of each object (`at_loc[obj] = loc`) and which package is in
       which vehicle (`in_veh[pkg] = veh`). This is done by iterating through
       the state facts once.
    4. Iterate through each package `p` for which a goal location `goal_l`
       was identified during initialization (`self.package_goals`).
    5. For the current package `p`, check if the fact `(at p goal_l)` is present
       in the current state. If it is, the package has reached its goal and
       contributes 0 to the heuristic for this package. Continue to the next package.
    6. If the package `p` is not at its goal location, determine its current status:
       a. If `p` is found as a key in the `at_loc` dictionary, it means the package
          is currently at location `current_l = at_loc[p]`.
          - The estimated cost for this package is calculated as:
            1 (for the pick-up action) + `self.distances[current_l][goal_l]` (for the drive actions) + 1 (for the drop action).
          - If `self.distances[current_l][goal_l]` is `float('inf')`, it means the goal location is unreachable from the package's current location, and the heuristic returns `float('inf')` for the entire state, indicating unsolvability.
          - Add this calculated cost to the total heuristic `h`.
       b. If `p` is found as a key in the `in_veh` dictionary, it means the package
          is currently inside vehicle `v = in_veh[p]`.
          - Find the current location of vehicle `v` from the `at_loc` dictionary: `current_l = at_loc[v]`.
          - The estimated cost for this package is calculated as:
            `self.distances[current_l][goal_l]` (for the drive actions) + 1 (for the drop action).
          - If `self.distances[current_l][goal_l]` is `float('inf')`, the heuristic returns `float('inf')`.
          - Add this calculated cost to the total heuristic `h`.
       c. If the package `p` is not found in either the `at_loc` or `in_veh` dictionaries,
          it represents an unexpected or potentially invalid state. The heuristic
          returns `float('inf')` in this case, treating the state as effectively
          unsolvable or unreachable from a valid state.
    7. After processing all goal packages, the accumulated value in `h` is returned
       as the heuristic estimate for the state.
    """
    def __init__(self, task):
        self.task = task
        self.location_graph = {}
        self.distances = {}
        self.package_goals = {}

        # 1. Build location graph from road facts and identify all locations
        locations = set()
        for fact_str in task.static:
            if fact_str.startswith('(road '):
                _, l1, l2 = self._parse_fact(fact_str)
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                if l2 not in self.location_graph:
                    self.location_graph[l2] = []
                # Assuming roads are bidirectional based on example PDDL
                self.location_graph[l1].append(l2)
                self.location_graph[l2].append(l1)

        # Add any locations mentioned in initial state or goals but not in road facts
        # This handles cases with isolated locations
        for fact_str in task.initial_state:
             if fact_str.startswith('(at '):
                 _, obj, loc = self._parse_fact(fact_str)
                 locations.add(loc)
        for fact_str in task.goals:
             if fact_str.startswith('(at '):
                 _, obj, loc = self._parse_fact(fact_str)
                 locations.add(loc)

        # Ensure all identified locations are keys in the graph dictionary,
        # even if they have no neighbors (isolated nodes).
        for loc in locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []


        # 2. Compute all-pairs shortest paths using BFS
        for start_node in self.location_graph:
            self.distances[start_node] = self._bfs(self.location_graph, start_node)

        # 3. Extract package goals
        for fact_str in task.goals:
            if fact_str.startswith('(at '):
                _, package, goal_loc = self._parse_fact(fact_str)
                self.package_goals[package] = goal_loc

    def _parse_fact(self, fact_string):
        """Helper to parse PDDL fact string."""
        # Remove surrounding parentheses and split by space
        # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
        parts = fact_string.strip("()").split()
        return tuple(parts)

    def _bfs(self, graph, start_node):
        """Helper to perform BFS for shortest paths."""
        # Initialize distances for all nodes in the graph
        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in graph keys before iterating neighbors
            # This handles isolated nodes correctly where graph[current_node] might not exist
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # If neighbor hasn't been visited yet
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, state):
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0
        at_loc = {}
        in_veh = {}

        # Pre-parse state for quick lookups of object locations and package contents
        for fact_str in state:
            parsed = self._parse_fact(fact_str)
            if parsed[0] == 'at':
                # Fact is (at obj loc)
                _, obj, loc = parsed
                at_loc[obj] = loc
            elif parsed[0] == 'in':
                # Fact is (in pkg veh)
                _, pkg, veh = parsed
                in_veh[pkg] = veh

        # Calculate cost for each package that is not at its goal location
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at its goal location
            if package in at_loc and at_loc[package] == goal_loc:
                continue # This package is done, contributes 0

            cost_for_package = 0
            current_loc = None

            if package in at_loc:
                # Package is currently at a location (not the goal location)
                current_loc = at_loc[package]
                # Estimated cost: pick-up (1) + drive from current_loc to goal_loc + drop (1)
                # Need to check if current_loc and goal_loc are in our distance map
                if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                     drive_cost = self.distances[current_loc][goal_loc]
                     if drive_cost == float('inf'):
                         # Goal location is unreachable from the package's current location
                         return float('inf') # State is likely unsolvable
                     cost_for_package = 1 + drive_cost + 1
                else:
                     # This case should ideally not happen if all relevant locations
                     # were included when building the graph, but serves as a safety.
                     return float('inf') # Unreachable location

            elif package in in_veh:
                # Package is currently inside a vehicle
                vehicle = in_veh[package]
                # Find the location of the vehicle
                if vehicle in at_loc:
                    current_loc = at_loc[vehicle]
                    # Estimated cost: drive vehicle from current_loc to goal_loc + drop (1)
                    if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                        drive_cost = self.distances[current_loc][goal_loc]
                        if drive_cost == float('inf'):
                            # Goal location is unreachable from the vehicle's current location
                            return float('inf') # State is likely unsolvable
                        cost_for_package = drive_cost + 1
                    else:
                        # Safety check for unreachable location
                        return float('inf')
                else:
                    # Vehicle location is unknown - indicates an invalid state representation
                    # Treat as unsolvable from this state
                    return float('inf')

            else:
                # Package is not found in 'at' or 'in' facts in the state.
                # This indicates an invalid state representation.
                # Treat as unsolvable from this state.
                return float('inf')

            # Add the estimated cost for this package to the total heuristic value
            h += cost_for_package

        return h
