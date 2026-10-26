from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, though valid PDDL facts are expected.
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    package independently, ignoring vehicle capacity and potential synergies
    from moving multiple packages together.

    # Assumptions
    - All actions (drive, pick-up, drop) have a cost of 1.
    - Vehicle capacity constraints are ignored.
    - A suitable vehicle is assumed to be available whenever and wherever
      a package needs to be picked up or dropped off.
    - The road network is static and provides connections between locations.
    - Shortest path distances in the road network represent the minimum
      number of drive actions needed between two locations.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Identifies all locations and builds the road network graph based on
      static `(road ?l1 ?l2)` facts.
    - Computes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).
    - Identifies vehicles based on their appearance in relevant initial
      state or static facts (e.g., `capacity`, `in`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Identify Current Positions:** Create a mapping of each locatable object
        (packages and vehicles) to its current position. A package's position
        is either the location it is `(at ?p ?l)` or the vehicle it is `(in ?p ?v)`.
        A vehicle's position is the location it is `(at ?v ?l)`.

    2.  **Iterate Through Packages:** For each package that has a specified
        goal location in the task:

        a.  **Get Goal:** Retrieve the package's goal location.

        b.  **Determine Current Physical Location:**
            -   If the package's current position is a vehicle `v`, find the
                physical location of vehicle `v`. This is the package's current
                physical location.
            -   If the package's current position is a location `l`, this is
                the package's current physical location.

        c.  **Calculate Package Cost:**
            -   If the package is currently `(at ?p ?l_current)` at its goal
                location `l_goal` (`l_current == l_goal`), the cost for this
                package is 0.
            -   If the package is currently `(at ?p ?l_current)` at a location
                `l_current` that is not its goal `l_goal`:
                The estimated cost is 1 (pick-up) + `dist(l_current, l_goal)` (drive) + 1 (drop).
                `dist(l_current, l_goal)` is the shortest path distance in the road network.
            -   If the package is currently `(in ?p ?v)` inside a vehicle `v`,
                and the vehicle is `(at ?v ?l_current)` at a location `l_current`:
                -   If `l_current` is the goal location `l_goal`:
                    The estimated cost is 1 (drop).
                -   If `l_current` is not the goal location `l_goal`:
                    The estimated cost is `dist(l_current, l_goal)` (drive) + 1 (drop).

        d.  **Handle Unreachable Goals:** If the goal location is not reachable
            from the package's current physical location (or the vehicle's
            current location if the package is inside a vehicle) via the road
            network, the heuristic returns infinity to indicate an unsolvable
            state from this point.

    3.  **Sum Costs:** The total heuristic value is the sum of the estimated
        costs for all packages with goal locations. This sum is 0 if and only
        if all packages are currently `(at ?p ?l_goal)` at their respective
        goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        packages_in_goals = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at' and len(parts) > 2:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                packages_in_goals.add(package)

        # 2. Identify locations and build road graph.
        locations = set()
        road_facts = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'road' and len(parts) > 2:
                road_facts.append(fact)
                locations.add(parts[1])
                locations.add(parts[2])

        self.road_graph = {loc: set() for loc in locations}
        for fact in road_facts:
            parts = get_parts(fact)
            if len(parts) > 2:
                l1, l2 = parts[1], parts[2]
                # Only add roads between identified locations
                if l1 in self.road_graph and l2 in self.road_graph:
                    self.road_graph[l1].add(l2)
                    self.road_graph[l2].add(l1) # Assuming roads are bidirectional

        # 3. Compute all-pairs shortest paths using BFS.
        self.distance_map = {}
        for start_node in self.road_graph:
            distances = {node: float('inf') for node in self.road_graph}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # Check if current_node is a valid key in the graph
                if current_node not in self.road_graph:
                    continue # Should not happen if locations set is built correctly

                for neighbor in self.road_graph.get(current_node, []): # Use .get for safety
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

            for end_node, dist in distances.items():
                if dist != float('inf'):
                    self.distance_map[(start_node, end_node)] = dist

        # 4. Identify vehicles.
        # Collect all objects mentioned in initial state and static facts.
        all_objects = set()
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts:
                 all_objects.update(parts[1:])

        # Infer vehicles: objects that are not locations and appear in vehicle-specific predicates
        self.vehicles = set()
        for obj in all_objects:
            if obj not in locations and obj not in packages_in_goals:
                 # Check if it appears as a vehicle in initial state or static facts
                 is_vehicle = False
                 for fact in initial_state | static_facts:
                      parts = get_parts(fact)
                      if parts:
                          if (parts[0] == 'capacity' and len(parts) > 1 and parts[1] == obj) or \
                             (parts[0] == 'in' and len(parts) > 2 and parts[2] == obj):
                              is_vehicle = True
                              break
                 if is_vehicle:
                      self.vehicles.add(obj)

        # Store locations set for quick lookup in __call__
        self.locations = locations


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state.

        # 1. Identify Current Positions of packages and vehicles.
        current_pos_map = {} # obj -> location or vehicle
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'at' and len(parts) > 2:
                    obj, loc = parts[1], parts[2]
                    current_pos_map[obj] = loc
                elif parts[0] == 'in' and len(parts) > 2:
                    pkg, veh = parts[1], parts[2]
                    current_pos_map[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # 2. Iterate through packages that have a goal location.
        for package_name, goal_location_name in self.goal_locations.items():
            # If package is not in the current state's position map, it's an issue.
            # Assuming valid states always contain position info for relevant objects.
            if package_name not in current_pos_map:
                 # This package's state is unknown. Cannot compute heuristic.
                 # Return infinity as it might be an unreachable state.
                 return float('inf')

            current_pos = current_pos_map[package_name]

            # 3. Determine package's physical location and calculate cost.
            package_cost = 0

            if current_pos in self.vehicles: # Package is in a vehicle
                vehicle_name = current_pos
                # Find vehicle's location
                if vehicle_name not in current_pos_map:
                     # Vehicle location unknown. Cannot compute heuristic.
                     return float('inf') # Unreachable

                current_location_name = current_pos_map[vehicle_name]

                # Check if vehicle's current location is a known location
                if current_location_name not in self.locations:
                     # Vehicle is at an unknown location. Cannot compute heuristic.
                     return float('inf') # Unreachable

                # Package is in vehicle at current_location_name
                if current_location_name == goal_location_name:
                    # Needs 1 drop action to reach final (at) goal state
                    package_cost = 1
                else:
                    # Needs drive + drop
                    # Check if path exists from vehicle's location to goal.
                    if (current_location_name, goal_location_name) not in self.distance_map:
                         # Goal location unreachable from current location.
                         return float('inf') # Unreachable

                    drive_cost = self.distance_map[(current_location_name, goal_location_name)]
                    package_cost = drive_cost + 1 # drive + drop

            elif current_pos in self.locations: # Package is at a location
                current_location_name = current_pos

                if current_location_name != goal_location_name:
                    # Needs pick + drive + drop
                    # Check if path exists from current location to goal.
                    if (current_location_name, goal_location_name) not in self.distance_map:
                         # Goal location unreachable from current location.
                         return float('inf') # Unreachable

                    drive_cost = self.distance_map[(current_location_name, goal_location_name)]
                    package_cost = 1 + drive_cost + 1 # pick + drive + drop
                else:
                    # Package is already at goal location (and not in a vehicle). Cost is 0.
                    package_cost = 0

            else:
                # Package's current_pos is neither a known vehicle nor a known location.
                # Invalid state representation?
                return float('inf') # Unreachable

            total_cost += package_cost

        # 4. Return the total estimated cost.
        return total_cost
