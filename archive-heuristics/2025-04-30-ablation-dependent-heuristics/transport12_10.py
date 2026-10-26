import re
from collections import deque
import sys

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum actions
        required for each package that is not yet at its goal location.
        The cost for a package depends on whether it is currently at a location
        or inside a vehicle, and the shortest road distance to its goal location.
        It assumes packages can be transported independently and simplifies
        vehicle availability and capacity constraints.

    Assumptions:
        - The road network is static and provides shortest path distances.
        - Packages are either at a location or inside a vehicle in any valid state.
        - The goal only specifies the final location for certain packages using `(at package location)` facts.
        - All locations mentioned in the initial state and goal are part of the
          connected road network relevant to package movement.
        - Facts are represented as strings like '(predicate arg1 arg2)'.
        - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles.

    Heuristic Initialization:
        1. Parses static facts to build the road network graph.
        2. Computes all-pairs shortest path distances between locations using BFS.
        3. (Capacity hierarchy is not used in this simple heuristic but could be parsed here).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state using `task.goal_reached(state)`. If yes, return 0.
        2. Initialize the total heuristic value `h` to 0.
        3. Create temporary dictionaries to store the current location of packages
           (`package_location`), packages inside vehicles (`package_in_vehicle`),
           and vehicle locations (`vehicle_location`) by parsing the facts in the current state.
           Distinguish packages and vehicles based on their object name prefix ('p' vs 'v').
        4. Iterate through each goal fact specified in `task.goals`.
        5. For each goal fact `(at p goal_loc)`:
           a. Check if the exact goal fact string `(at p goal_loc)` is present in the current `state`. If yes, this package is already at its goal location, continue to the next goal fact.
           b. If `p` is not at `goal_loc`, find its current status using the dictionaries built in step 3:
              - If `p` is found as a key in `package_location`:
                - Get its current location `current_loc = package_location[p]`.
                - The package needs to be picked up, driven to `goal_loc`, and dropped.
                - Add `2` (for pick-up and drop actions) plus the shortest road distance from `current_loc` to `goal_loc` to `h`. If the distance is infinite (unreachable), the problem is likely unsolvable from this state, return infinity.
              - If `p` is found as a key in `package_in_vehicle`:
                - Get the vehicle `v = package_in_vehicle[p]`.
                - Find the vehicle's current location `current_v_loc = vehicle_location.get(v)`.
                - If the vehicle's location is known:
                  - The package needs to be driven from `current_v_loc` to `goal_loc` and dropped.
                  - Add `1` (for the drop action) plus the shortest road distance from `current_v_loc` to `goal_loc` to `h`. If the distance is infinite, return infinity.
                - If the vehicle's location is unknown (should not happen in valid states), return infinity.
              - If `p` is neither in `package_location` nor `package_in_vehicle` (indicates an invalid state), return infinity.
        6. Return the total accumulated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        @param task: The planning task object.
        """
        self.task = task
        self.distances = self._compute_all_pairs_shortest_paths(task.static, task.initial_state, task.goals)
        # Optional: Parse capacity hierarchy if needed for a more complex heuristic
        # self.size_order = self._parse_capacity_hierarchy(task.static)

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and arguments."""
        # Use regex to extract predicate and arguments
        # Handles facts like '(at obj loc)' or '(capacity v size)'
        match = re.match(r"\((\S+)(.*)\)", fact_string)
        if not match:
            return None, []
        predicate = match.group(1)
        # Split arguments, handling potential multiple spaces
        args = match.group(2).strip().split()
        return predicate, args

    def _compute_all_pairs_shortest_paths(self, static_facts, initial_state, goals):
        """
        Builds the road network graph and computes shortest paths between all locations.

        @param static_facts: The frozenset of static facts from the task.
        @param initial_state: The frozenset of initial state facts.
        @param goals: The frozenset of goal facts.
        @return: A dictionary distances[l1][l2] = shortest_distance(l1, l2).
                 Returns float('inf') if l2 is unreachable from l1.
        """
        roads = {}
        locations = set()

        # Add locations from road facts
        for fact_string in static_facts:
            pred, args = self._parse_fact(fact_string)
            if pred == 'road':
                if len(args) == 2:
                    l1, l2 = args
                    locations.add(l1)
                    locations.add(l2)
                    roads.setdefault(l1, set()).add(l2)
                # else: ignore malformed road fact

        # Add locations mentioned in initial state and goals, even if no road facts exist for them
        for fact_string in initial_state | goals:
             pred, args = self._parse_fact(fact_string)
             if pred == 'at' and len(args) == 2:
                  obj, loc = args
                  locations.add(loc)
             elif pred == 'in' and len(args) == 2:
                  # 'in' facts don't mention location directly, vehicle location is needed
                  pass # Handled by processing 'at' facts for vehicles


        distances = {}
        for start_loc in locations:
            distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[start_loc][start_loc] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in roads.get(current_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # Ensure all pairs of locations mentioned in the problem have a distance entry,
        # using infinity for unreachable pairs.
        all_locations_list = list(locations)
        for l1 in all_locations_list:
            distances.setdefault(l1, {})
            for l2 in all_locations_list:
                 distances[l1].setdefault(l2, float('inf'))

        return distances

    # Optional: Implementation for parsing capacity hierarchy
    # def _parse_capacity_hierarchy(self, static_facts):
    #     # This would involve parsing '(capacity-predecessor s1 s2)' facts
    #     # to understand the size ordering (e.g., c0 < c1 < c2).
    #     # It's not strictly necessary for this simple heuristic but could be
    #     # used in a more complex version that considers vehicle capacity.
    #     pass


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
                 Returns float('inf') if a required location is unreachable.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Extract dynamic information from the current state
        package_location = {}
        package_in_vehicle = {}
        vehicle_location = {}

        for fact_string in state:
            pred, args = self._parse_fact(fact_string)
            if pred == 'at':
                if len(args) == 2:
                    obj, loc = args
                    # Distinguish packages and vehicles based on naming convention
                    if obj.startswith('p'): # Assuming 'p' prefix for packages
                        package_location[obj] = loc
                    elif obj.startswith('v'): # Assuming 'v' prefix for vehicles
                        vehicle_location[obj] = loc
            elif pred == 'in':
                 if len(args) == 2:
                    p, v = args
                    # Assuming 'p' is package, 'v' is vehicle
                    if p.startswith('p') and v.startswith('v'):
                        package_in_vehicle[p] = v
            # Ignore capacity facts for this simple heuristic

        h = 0
        # Iterate through goal facts to find misplaced packages
        for goal_fact_string in self.task.goals:
            pred, args = self._parse_fact(goal_fact_string)
            if pred == 'at': # We only care about package locations in the goal
                if len(args) == 2:
                    p, goal_loc = args

                    # Check if package p is already at goal_loc
                    if goal_fact_string in state:
                         continue # Package is already at its goal location

                    # Package is not at its goal location, calculate cost contribution
                    if p in package_location:
                        # Package is at a location
                        current_loc = package_location[p]
                        # Cost: pick-up (1) + drive (distance) + drop (1)
                        # Need distance from current_loc to goal_loc
                        dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                        if dist == float('inf'):
                             # Goal is unreachable from current location - problem likely unsolvable
                             return float('inf')
                         # Add cost for this package: pick-up (1) + drop (1) + drive (dist)
                        h += 2 + dist

                    elif p in package_in_vehicle:
                        # Package is inside a vehicle
                        v = package_in_vehicle[p]
                        # Need vehicle's current location
                        current_v_loc = vehicle_location.get(v)
                        if current_v_loc is not None:
                            # Cost: drive (distance) + drop (1)
                            # Need distance from vehicle's current_loc to goal_loc
                            dist = self.distances.get(current_v_loc, {}).get(goal_loc, float('inf'))
                            if dist == float('inf'):
                                # Goal is unreachable from vehicle's current location
                                return float('inf')
                            # Add cost for this package: drop (1) + drive (dist)
                            h += 1 + dist
                        else:
                            # Package is in a vehicle, but vehicle location is unknown?
                            # This indicates an invalid state according to domain rules.
                            # Treat as unreachable or error.
                            # print(f"Warning: Vehicle {v} containing package {p} has no known location.")
                            return float('inf')
                    else:
                        # Package is not at a location and not in a vehicle?
                        # This indicates an invalid state according to domain rules.
                        # print(f"Warning: Package {p} is neither at a location nor in a vehicle.")
                        return float('inf')
                # else: ignore malformed goal fact

        return h
