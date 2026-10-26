import sys
from collections import deque

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the minimum estimated actions required for each package that is not
        yet at its goal location. It considers whether a package is currently
        at a location or inside a vehicle and uses precomputed shortest path
        distances on the road network to estimate driving costs. Vehicle
        capacity and availability constraints are ignored for efficiency.

    Assumptions:
        - The road network defined by 'road' facts is static.
        - 'capacity-predecessor' facts are static.
        - The goal state specifies the final location for each package
          using 'at' predicates.
        - The road network is used for all vehicle movements.
        - Vehicle capacity and the number of available vehicles are not
          considered when estimating the cost for individual packages.
          This makes the heuristic non-admissible but potentially more
          informative for greedy search by focusing on the geometric distance.

    Heuristic Initialization:
        The constructor `__init__` takes the `Task` object as input. It
        performs the following precomputations based on the static information
        and the goal state:
        1.  **Identify Objects**: Parses the initial state, goal state, and
            static facts to identify all packages, vehicles, locations, and
            sizes involved in the problem. This helps in later parsing of states.
        2.  **Package Goals**: Creates a dictionary `self.package_goals` mapping
            each package object name to its target location object name as
            specified in the task's goal state.
        3.  **Road Network and Shortest Paths**: Builds an adjacency list
            representation of the road network graph from the 'road' facts
            in the static information. It then computes all-pairs shortest
            path distances between all identified locations using Breadth-First
            Search (BFS) starting from each location. These distances are stored
            in the dictionary `self.distances`, where keys are `(start_location,
            end_location)` tuples and values are the minimum number of 'drive'
            actions required. Unreachable locations will not have an entry,
            and lookups for such pairs will return a large value (`sys.maxsize`).

    Step-By-Step Thinking for Computing Heuristic:
        The `__call__` method takes the current `state` (a frozenset of fact
        strings) as input and computes the heuristic value:
        1.  Initialize the total heuristic value `h` to 0.
        2.  Parse the current `state` to determine the current status of
            relevant objects:
            -   Create a dictionary `package_current_location` mapping packages
                that are currently `at` a location to that location.
            -   Create a dictionary `package_current_vehicle` mapping packages
                that are currently `in` a vehicle to that vehicle.
            -   Create a dictionary `vehicle_current_location` mapping vehicles
                that are currently `at` a location to that location.
        3.  Iterate through each package `p` that has a defined goal location
            `goal_l` in `self.package_goals`.
        4.  For the current package `p` and its goal location `goal_l`:
            -   Check if the fact `'(at {} {})'.format(p, goal_l)` is present
                in the current `state`. If it is, the package is already at
                its goal location, contributes 0 to the heuristic, and we
                proceed to the next package.
            -   If the package `p` is not at its goal location, determine its
                current status:
                -   If `p` is currently `at` a location `l_p` (i.e., `p` is a key
                    in `package_current_location`):
                    -   The package needs to be picked up, transported, and dropped.
                    -   The estimated number of actions is 1 (pick-up) + the
                        shortest distance from `l_p` to `goal_l` + 1 (drop).
                    -   Look up the distance `dist = self.distances.get((l_p, goal_l), sys.maxsize)`.
                    -   If `dist` is finite, add `2 + dist` to `h`.
                    -   If `dist` is `sys.maxsize` (indicating unreachable), add
                        `sys.maxsize` to `h` (this state is likely unsolvable).
                -   If `p` is currently `in` a vehicle `v` (i.e., `p` is a key
                    in `package_current_vehicle`):
                    -   Find the current location `l_v` of vehicle `v` from
                        `vehicle_current_location`.
                    -   If `l_v == goal_l`: The package is in a vehicle already
                        at the goal location. It just needs to be dropped.
                        -   The estimated number of actions is 1 (drop).
                        -   Add `1` to `h`.
                    -   If `l_v != goal_l`: The package is in a vehicle not at
                        the goal location. It needs to be transported and dropped.
                        -   The estimated number of actions is the shortest
                            distance from `l_v` to `goal_l` + 1 (drop).
                        -   Look up the distance `dist = self.distances.get((l_v, goal_l), sys.maxsize)`.
                        -   If `dist` is finite, add `1 + dist` to `h`.
                        -   If `dist` is `sys.maxsize`, add `sys.maxsize` to `h`.
                -   (Edge case: If a package is not found in either
                   `package_current_location` or `package_current_vehicle`,
                   which should not happen in a valid state representation,
                   add `sys.maxsize` to `h`).
        5.  Return the total accumulated heuristic value `h`. The heuristic is 0
            if and only if all packages are found to be at their goal locations
            during the iteration.

    """

    def __init__(self, task):
        # Identify all objects by type based on predicates they appear in
        self._identify_objects(task)

        # 1. Parse goal state for package goals
        self.package_goals = {}
        for goal_fact_string in task.goals:
            # Goal facts are expected to be '(at package location)'
            if goal_fact_string.startswith('(at '):
                parts = self._parse_fact(goal_fact_string)
                if len(parts) == 3: # Ensure it's a binary predicate like (at ?p ?l)
                    predicate, package, location = parts
                    if predicate == 'at' and package in self.all_packages and location in self.all_locations:
                         self.package_goals[package] = location
                    # Ignore other goal facts if any (e.g., capacity goals, though unlikely in transport)

        # 2. Build road network graph and compute all-pairs shortest paths
        self.distances, self.locations_list = self._build_graph_and_compute_distances(task.static)

    def __call__(self, state):
        # 1. Parse current state to find locations of packages and vehicles
        package_current_location = {}
        package_current_vehicle = {}
        vehicle_current_location = {}

        for fact_string in state:
            parts = self._parse_fact(fact_string)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]

            if predicate == 'at' and len(parts) == 3:
                _, obj, loc = parts
                if obj in self.all_packages and loc in self.all_locations:
                    package_current_location[obj] = loc
                elif obj in self.all_vehicles and loc in self.all_locations:
                    vehicle_current_location[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                _, package, vehicle = parts
                if package in self.all_packages and vehicle in self.all_vehicles:
                    package_current_vehicle[package] = vehicle
            # Ignore capacity facts in state for heuristic calculation

        # 2. Compute heuristic based on packages not at their goal
        h = 0
        for package, goal_l in self.package_goals.items():
            # Check if package is already at goal location
            if package in package_current_location and package_current_location[package] == goal_l:
                continue # Package is at goal location, contributes 0

            # Package is not at goal location. Where is it?
            if package in package_current_location: # Package is at some location l_p
                l_p = package_current_location[package]
                # Needs pick-up (1), drive (dist), drop (1)
                dist = self.distances.get((l_p, goal_l), sys.maxsize)
                if dist == sys.maxsize:
                     # If goal is unreachable from package location, this state is likely unsolvable
                     h += sys.maxsize # Add a large value
                else:
                     h += 2 + dist
            elif package in package_current_vehicle: # Package is in some vehicle v
                v = package_current_vehicle[package]
                if v in vehicle_current_location: # Vehicle v is at some location l_v
                    l_v = vehicle_current_location[v]
                    if l_v == goal_l:
                        # Needs drop (1)
                        h += 1
                    else:
                        # Needs drive (dist), drop (1)
                        dist = self.distances.get((l_v, goal_l), sys.maxsize)
                        if dist == sys.maxsize:
                            # If goal is unreachable from vehicle location, large value
                            h += sys.maxsize # Add a large value
                        else:
                            h += 1 + dist
                else:
                    # Vehicle location not found in state? Should not happen in valid states.
                    h += sys.maxsize # Add a large value
            else:
                 # Package location/vehicle not found in state? Should not happen.
                 h += sys.maxsize # Add a large value

        # If h is still 0 after checking all packages, it means all packages
        # are at their goal locations, so it's a goal state.
        # If any package contributed sys.maxsize, the total h will be sys.maxsize.
        # This is fine for greedy search.

        return h

    def _parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into a tuple."""
        # Remove leading/trailing parentheses and split by space
        fact_string = fact_string.strip()
        if not fact_string.startswith('(') or not fact_string.endswith(')'):
            return None # Not a valid fact string format

        content = fact_string[1:-1].strip()
        if not content:
            return None # Empty fact

        parts = content.split()
        return tuple(parts)

    def _build_graph_and_compute_distances(self, static_facts):
        """Builds road graph and computes all-pairs shortest paths."""
        graph = {}
        locations = set()

        for fact_string in static_facts:
            parts = self._parse_fact(fact_string)
            if parts and parts[0] == 'road' and len(parts) == 3:
                _, l1, l2 = parts
                locations.add(l1)
                locations.add(l2)
                graph.setdefault(l1, []).append(l2)
                # Assuming bidirectional roads based on example
                graph.setdefault(l2, []).append(l1)

        distances = {}
        locations_list = sorted(list(locations)) # Consistent order

        # Handle case with no locations/roads
        if not locations_list:
             return {}, []

        # Compute shortest paths using BFS from each location
        for start_l in locations_list:
            q = deque([(start_l, 0)])
            visited = {start_l}
            distances[(start_l, start_l)] = 0

            while q:
                curr_l, dist = q.popleft()

                # Get neighbors, handle locations not in graph keys (if any)
                for next_l in graph.get(curr_l, []):
                    if next_l not in visited:
                        visited.add(next_l)
                        distances[(start_l, next_l)] = dist + 1
                        q.append((next_l, dist + 1))

        return distances, locations_list

    def _identify_objects(self, task):
        """Identifies objects by type from task definition."""
        self.all_packages = set()
        self.all_vehicles = set()
        self.all_locations = set()
        self.all_sizes = set()

        # From goal state (packages and locations)
        for goal_fact_string in task.goals:
             parts = self._parse_fact(goal_fact_string)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 _, package, location = parts
                 self.all_packages.add(package)
                 self.all_locations.add(location)

        # From initial state
        for fact_string in task.initial_state:
             parts = self._parse_fact(fact_string)
             if not parts: continue
             predicate = parts[0]

             if predicate == 'at' and len(parts) == 3:
                 _, obj, loc = parts
                 # Cannot determine type from 'at' alone, but add location
                 self.all_locations.add(loc)
             elif predicate == 'in' and len(parts) == 3:
                 _, package, vehicle = parts
                 self.all_packages.add(package)
                 self.all_vehicles.add(vehicle)
             elif predicate == 'capacity' and len(parts) == 3:
                 _, vehicle, size = parts
                 self.all_vehicles.add(vehicle)
                 self.all_sizes.add(size)

        # From static facts
        for fact_string in task.static:
             parts = self._parse_fact(fact_string)
             if not parts: continue
             predicate = parts[0]

             if predicate == 'road' and len(parts) == 3:
                 _, l1, l2 = parts
                 self.all_locations.add(l1)
                 self.all_locations.add(l2)
             elif predicate == 'capacity-predecessor' and len(parts) == 3:
                 _, s1, s2 = parts
                 self.all_sizes.add(s1)
                 self.all_sizes.add(s2)
             # If capacity facts were in static, we'd add vehicles here too.
             # Assuming capacity is only in initial state based on example.

        # Ensure all objects mentioned in goals/init/static are categorized
        # This step populates the sets based on predicate usage.
        # Objects in 'at' not identified as package/vehicle yet could be ignored
        # or added to a generic 'locatable' set if needed, but for this heuristic
        # we only care about packages (for goals) and vehicles (for transport).
