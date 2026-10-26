# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple."""
    # Remove surrounding brackets and split by spaces
    # Example: '(at p1 l1)' -> ('at', 'p1', 'l1')
    parts = fact_string[1:-1].split()
    return tuple(parts)

from collections import deque, defaultdict

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        This heuristic estimates the cost to reach the goal by summing up
        the minimum actions required for each package that is not yet at
        its goal location. For a package not at its goal, the estimated cost
        includes:
        - 1 action for pick-up (if not already in a vehicle).
        - The shortest path distance (number of drive actions) for a vehicle
          to move the package from its current location (or the vehicle's
          current location if inside a vehicle) to the package's goal location.
        - 1 action for drop-off.
        The heuristic ignores vehicle capacity constraints and assumes any
        package can be picked up and transported by any vehicle.

    Assumptions:
        - The road network is static and provided by '(road l1 l2)' facts.
        - The goal is specified by '(at package location)' facts for some packages.
        - All packages relevant to the goal are initially either '(at package location)'
          or '(in package vehicle)'.
        - The road network is connected enough that goal locations are reachable
          from initial package/vehicle locations.
        - Package names start with 'p' and vehicle names start with 'v'.

    Heuristic Initialization:
        1. Parse the static facts to build the road network graph from '(road l1 l2)' facts.
        2. Compute the shortest path distance between all pairs of locations using BFS.
           This distance represents the minimum number of 'drive' actions required
           to travel between two locations.
        3. Identify the goal location for each package specified in the task's goals.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. For each package `p` that has a goal location `loc_p_goal` (identified during initialization):
        3. Check if the fact `(at p loc_p_goal)` is present in the current state.
           This is done by parsing state facts and checking the parsed tuples.
           If it is, the package is already at its goal, and contributes 0 to the heuristic.
           Continue to the next package.
        4. If the package `p` is not at its goal, find its current status in the state:
           - Look for a fact `(at p loc_p_current)`.
           - Look for a fact `(in p v)`.
           A dictionary `current_locations` is built from the state facts for quick lookup.
        5. If `(at p loc_p_current)` is found in `current_locations`:
           - The package needs to be picked up (1 action).
           - It needs to be transported from `loc_p_current` to `loc_p_goal`. The minimum
             drive actions required is the shortest path distance `dist(loc_p_current, loc_p_goal)`.
           - It needs to be dropped off (1 action).
           - Add `1 + dist(loc_p_current, loc_p_goal) + 1` to `h`.
           - If `loc_p_goal` is unreachable from `loc_p_current`, return infinity.
        6. If `(in p v)` is found in `current_locations`:
           - The package is already in a vehicle `v`.
           - Find the current location of vehicle `v` by looking for `(at v loc_v_current)`
             in the `current_locations` dictionary.
           - The package needs to be transported from `loc_v_current` (inside `v`)
             to `loc_p_goal`. The minimum drive actions required is
             `dist(loc_v_current, loc_p_goal)`.
           - It needs to be dropped off (1 action).
           - Add `dist(loc_v_current, loc_p_goal) + 1` to `h`.
           - If `loc_p_goal` is unreachable from `loc_v_current`, return infinity.
           - If the vehicle's location is not found, return infinity.
        7. If the package's location/status is not found in `current_locations` (should not happen in valid problems),
           return infinity.
        8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.package_goals = {}
        self.road_graph = defaultdict(list)
        self.locations = set()
        self.distance_map = {} # Stores shortest path distances dist[l1][l2]

        # 1. Parse static facts and build road graph
        for fact_str in task.static:
            fact = parse_fact(fact_str)
            if fact[0] == 'road':
                l1, l2 = fact[1], fact[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)
            # We ignore capacity-predecessor for this heuristic

        # 2. Compute all-pairs shortest paths
        self._compute_all_pairs_shortest_paths()

        # 3. Identify package goals
        # Goal facts are in task.goals
        for goal_fact_str in task.goals:
            goal_fact = parse_fact(goal_fact_str)
            # Assuming goal facts are always simple predicates like (at obj loc)
            # and we are only interested in package goals.
            # We assume objects starting with 'p' are packages based on examples.
            if goal_fact[0] == 'at' and len(goal_fact) == 3:
                 obj_name = goal_fact[1]
                 if obj_name.startswith('p'): # Simple check for package name pattern
                     self.package_goals[obj_name] = goal_fact[2]


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        for start_loc in self.locations:
            self.distance_map[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distance_map[start_loc][start_loc] = 0 # Distance to self is 0

            while queue:
                current_loc, dist = queue.popleft()

                # Check if current_loc is in graph (handle locations with no roads)
                # This might happen if a location exists but has no roads connected
                if current_loc not in self.road_graph:
                    continue

                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distance_map[start_loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach a goal state.
        """
        h = 0
        # Parse state facts and store relevant ones for quick lookup
        # {object_name: ('at', location)} or {object_name: ('in', vehicle)}
        current_locations = {}
        for fact_str in state:
             fact = parse_fact(fact_str)
             if fact[0] == 'at' and len(fact) == 3:
                 current_locations[fact[1]] = ('at', fact[2])
             elif fact[0] == 'in' and len(fact) == 3:
                 current_locations[fact[1]] = ('in', fact[2])


        # Iterate through packages that have a goal
        for package, goal_loc in self.package_goals.items():
            # Check if package is already at goal
            # We need to check if the package is *at* the goal location.
            # It being *in* a vehicle at the goal location is not the goal state for the package.
            package_at_goal = False
            if package in current_locations:
                 status, loc_or_veh = current_locations[package]
                 if status == 'at' and loc_or_veh == goal_loc:
                      package_at_goal = True

            if package_at_goal:
                continue # Package is at goal, contributes 0

            # Package is not at goal, calculate its contribution
            if package in current_locations:
                status, loc_or_veh = current_locations[package]

                if status == 'at':
                    # Package is at a location, needs pick-up, drive, drop
                    current_loc = loc_or_veh
                    # Cost = pick-up (1) + drive (dist) + drop (1)
                    drive_cost = self.distance_map.get(current_loc, {}).get(goal_loc)

                    if drive_cost is None:
                         # Goal location unreachable from current package location
                         # This state is likely not on a path to the goal
                         return float('inf')

                    h += 1 + drive_cost + 1

                elif status == 'in':
                    # Package is in a vehicle, needs drive, drop
                    vehicle = loc_or_veh
                    # Find vehicle's current location
                    if vehicle in current_locations and current_locations[vehicle][0] == 'at':
                        current_veh_loc = current_locations[vehicle][1]
                        # Cost = drive (dist) + drop (1)
                        drive_cost = self.distance_map.get(current_veh_loc, {}).get(goal_loc)

                        if drive_cost is None:
                             # Goal location unreachable from current vehicle location
                             # This state is likely not on a path to the goal
                             return float('inf')

                        h += drive_cost + 1
                    else:
                         # Vehicle location not found? Should not happen in valid states.
                         # Or vehicle is also 'in' something else? (Not possible in this domain)
                         # This state is likely not on a path to the goal
                         return float('inf')
            else:
                 # Package not found in state facts? Should not happen in valid states.
                 # This state is likely not on a path to the goal
                 return float('inf')

        return h
