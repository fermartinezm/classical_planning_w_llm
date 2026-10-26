import math
from collections import deque

class transportHeuristic:
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum required
        actions for each package that is not yet at its goal location.
        The minimum actions for a package depend on whether it's on the ground
        or inside a vehicle, and the shortest road distance between its current
        location and its goal location.

    Assumptions:
        - The goal state primarily consists of (at package location) facts.
          The heuristic focuses only on these goals.
        - The road network is static and provided in the static facts.
        - Vehicle capacity constraints are ignored for simplicity in this
          non-admissible heuristic.
        - The cost of moving vehicles without the target package (e.g., to
          reach a package's initial location) is implicitly included in the
          cost calculation for packages on the ground (2 + distance), assuming
          a vehicle needs to arrive and depart.
        - Object types (package, vehicle) are inferred from the fact structure
          and naming conventions (e.g., 'p' prefix for packages, 'v' prefix
          for vehicles) as type information is not explicitly available in
          the state facts themselves in the provided representation.
        - All packages with goal locations are present in the state facts
          (either at a location or in a vehicle).

    Heuristic Initialization:
        The constructor processes the static facts and goal facts from the
        Task object.
        1. It extracts the goal location for each package from the goal facts
           (assuming goals are only of the form (at package location)). This
           is stored in a dictionary `self.package_goals`.
        2. It builds a graph representing the road network from (road l1 l2) facts
           and collects all unique locations.
        3. It computes the shortest path distance (number of drive actions)
           between all pairs of locations using BFS. This is stored in a
           dictionary `self.distances`. Unreachable locations have a distance of infinity.

    Step-By-Step Thinking for Computing Heuristic:
        The __call__ method takes the current state and the task object.
        1. Initialize the total heuristic value `h` to 0.
        2. Create dictionaries to quickly look up package locations, package
           contents, and vehicle locations in the current state. Iterate
           through the state facts:
           - If a fact is `(at obj loc)`, check if `obj` looks like a package
             (e.g., starts with 'p') or a vehicle (e.g., starts with 'v')
             and store its location in `package_locations` or `vehicle_locations`
             respectively.
           - If a fact is `(in pkg veh)`, store that `pkg` is in `veh` in
             `package_in_vehicle`.
        3. Iterate through each package `p` that has a goal location `goal_l`
           defined in `self.package_goals`.
        4. Determine the current status and location of package `p`:
           - If `p` is found in `package_locations`, its current location
             `current_l` is `package_locations[p]`, and it's on the ground.
           - If `p` is not in `package_locations` but is found in
             `package_in_vehicle` (say, in vehicle `v`), find `v`'s location
             `current_l` from `vehicle_locations[v]`. The package is inside
             a vehicle.
           - If `p` is not found in either lookup (which shouldn't happen in
             a valid state for a package with a goal), it's an error case
             or the state is invalid.
        5. If the package's current location `current_l` is determined:
           - If `current_l == goal_l` AND the package is on the ground (not
             in a vehicle), this package is satisfied and contributes 0 to `h`.
           - If `current_l != goal_l` AND the package is on the ground:
             - The package needs to be picked up, the vehicle needs to drive
               from `current_l` to `goal_l`, and the package needs to be dropped.
             - Add `2 + self.distances[current_l][goal_l]` to `h`. If the
               distance is infinity, the state is likely unsolvable, and we
               can return infinity immediately.
           - If the package is inside a vehicle at `current_l`:
             - If `current_l == goal_l`: The vehicle is at the goal location;
               the package just needs to be dropped. Add 1 to `h`.
             - If `current_l != goal_l`: The vehicle needs to drive from
               `current_l` to `goal_l`, and then the package needs to be dropped.
               Add `1 + self.distances[current_l][goal_l]` to `h`. If the
               distance is infinity, return infinity.
        6. After processing all packages with goals, return the total heuristic
           value `h`.
    """

    def __init__(self, task):
        self.task = task
        self.package_goals = {}
        self.locations = set()
        self.graph = {}
        self.distances = {}

        # 1. Extract goal locations for packages
        # Assuming goal facts are like '(at package location)'
        for goal_fact in self.task.goals:
            parts = goal_fact.strip('()').split()
            if parts[0] == 'at' and len(parts) == 3:
                package_name = parts[1]
                location_name = parts[2]
                # Basic check if it looks like a package name based on examples
                if package_name.startswith('p'):
                    self.package_goals[package_name] = location_name

        # 2. Build road network graph and collect all locations
        for static_fact in self.task.static:
            parts = static_fact.strip('()').split()
            if parts[0] == 'road' and len(parts) == 3:
                l1 = parts[1]
                l2 = parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.graph:
                    self.graph[l1] = set()
                self.graph[l1].add(l2)

        # Add locations from goals to the set of all locations, just in case
        # a goal location is not mentioned in any road fact (unlikely in valid PDDL)
        for goal_l in self.package_goals.values():
             self.locations.add(goal_l)

        # Ensure all locations have an entry in the graph dictionary
        for loc in self.locations:
             if loc not in self.graph:
                 self.graph[loc] = set() # Isolated location

        # 3. Compute all-pairs shortest paths using BFS
        for start_node in self.locations:
            self.distances[start_node] = {}
            for loc in self.locations:
                self.distances[start_node][loc] = math.inf

            queue = deque([(start_node, 0)])
            self.distances[start_node][start_node] = 0

            while queue:
                (current_node, current_dist) = queue.popleft()

                # If current_node is in graph (it should be if added from locations set)
                if current_node in self.graph:
                    for neighbor in self.graph[current_node]:
                        # Use .get for safety, though neighbor should be in self.locations
                        if self.distances[start_node].get(neighbor, math.inf) == math.inf:
                            self.distances[start_node][neighbor] = current_dist + 1
                            queue.append((neighbor, current_dist + 1))


    def __call__(self, state, task):
        h = 0

        # Quick lookups for current state
        package_locations = {}
        package_in_vehicle = {}
        vehicle_locations = {}

        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'at' and len(parts) == 3:
                obj_name = parts[1]
                loc_name = parts[2]
                # Infer type based on naming convention from examples
                if obj_name.startswith('p'):
                    package_locations[obj_name] = loc_name
                elif obj_name.startswith('v'):
                    vehicle_locations[obj_name] = loc_name
            elif parts[0] == 'in' and len(parts) == 3:
                 # Assuming '(in package vehicle)'
                 package_name = parts[1]
                 vehicle_name = parts[2]
                 # Infer type based on naming convention
                 if package_name.startswith('p') and vehicle_name.startswith('v'):
                     package_in_vehicle[package_name] = vehicle_name

        # Calculate cost for each package with a goal
        for package_name, goal_l in self.package_goals.items():
            current_l = None
            is_in_vehicle = False

            if package_name in package_locations:
                current_l = package_locations[package_name]
                is_in_vehicle = False
            elif package_name in package_in_vehicle:
                vehicle_name = package_in_vehicle[package_name]
                if vehicle_name in vehicle_locations:
                    current_l = vehicle_locations[vehicle_name]
                    is_in_vehicle = True
                else:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # This state is likely invalid or problematic.
                    # Treat as unreachable goal for this package.
                    return math.inf # Return infinity immediately
            else:
                # Package is not in package_locations or package_in_vehicle.
                # Its location is unknown. This shouldn't happen for packages with goals
                # in valid states. If it does, treat as unreachable.
                 return math.inf # Return infinity immediately


            # Check if package is already at goal AND on the ground
            if current_l == goal_l and not is_in_vehicle:
                 # Package is on the ground at the goal location. Done.
                 continue # Cost is 0 for this package

            # Package is not at goal location on the ground. Calculate steps needed.
            distance_to_goal = self.distances.get(current_l, {}).get(goal_l, math.inf)

            if distance_to_goal == math.inf:
                 # Goal location is unreachable from current location.
                 return math.inf # Return infinity immediately

            if is_in_vehicle:
                # Package is inside a vehicle at current_l
                # Needs drive (if current_l != goal_l) + drop
                # If current_l == goal_l, distance_to_goal is 0. Cost is 1 + 0 = 1 (drop)
                # If current_l != goal_l, distance_to_goal > 0. Cost is 1 + distance (drive + drop)
                h += 1 + distance_to_goal
            else:
                # Package is on the ground at current_l (which is not goal_l)
                # Needs pick-up + drive + drop
                h += 2 + distance_to_goal # 1 for pick-up, distance_to_goal for drive, 1 for drop

        return h
