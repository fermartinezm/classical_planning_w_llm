from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for all packages that are not yet at their final destination. The cost for
    a single package is estimated based on whether it needs to be picked up,
    driven, and dropped.

    # Assumptions
    - The cost of each action (pick-up, drop, drive) is 1.
    - Vehicles are assumed to be available where needed for pick-up and transport.
      This ignores vehicle capacity and the need to move a vehicle to a package's
      location if none is present.
    - The shortest path distance between any two locations is precomputed.
    - The goal for a package is always to be on the ground at a specific location.
    - Any object in an '(at obj loc)' fact in the state that is not a package
      (as identified from goal facts) is assumed to be a vehicle for the purpose
      of finding the location of a vehicle carrying a package.

    # Heuristic Initialization
    - Identify all locations from the static facts (`road` predicates), initial state (`at` predicates), and goal state (`at` predicates).
    - Build a graph of locations based on `road` predicates. Assume roads are bidirectional based on example instances.
    - Compute the shortest path distance between all pairs of locations using BFS.
    - Extract the goal location for each package from the task's goal conditions (`at` predicates). Identify packages based on these goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state using the task's goal_reached method. If yes, return 0.
    2. Determine the current status of each package that has a goal:
       - Is it on the ground at a location `l` (`(at p l)`)?
       - Is it inside a vehicle `v` (`(in p v)`)? If so, find the vehicle's current location `(at v l)`.
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `goal_l`:
       a. If `p` is currently on the ground at `current_l`:
          - If `current_l` is the same as `goal_l`: The package is at its goal, cost is 0 for this package.
          - If `current_l` is different from `goal_l`:
             - Estimated cost for this package = 1 (pick-up) + distance(`current_l`, `goal_l`) (drive) + 1 (drop).
             - If the distance is infinite (unreachable), the state is likely on an unsolvable path; return infinity.
       b. If `p` is currently inside a vehicle `v` which is at `current_l`:
          - If `current_l` is the same as `goal_l`:
             - Estimated cost for this package = 1 (drop).
          - If `current_l` is different from `goal_l`:
             - Estimated cost for this package = distance(`current_l`, `goal_l`) (drive) + 1 (drop).
             - If the distance is infinite (unreachable), return infinity.
       c. Add the estimated cost for this package to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing location distances.
        """
        self.task = task # Store task to use goal_reached
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Identify all locations and build the road graph
        locations = set()
        road_graph = {} # Adjacency list: {loc: [neighbor1, neighbor2, ...]}

        # Collect locations from road facts (static)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                road_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional based on example instances
                road_graph.setdefault(l2, []).append(l1)

        # Collect locations from initial 'at' facts (dynamic initial state)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)
                 road_graph.setdefault(loc, []) # Ensure all locations are keys even if no roads connect them initially

        # Collect locations from goal 'at' facts
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 locations.add(loc)
                 road_graph.setdefault(loc, []) # Ensure all locations are keys

        self.locations = list(locations) # Store as list
        self.road_graph = road_graph

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {} # {(l1, l2): distance}

        for start_l in self.locations:
            q = deque([(start_l, 0)]) # Queue stores (location, distance)
            visited = {start_l}
            self.distances[(start_l, start_l)] = 0

            while q:
                current_l, dist = q.popleft()

                # Ensure current_l is a key in road_graph, even if it has no roads
                if current_l not in self.road_graph:
                    continue

                for neighbor_l in self.road_graph[current_l]:
                    if neighbor_l not in visited:
                        visited.add(neighbor_l)
                        self.distances[(start_l, neighbor_l)] = dist + 1
                        q.append((neighbor_l, dist + 1))

        # 3. Store goal locations for each package and identify packages
        self.goal_locations = {}
        self.packages = set()
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
                self.packages.add(package)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (dynamic facts only).

        # Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # Track where packages and vehicles are currently located.
        # package_state: {package: {'location': loc_or_vehicle, 'in_vehicle': vehicle_obj or None}}
        package_state = {}
        # vehicle_location: {vehicle: location}
        vehicle_location = {}

        # Populate package_state and vehicle_location from current state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                _, obj, loc = parts
                if obj in self.packages:
                     package_state[obj] = {'location': loc, 'in_vehicle': None}
                else:
                     # Assume any other object 'at' a location is a vehicle
                     vehicle_location[obj] = loc
            elif parts[0] == "in":
                _, package, vehicle = parts
                # Ensure package is one we care about (i.e., has a goal)
                if package in self.packages:
                    package_state[package] = {'location': vehicle, 'in_vehicle': vehicle}
                # We don't need to add vehicle to vehicle_location here, it will be found by its 'at' fact


        total_cost = 0  # Initialize action cost counter.

        # Iterate over packages that have a goal location defined
        for package, goal_location in self.goal_locations.items():
            # If package is not in package_state, it means it's not 'at' or 'in' in the current state.
            # This indicates an invalid state or problem definition. Treat as unreachable.
            if package not in package_state:
                 return float('inf')

            p_state = package_state[package]
            current_loc_or_vehicle = p_state['location']
            in_vehicle = p_state['in_vehicle']

            # Check if the package is already at its goal location on the ground
            # The goal is (at package goal_location), so being 'in' a vehicle at goal_location is NOT a goal.
            if in_vehicle is None and current_loc_or_vehicle == goal_location:
                # Package is on the ground at the goal, cost is 0 for this package
                continue

            # Package is not at the goal location on the ground. Calculate cost.
            cost_for_package = 0

            if in_vehicle is None: # Package is on the ground
                current_loc = current_loc_or_vehicle
                # Cost: pick-up + drive + drop
                cost_for_package += 1 # pick-up action

                # Add drive cost
                drive_cost = self.distances.get((current_loc, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # If goal location is unreachable from current location, this path is unsolvable.
                    return float('inf') # Return infinity immediately

                cost_for_package += drive_cost # drive actions
                cost_for_package += 1 # drop action

            else: # Package is in a vehicle
                vehicle = in_vehicle
                # Need vehicle's location to know where the package currently is
                if vehicle not in vehicle_location:
                    # Vehicle containing package is not 'at' any location in the state? Invalid state.
                    return float('inf') # Vehicle location unknown

                current_loc = vehicle_location[vehicle]

                # If vehicle is at goal location, only drop is needed
                if current_loc == goal_location:
                    cost_for_package += 1 # drop action
                else: # Vehicle needs to drive, then drop
                    # Add drive cost
                    drive_cost = self.distances.get((current_loc, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         return float('inf') # Return infinity immediately

                    cost_for_package += drive_cost # drive actions
                    cost_for_package += 1 # drop action

            total_cost += cost_for_package

        return total_cost
