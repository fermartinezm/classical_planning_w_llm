from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and potential
    synergies/conflicts between packages needing the same vehicle.

    # Assumptions
    - The primary goal is to move packages to specific locations.
    - Vehicle capacity is not a hard constraint in the heuristic calculation;
      it assumes a vehicle with sufficient capacity is available when needed.
    - The cost of moving a vehicle between two locations is the shortest path
      distance in the road network.
    - Each package requires a pick-up action (if on the ground) and a drop
      action (if in a vehicle) as part of its journey, in addition to vehicle
      movement.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current effective location:
       - If the package is on the ground at location L, its effective location is L.
       - If the package is inside a vehicle V, its effective location is the
         current location of vehicle V.
    2. If the package is already on the ground at its goal location, the cost for this package is 0.
    3. If the package is inside a vehicle which is currently at the package's goal location,
       it only needs to be dropped. Cost: 1 (drop).
    4. If the package is on the ground at a location L_curr (not the goal L_goal):
       - It needs to be picked up (1 action).
       - A vehicle needs to drive from L_curr to L_goal (shortest path distance).
       - It needs to be dropped at L_goal (1 action).
       - Estimated cost: 1 (pick-up) + shortest_path_distance(L_curr, L_goal) + 1 (drop).
    5. If the package is inside a vehicle V at a location L_v_curr (not the goal L_goal):
       - Vehicle V needs to drive from L_v_curr to L_goal (shortest path distance).
       - It needs to be dropped at L_goal (1 action).
       - Estimated cost: shortest_path_distance(L_v_curr, L_goal) + 1 (drop).
    6. The total heuristic value is the sum of the estimated costs for all packages.
    7. If any required location is unreachable via the road network, the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and precomputing shortest paths in the road network.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "at":
                 package, location = args
                 self.goal_locations[package] = location
             # Assuming package goals are always (at ?p ?l) based on domain/examples

        # 2. Build the road network graph
        self.road_graph = {}
        locations = set() # Collect all locations mentioned in road facts
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                self.road_graph.setdefault(l1, []).append(l2)
                locations.add(l1)
                locations.add(l2)

        # Add locations from goals and initial state that might not be in road facts
        # (e.g., a single isolated location)
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "at":
                 locations.add(args[1]) # Add goal location

        # We need access to the initial state to get all locations mentioned there
        # However, the __init__ method only receives the task object, which contains
        # initial_state. Let's iterate through initial_state facts to get all locations.
        # This is slightly less clean as it uses initial_state in __init__, but necessary
        # to ensure all relevant locations are included in the BFS graph.
        for fact in task.initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "at":
                 # args[0] is the object, args[1] is the location
                 locations.add(args[1])
             elif predicate == "road": # Should be in static, but double check
                 locations.add(args[0])
                 locations.add(args[1])


        # 3. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        all_locations_list = list(locations) # Convert set to list for consistent iteration order (optional)
        for start_node in all_locations_list:
            self.shortest_paths[start_node] = self._bfs(start_node, all_locations_list)

    def _bfs(self, start_node, all_locations):
        """Performs BFS from start_node to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            current_dist = distances[current_node]

            # Check if current_node has outgoing roads defined
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Only update if we found a shorter path (or the first path)
                    if distances[neighbor] > current_dist + 1:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0

        # Map current locations of locatables (packages and vehicles)
        current_locations = {} # Maps object name (package or vehicle) to its location string
        package_in_vehicle = {} # Maps package name to vehicle name if inside

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                current_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        # Iterate through each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Determine the package's effective current location
            current_package_location = None
            is_in_vehicle = False

            if package in package_in_vehicle:
                # Package is in a vehicle, its effective location is the vehicle's location
                vehicle = package_in_vehicle[package]
                # The vehicle must be somewhere
                current_package_location = current_locations.get(vehicle)
                is_in_vehicle = True
            else:
                # Package is on the ground
                current_package_location = current_locations.get(package)

            # If the package isn't mentioned in 'at' or 'in' facts, something is wrong
            # or it's not part of the initial state/goals being considered.
            # Assuming all goal packages are in the initial state and tracked.
            if current_package_location is None:
                 # This case should ideally not happen in a well-formed problem instance
                 # where goal objects are always present in the initial state.
                 # If it does, this package is untraceable, assume infinite cost or skip?
                 # Skipping might lead to underestimation if the package needs to be created/moved.
                 # Returning infinity is safer for pruning.
                 # print(f"Warning: Package {package} not found in state.") # Debugging
                 return float('inf')


            # Calculate cost for this package
            cost_for_package = 0

            if is_in_vehicle:
                # Package is in a vehicle at current_package_location
                # Needs to drive vehicle to goal_location and drop package
                drive_cost = self.shortest_paths.get(current_package_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal location is unreachable from current vehicle location
                     return float('inf')

                # Cost is drive actions + 1 drop action
                cost_for_package = drive_cost + 1
            else:
                # Package is on the ground at current_package_location
                # Needs pick-up, drive vehicle to goal_location, and drop package
                drive_cost = self.shortest_paths.get(current_package_location, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal location is unreachable from current package location
                     return float('inf')

                # Cost is 1 pick-up action + drive actions + 1 drop action
                cost_for_package = 1 + drive_cost + 1 # pick-up + drive + drop

            total_cost += cost_for_package

        return total_cost

