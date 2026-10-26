# Required imports
from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)

    # Check if the number of parts matches the number of args,
    # unless the pattern ends with a wildcard.
    if len(parts) != len(args):
        if not args or args[-1] != '*':
            return False
        # If it ends with '*', the number of parts must be at least len(args) - 1
        if len(parts) < len(args) - 1:
             return False

    # Check each part against the corresponding arg pattern
    # Use min length to handle trailing '*'
    min_len = min(len(parts), len(args))
    for i in range(min_len):
        if not fnmatch(parts[i], args[i]):
            return False

    # If we reached here, all explicit args matched.
    # If pattern had more args than parts (and didn't end with *), it failed the length check.
    # If parts had more elements than args (and pattern didn't end with *), it failed the length check.
    # If pattern ended with *, the prefix matched, and we are done.
    return True


# Inherit from Heuristic base class if available
# class transportHeuristic(Heuristic):
class transportHeuristic: # Define class name as requested
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost for each package independently, assuming vehicles
    are available when needed and capacity is sufficient for one package.
    The cost for a package is the number of pick-up/drop actions plus
    the shortest path distance for the vehicle carrying it.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and precomputing shortest path distances between locations.
        """
        # Assuming task object has attributes: goals, static, initial_state
        # If inheriting from Heuristic, call super().__init__(task)
        # super().__init__(task)

        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all locations

        # 1. Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location

        # 2. Build the road network graph and collect all relevant locations.
        self.road_graph = {}
        all_relevant_locations = set()

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)
                all_relevant_locations.add(l1)
                all_relevant_locations.add(l2)

        # Add locations from initial state and goals that might not be in road facts
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  all_relevant_locations.add(loc)
        for goal_loc in self.package_goals.values():
             all_relevant_locations.add(goal_loc)

        self.locations = all_relevant_locations # Set of all locations

        # Ensure all locations are keys in the graph, even if they have no outgoing roads
        for loc in self.locations:
             if loc not in self.road_graph:
                  self.road_graph[loc] = []


        # 3. Precompute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.locations:
            self.distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()

                # Check if current_loc has outgoing roads
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Capacity information is not used in this simple heuristic.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Track current location/status of packages and vehicles
        package_status = {} # {package: {'loc': loc_or_vehicle, 'is_in_vehicle': bool}}
        vehicle_locations = {} # {vehicle: loc}

        # Iterate through the state facts to find package and vehicle locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue

            predicate = parts[0]

            if predicate == "at":
                # Fact is (at obj loc)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Check if the object is a package we care about (in package_goals)
                    if obj in self.package_goals:
                         package_status[obj] = {'loc': loc, 'is_in_vehicle': False}
                    # Check if the object is a vehicle (starts with 'v' based on examples)
                    # This is a heuristic-specific assumption based on examples.
                    # A more robust way would be to check object types from the problem file,
                    # but we don't have access to that here. Relying on naming convention.
                    # Or, check if the object appears as the second argument in an '(in package vehicle)' fact.
                    # Let's stick to the 'v' convention for simplicity based on examples.
                    elif obj.startswith('v'):
                         vehicle_locations[obj] = loc
            elif predicate == "in":
                 # Fact is (in package vehicle)
                 if len(parts) == 3:
                     package, vehicle = parts[1], parts[2]
                     if package in self.package_goals:
                          # When in a vehicle, the 'loc' stored is the vehicle name
                          package_status[package] = {'loc': vehicle, 'is_in_vehicle': True}

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # If package is not mentioned in the current state facts we parsed,
            # it's in an unknown state, which shouldn't happen in a valid problem.
            # We skip it or return infinity. Let's skip for now.
            if package not in package_status:
                 # This package is required for the goal but its state is unknown.
                 # This implies an unsolvable state or a parsing issue.
                 # Returning infinity is appropriate for an unsolvable state.
                 return float('inf')


            current_status = package_status[package]
            current_loc_or_vehicle = current_status['loc']
            is_in_vehicle = current_status['is_in_vehicle']

            # If package is already at goal location (and not inside a vehicle), cost is 0 for this package.
            # Note: A package *in* a vehicle at the goal location is *not* at the goal location yet.
            if not is_in_vehicle and current_loc_or_vehicle == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal location or is in a vehicle. Calculate estimated cost.
            package_cost = 0

            if is_in_vehicle:
                # Package is inside a vehicle. The current_loc_or_vehicle is the vehicle name.
                vehicle_name = current_loc_or_vehicle
                vehicle_loc = vehicle_locations.get(vehicle_name)

                if vehicle_loc is None:
                    # Vehicle location not found in state - indicates an inconsistent state.
                    # Treat as unsolvable.
                    return float('inf')

                # Cost to move package from current vehicle location to goal location + drop
                start_loc = vehicle_loc
                end_loc = goal_location

                # Check if path exists
                if (start_loc, end_loc) not in self.distances:
                    # Goal location unreachable from current vehicle location
                    return float('inf')

                package_cost += self.distances[(start_loc, end_loc)] # Drive cost
                package_cost += 1 # Drop action

            else:
                # Package is on the ground. current_loc_or_vehicle is the package's location.
                start_loc = current_loc_or_vehicle
                end_loc = goal_location

                # Check if path exists
                if (start_loc, end_loc) not in self.distances:
                     # Goal location unreachable from package location
                     return float('inf')

                # Cost to pick up + move package from current location to goal location + drop
                # Assumes a vehicle is available at start_loc with capacity.
                package_cost += 1 # Pick-up action
                package_cost += self.distances[(start_loc, end_loc)] # Drive cost
                package_cost += 1 # Drop action

            total_cost += package_cost

        return total_cost
