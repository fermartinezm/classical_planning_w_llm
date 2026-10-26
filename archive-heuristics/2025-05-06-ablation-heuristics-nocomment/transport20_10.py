from fnmatch import fnmatch
from collections import deque
import math

# Helper functions
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Assumes fact is like '(predicate arg1 arg2 ...)'
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts in the fact
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class exists as shown in examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class if not provided in the execution environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Estimates the cost to reach the goal by summing the estimated costs
    for each package that is not yet at its goal location. The cost for
    a package is estimated based on its current state (at a location or
    in a vehicle) and the shortest path distance to its goal location.
    This heuristic is non-admissible but aims to guide a greedy search
    efficiently by focusing on moving misplaced packages towards their goals.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing necessary information
        from the task definition, such as goal locations for packages
        and shortest path distances between all pairs of locations.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization:
        # 1. Store goal locations for each package from the task goals.
        # 2. Build the road network graph from static facts.
        # 3. Identify all relevant locations present in the problem (from roads, initial state, goals).
        # 4. Compute all-pairs shortest path distances between all relevant locations using BFS.

        # 1. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                # Ensure the goal fact has the expected structure (predicate, obj, loc)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # else: Unexpected goal format - ignored


        # 2. Build road graph and 3. Identify all relevant locations
        graph = {}
        all_locations_in_problem = set()

        # Add locations from road facts and build graph
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, l1, l2 = parts
                    graph.setdefault(l1, []).append(l2)
                    graph.setdefault(l2, []).append(l1) # Roads are typically bidirectional
                    all_locations_in_problem.add(l1)
                    all_locations_in_problem.add(l2)
                # else: Unexpected road fact format - ignored


        # Add locations mentioned in initial state 'at' facts
        # This ensures locations with packages/vehicles but no roads are included
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, obj, loc = parts
                     # We need the location object itself
                     all_locations_in_problem.add(loc)
                 # else: Unexpected at fact format - ignored

        # Add locations mentioned in goal 'at' facts
        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 3:
                     _, obj, loc = parts
                     all_locations_in_problem.add(loc)
                 # else: Unexpected at fact format - ignored


        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in all_locations_in_problem:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Update distance if this is the first path found or a shorter one
                # (BFS guarantees shortest on unweighted graph, so first time is shortest)
                if (start_node, current_loc) not in self.distances or self.distances[(start_node, current_loc)] > dist:
                     self.distances[(start_node, current_loc)] = dist

                # Get neighbors from the graph, handle locations not in graph (isolated locations)
                neighbors = graph.get(current_loc, [])

                for next_loc in neighbors:
                    if next_loc not in visited:
                        visited.add(next_loc)
                        q.append((next_loc, dist + 1))

        # Distances for unconnected locations will be missing in self.distances.
        # Accessing with .get(..., math.inf) will handle this during __call__.


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimate of the number of actions required to reach a goal state.
            Returns math.inf if the goal is unreachable from this state based
            on the precomputed distances for any package.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic:
        # 1. Identify the current location or containing vehicle for each package present in the state.
        # 2. Identify the current location for each vehicle present in the state.
        # 3. Initialize total heuristic cost to 0.
        # 4. For each package that has a goal location defined in the task:
        #    a. Check if the package is already at its goal location in the current state. If yes, contribute 0 to cost for this package.
        #    b. If not at the goal location, determine its current status (at a location or in a vehicle) from the state facts.
        #    c. If the package's status is unknown (not found in 'at' or 'in' facts), the goal might be unreachable or the state is malformed. Return infinity.
        #    d. If the package is at a location `l`:
        #       - It needs to be picked up (estimated 1 action).
        #       - It needs to be transported from `l` to its goal location `goal_l` (estimated `distance(l, goal_l)` drive actions).
        #       - It needs to be dropped at `goal_l` (estimated 1 action).
        #       - Add `1 + distance(l, goal_l) + 1` to the total cost for this package.
        #    e. If the package is in a vehicle `v`:
        #       - Find the current location `v_loc` of vehicle `v` from the state facts.
        #       - If the vehicle's location is unknown, the goal might be unreachable or the state is malformed. Return infinity.
        #       - It needs to be transported from `v_loc` to its goal location `goal_l` (estimated `distance(v_loc, goal_l)` drive actions).
        #       - It needs to be dropped at `goal_l` (estimated 1 action).
        #       - Add `distance(v_loc, goal_l) + 1` to the total cost for this package.
        #    f. If the required distance for any package is infinite (unreachable), the total heuristic is infinite.
        # 5. Return the total accumulated cost.

        # 1. Identify current package locations/vehicles and 2. Identify current vehicle locations
        package_locations = {} # Maps package -> location or vehicle
        vehicle_locations = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Simple check based on name prefix (assuming standard naming like p1, v1)
                    if obj.startswith('v'):
                        vehicle_locations[obj] = loc
                    elif obj.startswith('p'):
                        package_locations[obj] = loc
                    # Add other locatable types if necessary
                # else: Unexpected format - ignored

            elif predicate == 'in':
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    package_locations[package] = vehicle # Package is inside a vehicle
                 # else: Unexpected format - ignored

        # 3. Initialize total heuristic cost
        total_cost = 0

        # 4. For each package that has a goal location defined
        for package, goal_location in self.goal_locations.items():
            # a. Check if the package is already at its goal location
            if f'(at {package} {goal_location})' in state:
                 continue # Package is already at its goal location

            # b. Determine current status (at location or in vehicle)
            current_status = package_locations.get(package)

            # c. If status is unknown, goal is likely unreachable from this state
            if current_status is None:
                 return math.inf # Package state unknown, assume unreachable goal

            # d. If the package is in a vehicle
            if current_status in vehicle_locations: # current_status is a vehicle name
                vehicle = current_status
                vehicle_loc = vehicle_locations.get(vehicle) # Location of the vehicle

                # If vehicle location is unknown, goal is likely unreachable
                if vehicle_loc is None:
                     return math.inf # Vehicle state unknown, assume unreachable goal

                # Package is in a vehicle, needs to be driven to goal_location and dropped
                drive_cost = self.distances.get((vehicle_loc, goal_location), math.inf)
                drop_cost = 1
                cost_for_package = drive_cost + drop_cost

            # e. If the package is at a location
            else: # current_status is a location name
                package_loc = current_status

                # Package is at a location, needs to be picked up, driven to goal_location, and dropped
                pick_cost = 1
                drive_cost = self.distances.get((package_loc, goal_location), math.inf)
                drop_cost = 1
                cost_for_package = pick_cost + drive_cost + drop_cost

            # f. If any part of the path is unreachable, the cost for this package is infinite
            if cost_for_package == math.inf:
                 return math.inf # Goal unreachable

            total_cost += cost_for_package

        # 5. Return the total accumulated cost.
        return total_cost
