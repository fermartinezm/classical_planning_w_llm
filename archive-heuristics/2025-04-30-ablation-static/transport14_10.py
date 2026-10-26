from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided elsewhere
# In a real planning system, this would be imported.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class for standalone execution."""
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    to its goal location, summing the individual costs. It relaxes vehicle
    capacity constraints and assumes vehicles are always available when needed.
    The cost for a package is estimated based on its current state:
    - If on the ground, it needs pickup, transport (driving), and drop.
    - If in a vehicle, it needs transport (driving) and drop.
    The transport cost is estimated by the shortest path distance in the road network.

    # Assumptions
    - Goals are always of the form `(at ?p ?l)`.
    - The road network is static.
    - Vehicle capacity and availability are not bottlenecks (relaxed).
    - Action costs are uniform (each action costs 1).

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build the road network graph from `(road ?l1 ?l2)` facts.
    - Precompute all-pairs shortest path distances between locations using BFS.
    - Extract capacity size ordering from `(capacity-predecessor ?s1 ?s2)` facts (though not directly used in the current simple cost calculation, it's good practice to parse).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Determine the current location or containment status for every package and vehicle by examining the state facts.
       - Store package status: `(at p loc)` or `(in p vehicle)`.
       - Store vehicle locations: `(at v loc)`.
    3. For each package `p` that has a goal location `goal_loc`:
       - If `p` is already at `goal_loc` (i.e., `(at p goal_loc)` is in the state), the cost for this package is 0.
       - If `p` is on the ground at `current_loc` (`(at p current_loc)` in state, and `current_loc != goal_loc`):
         - Estimate cost as 1 (pick-up) + shortest_distance(`current_loc`, `goal_loc`) (drive actions) + 1 (drop).
         - If `goal_loc` is unreachable from `current_loc`, the total heuristic is infinity.
       - If `p` is inside a vehicle `v` (`(in p v)` in state):
         - Find the vehicle's current location `vehicle_loc` (`(at v vehicle_loc)` in state).
         - If `vehicle_loc == goal_loc`: Estimate cost as 1 (drop).
         - If `vehicle_loc != goal_loc`: Estimate cost as shortest_distance(`vehicle_loc`, `goal_loc`) (drive actions) + 1 (drop).
         - If `goal_loc` is unreachable from `vehicle_loc`, the total heuristic is infinity.
    4. Sum the estimated costs for all packages.
    5. Return the total sum. If any package's goal is unreachable, return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task) # Call base class constructor

        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.package_goals[package] = location
                # Ignore other types of 'at' goals if any (e.g., vehicle goals)
                # as the heuristic focuses on package delivery.

        # 2. Build the road network graph and collect all locations.
        self.adj_list = {}
        all_locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                if len(args) == 2:
                    l1, l2 = args
                    if l1 not in self.adj_list:
                        self.adj_list[l1] = []
                    if l2 not in self.adj_list:
                         self.adj_list[l2] = [] # Ensure all locations are keys even if no outgoing roads
                    self.adj_list[l1].append(l2)
                    all_locations.add(l1)
                    all_locations.add(l2)

        # Add locations from initial state and goals that might not be in road facts
        # (e.g., a single isolated location where a package starts or needs to go).
        initial_state_locations = set()
        for fact in task.initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "at" and len(args) == 2:
                 locatable, loc = args
                 # Add the location to the set of all locations
                 all_locations.add(loc)
                 # Ensure it's in the adj_list structure even if isolated
                 if loc not in self.adj_list:
                     self.adj_list[loc] = []

        goal_locations_set = set(self.package_goals.values())
        all_locations.update(goal_locations_set)

        # Ensure all collected locations are in the adjacency list structure
        for loc in all_locations:
             if loc not in self.adj_list:
                 self.adj_list[loc] = []

        self.all_locations = list(all_locations) # Store as list if needed, dict lookup is fine

        # 3. Precompute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Extract capacity size ordering (optional for this simple heuristic, but included for completeness).
        # This part is not used in the current heuristic calculation but demonstrates parsing static facts.
        size_predecessors = {} # s2 -> s1 if (capacity-predecessor s1 s2)
        size_successors = {} # s1 -> s2 if (capacity-predecessor s1 s2)
        all_sizes = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "capacity-predecessor":
                if len(args) == 2:
                    s1, s2 = args
                    size_predecessors[s2] = s1
                    size_successors[s1] = s2
                    all_sizes.add(s1)
                    all_sizes.add(s2)

        self.size_to_int = {}
        self.int_to_size = {}

        if all_sizes:
            # Find the smallest size string (largest capacity) - it's not a key in size_successors
            smallest_size_str = None
            successor_keys = set(size_successors.keys())
            for s in all_sizes:
                if s not in successor_keys:
                    smallest_size_str = s # e.g., c3 in (c0 c1)(c1 c2)(c2 c3)
                    break

            if smallest_size_str is not None:
                 # Assign integer values: largest capacity gets highest int
                 current_size_str = smallest_size_str
                 current_int = len(all_sizes) - 1
                 while current_size_str is not None:
                     self.size_to_int[current_size_str] = current_int
                     self.int_to_size[current_int] = current_size_str
                     current_int -= 1
                     current_size_str = size_predecessors.get(current_size_str) # Move to the next larger size string (smaller capacity)


    def _bfs(self, start_loc):
        """
        Performs BFS from a start location to find distances to all reachable locations.
        Returns a dictionary {location: distance}. Unreachable locations have distance infinity.
        """
        distances_from_start = {loc: float('inf') for loc in self.all_locations}
        # Only proceed if the start_loc is a known location in our graph structure
        if start_loc not in self.all_locations:
             return distances_from_start # All distances remain infinity

        distances_from_start[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances_from_start[current_loc]

            # Check if current_loc is in adj_list (it should be if collected properly)
            if current_loc in self.adj_list:
                for neighbor in self.adj_list[current_loc]:
                    if distances_from_start[neighbor] == float('inf'):
                        distances_from_start[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances_from_start

    def get_distance(self, loc1, loc2):
        """
        Returns the precomputed shortest distance between loc1 and loc2.
        Returns float('inf') if either location is unknown or unreachable.
        """
        # Check if both locations are known from the precomputation phase
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This means loc1 was not a starting point for BFS or loc2 wasn't reached.
             # In a well-formed problem, all relevant locations should be in self.all_locations
             # and BFS should explore connectivity. If not found here, they are likely
             # disconnected or malformed. Treat as unreachable.
             return float('inf')

        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings).

        # Check if goal is reached
        # Note: The task.goal_reached method checks if ALL goals are met.
        # Our heuristic calculates cost towards meeting package goals specifically.
        # If task.goal_reached is True, our heuristic should be 0.
        if self.task.goal_reached(state):
             return 0

        # Track where packages and vehicles are currently located or contained.
        package_status = {} # {package: ('at', loc) or ('in', vehicle)}
        vehicle_locations = {} # {vehicle: loc}
        # vehicle_capacities = {} # {vehicle: size_string} # Not used in this simple heuristic

        # Populate status dictionaries from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Need to distinguish packages from vehicles.
                # We assume objects listed in package_goals are packages.
                if obj in self.package_goals:
                    package_status[obj] = ('at', loc)
                else:
                     # Assume anything else with 'at' is a vehicle for this domain structure
                     vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.package_goals: # Only track packages we care about
                    package_status[package] = ('in', vehicle)

            # elif predicate == "capacity" and len(parts) == 3:
            #     vehicle, size = parts[1], parts[2]
            #     vehicle_capacities[vehicle] = size # Not used in this heuristic

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.package_goals.items():
            # If a package is not found in the current state facts, it's an unexpected state.
            # Treat this as an unreachable goal for safety, although it shouldn't happen
            # in valid state transitions from a well-formed initial state.
            if package not in package_status:
                 return float('inf')

            status, current_loc_or_vehicle = package_status[package]

            # Case 1: Package is already at the goal location on the ground.
            # This check is implicitly handled by the cost calculation below,
            # but explicitly checking might save a tiny bit of computation if goal is met.
            # However, the goal check at the start of __call__ is the primary way
            # to return 0 for the goal state. We calculate cost for *all* packages
            # not yet satisfying their individual (at p l) goal.

            # Case 2: Package is on the ground at current_loc, needs to reach goal_location.
            if status == 'at':
                current_loc = current_loc_or_vehicle
                # If the package is already at its goal location on the ground, cost is 0 for this package.
                if current_loc == goal_location:
                    continue

                # Needs pick-up, drive, drop.
                dist = self.get_distance(current_loc, goal_location)
                if dist == float('inf'):
                    # Goal is unreachable for this package.
                    return float('inf')
                total_cost += 1  # pick-up action
                total_cost += dist # drive actions
                total_cost += 1  # drop action

            # Case 3: Package is inside a vehicle, needs to reach goal_location.
            elif status == 'in':
                vehicle = current_loc_or_vehicle
                # Find the vehicle's location.
                if vehicle not in vehicle_locations:
                    # Vehicle location unknown. Cannot move package.
                    return float('inf')

                vehicle_loc = vehicle_locations[vehicle]

                # If the vehicle is already at the package's goal location, needs only drop.
                if vehicle_loc == goal_location:
                    total_cost += 1 # drop action
                else:
                    # Needs drive and drop.
                    dist = self.get_distance(vehicle_loc, goal_location)
                    if dist == float('inf'):
                        # Goal is unreachable for this package (via this vehicle's current path).
                        return float('inf')
                    total_cost += dist # drive actions
                    total_cost += 1  # drop action

        return total_cost

# Note: To run this code standalone for testing, you would need dummy
# Task and Node classes that mimic the structure used by the planner.
# The provided code assumes these classes exist and are used to pass
# the task definition and current state to the heuristic.
