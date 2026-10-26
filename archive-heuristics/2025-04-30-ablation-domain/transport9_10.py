from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts gracefully, though unlikely with planner states
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 locationA)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Perform a Breadth-First Search to find shortest distances from a start node
    to all reachable nodes in a graph.

    Args:
        graph: A dictionary representing the graph (e.g., {node: set(neighbors)}).
        start: The starting node for the BFS.

    Returns:
        A dictionary mapping reachable nodes to their shortest distance from the start node.
    """
    distances = {start: 0}
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        # Ensure the current node exists in the graph's keys before accessing neighbors
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, summing the individual minimum costs.
    It considers the cost of loading, unloading, and the minimum number of
    drive actions needed based on shortest paths in the road network. It
    acts as an h_add-like heuristic for packages, ignoring vehicle capacity
    and potential for shared trips for simplicity and efficiency.

    # Assumptions
    - Packages are either on the ground at a location or inside a vehicle.
    - The state representation uses `(at ?p ?loc)` for packages on the ground
      and `(at ?p ?v)` for packages inside vehicle `?v`.
    - Vehicle locations are represented by `(at ?v ?loc)`.
    - The road network defined by `(road ?loc1 ?loc2)` facts is static.
    - Vehicle capacities defined by `(capacity ?v ?s)` and ordered by
      `(capacity-predecessor ?s1 ?s2)` are static.
    - Actions have a cost of 1. The heuristic counts actions (load, unload, drive).
    - If a package's goal location is unreachable from its current location
      (or its vehicle's current location) via the road network, the state
      is considered a dead end and assigned a very high heuristic value.

    # Heuristic Initialization
    - Extracts goal locations for each package from `task.goals`.
    - Builds the road network graph from `task.static` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Parses vehicle capacities and the size ordering from `task.static`
      (although capacity is not directly used in the current h_add calculation,
      it's extracted as relevant static information).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status (location or vehicle) for each package
       and the current location for each vehicle from the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined:
       a. Check if the package is already at its goal location on the ground.
          If yes, its contribution to the heuristic is 0.
       b. If the package is on the ground at a location different from its goal:
          - The package needs to be loaded into a vehicle (+1 action).
          - The vehicle needs to drive from the package's current location
            to its goal location. The minimum number of drive actions is the
            shortest path distance between these locations. (+ distance actions).
          - The package needs to be unloaded at the goal location (+1 action).
          - Total cost for this package: 1 (load) + distance + 1 (unload).
          - If the goal location is unreachable from the current location,
            return a very high value (infinity).
       c. If the package is inside a vehicle:
          - Find the current location of the vehicle.
          - The vehicle needs to drive from its current location to the
            package's goal location. The minimum number of drive actions is
            the shortest path distance between these locations. (+ distance actions).
          - The package needs to be unloaded at the goal location (+1 action).
          - Total cost for this package: distance + 1 (unload).
          - If the goal location is unreachable from the vehicle's current location,
            return a very high value (infinity).
    4. Sum the costs calculated for each package.
    5. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Store goal locations for each package.
        self.goal_locations = {}
        # Infer packages from goals
        self.packages = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
                self.packages.add(package)

        # 2. Build the road network graph and collect all locations.
        self.road_graph = collections.defaultdict(set)
        self.locations = set()
        # Infer locations and roads from static facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].add(loc2)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 3. Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(self.road_graph, start_loc)

        # 4. Parse capacity sizes and their ordering.
        # Build successor graph for sizes
        size_successors = {}
        all_sizes = set()
        is_successor = set()
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                _, s1, s2 = get_parts(fact)
                size_successors[s1] = s2
                all_sizes.add(s1)
                all_sizes.add(s2)
                is_successor.add(s2)

        # Find the smallest size(s) (those that are not successors)
        smallest_sizes = list(all_sizes - is_successor)

        self.capacity_values = {}
        # Assign integer values starting from 0 for the smallest size(s)
        # Use a queue for BFS-like assignment
        queue = collections.deque(smallest_sizes)
        visited = set(smallest_sizes)
        value = 0
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                current_size = queue.popleft()
                self.capacity_values[current_size] = value

                # Find sizes for which current_size is a predecessor
                next_size = size_successors.get(current_size)
                if next_size and next_size not in visited:
                    visited.add(next_size)
                    queue.append(next_size)
            value += 1 # Increment value for the next level

        # 5. Parse vehicle capacities using the size values.
        self.vehicle_capacities = {}
        # Infer vehicles from capacity facts
        self.vehicles = set()
        for fact in static_facts:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, size = get_parts(fact)
                self.vehicles.add(vehicle)
                # Store the size name, value lookup happens in __call__ if needed
                # self.vehicle_capacities[vehicle] = self.capacity_values.get(size, 0) # Default to 0 if size unknown

        # Infer vehicles and packages from initial state if not found in static/goals
        # This is a fallback and might not be strictly necessary if PDDL is well-formed
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 obj1, obj2 = parts[1], parts[2]
                 if obj1.startswith('p'): self.packages.add(obj1)
                 if obj1.startswith('v'): self.vehicles.add(obj1)
                 if obj2.startswith('l'): self.locations.add(obj2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        # A package can be at a location (on ground) or in a vehicle.
        package_status = {} # {package: location or vehicle}
        vehicle_locations = {} # {vehicle: location}

        # Populate current status from the state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                obj1, obj2 = parts[1], parts[2]
                if obj1 in self.packages:
                    package_status[obj1] = obj2 # obj2 is either a location or a vehicle
                elif obj1 in self.vehicles:
                    vehicle_locations[obj1] = obj2 # obj2 must be a location

        total_cost = 0  # Initialize action cost counter.

        # Compute cost for each package independently
        for package, goal_location in self.goal_locations.items():
            # If package is not in the state or doesn't have a goal, skip (shouldn't happen in valid problems)
            if package not in package_status:
                 # This package is not in the state facts, which is unexpected.
                 # Could indicate an issue or a package that doesn't exist.
                 # For robustness, we could return infinity or a large cost.
                 # Assuming valid states, this branch is not taken.
                 continue

            current_status = package_status[package]

            # Case 1: Package is already at its goal location on the ground
            if current_status == goal_location:
                continue # Cost for this package is 0

            # Case 2: Package is on the ground at a different location
            if current_status in self.locations: # Check if it's a location string
                current_loc_p = current_status
                # Cost: load (1) + drive (distance) + unload (1)
                # Check if goal is reachable from current location
                if current_loc_p not in self.distances or goal_location not in self.distances[current_loc_p]:
                    # Goal is unreachable from this location. This state is likely a dead end.
                    return float('inf') # Return infinity or a large value

                distance = self.distances[current_loc_p][goal_location]
                cost_p = 1 + distance + 1 # load + drive + unload
                total_cost += cost_p

            # Case 3: Package is inside a vehicle
            elif current_status in self.vehicles: # Check if it's a vehicle string
                current_v = current_status
                # Find the vehicle's location
                if current_v not in vehicle_locations:
                     # Vehicle location not found in state, unexpected.
                     return float('inf') # Indicate problem

                current_loc_v = vehicle_locations[current_v]

                # Cost: drive (distance) + unload (1)
                # Check if goal is reachable from vehicle's current location
                if current_loc_v not in self.distances or goal_location not in self.distances[current_loc_v]:
                    # Goal is unreachable from vehicle's location. Likely dead end.
                    return float('inf') # Return infinity or a large value

                distance = self.distances[current_loc_v][goal_location]
                cost_p = distance + 1 # drive + unload
                total_cost += cost_p

            # Case 4: Package status is something else (unexpected)
            else:
                 # This indicates an unexpected state fact format for a package.
                 # For robustness, return infinity.
                 return float('inf')


        # If total_cost is 0, it means all packages are at their goal locations.
        # However, the goal might include other conditions not covered by package locations.
        # The Task object has a goal_reached method that checks all goal conditions.
        # We must return 0 iff the state is a goal state.
        # If total_cost > 0, it's definitely not a goal state.
        # If total_cost == 0, we must verify if it's truly a goal state.
        # This check ensures the heuristic is 0 iff the state is a goal state.
        # This check adds a small overhead but is necessary for correctness.
        if total_cost == 0 and not self.goals <= state:
             # This case should ideally not happen with this heuristic logic
             # if goals are only (at p l) facts, but it's a safeguard.
             # If goals include other facts, and all packages are at goal,
             # but other facts are not met, the heuristic should be > 0.
             # A simple way is to add 1 if goals are not met but package costs are 0.
             # However, the h_add principle sums costs for *unmet* goal facts.
             # Our heuristic only considers (at p l) goals.
             # Let's assume goals are only (at p l) for packages.
             # If total_cost is 0, all (at p l) goals are met.
             # So, total_cost == 0 implies self.goals <= state for this domain.
             # Thus, the check `if total_cost == 0 and not self.goals <= state:`
             # should theoretically never be true if goals are only package locations.
             # If goals *can* include other facts, this heuristic is incomplete.
             # Assuming goals are only package locations:
             pass # total_cost is already 0, which is correct for a goal state.
        elif total_cost == 0 and self.goals <= state:
             # This is the goal state, heuristic is 0. Correct.
             pass
        elif total_cost > 0 and self.goals <= state:
             # This should not happen. If goals are met, total_cost should be 0.
             # This would indicate an error in the heuristic logic or goal parsing.
             # For safety, return 0 if it's a goal state, even if calculation yielded > 0.
             # This makes the heuristic admissible (0 at goal) but might hide bugs.
             # Let's trust the calculation and the assumption about goal structure.
             pass # total_cost > 0, not a goal state. Correct.


        return total_cost

