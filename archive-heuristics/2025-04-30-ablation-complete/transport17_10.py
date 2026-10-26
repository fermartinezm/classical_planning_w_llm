# Required imports
from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for invalid facts
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package to its goal location, summing the costs
    independently for each package. The cost includes picking up the package
    (if on the ground), driving the required distance, and dropping the package
    at the goal location. Vehicle capacity and availability are ignored.

    # Assumptions
    - The goal is to have specific packages at specific locations, represented by `(at ?p ?l)` predicates in the goal.
    - Any vehicle can pick up any package (ignoring size/capacity constraints).
    - A vehicle is always available when needed at the package's location or
      the vehicle's location if the package is in transit.
    - The cost of driving between two locations is the shortest path distance
      on the road network, where each road segment costs 1 drive action.
    - Roads are bidirectional as defined in the domain file.
    - Objects starting with 'v' are vehicles, others are packages (based on common naming conventions in this domain).

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals. Only `(at ?p ?l)` goals are considered.
    - Build the road network graph from the static facts (`road` predicates).
    - Compute all-pairs shortest paths on the road network graph using BFS to find
      distances between any two locations. These distances represent the minimum
      number of `drive` actions required.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package that has a goal.
       Also, identify the current location of every vehicle. This information is parsed from the state facts (`at` and `in` predicates).
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `goal_l` (extracted during initialization):
       a. Check if the goal fact `(at p goal_l)` is already true in the current state.
          If it is, this package is already at its final destination, and contributes 0 to the heuristic.
       b. If the goal fact is not true, find the package's current state (`(at p current_l)` or `(in p v)`).
       c. Determine the package's effective current location (`current_l`). If the package is `in` a vehicle, its location is the vehicle's location.
       d. If the package's state or location cannot be determined (e.g., vehicle location unknown), or if the current or goal location is not part of the road network graph, add a large penalty to the total heuristic. This indicates a potentially unreachable or invalid state.
       e. If the locations are valid and reachable (distance is finite):
          - Get the shortest path distance (`distance`) between `current_l` and `goal_l` from the precomputed distances.
          - If the package is currently on the ground at `current_l` (`(at p current_l)` was found in step 3b):
            - The estimated cost for this package is 1 (pick-up action) + `distance` (drive actions) + 1 (drop action).
            - Add this cost to the total heuristic.
          - If the package is currently inside a vehicle at `current_l` (`(in p v)` was found in step 3b):
            - The estimated cost for this package is `distance` (drive actions) + 1 (drop action).
            - Add this cost to the total heuristic.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph to compute shortest paths.
        """
        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        # task.goals can be a frozenset of literals (for conjunctions) or a single literal string
        if isinstance(task.goals, frozenset):
             goal_literals = task.goals
        else: # Single goal literal
             goal_literals = {task.goals} # Treat as a set with one element

        for goal_literal in goal_literals:
            predicate, *args = get_parts(goal_literal)
            # We only care about (at ?p ?l) goals for packages
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Assuming objects starting with 'p' are packages based on examples
                if package.startswith('p'):
                    self.goal_locations[package] = location
            # Ignore other goal types or non-package 'at' goals

        # 2. Build the road network graph.
        self.road_graph = defaultdict(list)
        locations = set()
        for fact in task.static:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2: # Ensure it's a (road l1 l2) predicate
                l1, l2 = args
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Assuming roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store list of locations

        # 3. Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Perform Breadth-First Search to find shortest distances from start_node
        to all other nodes in the road graph.
        Returns a dictionary mapping reachable nodes to their distance from start_node.
        Nodes not reachable will have distance float('inf').
        """
        distances = {node: float('inf') for node in self.locations}
        if start_node not in distances:
             # Start node is not in the graph (e.g., isolated location not mentioned in roads)
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Map current state of packages and vehicles
        package_current_state = {} # {package: ('at', loc) or ('in', vehicle)}
        vehicle_at_loc = {} # {vehicle: location}

        # First pass: find vehicle locations and package states (at/in)
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                # Assuming objects starting with 'v' are vehicles, others are packages
                if obj.startswith('v'):
                     vehicle_at_loc[obj] = loc
                elif obj.startswith('p'): # Assuming objects starting with 'p' are packages
                     package_current_state[obj] = ('at', loc)
                # Ignore other 'at' predicates if any
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                # Assuming objects starting with 'p' are packages
                if package.startswith('p') and vehicle.startswith('v'):
                    package_current_state[package] = ('in', vehicle)
                # Ignore other 'in' predicates if any

        total_cost = 0
        LARGE_PENALTY = 1000 # Penalty for unreachable or unknown states

        # Iterate through packages that have a goal location
        for package, goal_l in self.goal_locations.items():
            # Check if the goal is already satisfied for this package
            if f"(at {package} {goal_l})" in state:
                continue # Package is already at its goal, cost is 0 for this package

            # Find the package's current location/state
            if package not in package_current_state:
                 total_cost += LARGE_PENALTY
                 continue

            state_type, obj_or_loc = package_current_state[package]
            current_l = None

            if state_type == 'at':
                current_l = obj_or_loc # Package is on the ground at this location
            elif state_type == 'in':
                vehicle = obj_or_loc
                if vehicle in vehicle_at_loc:
                    current_l = vehicle_at_loc[vehicle] # Package is in vehicle at vehicle's location
                else:
                    total_cost += LARGE_PENALTY # Vehicle location unknown
                    continue # Cannot estimate further for this package

            # Check if current_l and goal_l are valid locations in our graph
            if current_l not in self.locations or goal_l not in self.locations:
                 total_cost += LARGE_PENALTY # Location not in graph
                 continue

            # Get the distance between current location and goal location
            # Check if current_l exists as a key in self.distances before accessing
            if current_l not in self.distances or goal_l not in self.distances[current_l]:
                 # This should ideally not happen if current_l and goal_l are in self.locations,
                 # but defensive check in case BFS failed or graph is weird.
                 total_cost += LARGE_PENALTY
                 continue

            distance = self.distances[current_l][goal_l]

            if distance == float('inf'):
                 # No path exists between current location and goal location
                 total_cost += LARGE_PENALTY # Unreachable goal
                 continue

            # Calculate cost based on current state
            if state_type == 'at':
                # Package is on the ground at current_l, needs pick + drive + drop
                total_cost += 1 # pick-up
                total_cost += distance # drive actions (1 per road segment)
                total_cost += 1 # drop
            elif state_type == 'in':
                # Package is in a vehicle at current_l, needs drive + drop
                total_cost += distance # drive actions (1 per road segment)
                total_cost += 1 # drop

        return total_cost
