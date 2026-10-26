# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic # This import should be present in the actual environment

import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # In a real system, you might log a warning or raise an error
         # print(f"Warning: Unexpected fact format: {fact}")
         return []
    return fact[1:-1].split()

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

def bfs(graph, start):
    """
    Performs BFS on a graph to find shortest distances from a start node.
    graph: adjacency dictionary {node: [neighbor1, neighbor2, ...]}
    start: the starting node
    Returns: dictionary {node: distance}
    """
    distances = {node: float('inf') for node in graph}
    if start not in graph:
        # Start node not in graph, no paths possible from here
        return distances

    distances[start] = 0
    queue = collections.deque([start])

    while queue:
        current = queue.popleft()
        # Ensure current node is still valid in graph keys (should be if graph is built correctly)
        if current not in graph:
             continue
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

def all_pairs_shortest_paths(graph):
    """
    Computes shortest paths between all pairs of nodes in a graph.
    graph: adjacency dictionary {node: [neighbor1, neighbor2, ...]}
    Returns: dictionary {start_node: {end_node: distance}}
    """
    all_distances = {}
    # Collect all unique nodes that appear as keys or values in the graph
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    # Create a graph structure that includes all identified nodes, even if isolated
    graph_inclusive = {node: graph.get(node, []) for node in all_nodes}

    for start_node in graph_inclusive:
        all_distances[start_node] = bfs(graph_inclusive, start_node)
    return all_distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, ignoring vehicle capacity constraints
    beyond the implicit single-package assumption suggested by the PDDL,
    and assuming a vehicle is always available when needed. The cost for
    each package is calculated independently and summed up.

    # Assumptions:
    - Each vehicle can carry at most one package at a time (inferred from
      the capacity predicate structure).
    - A suitable vehicle is always available at the required location
      to pick up or drop off a package.
    - The 'drive' action cost is 1, and the cost of picking up or dropping
      a package is 1.
    - The shortest path distance between locations is used for drive costs.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representing the locations and roads from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state is the goal state. If yes, the heuristic is 0.
    2. Identify the current location of every locatable object (vehicles and packages).
       This includes packages on the ground (`(at p l)`) and packages inside vehicles (`(in p v)`),
       finding the vehicle's location (`(at v l)`).
    3. Initialize the total heuristic cost to 0.
    4. For each package whose goal location is specified in the task goals:
       a. Check if the package is already at its goal location in the current state. If yes,
          this package contributes 0 to the heuristic and we move to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current physical location (`l_current`). This is either
             its location if it's on the ground, or the location of the vehicle it's in.
             If the package's location cannot be determined from the state, skip this package
             or return infinity if this indicates an unsolvable state.
          ii. Get the package's goal location (`l_goal`).
          iii. Find the shortest path distance (`d`) between `l_current` and `l_goal` using
              the precomputed distances. If no path exists between `l_current` and `l_goal`,
              the distance is infinity, indicating the goal is unreachable for this package
              from its current location.
          iv. If the distance `d` is infinity, the state is likely unsolvable, so return
              infinity for the total heuristic cost.
          v. Calculate the estimated cost for this package:
              - If the package is on the ground: It needs a pick-up (1 action), driving
                from `l_current` to `l_goal` (`d` actions), and a drop (1 action).
                Total cost: `d + 2`.
              - If the package is inside a vehicle: It needs driving from `l_current`
                to `l_goal` (`d` actions) and a drop (1 action). Total cost: `d + 1`.
          vi. Add this estimated cost to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                # Goal is (at package location)
                package, location = args
                self.goal_locations[package] = location
            # Ignore other goal predicates if any

        # Build the location graph from road facts.
        self.location_graph = collections.defaultdict(list)
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2:
                loc1, loc2 = args
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Roads are bidirectional

        # Precompute all-pairs shortest path distances.
        # The all_pairs_shortest_paths helper is designed to include all nodes
        # found in the graph keys/values, ensuring comprehensive distance calculation.
        self.distances = all_pairs_shortest_paths(self.location_graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Track where locatable objects (vehicles, packages on ground) are currently located.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                # (at object location)
                obj, location = args
                current_locations[obj] = location

        # Track which package is in which vehicle.
        package_in_vehicle = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "in" and len(args) == 2:
                # (in package vehicle)
                package, vehicle = args
                package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location specified
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location
            if match(f"(at {package} {goal_location})", "at", package, goal_location) in state:
                 continue # Package is already at goal, contributes 0 cost

            # Find the package's current physical location
            l_current = None
            if package in package_in_vehicle:
                # Package is in a vehicle, its location is the vehicle's location
                vehicle = package_in_vehicle[package]
                if vehicle in current_locations:
                    l_current = current_locations[vehicle]
                # else: vehicle location unknown? This state might be invalid or unsolvable.
            elif package in current_locations:
                # Package is on the ground
                l_current = current_locations[package]
            # else: package location unknown? This state might be invalid or unsolvable.

            if l_current is None:
                 # If a package's current location cannot be determined, it's likely an unsolvable state
                 # or a state representation issue. Return infinity.
                 # print(f"Warning: Location of package {package} is unknown in state. Returning inf.")
                 return float('inf')


            # Get the shortest path distance from current location to goal location
            # Use .get() with default {} to handle cases where l_current might not be in self.distances keys
            # (e.g., if l_current was not part of any road fact or goal location)
            distance = self.distances.get(l_current, {}).get(goal_location, float('inf'))

            # If goal is unreachable from the package's current location, return infinity for the whole state
            if distance == float('inf'):
                 # print(f"Warning: Goal location {goal_location} unreachable from {l_current} for package {package}. Returning inf.")
                 return float('inf')

            # Calculate cost based on package's state (in vehicle or on ground)
            if package in package_in_vehicle:
                # Package is in a vehicle: needs drive + drop
                cost_for_package = distance + 1
            else:
                # Package is on the ground: needs pick-up + drive + drop
                cost_for_package = distance + 2

            total_cost += cost_for_package

        return total_cost
