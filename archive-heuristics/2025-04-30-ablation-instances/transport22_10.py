from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path on road network
def bfs(start, graph):
    """Computes shortest path distances from start node to all reachable nodes in a graph."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        # Check if current is in graph. Sometimes nodes might be in distances but not have edges.
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, nodes):
    """Computes shortest path distances between all pairs of nodes in a graph."""
    all_distances = {}
    # Ensure all nodes are included even if they have no roads defined (isolated nodes)
    full_graph = {node: graph.get(node, []) for node in nodes}
    for start_node in nodes:
        all_distances[start_node] = bfs(start_node, full_graph)
    return all_distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    package independently, ignoring vehicle capacity constraints and vehicle
    availability beyond what is needed for a single package's journey.
    The estimated cost for a single package not at its goal is:
    (1 if package is on the ground) + (shortest road distance from package's
    current physical location to its goal location) + (1 for dropping the package).

    # Assumptions:
    - The road network is static and provides connections between locations.
    - Packages can only be moved by vehicles.
    - Vehicle capacity is ignored for the purpose of summing costs per package.
      The heuristic assumes a vehicle is available and has capacity when needed
      for a package's individual journey estimate.
    - The cost of each action (drive, pick-up, drop) is 1.
    - The goal is defined primarily by a set of (at ?p ?l) facts for specific packages.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Build the road network graph from the static 'road' facts.
    - Compute all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the goal location for each package that needs to be transported (based on the task goals).
    2. For each such package:
       a. Check if the package is already at its goal location in the current state. If yes, its cost is 0.
       b. If not at the goal, determine its current status: Is it on the ground at a location `l`, or is it inside a vehicle `v`?
       c. If it's inside a vehicle `v`, find the current physical location `l_v` of that vehicle. The package's current physical location is `l_v`.
       d. If it's on the ground at location `l`, its current physical location is `l`.
       e. Calculate its estimated cost:
          - Add 1 if the package is currently on the ground (for the pick-up action).
          - Find the shortest road distance from its current physical location to its goal location. Add this distance. If the goal is unreachable, the cost is infinity.
          - Add 1 for the drop action (needed once the vehicle reaches the goal location).
    3. The total heuristic value is the sum of the estimated costs for all packages that are not yet at their goal locations. If any package's goal is unreachable, the total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network, and computing shortest paths.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in task.goals:
            # Assuming goals are primarily (at ?p ?l) for packages
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # We assume objects in 'at' goals are packages based on domain structure
                 self.package_goals[obj] = loc

        # Build the road network graph and list of all locations.
        self.road_network = {}
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_network:
                    self.road_network[l1] = []
                self.road_network[l1].append(l2)
                self.locations.add(l1)
                self.locations.add(l2)

        # Compute all-pairs shortest path distances.
        # Ensure all locations mentioned in goals or roads are included.
        all_relevant_locations = self.locations.union(set(self.package_goals.values()))
        self.distance = compute_all_pairs_shortest_paths(self.road_network, list(all_relevant_locations))

        # Capacity information is not used in this simple sum-of-costs heuristic
        # as it ignores vehicle constraints for the estimate.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where locatables (packages and vehicles) are currently located.
        # This includes both physical location (at) and containment (in).
        current_locations = {} # obj -> location
        package_in_vehicle = {} # package -> vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 package_in_vehicle[package] = vehicle
                 # Note: package's physical location is the vehicle's location,
                 # which we look up later using current_locations[vehicle].

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location
            if (f"(at {package} {goal_location})" in state):
                continue # Package is already delivered, cost is 0 for this package

            # Find the package's current physical location
            current_physical_location = None
            is_on_ground = True

            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle in current_locations:
                    current_physical_location = current_locations[vehicle]
                    is_on_ground = False
                else:
                    # Vehicle location unknown - indicates an invalid state or unreachable goal
                    return float('inf')
            elif package in current_locations:
                # Package is on the ground
                current_physical_location = current_locations[package]
                is_on_ground = True
            else:
                 # Package location unknown - indicates an invalid state or unreachable goal
                 return float('inf')

            # Calculate estimated cost for this package
            package_cost = 0

            # 1. Cost to pick up (if on ground)
            if is_on_ground:
                package_cost += 1 # pick-up action

            # 2. Cost to drive vehicle from current location to goal location
            # Check if locations are valid and reachable
            if current_physical_location not in self.distance or goal_location not in self.distance.get(current_physical_location, {}):
                 # Current or Goal location is not in the known locations or unreachable
                 return float('inf') # Cannot reach goal

            drive_cost = self.distance[current_physical_location][goal_location]

            if drive_cost == float('inf'):
                 # Goal location is unreachable via roads
                 return float('inf')

            package_cost += drive_cost # drive action(s)

            # 3. Cost to drop (always needed if not already at goal)
            package_cost += 1 # drop action

            total_cost += package_cost

        # The heuristic is 0 if and only if all goal packages are at their goal locations.
        # This is implicitly handled by the loop: if all packages in package_goals
        # are already at their goal, the loop continues for all of them, and total_cost remains 0.
        # If the task goal includes other predicates, this heuristic might return 0
        # even if the full task goal is not met. This is acceptable for a non-admissible
        # domain-dependent heuristic focused on the primary transport task.

        return total_cost
