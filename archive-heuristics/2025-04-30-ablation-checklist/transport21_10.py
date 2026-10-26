import collections
from fnmatch import fnmatch
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
    # Ensure the number of parts matches the number of arguments, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start):
    """
    Perform a Breadth-First Search on a graph to find shortest distances
    from a start node to all other reachable nodes.

    Args:
        graph: An adjacency list representation of the graph (dict: node -> list of neighbors).
        start: The starting node for the BFS.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = collections.deque([start])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes that might be in distances but not graph (e.g. if graph is empty)
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package independently.
    The cost for a package includes pick-up, drive actions (based on shortest path
    in the road network), and drop-off. It ignores vehicle capacity constraints
    and the need to move a vehicle to a package's location if it's on the ground.

    # Assumptions
    - The goal is to move packages to specific locations.
    - Any vehicle can transport any package (capacity is ignored).
    - A vehicle is available at the package's location when needed for pick-up.
    - Drive actions have a cost of 1, pick-up costs 1, drop costs 1.
    - The road network is unweighted.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build the road network graph from static facts.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the package's current status: on the ground at a location, or inside a vehicle.
    2. If the package is on the ground at `l_current` and `l_current` is not the goal `l_goal`:
       - Estimate cost as 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) (drive) + 1 (drop).
    3. If the package is inside a vehicle `v`, and `v` is at `l_v`:
       - If `l_v` is not the goal `l_goal`: Estimate cost as shortest_distance(`l_v`, `l_goal`) (drive) + 1 (drop).
       - If `l_v` is the goal `l_goal`: Estimate cost as 1 (drop).
    4. If the package is already on the ground at its goal location, the cost for this package is 0.
    5. The total heuristic value is the sum of the estimated costs for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph for shortest path calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].append(loc2)
                self.road_graph[loc2].append(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.goal_locations.values():
             locations.add(loc)
        # Also add locations mentioned in initial state 'at' facts for packages/vehicles
        # (This info isn't available in __init__ from the Task object directly,
        # but the locations set should ideally cover all possible locations in the problem)
        # A more robust way would be to parse the initial state from the task object
        # or rely on the fact that all relevant locations are mentioned in static/init/goal.
        # For now, assuming locations from roads and goals cover the necessary nodes.
        for loc in locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = [] # Add isolated locations to graph structure

        # Precompute all-pairs shortest paths.
        self.distances = {}
        for start_loc in self.road_graph:
            self.distances[start_loc] = bfs(self.road_graph, start_loc)

    def get_distance(self, loc1, loc2):
        """Safely get the precomputed distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This should ideally not happen if all locations are in the graph
             # but handle defensively. Infinite distance means unreachable.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        current_locations = {} # Maps locatable (pkg or vehicle) to location
        in_vehicle_map = {}    # Maps package to vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                locatable, location = parts[1], parts[2]
                current_locations[locatable] = location
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                in_vehicle_map[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if package in current_locations and current_locations[package] == goal_location:
                continue # Package is already at the goal location on the ground

            # Package is not at its goal location on the ground.
            # It's either on the ground elsewhere or inside a vehicle.

            if package in current_locations:
                # Package is on the ground at current_locations[package]
                current_location = current_locations[package]
                # Cost: pick-up (1) + drive (dist) + drop (1)
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'):
                    return float('inf') # Goal unreachable
                total_cost += 1 + dist + 1

            elif package in in_vehicle_map:
                # Package is inside a vehicle
                vehicle = in_vehicle_map[package]
                # Find the vehicle's location
                if vehicle not in current_locations:
                    # This state should ideally not happen in a valid problem
                    # A package cannot be in a vehicle if the vehicle has no location
                    # Return infinity or a large cost to penalize this state
                    return float('inf') # Invalid state or unreachable goal

                vehicle_location = current_locations[vehicle]

                # Cost: drive from vehicle_location to goal (dist) + drop (1)
                dist = self.get_distance(vehicle_location, goal_location)
                if dist == float('inf'):
                     return float('inf') # Goal unreachable
                total_cost += dist + 1

            else:
                 # Package is not 'at' any location and not 'in' any vehicle.
                 # This state should not be reachable in a valid problem unless
                 # the package somehow disappeared. Treat as unreachable goal.
                 return float('inf')


        return total_cost

