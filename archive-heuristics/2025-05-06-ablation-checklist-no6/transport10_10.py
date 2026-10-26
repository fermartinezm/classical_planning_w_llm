# Import necessary modules
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
# This is just for ensuring the code structure is correct, the actual
# base class will be provided by the planning framework.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


# Helper functions (copy from Logistics example, slightly adapted for robustness)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if all provided args match the corresponding parts using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# BFS helper functions for shortest path calculation
def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'road' and len(parts) == 3:
            l1, l2 = parts[1], parts[2]
            graph.setdefault(l1, set()).add(l2)
            graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional
    return graph

def compute_shortest_paths(graph):
    """Computes shortest path distances from all locations to all other locations using BFS."""
    distances = {}
    # Collect all unique locations from the graph keys and values
    locations = set(graph.keys())
    for neighbors in graph.values():
        locations.update(neighbors)
    locations = list(locations) # Convert to list for consistent iteration order

    for start_node in locations:
        distances[start_node] = {}
        queue = deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            current_node, dist = queue.popleft()
            distances[start_node][current_node] = dist
            # Ensure current_node exists in the graph keys before accessing neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the cost to reach the goal by summing the minimum actions
    required for each package not yet at its goal location.

    # Assumptions:
    - Capacity constraints are ignored.
    - Vehicle availability is ignored.
    - Roads are bidirectional.
    - Shortest path distance is used for driving cost.
    - Each package movement is considered independently.

    # Heuristic Initialization
    - Extract goal locations for packages.
    - Build the road network graph from static facts.
    - Precompute shortest path distances between all locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not yet at its goal location:
    1. Determine the package's current state: Is it at a location or inside a vehicle?
    2. If the package is at a location `l`:
       - The estimated cost for this package is 1 (pick-up) + shortest_path(l, goal_l) (drive) + 1 (drop).
       - This assumes a vehicle is available at `l` and has capacity.
    3. If the package is inside a vehicle `v`:
       - Find the current location `vehicle_l` of vehicle `v`.
       - The estimated cost for this package is shortest_path(vehicle_l, goal_l) (drive) + 1 (drop).
       - This assumes the vehicle can drive directly to the goal location and has capacity to drop.
    4. Sum the estimated costs for all packages not at their goal.
    5. If any required shortest path does not exist, the state is considered unreachable, and the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and precomputing shortest path distances between locations.
        """
        # Call the base class constructor
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph from static facts.
        self.road_graph = build_road_graph(self.static)

        # Precompute shortest path distances between all locations.
        self.shortest_paths = compute_shortest_paths(self.road_graph)

        # Collect all known locations from the graph for quick lookup
        self._all_locations = set(self.road_graph.keys())
        for neighbors in self.road_graph.values():
            self._all_locations.update(neighbors)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Track where packages and vehicles are currently located.
        # This maps object name (package or vehicle) to its location (location name)
        # or the vehicle it is inside (vehicle name).
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                current_locations[obj] = location
            elif parts and parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                current_locations[package] = vehicle # Store the vehicle name

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location.
            if (f"(at {package} {goal_location})" in state):
                 continue # Package is already at the goal, cost is 0 for this package.

            # Package is not at the goal. Calculate cost to move it.
            current_location_or_vehicle = current_locations.get(package)

            if current_location_or_vehicle is None:
                 # This package is not found in the state (neither at nor in).
                 # Assume unreachable.
                 return float('inf')

            # Find the actual physical location of the package (either where it is, or where its vehicle is)
            package_physical_location = None
            is_in_vehicle = False

            if current_location_or_vehicle in self._all_locations:
                 # current_location_or_vehicle is a location name
                 package_physical_location = current_location_or_vehicle
                 is_in_vehicle = False
            else:
                 # current_location_or_vehicle is likely a vehicle name
                 vehicle_name = current_location_or_vehicle
                 package_physical_location = current_locations.get(vehicle_name)
                 is_in_vehicle = True

            if package_physical_location is None:
                 # Could not find the physical location (either package at unknown loc, or in vehicle at unknown loc)
                 # Assume unreachable.
                 return float('inf')

            # Get the shortest path cost from the package's current physical location to its goal location.
            drive_cost = self.get_shortest_path_cost(package_physical_location, goal_location)

            if drive_cost is None: # No path exists between physical location and goal
                 # Assume unreachable.
                 return float('inf')

            # Calculate cost based on whether the package is currently at a location or in a vehicle
            if is_in_vehicle:
                # Package is in a vehicle. Cost = drive + drop
                total_cost += drive_cost + 1 # drive + drop
            else:
                # Package is at a location. Cost = pick + drive + drop
                total_cost += 1 + drive_cost + 1 # pick + drive + drop


        return total_cost

    def get_shortest_path_cost(self, start_loc, end_loc):
        """
        Retrieves the precomputed shortest path cost between two locations.
        Returns None if either location is not in the graph or no path exists.
        """
        # Check if both locations are known locations in the graph.
        if start_loc not in self._all_locations or end_loc not in self._all_locations:
             return None # Indicate unreachable

        # Check if end_loc is reachable from start_loc in the precomputed distances.
        if start_loc in self.shortest_paths and end_loc in self.shortest_paths[start_loc]:
             return self.shortest_paths[start_loc][end_loc]
        else:
             # This happens if end_loc is not reachable from start_loc (different components).
             return None # Indicate unreachable
