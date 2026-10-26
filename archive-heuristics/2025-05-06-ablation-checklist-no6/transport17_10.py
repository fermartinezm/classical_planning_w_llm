from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the required number of actions to reach a goal state by summing
    the estimated minimum actions for each package not yet at its goal location.
    Assumes infinite vehicle capacity and availability for distance calculations,
    and does not explicitly model vehicle capacity constraints or the cost
    of moving vehicles to packages on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing location distances and
        extracting package goal locations.
        """
        super().__init__(task)

        # Extract package goal locations from the goal state
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.package_goals[package] = location

        # Build location graph and compute all-pairs shortest paths
        locations = set()
        roads = []

        # Collect locations and roads from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                roads.append((l1, l2))

        # Collect locations from initial state 'at' facts
        # This ensures we include locations that might not have roads connected in static facts
        # but appear in the problem (e.g., initial vehicle/package locations).
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # The second argument is the location
                 locations.add(parts[2])

        # Build adjacency list graph
        graph = {l: [] for l in locations}
        for l1, l2 in roads:
            graph[l1].append(l2)

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in locations:
            self.dist[start_node] = {}
            queue = [(start_node, 0)]
            visited = {start_node}
            while queue:
                (current_loc, d) = queue.pop(0)
                self.dist[start_node][current_loc] = d
                # Check if current_loc is in graph before iterating neighbors
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        The heuristic sums the estimated cost for each package not at its goal.
        - If a package is on the ground at location L and needs to go to G:
          Estimated cost = 1 (pickup) + dist(L, G) (drive) + 1 (drop)
        - If a package is in a vehicle at location L and needs to go to G:
          Estimated cost = dist(L, G) (drive) + 1 (drop)
        - If a package is at its goal location: Cost is 0.

        Capacity constraints and vehicle availability for pickup are ignored.
        Drive costs are potentially double-counted if multiple packages share a vehicle/route.
        Returns float('inf') if a goal location is unreachable from the package's
        current location (or its vehicle's location).
        """
        state = node.state

        # Parse current locations/containment from the state
        current_locations = {} # obj -> loc (for packages and vehicles)
        package_in_vehicle = {} # package -> vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                p, v = parts[1], parts[2]
                package_in_vehicle[p] = v

        h = 0
        # Iterate through each package that has a goal location
        for package, goal_location in self.package_goals.items():

            # Check if package is already at its goal location in the current state
            if f"(at {package} {goal_location})" in state:
                continue # Package is at goal, cost is 0 for this package

            # If package is not at goal, estimate cost based on its current state

            # Case 1: Package is on the ground
            if package in current_locations:
                 current_l = current_locations[package]
                 # Package is on the ground at current_l, needs to go to goal_location
                 # Estimated cost: pickup (1) + drive (dist) + drop (1)
                 drive_cost = self.dist.get(current_l, {}).get(goal_location, float('inf'))

                 if drive_cost == float('inf'):
                     # Goal is unreachable from current location on the ground
                     # This state is likely unsolvable or requires complex vehicle positioning
                     # Return infinity or a very large number to prune this path
                     return float('inf')

                 h += 2 + drive_cost

            # Case 2: Package is in a vehicle
            elif package in package_in_vehicle:
                 vehicle = package_in_vehicle[package]
                 if vehicle in current_locations:
                     vehicle_l = current_locations[vehicle]
                     # Package is in vehicle at vehicle_l, needs to go to goal_location
                     # Estimated cost: drive (dist) + drop (1)
                     drive_cost = self.dist.get(vehicle_l, {}).get(goal_location, float('inf'))

                     if drive_cost == float('inf'):
                         # Goal is unreachable from vehicle's current location
                         return float('inf')

                     h += 1 + drive_cost
                 else:
                     # Package is in a vehicle, but vehicle location is unknown.
                     # This indicates an inconsistent state representation or a problem
                     # with state parsing. Treat as unreachable goal.
                     return float('inf')
            # Case 3: Package is not at a location and not in a vehicle.
            # This indicates an inconsistent state representation.
            # Treat as unreachable goal.
            else:
                 return float('inf')

        return h
