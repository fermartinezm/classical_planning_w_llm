# Imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if needed

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip, which stops at the shortest sequence.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class
# If a base class 'Heuristic' is provided (e.g., from heuristics.heuristic_base),
# inherit from it: class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions required to move each package to its goal location.
    Ignores vehicle capacity and assumes vehicles are available when needed for ground packages.
    Uses shortest path (BFS) for drive costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network for shortest path calculations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        # If inheriting from Heuristic, call super().__init__(task)
        # super().__init__(task)
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goals are typically (at ?p ?l)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location
            # Ignore other potential goal types or malformed goals

        # Build the road graph from static facts.
        self.road_graph = self._build_road_graph(self.static)

    def _build_road_graph(self, static_facts):
        """Builds an adjacency list representation of the road network."""
        graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                graph.setdefault(loc1, []).append(loc2)
                # The example PDDL lists roads bidirectionally, so a directed graph is sufficient.
        return graph

    def bfs(self, start_loc, end_loc):
        """
        Performs BFS to find the shortest path distance between two locations.
        Returns the distance or float('inf') if unreachable.
        """
        if start_loc == end_loc:
            return 0

        # Collect all locations mentioned in the road graph
        all_locations_in_graph = set(self.road_graph.keys())
        for neighbors in self.road_graph.values():
             all_locations_in_graph.update(neighbors)

        # If start or end location is not part of the road network, they are unreachable from each other
        if start_loc not in all_locations_in_graph or end_loc not in all_locations_in_graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Get neighbors, handle locations with no outgoing roads gracefully
            neighbors = self.road_graph.get(current_loc, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # Unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # A state is a goal state if all facts in self.goals are true.
        # Check this first for efficiency and correctness (h=0 only at goal).
        if self.goals <= state:
             return 0

        # Track current status of packages and vehicles
        package_status = {} # {package: {'type': 'at'/'in', 'location': loc_or_vehicle}}
        vehicle_locations = {} # {vehicle: location}

        # Populate package_status and vehicle_locations from the current state
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, location = get_parts(fact)
                # Simple type check based on common naming convention in examples
                if obj.startswith('v'): # Assuming vehicle
                     vehicle_locations[obj] = location
                elif obj.startswith('p'): # Assuming package
                     package_status[obj] = {'type': 'at', 'location': location}
                # Ignore other 'at' facts if any

            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                # Assuming 'in' only applies to packages in vehicles
                if package.startswith('p') and vehicle.startswith('v'):
                     package_status[package] = {'type': 'in', 'location': vehicle}

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach its goal
        for package, goal_location in self.package_goals.items():
            status = package_status.get(package)

            # If a package with a goal is not in the state, it's likely an issue
            # or it might have been removed in a way not covered by standard actions (unlikely in PDDL).
            # Treat as unreachable for safety.
            if status is None:
                 return float('inf')

            cost_for_package = 0 # Cost for this specific package

            if status['type'] == 'at':
                current_location = status['location']
                # If package is at its goal location on the ground, cost is 0 for this package.
                if current_location != goal_location:
                    # Package is on the ground, not at goal.
                    # Needs pick-up, drive, drop.
                    # Assume a vehicle is available at current_location.
                    drive_cost = self.bfs(current_location, goal_location)
                    if drive_cost == float('inf'):
                        return float('inf') # Goal unreachable for this package
                    cost_for_package = drive_cost + 2 # +1 for pick-up, +1 for drop

            elif status['type'] == 'in':
                vehicle = status['location'] # This is the vehicle name
                current_vehicle_location = vehicle_locations.get(vehicle)

                # If package is in a vehicle, but vehicle location is unknown, treat as unreachable.
                if current_vehicle_location is None:
                    return float('inf')

                # If package is in vehicle, and vehicle is at goal location, needs 1 drop action.
                # If package is in vehicle, and vehicle is not at goal, needs drive and drop.
                if current_vehicle_location != goal_location:
                    # Needs drive (with package), drop.
                    drive_cost = self.bfs(current_vehicle_location, goal_location)
                    if drive_cost == float('inf'):
                         return float('inf') # Goal unreachable for this package
                    cost_for_package = drive_cost + 1 # +1 for drop
                else: # current_vehicle_location == goal_location
                    # Needs drop.
                    cost_for_package = 1 # +1 for drop

            # If any package is unreachable, the total cost is infinity
            if cost_for_package == float('inf'):
                 return float('inf')

            total_cost += cost_for_package

        # The total heuristic is the sum of costs for each package.
        # The check `if self.goals <= state: return 0` at the beginning handles the case
        # where all packages are already at their goal locations on the ground.
        # If the state is not a goal state, total_cost will be > 0 (assuming reachable).

        return total_cost
