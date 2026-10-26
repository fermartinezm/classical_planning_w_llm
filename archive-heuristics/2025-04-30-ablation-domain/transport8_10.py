from fnmatch import fnmatch
import collections
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args if no wildcards are used
    # Or handle wildcards carefully. fnmatch handles this.
    # Simple check: if args is longer than parts, it can't match.
    if len(args) > len(parts):
        return False
    # Use zip to handle cases where parts might be longer than args (extra arguments ignored by match)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest distances from a source
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from a start_node in a graph using BFS.
    Assumes edge weights are 1.
    """
    distances = {start_node: 0}
    queue = collections.deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    that is not yet at its goal location (on the ground) to its goal location.
    It sums the estimated costs for each misplaced package independently,
    relaxing vehicle capacity and availability constraints.

    # Assumptions:
    - The cost of a 'drive' action between adjacent locations is 1.
    - The cost of 'load' and 'unload' actions is 1.
    - Vehicle capacity constraints are ignored for the heuristic calculation.
    - Vehicle availability is ignored; any package can be moved by an "ideal"
      vehicle that only incurs drive cost based on shortest path distance.
    - The goal for a package is to be on the ground at its goal location.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from 'road' facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every object (packages and vehicles)
       that is 'at' a location or 'in' a vehicle.
    2. For each package specified in the task goals:
       a. Check if the package is already at its goal location on the ground
          (i.e., the fact '(at package goal_location)' is in the state). If yes,
          the cost for this package is 0.
       b. If the package is not at its goal location on the ground, determine
          its current effective location:
          - If the package is 'at' a location, its effective location is that location.
          - If the package is 'in' a vehicle, its effective location is the location
            of that vehicle.
          - If the package's location cannot be determined (e.g., vehicle location unknown),
            the state might be unreachable or invalid; return infinity.
       c. Calculate the shortest path distance from the package's current effective
          location to its goal location using the precomputed distances. If the goal
          is unreachable from the current effective location, return infinity.
       d. Estimate the minimum actions needed for this package:
          - If the package is on the ground at its current effective location:
            It needs a 'load' action (1), a sequence of 'drive' actions (distance),
            and an 'unload' action (1). Total: 1 + distance + 1 = 2 + distance.
          - If the package is inside a vehicle at its current effective location:
            It needs a sequence of 'drive' actions (distance) and an 'unload'
            action (1). Total: distance + 1.
    3. The total heuristic value is the sum of the estimated costs for all
       misplaced packages.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task) # Call the base class constructor

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build road graph and compute all-pairs shortest paths
        self.road_graph = collections.defaultdict(list)
        self.locations = set()
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure fact has correct structure
                    _, loc1, loc2 = parts
                    self.road_graph[loc1].append(loc2)
                    self.locations.add(loc1)
                    self.locations.add(loc2)

        self.distances = {}
        # Compute shortest paths from every location to every other location
        for start_loc in self.locations:
            self.distances.update({(start_loc, end_loc): dist
                                   for end_loc, dist in bfs_shortest_paths(self.road_graph, start_loc).items()})

        # Capacity information is ignored for this heuristic's calculation,
        # but could be extracted here if needed for a more complex heuristic.
        # self.vehicle_capacities = {}
        # self.size_levels = {}
        # ... (parsing capacity and capacity-predecessor facts)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Extract current locations of objects and package contents of vehicles
        current_at = {} # obj -> location
        current_in = {} # package -> vehicle
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_at[obj] = loc
            elif parts and parts[0] == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                current_in[pkg] = veh

        total_cost = 0

        # Iterate through all packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})") in state:
                # This package goal is satisfied in this state
                continue

            # The package is not at its goal location on the ground. It needs to be moved.
            current_effective_location = None
            is_in_vehicle = False

            if package in current_at:
                # Package is on the ground at current_at[package]
                current_effective_location = current_at[package]
                is_in_vehicle = False
            elif package in current_in:
                # Package is inside a vehicle. Its effective location is the vehicle's location.
                vehicle = current_in[package]
                if vehicle in current_at:
                    current_effective_location = current_at[vehicle]
                    is_in_vehicle = True
                else:
                    # Vehicle carrying the package is not at any known location.
                    # This state might be invalid or unreachable.
                    # Return infinity as a high heuristic cost.
                    return float('inf')
            else:
                # Package is neither 'at' a location nor 'in' a vehicle. Invalid state?
                # Return infinity as a high heuristic cost.
                return float('inf')

            # Ensure the current effective location is a known location in the graph
            # If the package is at a location not in the road graph, it's likely isolated.
            # If its goal is also this location, the goal check above handles it (cost 0).
            # If its goal is different, it's unreachable.
            if current_effective_location not in self.locations:
                 if current_effective_location != goal_location:
                      return float('inf')
                 # If current_effective_location == goal_location, it means the package is at its goal
                 # but not necessarily on the ground, and the location is isolated.
                 # The initial check `(f"(at {package} {goal_location})") in state` handles the case
                 # where it's on the ground at the goal. If it's in a vehicle at the goal,
                 # it still needs unloading, but the location is isolated, so the vehicle
                 # couldn't have driven there from anywhere else. This case is tricky.
                 # Let's assume valid problem instances don't have goal locations that are isolated
                 # and not part of the road network unless the package starts there and is already at goal.
                 # The current logic handles the unreachable case correctly if the goal is *different*
                 # from the isolated current location. If current == goal and isolated,
                 # the initial check handles the 'on ground' case. If 'in vehicle' at isolated goal,
                 # the distance lookup will fail or return 0, leading to cost 1 (unload). This seems reasonable.
                 pass


            # Calculate the shortest distance from the current effective location to the goal location
            # Use get with default float('inf') if path doesn't exist.
            dist = self.distances.get((current_effective_location, goal_location), float('inf'))

            if dist == float('inf'):
                # Goal location is unreachable from the package's current effective location
                return float('inf')

            # Estimate actions needed for this package based on its current state (in vehicle or on ground)
            if is_in_vehicle:
                # Package is in a vehicle: needs drive actions (distance) + unload (1)
                cost_for_package = 1 + dist
            else:
                # Package is on the ground: needs load (1) + drive actions (distance) + unload (1)
                cost_for_package = 2 + dist

            total_cost += cost_for_package

        return total_cost
