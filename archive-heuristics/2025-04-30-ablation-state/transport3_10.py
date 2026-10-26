# Assuming heuristics.heuristic_base provides a base class named Heuristic
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts (optional, but useful)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation
def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    graph: adjacency list representation {node: [neighbor1, neighbor2, ...]}
    start_node: the node to start BFS from
    Returns: dictionary {node: distance}
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle cases where start_node is not in the graph
         return distances # All distances remain inf

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Should always be true if start_node was in graph
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


# Define the heuristic class
# Assuming Heuristic base class is available and imported elsewhere,
# or this class is used directly.
# If inheriting is strictly required for the environment, uncomment the import
# and the class definition line below and ensure heuristics.heuristic_base exists.
# from heuristics.heuristic_base import Heuristic
# class transportHeuristic(Heuristic):
class transportHeuristic:

    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move packages
    to their goal locations. It calculates a minimum cost for each package
    independently, summing these costs. The cost for a package depends on
    whether it's on the ground or in a vehicle, and the shortest road distance
    to its goal location.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Capacity constraints are ignored.
    - Vehicle availability for picking up packages on the ground is ignored
      (assumes a vehicle is available at the package's location when needed).
    - The shortest path between locations is always usable by a vehicle.
    - The heuristic sums individual package costs, potentially overestimating
      due to shared vehicle usage, but aiming for better guidance than simpler
      options (like ignoring distances).
    - If a location is unreachable from the package's current location (or vehicle's location),
      the heuristic returns infinity for the state, indicating it's likely unsolvable.

    # Heuristic Initialization
    - Parses goal facts to map each package to its goal location.
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each package `p` that has a goal location `l_goal` defined in the task goals:
       a. Determine the current status of package `p` in the current state:
          - Check if `(at p l_goal)` is a fact in the state. If yes, this package is at its goal on the ground; cost for this package is 0. Continue to the next package.
          - If not at the goal on the ground, check if `(at p l_current)` is a fact for some `l_current`. If yes, `p` is on the ground at `l_current`.
          - If not on the ground, check if `(in p v)` is a fact for some vehicle `v`. If yes, `p` is inside vehicle `v`. Find the location `l_v` of vehicle `v` by looking for `(at v l_v)` in the state. The package is effectively at `l_v`.
          - If the package's location/containment cannot be determined from the state facts (shouldn't happen for packages in goals), the state is malformed or unreachable; return infinity.
       b. If the package is not at its goal location on the ground, calculate the minimum estimated cost to get it there:
          - If `p` is on the ground at `l_current` (`l_current != l_goal`):
            - The cost involves picking it up (1 action), driving from `l_current` to `l_goal` (distance actions), and dropping it (1 action).
            - Get the shortest distance `d = self.distances.get(l_current, {}).get(l_goal, float('inf'))`. If `d` is infinity, the goal is unreachable from `l_current`, so the state is unsolvable; return infinity.
            - Cost for this package = 1 (pick) + `d` (drive) + 1 (drop).
          - If `p` is in vehicle `v` which is at `l_v`:
            - If `l_v == l_goal`: The package is in the vehicle at the goal location. The cost is just dropping the package (1 action).
            - If `l_v != l_goal`: The package is in the vehicle elsewhere. The cost involves driving from `l_v` to `l_goal` (distance actions) and dropping it (1 action).
            - Get the shortest distance `d = self.distances.get(l_v, {}).get(l_goal, float('inf'))`. If `d` is infinity, the goal is unreachable from `l_v`; return infinity.
            - Cost for this package = `d` (drive) + 1 (drop).
       c. Add the calculated cost for package `p` to the total heuristic cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the location graph from road facts.
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                l1, l2 = args
                locations.add(l1)
                locations.add(l2)
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                self.location_graph[l1].append(l2)

        # Ensure all locations mentioned in goals or roads are in the graph dictionary,
        # even if they have no outgoing roads. This prevents BFS from crashing
        # if a location is a start/goal but has no road facts involving it.
        all_relevant_locations = locations.union(set(self.goal_locations.values()))
        for loc in all_relevant_locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []


        # Compute all-pairs shortest paths.
        self.distances = {}
        for start_loc in self.location_graph:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Map locatable objects (packages, vehicles) to their current location if 'at'
        current_at_locations = {}
        # Map packages to the vehicle they are 'in'
        package_in_vehicle = {}

        # Build current state maps for quick lookup
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args # obj is locatable (vehicle or package)
                current_at_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args # package is in vehicle
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Consider each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if package in current_at_locations and current_at_locations[package] == goal_location:
                 continue # Package is done

            # Package is not at the goal location on the ground. Where is it?
            package_current_loc = None
            is_in_vehicle = False

            if package in current_at_locations:
                 # Package is on the ground somewhere
                 package_current_loc = current_at_locations[package]
                 is_in_vehicle = False
            elif package in package_in_vehicle:
                 # Package is in a vehicle. Find the vehicle's location.
                 vehicle = package_in_vehicle[package]
                 if vehicle not in current_at_locations:
                     # Vehicle location unknown - malformed state? Assume unreachable.
                     # A package can only be 'in' a vehicle if the vehicle exists and is 'at' a location.
                     # This indicates a potentially invalid state representation.
                     return float('inf')
                 package_current_loc = current_at_locations[vehicle] # Package is effectively at vehicle's location
                 is_in_vehicle = True
            else:
                 # Package is neither 'at' nor 'in' - malformed state? Assume unreachable.
                 # This could happen if a package exists but its location is not defined.
                 return float('inf')

            # Now calculate cost based on package_current_loc and is_in_vehicle
            if package_current_loc == goal_location:
                 # Package is at the goal location, but not on the ground (must be in a vehicle)
                 # It only needs to be dropped.
                 total_cost += 1
            else:
                 # Package is not at the goal location. It needs to be moved.
                 # Get the shortest distance from its current effective location to the goal.
                 # Handle cases where the location might not be in the pre-computed distances (e.g., malformed state)
                 if package_current_loc not in self.distances or goal_location not in self.distances.get(package_current_loc, {}):
                      # This case should ideally be caught by the BFS handling of start_node not in graph,
                      # but adding a check here is safer against unexpected data.
                      return float('inf')

                 drive_cost = self.distances[package_current_loc][goal_location]

                 if drive_cost == float('inf'):
                     # Goal is unreachable from the package's current effective location
                     return float('inf')

                 if is_in_vehicle:
                     # Package is in a vehicle, needs drive + drop
                     total_cost += drive_cost + 1
                 else:
                     # Package is on the ground, needs pick + drive + drop
                     total_cost += 1 + drive_cost + 1

        # The total_cost is the sum of costs for packages *not* yet at their goal on the ground.
        # If total_cost is 0, it means all packages listed in goal_locations were found
        # to be at their goal locations on the ground. This is exactly the goal condition.
        # So, h=0 iff goal state is reached.

        return total_cost
