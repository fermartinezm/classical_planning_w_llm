# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic
# If running standalone for testing, define a dummy base class
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


from fnmatch import fnmatch
from collections import deque, defaultdict
import math # For math.inf

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match_exact(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with an exact number of arguments.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated cost for each package that
    is not yet at its destination. The cost for a package depends on whether
    it is currently on the ground or inside a vehicle, and the shortest path
    distance in the road network from its current location (or its vehicle's
    location) to its goal location.

    # Assumptions:
    - Each package needs to reach a specific goal location specified in the task goals.
    - Capacity constraints of vehicles are ignored. It is assumed that a suitable
      vehicle is available or can be made available when needed to pick up/drop
      a package.
    - Reachability between locations is determined solely by the static 'road'
      facts; other constraints (like vehicle type, size compatibility, etc.)
      are ignored for distance calculation.
    - The cost of a 'drive' action is 1 per road segment traversed (shortest path distance).
    - The cost of 'pick-up' and 'drop' actions is 1 each.
    - All goal packages are assumed to be locatable objects present in the initial state
      or subsequent states via 'at' or 'in' facts.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representation of the road network from static 'road' facts.
    - Collects all relevant locations mentioned in static facts, initial state,
      and goals to ensure the road network graph covers all necessary nodes.
    - Computes shortest path distances between all pairs of these relevant locations
      using Breadth-First Search (BFS). These distances are stored for quick lookup
      during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (represented as a frozenset of facts):
    1. Parse the state to identify the current location of every locatable object
       (packages and vehicles) using '(at obj loc)' facts.
    2. Parse the state to identify which packages are currently inside which vehicles
        using '(in pkg veh)' facts.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each package that has a specified goal location according
       to the task goals:
       a. Check if the package is already at its goal location in the current state.
          If the fact '(at package goal_location)' is present, the cost for this
          package is 0; proceed to the next package.
       b. If the package is not at its goal, determine its current status:
          i. If the package is currently inside a vehicle (found in the
             package-in-vehicle mapping):
             - Get the name of the vehicle.
             - Find the current location of that vehicle from the
               current-locations mapping.
             - Calculate the shortest path distance from the vehicle's current
               location to the package's goal location using the pre-computed
               distances.
             - If the goal location is unreachable from the vehicle's location
               via the road network (distance is infinity), the state is
               considered to lead to an unsolvable path for this package;
               return infinity for the total heuristic.
             - The estimated cost for this package is the calculated drive
               distance plus 1 action for the 'drop' operation.
          ii. If the package is currently on the ground (found in the
              current-locations mapping but not in the package-in-vehicle mapping)
              and its location is not the goal location:
              - Get the package's current location.
              - Calculate the shortest path distance from the package's current
                location to its goal location.
              - If the goal location is unreachable from the package's location,
                return infinity for the total heuristic.
              - The estimated cost for this package is 1 action for 'pick-up',
                plus the calculated drive distance, plus 1 action for 'drop'.
          iii. If the package is a goal package but is not found in either the
               current-locations or package-in-vehicle mappings, this indicates
               an unexpected state (e.g., package missing); return infinity.
       c. Add the estimated cost for the current package to the total heuristic cost.
    5. After processing all goal packages, return the accumulated total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, road network,
        and computing shortest path distances.
        """
        super().__init__(task) # Call base class constructor if needed

        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match_exact(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location

        self.roads = defaultdict(list)
        locations = set()

        # Collect locations and build road graph from static facts
        for fact in self.static:
            if match_exact(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.roads[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Collect locations from initial state (packages and vehicles)
        for fact in self.initial_state:
             if match_exact(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 locations.add(loc)
             # 'in' facts don't give location directly, only vehicle location matters

        # Ensure all goal locations are included, even if not in roads/initial state
        locations.update(self.goal_locations.values())

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_l in locations:
            self.distances[start_l] = self._bfs(start_l) # BFS only needs graph, not all locations list

    def _bfs(self, start_node):
        """
        Performs BFS from start_node on the road network graph to find distances
        to all reachable locations.
        Returns a dictionary {location: distance}.
        Unreachable locations from start_node are not included in the result.
        """
        queue = deque([(start_node, 0)])
        visited = {start_node}
        distances = {start_node: 0}

        while queue:
            current_node, dist = queue.popleft()

            # Get neighbors from the road graph
            # Use .get() with empty list default for nodes with no outgoing roads
            neighbors = self.roads.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return distances

    def get_distance(self, l1, l2):
        """
        Returns the pre-computed shortest distance between l1 and l2.
        Returns float('inf') if l2 is unreachable from l1 or if l1 is not a
        known location from which distances were computed.
        """
        if l1 == l2:
            return 0
        # Check if l1 is a known start location for BFS and l2 was reached from l1
        if l1 in self.distances and l2 in self.distances[l1]:
            return self.distances[l1][l2]
        else:
            # l2 is unreachable from l1 in the road network, or l1 is not a valid location
            return math.inf

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Track current locations of locatable objects (packages and vehicles)
        current_locations = {} # {object_name: location_name}
        # Track which package is inside which vehicle
        package_in_vehicle = {} # {package_name: vehicle_name}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty or malformed facts
                continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1:]
                current_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1:]
                package_in_vehicle[pkg] = veh

        total_cost = 0

        # Iterate through all packages that have a goal location
        for package, goal_l in self.goal_locations.items():
            package_cost = 0

            # Check if package is already at its goal location
            # A package is at its goal if the fact (at package goal_l) is in the state
            if package in current_locations and current_locations[package] == goal_l:
                continue # Package is at goal, cost is 0 for this package

            # Determine package's current status (in vehicle or on ground)
            if package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find the vehicle's location
                if vehicle not in current_locations:
                    # Vehicle location unknown - indicates invalid state or parsing issue
                    # Treat as unreachable goal for this package
                    return math.inf # Return infinity for the whole state

                vehicle_l = current_locations[vehicle]

                # Cost: Drive vehicle from vehicle_l to goal_l + Drop package
                drive_cost = self.get_distance(vehicle_l, goal_l)
                if drive_cost == math.inf:
                    # Goal is unreachable from vehicle location
                    return math.inf # Problem likely unsolvable from this state

                package_cost = drive_cost + 1 # 1 for drop action

            elif package in current_locations:
                # Package is on the ground, not at goal
                current_l = current_locations[package]

                # Cost: Pick up package + Drive from current_l to goal_l + Drop package
                drive_cost = self.get_distance(current_l, goal_l)
                if drive_cost == math.inf:
                    # Goal is unreachable from package location
                    return math.inf # Problem likely unsolvable from this state

                package_cost = 1 + drive_cost + 1 # 1 for pick, 1 for drop

            else:
                # Package is a goal package but not found in 'at' or 'in' facts.
                # This suggests an invalid state representation or a package that
                # somehow disappeared. Treat as unreachable goal.
                return math.inf # Problem likely unsolvable from this state

            total_cost += package_cost

        return total_cost
