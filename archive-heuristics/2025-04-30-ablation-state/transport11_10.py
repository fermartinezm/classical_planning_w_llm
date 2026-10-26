import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe return empty list or raise error
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current state (on the ground or in a vehicle)
    to its goal location. It sums the estimated costs for each package independently,
    ignoring vehicle capacity and coordination.

    # Assumptions
    - The goal is defined by (at package location) facts.
    - Any location relevant to package goals is reachable from any other relevant location
      via the road network (or the heuristic returns infinity).
    - A suitable vehicle is always available when a package needs to be picked up.
    - Vehicle capacity constraints are ignored.
    - The cost of getting a vehicle to a package's location for pick-up is ignored.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build a graph of locations based on the 'road' facts.
    - Compute all-pairs shortest paths between locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Initialize the total heuristic cost to 0.
    2. Build a map of the current location or container for all locatable objects (packages and vehicles) in the state.
    3. For each package that has a goal location specified:
       a. Check if the package is already at its goal location (i.e., the state contains `(at package goal_location)`). If yes, the cost for this package is 0, continue to the next package.
       b. If the package is not at its goal, determine its current physical location. This is either the location where it is `at`, or the location of the vehicle it is `in`. If the package or its vehicle is not located, return infinity.
       c. Estimate the minimum actions needed for this package based on its current state and physical location:
          - If the package is currently on the ground at location L_curr (and L_curr is not the goal): It needs a pick-up (1 action), driving from L_curr to the goal location (cost is the shortest path distance), and a drop (1 action). Total: 1 + distance(L_curr, L_goal) + 1.
          - If the package is currently inside a vehicle V, and V is at location L_v (and L_v is not the goal): It needs driving from L_v to the goal location (cost is the shortest path distance), and a drop (1 action). Total: distance(L_v, L_goal) + 1.
          - If the package is currently inside a vehicle V, and V is at the goal location L_goal: It needs a drop (1 action). Total: 1.
       d. If the required driving path is unreachable (distance is infinite), return infinity for the total heuristic.
       e. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        shortest paths between all locations based on road facts.
        """
        # Assuming task object has attributes: goals, static
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically in the form (at package location)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3: # Ensure it's (at ?pkg ?loc)
                    _, package, location = parts
                    self.package_goals[package] = location

        # Build the location graph from static road facts.
        self.locations = set()
        graph = {} # Adjacency list: {location: [neighbor1, neighbor2, ...]}

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (road ?l1 ?l2)
                    _, l1, l2 = parts
                    self.locations.add(l1)
                    self.locations.add(l2)
                    if l1 not in graph:
                        graph[l1] = []
                    graph[l1].append(l2)

        # Ensure all locations mentioned in goals are included, even if isolated
        for loc in self.package_goals.values():
             self.locations.add(loc)
             if loc not in graph:
                 graph[loc] = [] # Add isolated locations to graph structure

        # Compute all-pairs shortest paths using BFS.
        self.distances = {} # {(start_loc, end_loc): distance}

        for start_loc in self.locations:
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while queue:
                curr_loc, dist = queue.popleft()

                # Get neighbors from the graph, handle locations with no outgoing roads
                neighbors = graph.get(curr_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Build a map of current locations/containers for all locatables (packages and vehicles).
        current_state_map = {} # {locatable: location_or_vehicle}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                _, obj, loc = parts
                current_state_map[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                _, package, vehicle = parts
                current_state_map[package] = vehicle # Package is *in* vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that need to reach a goal location.
        for package, goal_location in self.package_goals.items():

            # Check if the package is already at the goal location.
            # This is the exact goal condition for a package.
            if f"(at {package} {goal_location})" in state:
                # Package is already at its goal. Cost is 0 for this package.
                continue

            # Package is not at its goal location. Estimate cost to move it.

            # Find the package's current state (at location or in vehicle).
            package_current_container = current_state_map.get(package)

            if package_current_container is None:
                # Package is not mentioned in 'at' or 'in' facts. Invalid state?
                # Treat as unreachable goal for this package.
                return float('inf')

            # Determine if the package is in a vehicle or on the ground.
            # A container is a vehicle if it itself is a locatable object (i.e., has an entry in current_state_map, specifically an 'at' location).
            is_in_vehicle = package_current_container in current_state_map # Check if the container name is a known locatable (vehicle)

            package_physical_location = None
            if is_in_vehicle:
                vehicle = package_current_container
                # Find the physical location of the vehicle.
                package_physical_location = current_state_map.get(vehicle)
                if package_physical_location is None:
                    # Vehicle exists but is not 'at' any location. Invalid state?
                    return float('inf')
            else:
                # Package is not in a vehicle, its container is its physical location.
                package_physical_location = package_current_container

            # Now we have the package's current physical location (package_physical_location)
            # and its goal location (goal_location).
            # Estimate the cost to get the package from its current state to (at package goal_location).

            cost_this_package = 0

            # Case A: Package is on the ground at L_curr, L_curr != L_goal (handled by initial check)
            if not is_in_vehicle:
                current_location = package_physical_location # This is L_curr
                drive_cost = self.distances.get((current_location, goal_location))
                if drive_cost is None:
                    # Goal is unreachable from package's current location by driving.
                    return float('inf')
                cost_this_package = 1 + drive_cost + 1 # Pick + Drive + Drop

            # Case B: Package is in vehicle at L_v, L_v != L_goal
            elif package_physical_location != goal_location:
                vehicle_location = package_physical_location # This is L_v
                drive_cost = self.distances.get((vehicle_location, goal_location))
                if drive_cost is None:
                    # Goal is unreachable from vehicle's current location by driving.
                    return float('inf')
                cost_this_package = drive_cost + 1 # Drive + Drop

            # Case C: Package is in vehicle at L_v, L_v == L_goal
            else: # is_in_vehicle is True and package_physical_location == goal_location
                cost_this_package = 1 # Drop

            total_cost += cost_this_package

        return total_cost
