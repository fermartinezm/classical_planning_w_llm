from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts gracefully, though typically states are well-formed
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 locationA)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    that is not yet at its goal location. It sums the estimated costs for each
    such package independently, ignoring vehicle capacity and the specific
    vehicle used. The cost for a package includes the travel distance (number
    of drive actions) plus the necessary pick-up and drop actions.

    # Assumptions
    - Vehicle capacity constraints are ignored. Any vehicle can carry any package.
    - The availability and location of specific vehicles are simplified. It assumes
      a vehicle is available where needed to pick up a package and can travel
      directly to the destination.
    - The heuristic sums costs for packages independently, ignoring potential
      synergies (e.g., one vehicle transporting multiple packages).
    - Road network is static and bidirectional (inferred from example, domain doesn't enforce bidirectionality but typical for transport). If roads are strictly unidirectional, the BFS should only add the outgoing edge. Assuming bidirectional for simplicity based on example.

    # Heuristic Initialization
    - The road network is extracted from static facts to build a graph of locations.
    - All-pairs shortest path distances between locations are precomputed using BFS.
    - The goal location for each package is extracted from the task goals.
    - Packages and vehicles are identified from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. Identify the current status of each package that has a goal location:
       - Is the package on the ground at a specific location `l_current`?
       - Is the package inside a vehicle `v`? If so, find the current location `l_v` of that vehicle.

    2. For each package `p` that is not yet at its goal location `l_goal`:
       - If `p` is on the ground at `l_current`:
         - The estimated cost for this package is the shortest distance from `l_current` to `l_goal` (representing drive actions) plus 2 actions (1 for pick-up, 1 for drop).
       - If `p` is inside a vehicle `v` which is currently at `l_v`:
         - The estimated cost for this package is the shortest distance from `l_v` to `l_goal` (representing drive actions) plus 1 action (1 for drop). A pick-up is not needed as it's already in a vehicle.

    3. Sum the estimated costs for all packages that are not at their goal location.

    4. If any package's goal location is unreachable from its current location (or its vehicle's location), the heuristic returns infinity, indicating an unsolvable state.

    5. If all packages are at their goal locations, the heuristic returns 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal locations.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages from goals
        self.packages = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming 'at' goals for packages
                package, location = args
                self.goal_locations[package] = location
                self.packages.add(package)

        # Build the road network graph and identify all locations.
        self.locations = set()
        self.road_graph = {}  # Adjacency list: location -> [neighbor locations]

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                # Assuming roads are bidirectional based on typical transport domains
                self.road_graph.setdefault(l2, []).append(l1)

        # Identify vehicles from initial state (assuming anything 'at' a location
        # that is not a package we care about is a vehicle).
        self.vehicles = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj = parts[1]
                if obj not in self.packages:
                    self.vehicles.add(obj)
            # Also consider objects with capacity as vehicles
            elif parts[0] == "capacity":
                 vehicle = parts[1]
                 self.vehicles.add(vehicle)


        # Compute all-pairs shortest paths using BFS from each location.
        self.distances = {}  # distances[l1][l2] = shortest path distance
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)]) # Use deque for efficient BFS
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_loc][current_loc] = dist
                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value).
        """
        state = node.state  # Current world state (frozenset of facts).

        # Track the current status of packages and vehicles.
        # package_status: package -> {'type': 'at'/'in', 'location': loc/vehicle}
        package_status = {}
        # vehicle_locations: vehicle -> location
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_status[obj] = {'type': 'at', 'location': loc}
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                if package in self.packages:
                    package_status[package] = {'type': 'in', 'location': vehicle}

        total_cost = 0  # Initialize action cost counter.

        # Check if the current state is a goal state.
        # If all goal facts are in the state, the heuristic is 0.
        # This check is technically redundant if the loop below correctly calculates 0
        # when all packages are at their goals, but it's a clear way to ensure h=0 at goal.
        if self.goals <= state:
             return 0

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # If the package is not mentioned in the current state (e.g., not in initial state
            # and not a goal package), skip it. This shouldn't happen in valid problems.
            if package not in package_status:
                 # This package is not in the state, likely not relevant or an error.
                 # For robustness, we could return infinity, but assuming valid states.
                 continue

            status = package_status[package]
            current_type = status['type']
            current_loc_or_vehicle = status['location']

            # Check if the package is already at its goal location.
            is_at_goal = False
            if current_type == 'at' and current_loc_or_vehicle == goal_location:
                 is_at_goal = True
            # Note: If a package is 'in' a vehicle at the goal location, it's *not* yet at the goal (at predicate).

            if not is_at_goal:
                # Package is not yet at its goal. Calculate cost to move it.
                if current_type == 'at':
                    # Package is on the ground at current_location.
                    current_location = current_loc_or_vehicle
                    # Cost: Pick-up (1) + Drive (distance) + Drop (1)
                    drive_cost = self.distances.get(current_location, {}).get(goal_location, math.inf)

                    if drive_cost == math.inf:
                        # Goal is unreachable from the package's current location.
                        return math.inf

                    total_cost += drive_cost + 2 # 1 for pick-up, 1 for drop

                elif current_type == 'in':
                    # Package is in a vehicle. Find the vehicle's location.
                    vehicle = current_loc_or_vehicle
                    vehicle_location = vehicle_locations.get(vehicle)

                    if vehicle_location is None:
                        # Vehicle location not found in state. This indicates an invalid state
                        # or a vehicle not included in the initial 'at' facts.
                        # For robustness, return infinity.
                        return math.inf

                    # Cost: Drive (distance) + Drop (1)
                    drive_cost = self.distances.get(vehicle_location, {}).get(goal_location, math.inf)

                    if drive_cost == math.inf:
                        # Goal is unreachable from the vehicle's current location.
                        return math.inf

                    total_cost += drive_cost + 1 # 1 for drop

        # The loop only adds cost for packages *not* at their goal.
        # If all packages are at their goal, total_cost remains 0.
        # If any package goal is unreachable, we returned infinity earlier.
        return total_cost

