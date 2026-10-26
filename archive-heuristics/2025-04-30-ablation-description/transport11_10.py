from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define dummy Heuristic base class if not provided by the environment
# This allows the code to be runnable standalone for syntax checks,
# but the actual planner environment must provide the base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location,
    summing the costs for all packages independently. It uses precomputed shortest
    path distances on the road network for the drive cost.

    # Assumptions
    - The heuristic ignores vehicle capacity constraints.
    - The heuristic assumes vehicles are always available to pick up or drop packages
      at their current location.
    - The cost of each action (pick-up, drive, drop) is assumed to be 1.
    - The road network is static and bidirectional (if road l1 l2 exists, road l2 l1 exists).

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goals.
    - Build the road network graph from the static 'road' facts.
    - Compute all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location on the ground:
    1. Determine the package's current ground location. This is either directly
       given by an `(at package location)` fact, or indirectly by an `(in package vehicle)`
       fact combined with an `(at vehicle location)` fact.
    2. If the package is already at its goal location *inside* a vehicle, it only
       needs to be dropped (cost 1).
    3. If the package is at a location different from its goal location:
       - Calculate the shortest distance (minimum number of drive actions) between
         the package's current ground location and its goal location using the
         precomputed distances.
       - If the package is currently on the ground: It needs a pick-up (cost 1),
         the necessary drives (cost = distance), and a drop (cost 1). Total: 1 + distance + 1.
       - If the package is currently inside a vehicle: It needs the necessary drives
         (cost = distance) and a drop (cost 1). Total: distance + 1.
    4. Sum the estimated costs for all packages.
    5. If any package's goal location is unreachable from its current location via
       the road network, the heuristic value is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances on the road network.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goals are typically (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.package_goals[package] = location
            # Assuming only (at package location) goals for packages.

        # 2. Build the road network graph.
        self.road_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.package_goals.values():
             self.all_locations.add(loc)
             self.road_graph.setdefault(loc, []) # Add location with empty adjacency list if not in road facts

        # 3. Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in self.all_locations:
            self._bfs(start_loc)

    def _bfs(self, start_location):
        """Performs BFS from a start location to compute distances to all reachable locations."""
        distances_from_start = {loc: float('inf') for loc in self.all_locations}
        distances_from_start[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Get neighbors, handle locations not in road_graph keys but in all_locations
            neighbors = self.road_graph.get(current_loc, [])

            for neighbor in neighbors:
                if distances_from_start[neighbor] == float('inf'):
                    distances_from_start[neighbor] = distances_from_start[current_loc] + 1
                    queue.append(neighbor)

        # Store distances from this start location
        for end_loc, dist in distances_from_start.items():
            self.distances[(start_location, end_loc)] = dist

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Map locatable objects (packages, vehicles) to their direct predicate argument
        # (location string if 'at', or vehicle name string if 'in').
        locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, target = args
                locations[obj] = target

        total_cost = 0  # Initialize heuristic cost

        # Iterate through each package and its goal location
        for package, goal_location in self.package_goals.items():

            # Check if the package is already at the goal location on the ground
            if f"(at {package} {goal_location})" in state:
                # Package is already at the goal location on the ground. Cost is 0 for this package.
                continue

            # Package is not yet at the goal location on the ground.
            # Find its current status (location string or vehicle name).
            pkg_current_status = locations.get(package)

            # This should not happen in a valid state where all objects are located.
            # If it does, the state is malformed or the heuristic logic needs adjustment
            # for unexpected states. Returning infinity is a safe fallback.
            if pkg_current_status is None:
                 # print(f"Warning: Location of package {package} unknown in state.")
                 return float('inf') # Cannot estimate cost if package location is unknown

            # Determine the package's current ground location and if it's in a vehicle.
            pkg_ground_location = None
            package_is_in_vehicle = False

            if pkg_current_status in self.all_locations:
                # Package is on the ground at pkg_current_status.
                pkg_ground_location = pkg_current_status
                package_is_in_vehicle = False
            elif pkg_current_status in locations: # Check if status is a vehicle name whose location is known
                # Package is inside a vehicle. Find the vehicle's ground location.
                vehicle_name = pkg_current_status
                vehicle_current_location = locations.get(vehicle_name) # This gets the location string
                if vehicle_current_location is None:
                     # Vehicle location unknown. Should not happen in a valid state.
                     # print(f"Warning: Location of vehicle {vehicle_name} (carrying {package}) unknown.")
                     return float('inf') # Cannot estimate cost if vehicle location is unknown

                pkg_ground_location = vehicle_current_location
                package_is_in_vehicle = True
            else:
                 # pkg_current_status is neither a location nor a known vehicle with location.
                 # This indicates a state representation issue or an object type not handled.
                 # print(f"Warning: Package {package} status '{pkg_current_status}' is neither a location nor a known vehicle.")
                 return float('inf') # Cannot estimate cost

            # Now we know the package's current ground location (pkg_ground_location)
            # and its goal location (goal_location).

            # Calculate the shortest distance (drive actions) needed.
            dist = self.distances.get((pkg_ground_location, goal_location), float('inf'))

            # If the goal location is unreachable from the current location, return infinity.
            if dist == float('inf'):
                return float('inf')

            # Estimate cost for this package based on its state and distance.
            if pkg_ground_location == goal_location:
                 # Package is at the goal location, but not on the ground (must be in vehicle).
                 # Needs 1 drop action.
                 total_cost += 1
            elif package_is_in_vehicle:
                # Package is in a vehicle, not at the goal location.
                # Needs 'dist' drive actions + 1 drop action.
                total_cost += dist + 1
            else: # Package is on the ground, not at the goal location.
                # Needs 1 pick-up action + 'dist' drive actions + 1 drop action.
                total_cost += 1 + dist + 1

        return total_cost
